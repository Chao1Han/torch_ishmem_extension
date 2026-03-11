# Owner(s): ["oncall: distributed"]
# To run:
# python test/distributed/test_nvshmem_triton.py

import sys
import triton.language as tl

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._inductor.runtime.triton_compat import triton
# from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import IS_H100, requires_triton
import _ishmem_triton as ishmem


def requires_h100():
    return skip_but_pass_in_sandcastle_if(
        not IS_H100,
        "NVSHMEM requires H100. Skipping test on non-H100 GPU.",
    )


# So that tests are written in device-agnostic way
device_type = "xpu"
device_module = torch.get_device_module(device_type)


# Shared Triton JIT kernels

@requires_nvshmem
@triton.jit
def my_put_kernel(
    dest,
    src,
    nelems,
    pe,
):
    ishmem.put(dest, src, nelems, pe)


@requires_nvshmem
@triton.jit
def my_get_kernel(
    dest,
    src,
    nelems,
    pe,
    nbi: tl.constexpr,  # use nonblocking interface if True
):
    if nbi:
        ishmem.get_nbi(dest, src, nelems, pe)
        ishmem.quiet()
    else:
        ishmem.get(dest, src, nelems, pe)


@requires_nvshmem
@triton.jit
def my_putmem_signal_block_kernel(
    dst,
    src,
    size_bytes,
    signal,
    sig_val,
    sig_op,
    peer,
):
    ishmem.putmem_signal_block(dst, src, size_bytes, signal, sig_val, sig_op, peer)


@requires_nvshmem
@triton.jit
def my_signal_wait_until_kernel(signal, cmp_op, cmp_val):
    ishmem.signal_wait_until(signal, cmp_op, cmp_val)


@requires_nvshmem
@triton.jit
def my_signal_op_kernel(
    sig_addr,
    signal,
    sig_op,
    peer,
):
    ishmem.signal_op(sig_addr, signal, sig_op, peer)


@requires_nvshmem
@triton.jit
def my_wait_until_kernel(
    ivar,
    cmp_op,
    cmp_val,
):
    ishmem.wait_until(ivar, cmp_op, cmp_val)


@requires_nvshmem
@triton.jit
def my_fence_kernel():
    ishmem.fence()


@requires_nvshmem
@triton.jit
def my_put_with_fence_kernel(
    dst1,
    src1,
    dst2,
    src2,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    # First put
    ishmem.put(dst1, src1, nelems, peer)
    # Ensure the first put is ordered before the next.
    ishmem.fence()
    # Second put
    ishmem.put(dst2, src2, nelems, peer)
    # Order the second put before flag update.
    ishmem.fence()
    # Write the flag (single int64) to signal completion.
    ishmem.put(flag_dst, flag_src, 1, peer)


@requires_nvshmem
@triton.jit
def my_put_with_quiet_kernel(
    dst,
    src,
    flag_dst,
    flag_src,
    nelems,
    peer,
):
    # Put data
    ishmem.put(dst, src, nelems, peer)
    # Call quiet to ensure put is complete
    ishmem.quiet()
    # Only after quiet, set the completion flag
    # This ensures the data put is complete before flag is set
    ishmem.put(flag_dst, flag_src, 1, peer)


@requires_nvshmem
@triton.jit
def my_barrier_test_kernel(
    dst,
    src,
    nelems,
):
    # Testing barrier_all() requires coordinated operations across PEs within
    # the same kernel execution. Unlike other kernels that just wrap NVSHMEM
    # primitives, this one implements the full test logic to properly verify
    # device-side barrier synchronization.
    my_pe = ishmem.my_pe()
    n_pes = ishmem.n_pes()

    # Rank 0 broadcasts its value to all other ranks
    if my_pe == 0:
        # Write initial value
        p_src = src.to(tl.pointer_type(tl.int32))
        tl.store(p_src, 42)
        # Put to all other ranks
        i = 1
        while i < n_pes:
            ishmem.put(dst, src, nelems, i)
            i += 1

    # Synchronize all PEs
    ishmem.barrier_all()

    # Non-zero ranks increment the received value
    if my_pe != 0:
        p_dst = dst.to(tl.pointer_type(tl.int32))
        received = tl.load(p_dst)
        tl.store(p_dst, received + 1)


@requires_nvshmem
@triton.jit
def my_barrier_all_kernel():
    ishmem.barrier_all()


@requires_nvshmem
@triton.jit
def my_sync_test_kernel(
    local_data,
    remote_data,
    nelems,
):
    my_pe = ishmem.my_pe()
    n_pes = ishmem.n_pes()

    # Each PE writes a unique value to its local memory
    p_local = local_data.to(tl.pointer_type(tl.int32))
    unique_value = my_pe + 100  # PE 0 writes 100, PE 1 writes 101, etc.
    tl.store(p_local, unique_value)

    # sync_all() ensures local stores are visible to other PEs
    # but doesn't guarantee completion of any remote operations
    ishmem.sync_all()

    # Now each PE reads from the next PE's memory to verify visibility
    # PE 0 reads from PE 1, PE 1 reads from PE 2, ..., PE n-1 reads from PE 0
    next_pe = (my_pe + 1) % n_pes
    ishmem.get(remote_data, local_data, nelems, next_pe)

    # The get should now see the value that the next PE wrote locally
    # because sync_all() made those local stores visible


@requires_nvshmem
@triton.jit
def my_alltoall_kernel(
    team_handle,
    dst,
    src,
    nelems_per_pe,
):
    ishmem.alltoall(team_handle, dst, src, nelems_per_pe)


@requires_nvshmem
@triton.jit
def my_broadcast_kernel(
    team_handle,
    dst,
    src,
    nelems,
    pe_root,
):
    ishmem.broadcast(team_handle, dst, src, nelems, pe_root)


@requires_nvshmem
@triton.jit
def my_reduce_kernel(
    team_handle,
    dest_tensor,
    source_tensor,
    nreduce,
    operation: tl.constexpr,
):
    ishmem.reduce(team_handle, dest_tensor, source_tensor, nreduce, operation)

class ISHMEMTritonTest():
    def __init__(self):
        # ISHMEM specific environment setup
        os.environ['TORCH_SYMMMEM'] = 'ISHMEM'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend="xccl")
        rank = dist.rank
        os.environ['ZE_AFFINITY_MASK'] = str(rank)
        device_module.set_device(self.device)

    def _init_device(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)


    def test_triton_put(self) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        # Configuration
        nelems = 5  # number of elements to transfer
        dtype = torch.int64
        val = 42 + rank  # Each rank has different data

        # Create symmetric tensors
        src = symm_mem.empty(nelems, dtype=dtype, device=self.device)
        dst = symm_mem.empty(nelems, dtype=dtype, device=self.device).fill_(-999)

        # Fill source tensor with rank-specific pattern
        for i in range(nelems):
            src[i] = (
                val * 10 + i
            )  # Rank 0: [420, 421, 422, 423, 424], Rank 1: [430, 431, ...]

        # Rendezvous
        symm_mem.rendezvous(src, group=group_name)
        symm_mem.rendezvous(dst, group=group_name)

        # Synchronize before operation
        dist.barrier()

        peer = 1 - rank
        if rank == 0:
            # Rank 0 puts its data to Rank 1
            my_put_kernel[(1,)](
                dst,
                src,
                nelems,
                peer,
            )

        # Synchronize after operation
        dist.barrier()

        if rank == 1:
            # Verify that rank 1 received rank 0's data
            expected = [420 + i for i in range(nelems)]
            torch.testing.assert_close(
                dst, torch.tensor(expected, device=self.device, dtype=dtype)
            )

    def test_triton_get(self, nbi: bool = True) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        # Configuration
        numel = 8
        dtype = torch.int8
        val = 7

        # Create symmetric tensors
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        peer = 1 - rank
        if rank == 1:
            # Rank 1 gets data from rank 0 using tensor-aware API
            my_get_kernel[(1,)](
                out,
                inp,
                numel,
                peer,
                nbi=nbi,
            )
        if rank == 1:
            torch.testing.assert_close(
                out, val * torch.ones(numel, dtype=dtype, device=self.device)
            )

def main():
    ISHMEMTritonTest.__init__()
    ISHMEMTritonTest.test_triton_get()
    ISHMEMTritonTest.test_triton_put()

if __name__ == "__main__":
    main()