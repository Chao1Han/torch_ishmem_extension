# Owner(s): ["oncall: distributed"]
# To run:
# python test/distributed/test_nvshmem_triton.py

import os
import sys
import triton.language as tl

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch._inductor.runtime.triton_compat import triton
from _ishmem_triton import requires_nvshmem
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

#initialize

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'

os.environ['TORCH_SYMMMEM'] = 'ISHMEM'
dist.init_process_group(backend="xccl")
os.environ['ZE_AFFINITY_MASK'] = str(dist.get_rank())


# So that tests are written in device-agnostic way
device_type = "xpu"
device_module = torch.get_device_module(device_type)

# @requires_nvshmem
@triton.jit
def my_get_kernel(
    dest,
    src,
    nelems,
    pe,
    nbi: tl.constexpr,  # use nonblocking interface if True
):
    # ishmem.get(dest, src, nelems, pe)

    tl.device_print("In get triton kernel")
    
    # if nbi:
    #     ishmem.get_nbi(dest, src, nelems, pe)
    #     ishmem.quiet()
    # else:
    #     ishmem.get(dest, src, nelems, pe)


class ISHMEMTritonTest():
    def __init__(self):
        # ISHMEM specific environment setup
        self.rank = dist.get_rank()
        device_module.set_device(self.device)

    def _init_device(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device(device_type, self.rank)


    # def test_triton_put(self) -> None:
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

    def test_triton_get(self, nbi: bool = False) -> None:
        torch.manual_seed(42 + self.rank)
        self._init_device()

        group_name = dist.distributed_c10d._get_default_group().group_name
        rank = self.rank

        # Configuration
        numel = 8
        dtype = torch.int8
        val = 7

        print("Start to create symmetric tensors \n", flush=True)
        inp = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(
            val if rank == 0 else -1
        )
        out = symm_mem.empty(numel, dtype=dtype, device=self.device).fill_(-1)
        
        print("Start to rendezvous \n", flush=True)
        symm_mem.rendezvous(inp, group=group_name)
        symm_mem.rendezvous(out, group=group_name)

        dist.barrier()
        torch.xpu.synchronize() # just check
        print("Start to call triton kernel \n", flush=True)
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
        # if rank == 1:
        #     torch.testing.assert_close(
        #         out, val * torch.ones(numel, dtype=dtype, device=self.device)
        #     )
        dist.barrier()
        print("Done to test_triton_get \n", flush=True)

def main():
    test = ISHMEMTritonTest()
    try:
        test.test_triton_get()
        # test.test_triton_put()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()