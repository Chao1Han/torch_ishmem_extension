#!/usr/bin/env python3
"""
Performance comparison: ISHMEM TokenDispatcher vs NaiveAll2AllManager

This script compares the performance of:
1. ISHMEM-based TokenDispatcher (using all_to_all_vdev_2d)
2. NaiveAll2AllManager style: broadcast (dispatch) and all_reduce (combine)

Both dispatchers use the SAME input tensors for fair comparison.

Usage:
    cd /home/sdp/hanchao/symm
    source env.sh
    mpirun -n 2 python torch_ishmem_extension/benchmark_dispatch_combine.py
    mpirun -n 4 python torch_ishmem_extension/benchmark_dispatch_combine.py
"""

import os
import time
import argparse

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
# # Get rank/world_size from MPI environment FIRST
# rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', os.environ.get('PMI_RANK', '0')))
# world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', os.environ.get('PMI_SIZE', '2')))

# ISHMEM specific environment setup
os.environ['TORCH_SYMMMEM'] = 'ISHMEM'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['ZE_AFFINITY_MASK'] = str(rank)

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


# =====================================================
# NaiveAll2AllManager - Broadcast/AllReduce based
# =====================================================
class NaiveAll2AllManager:
    """
    A naive implementation of all2all communication.
    It uses broadcast (dispatch) and all-reduce (combine).
    The main purpose is for testing and debugging.
    """

    def __init__(self, group=None):
        self.group = group
        self.rank = rank
        self.world_size = world_size

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        sizes: list[int],
    ) -> torch.Tensor:
        """
        Dispatch: Gather hidden_states from all ranks using broadcast.
        
        Input: hidden_states [local_tokens, hidden_dim]
        Output: gathered [total_tokens, hidden_dim]
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_dim = hidden_states.shape[1]
        total_tokens = sum(sizes)
        
        # Calculate cumulative offsets
        cu_tokens = [0]
        for s in sizes:
            cu_tokens.append(cu_tokens[-1] + s)
        
        # Pre-allocate output buffer
        buffer = torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
        buffer.zero_()  # Initialize to avoid undefined memory
        
        # Copy local data to buffer
        start_idx = cu_tokens[self.rank]
        end_idx = cu_tokens[self.rank + 1]
        local_size = sizes[self.rank]
        buffer[start_idx:end_idx, :].copy_(hidden_states[:local_size])
        
        # Sync before broadcast
        torch.xpu.synchronize()
        
        # Broadcast from each rank
        for idx in range(self.world_size):
            s = cu_tokens[idx]
            e = cu_tokens[idx + 1]
            dist.broadcast(buffer[s:e, :].contiguous(), src=idx, group=self.group)
        
        return buffer

    def combine(
        self,
        hidden_states: torch.Tensor,
        sizes: list[int],
    ) -> torch.Tensor:
        """
        Combine: Reduce hidden_states across all ranks using all_reduce + slice.
        
        Input: hidden_states [total_tokens, hidden_dim]
        Output: local_hidden [local_tokens, hidden_dim]
        """
        # Calculate cumulative offsets
        cu_tokens = [0]
        for s in sizes:
            cu_tokens.append(cu_tokens[-1] + s)
        
        start_idx = cu_tokens[self.rank]
        end_idx = cu_tokens[self.rank + 1]
        
        # AllReduce + slice
        all_hidden = hidden_states.clone()
        dist.all_reduce(all_hidden, group=self.group)
        output = all_hidden[start_idx:end_idx, :].contiguous()
        
        return output


# =====================================================
# ISHMEMDispatcher - ISHMEM all_to_all_vdev_2d based
# =====================================================
class ISHMEMDispatcher:
    """
    ISHMEM-based token dispatcher using all_to_all_vdev_2d.
    """

    def __init__(self, group_name: str, num_experts: int, max_tokens: int, 
                 hidden_dim: int, dtype: torch.dtype, device: torch.device):
        self.group_name = group_name
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.nsplits = num_experts * world_size
        self.align = 8
        
        # Pre-allocate symmetric memory buffers
        max_inp_numel = max_tokens
        max_out_numel = max_tokens * world_size
        
        self.inp_buffer = symm_mem.empty(max_inp_numel, hidden_dim, dtype=dtype, device=device)
        self.out_buffer = symm_mem.empty(max_out_numel, hidden_dim, dtype=dtype, device=device)
        self.combine_out = symm_mem.empty(max_inp_numel, hidden_dim, dtype=dtype, device=device)
        
        self.in_splits = symm_mem.empty(self.nsplits, dtype=torch.int64, device=device)
        self.out_splits_offsets = symm_mem.empty((2, self.nsplits), dtype=torch.int64, device=device)
        self.in_splits_offsets = symm_mem.empty((2, self.nsplits), dtype=torch.int64, device=device)
        self.out_splits_offsets_combine = symm_mem.empty((2, self.nsplits), dtype=torch.int64, device=device)
    
    def dispatch(
        self,
        hidden_states: torch.Tensor,
        splits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dispatch using all_to_all_vdev_2d.
        
        Input: hidden_states [local_tokens, hidden_dim]
               splits [num_experts * world_size] - tokens per (rank, expert) pair
        Output: gathered [total_recv_tokens, hidden_dim]
        """
        inp_numel = hidden_states.shape[0]
        
        # Copy input to symmetric buffer
        self.inp_buffer[:inp_numel].copy_(hidden_states)
        self.in_splits.copy_(splits)
        
        # Dispatch: rank-major -> expert-major
        torch.ops.ishmem_ext.all_to_all_vdev_2d(
            self.inp_buffer, self.out_buffer, 
            self.in_splits, self.out_splits_offsets, 
            self.group_name, self.align
        )
        
        # Get output size from out_splits_offsets
        torch.xpu.synchronize()
        out_numel = self.out_splits_offsets[1, -1].item()
        
        return self.out_buffer[:out_numel]
    
    def combine(self) -> torch.Tensor:
        """
        Combine using all_to_all_vdev_2d_offset.
        
        Returns: combined [local_tokens, hidden_dim]
        """
        # Copy offsets for combine
        self.in_splits_offsets.copy_(self.out_splits_offsets)
        
        # Combine: expert-major -> rank-major
        torch.ops.ishmem_ext.all_to_all_vdev_2d_offset(
            self.out_buffer, self.combine_out,
            self.in_splits_offsets, self.out_splits_offsets_combine,
            self.group_name
        )
        
        torch.xpu.synchronize()
        out_numel = self.out_splits_offsets_combine[1, -1].item()
        
        return self.combine_out[:out_numel]


def setup_distributed():
    """Setup distributed environment."""
    dist.init_process_group(backend="xccl", rank=rank, world_size=world_size)
    
    # Import ISHMEM extension after dist init
    import torch_ishmem_extension as ishmem_ext
    
    symm_mem.set_backend("ISHMEM")
    
    device = torch.device("xpu")
    
    # Sync all ranks before returning
    dist.barrier()
    
    return device


def run_benchmark(num_tokens, hidden_dim, num_experts, warmup=5, iterations=20):
    """Run benchmarks for given parameters."""
    device = setup_distributed()
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    
    dtype = torch.bfloat16
    nsplits = num_experts * world_size
    
    # Use SAME random seed across all ranks for consistent splits
    torch.manual_seed(42)  # Same seed for all ranks
    tokens_per_split = num_tokens // nsplits
    splits = torch.randint(
        max(1, tokens_per_split - tokens_per_split // 2),
        tokens_per_split + tokens_per_split // 2 + 1,
        (nsplits,), dtype=torch.int64, device=device
    )
    
    # Total tokens for this rank (same for all ranks now)
    local_tokens = splits.sum().item()
    max_tokens = local_tokens + 1000  # Some slack
    
    # Create input tensor (can use different data per rank, but same size)
    torch.manual_seed(12345 + rank)  # Different data per rank
    input_tensor = torch.randn(local_tokens, hidden_dim, dtype=dtype, device=device)
    
    # sizes for NaiveAll2AllManager (per-rank token counts - now same for all ranks)
    sizes = [local_tokens] * world_size
    
    # Create XPU events for timing
    naive_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    naive_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ishmem_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ishmem_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Benchmark: num_tokens={num_tokens}, hidden_dim={hidden_dim}, "
              f"num_experts={num_experts}, world_size={world_size}")
        print(f"Local tokens: {local_tokens}, Hidden dim: {hidden_dim}")
        print(f"{'='*70}")
    
    # =====================================================
    # Benchmark NaiveAll2AllManager
    # =====================================================
    dispatcher_naive = NaiveAll2AllManager(group=dist.group.WORLD)
    
    # Warmup
    try:
        for _ in range(warmup):
            gathered = dispatcher_naive.dispatch(input_tensor, sizes)
            output = dispatcher_naive.combine(gathered, sizes)
        
        torch.xpu.synchronize()
        dist.barrier()
        
        # Benchmark with e2e time
        start_time = time.perf_counter()
        for i in range(iterations):
            naive_begin_events[i].record()
            gathered = dispatcher_naive.dispatch(input_tensor, sizes)
            output_naive = dispatcher_naive.combine(gathered, sizes)
            naive_end_events[i].record()
        
        torch.xpu.synchronize()
        dist.barrier()
        end_time = time.perf_counter()
        naive_e2e_time = (end_time - start_time) / iterations * 1000
        
        # Calculate XPU event times
        naive_xpu_times = [
            naive_begin_events[i].elapsed_time(naive_end_events[i]) 
            for i in range(iterations)
        ]
        naive_xpu_avg = sum(naive_xpu_times) / len(naive_xpu_times)
        naive_xpu_min = min(naive_xpu_times)
        naive_xpu_max = max(naive_xpu_times)
        
        if rank == 0:
            print(f"[NaiveAll2All] Broadcast + AllReduce:")
            print(f"    E2E time:  {naive_e2e_time:.3f} ms")
            print(f"    XPU time:  avg={naive_xpu_avg:.3f} ms, min={naive_xpu_min:.3f} ms, max={naive_xpu_max:.3f} ms")
    except Exception as e:
        if rank == 0:
            print(f"[NaiveAll2All] Failed: {e}")
            import traceback
            traceback.print_exc()
        naive_e2e_time = float('inf')
        naive_xpu_avg = float('inf')
    
    dist.barrier()
    
    # =====================================================
    # Benchmark ISHMEMDispatcher
    # =====================================================
    try:
        # Initialize ISHMEMDispatcher after Naive benchmark
        dispatcher_ishmem = ISHMEMDispatcher(
            group_name=group_name,
            num_experts=num_experts,
            max_tokens=max_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device
        )
        
        # Warmup
        for _ in range(warmup):
            gathered = dispatcher_ishmem.dispatch(input_tensor, splits)
            output = dispatcher_ishmem.combine()
        
        torch.xpu.synchronize()
        dist.barrier()
        
        # Benchmark with e2e time
        start_time = time.perf_counter()
        for i in range(iterations):
            ishmem_begin_events[i].record()
            gathered = dispatcher_ishmem.dispatch(input_tensor, splits)
            output_ishmem = dispatcher_ishmem.combine()
            ishmem_end_events[i].record()
        
        torch.xpu.synchronize()
        dist.barrier()
        end_time = time.perf_counter()
        ishmem_e2e_time = (end_time - start_time) / iterations * 1000
        
        # Calculate XPU event times
        ishmem_xpu_times = [
            ishmem_begin_events[i].elapsed_time(ishmem_end_events[i]) 
            for i in range(iterations)
        ]
        ishmem_xpu_avg = sum(ishmem_xpu_times) / len(ishmem_xpu_times)
        ishmem_xpu_min = min(ishmem_xpu_times)
        ishmem_xpu_max = max(ishmem_xpu_times)
        
        if rank == 0:
            print(f"[ISHMEM]       all_to_all_vdev_2d:")
            print(f"    E2E time:  {ishmem_e2e_time:.3f} ms")
            print(f"    XPU time:  avg={ishmem_xpu_avg:.3f} ms, min={ishmem_xpu_min:.3f} ms, max={ishmem_xpu_max:.3f} ms")
        
    except Exception as e:
        if rank == 0:
            print(f"[ISHMEM] Failed: {e}")
            import traceback
            traceback.print_exc()
        ishmem_e2e_time = float('inf')
        ishmem_xpu_avg = float('inf')
    
    # =====================================================
    # Summary
    # =====================================================
    if rank == 0:
        print(f"\n{'='*50}")
        print("Summary:")
        print(f"{'='*50}")
        if ishmem_xpu_avg < float('inf') and naive_xpu_avg < float('inf'):
            e2e_speedup = naive_e2e_time / ishmem_e2e_time
            xpu_speedup = naive_xpu_avg / ishmem_xpu_avg
            print(f"E2E Speedup (ISHMEM vs Naive): {e2e_speedup:.2f}x")
            print(f"XPU Speedup (ISHMEM vs Naive): {xpu_speedup:.2f}x")
            if xpu_speedup > 1:
                print(f"ISHMEM is {xpu_speedup:.2f}x faster!")
            else:
                print(f"Naive is {1/xpu_speedup:.2f}x faster!")
        else:
            print("Speedup: N/A (one or both failed)")
        print(f"{'='*50}\n")
    
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Benchmark ISHMEM vs NaiveAll2All")
    parser.add_argument("--num-tokens", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--num-experts", type=int, default=8, help="Number of experts per rank")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()
    
    run_benchmark(
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        warmup=args.warmup,
        iterations=args.iterations
    )


if __name__ == "__main__":
    main()
