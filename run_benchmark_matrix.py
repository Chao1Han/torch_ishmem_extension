#!/usr/bin/env python3
"""
Benchmark Matrix Runner for ISHMEM vs NaiveAll2All vs AgRsAll2All

This script runs benchmarks across multiple configurations and generates
a summary table in Markdown and CSV formats.

Usage:
    mpirun -n 2 python torch_ishmem_extension/run_benchmark_matrix.py --preset full --no-xpu-events
    mpirun -n 4 python torch_ishmem_extension/run_benchmark_matrix.py --preset full --no-xpu-events
    mpirun -n 8 python torch_ishmem_extension/run_benchmark_matrix.py --preset full --no-xpu-events
"""

import os
import time
import argparse
import json
from datetime import datetime

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

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
    def __init__(self, group=None):
        self.group = group
        self.rank = rank
        self.world_size = world_size

    def dispatch(self, hidden_states: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_dim = hidden_states.shape[1]
        total_tokens = sum(sizes)
        
        cu_tokens = [0]
        for s in sizes:
            cu_tokens.append(cu_tokens[-1] + s)
        
        buffer = torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
        buffer.zero_()
        
        start_idx = cu_tokens[self.rank]
        end_idx = cu_tokens[self.rank + 1]
        local_size = sizes[self.rank]
        buffer[start_idx:end_idx, :].copy_(hidden_states[:local_size])
        
        torch.xpu.synchronize()
        
        for idx in range(self.world_size):
            s = cu_tokens[idx]
            e = cu_tokens[idx + 1]
            dist.broadcast(buffer[s:e, :].contiguous(), src=idx, group=self.group)
        
        return buffer

    def combine(self, hidden_states: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        cu_tokens = [0]
        for s in sizes:
            cu_tokens.append(cu_tokens[-1] + s)
        
        start_idx = cu_tokens[self.rank]
        end_idx = cu_tokens[self.rank + 1]
        
        all_hidden = hidden_states.clone()
        dist.all_reduce(all_hidden, group=self.group)
        output = all_hidden[start_idx:end_idx, :].contiguous()
        
        return output


# =====================================================
# AgRsAll2AllManager - AllGather/ReduceScatter based
# =====================================================
class AgRsAll2AllManager:
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    This is the approach used in vLLM's AgRsAll2AllManager.
    """
    def __init__(self, group=None):
        self.group = group
        self.rank = rank
        self.world_size = world_size

    def dispatch(self, hidden_states: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        """
        Gather hidden_states from all ranks using all_gather_into_tensor.
        
        Input: hidden_states [local_tokens, hidden_dim]
        Output: gathered [total_tokens, hidden_dim]
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_dim = hidden_states.shape[1]
        local_tokens = hidden_states.shape[0]
        
        # For simplicity, assume all ranks have same size (sizes are equal)
        # Use all_gather_into_tensor for efficiency
        total_tokens = local_tokens * self.world_size
        output = torch.empty(total_tokens, hidden_dim, dtype=dtype, device=device)
        
        dist.all_gather_into_tensor(output, hidden_states.contiguous(), group=self.group)
        
        return output

    def combine(self, hidden_states: torch.Tensor, sizes: list[int]) -> torch.Tensor:
        """
        Reduce-scatter hidden_states across all ranks.
        
        Input: hidden_states [total_tokens, hidden_dim]
        Output: local_hidden [local_tokens, hidden_dim]
        """
        total_tokens = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        local_tokens = total_tokens // self.world_size
        
        output = torch.empty(local_tokens, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)
        
        dist.reduce_scatter_tensor(output, hidden_states.contiguous(), group=self.group)
        
        return output


# =====================================================
# ISHMEMDispatcher - ISHMEM all_to_all_vdev_2d based
# =====================================================
class ISHMEMDispatcher:
    def __init__(self, group_name: str, num_experts: int, max_tokens: int, 
                 hidden_dim: int, dtype: torch.dtype, device: torch.device):
        self.group_name = group_name
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.nsplits = num_experts * world_size
        self.align = 8
        
        max_inp_numel = max_tokens
        max_out_numel = max_tokens * world_size
        
        self.inp_buffer = symm_mem.empty(max_inp_numel, hidden_dim, dtype=dtype, device=device)
        self.out_buffer = symm_mem.empty(max_out_numel, hidden_dim, dtype=dtype, device=device)
        self.combine_out = symm_mem.empty(max_inp_numel, hidden_dim, dtype=dtype, device=device)
        
        self.in_splits = symm_mem.empty(self.nsplits, dtype=torch.int64, device=device)
        self.out_splits_offsets = symm_mem.empty((2, self.nsplits), dtype=torch.int64, device=device)
        self.in_splits_offsets = symm_mem.empty((2, self.nsplits), dtype=torch.int64, device=device)
        self.out_splits_offsets_combine = symm_mem.empty((2, self.nsplits), dtype=torch.int64, device=device)
    
    def dispatch(self, hidden_states: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
        inp_numel = hidden_states.shape[0]
        
        self.inp_buffer[:inp_numel].copy_(hidden_states)
        self.in_splits.copy_(splits)
        
        torch.ops.ishmem_ext.all_to_all_vdev_2d(
            self.inp_buffer, self.out_buffer, 
            self.in_splits, self.out_splits_offsets, 
            self.group_name, self.align
        )
        
        torch.xpu.synchronize()
        out_numel = self.out_splits_offsets[1, -1].item()
        
        return self.out_buffer[:out_numel]
    
    def combine(self) -> torch.Tensor:
        self.in_splits_offsets.copy_(self.out_splits_offsets)
        
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
    import torch_ishmem_extension as ishmem_ext
    symm_mem.set_backend("ISHMEM")
    device = torch.device("xpu")
    dist.barrier()
    return device


def run_single_benchmark(num_tokens, hidden_dim, num_experts, device, group_name, 
                         warmup=5, iterations=20, use_xpu_events=True):
    """Run a single benchmark configuration and return results."""
    dtype = torch.bfloat16
    nsplits = num_experts * world_size
    
    # Use SAME random seed across all ranks
    torch.manual_seed(42)
    tokens_per_split = num_tokens // nsplits
    splits = torch.randint(
        max(1, tokens_per_split - tokens_per_split // 2),
        tokens_per_split + tokens_per_split // 2 + 1,
        (nsplits,), dtype=torch.int64, device=device
    )
    
    local_tokens = splits.sum().item()
    max_tokens = local_tokens + 1000
    
    torch.manual_seed(12345 + rank)
    input_tensor = torch.randn(local_tokens, hidden_dim, dtype=dtype, device=device)
    sizes = [local_tokens] * world_size
    
    results = {
        "num_tokens": num_tokens,
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
        "local_tokens": local_tokens,
        "world_size": world_size,
    }
    
    # =====================================================
    # Benchmark NaiveAll2AllManager
    # =====================================================
    dispatcher_naive = NaiveAll2AllManager(group=dist.group.WORLD)
    
    # Create separate events for dispatch and combine
    if use_xpu_events:
        naive_dispatch_begin = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        naive_dispatch_end = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        naive_combine_begin = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        naive_combine_end = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    
    try:
        for _ in range(warmup):
            gathered = dispatcher_naive.dispatch(input_tensor, sizes)
            output = dispatcher_naive.combine(gathered, sizes)
        
        torch.xpu.synchronize()
        dist.barrier()
        
        if use_xpu_events:
            # Use XPU events for precise GPU timing
            for i in range(iterations):
                naive_dispatch_begin[i].record()
                gathered = dispatcher_naive.dispatch(input_tensor, sizes)
                naive_dispatch_end[i].record()
                naive_combine_begin[i].record()
                output_naive = dispatcher_naive.combine(gathered, sizes)
                naive_combine_end[i].record()
            
            torch.xpu.synchronize()
            dist.barrier()
            
            naive_dispatch_times = [naive_dispatch_begin[i].elapsed_time(naive_dispatch_end[i]) for i in range(iterations)]
            naive_combine_times = [naive_combine_begin[i].elapsed_time(naive_combine_end[i]) for i in range(iterations)]
            naive_total_times = [d + c for d, c in zip(naive_dispatch_times, naive_combine_times)]
            
            results["naive_dispatch_avg_ms"] = sum(naive_dispatch_times) / len(naive_dispatch_times)
            results["naive_combine_avg_ms"] = sum(naive_combine_times) / len(naive_combine_times)
            results["naive_xpu_avg_ms"] = sum(naive_total_times) / len(naive_total_times)
            results["naive_xpu_min_ms"] = min(naive_total_times)
            results["naive_xpu_max_ms"] = max(naive_total_times)
            results["naive_e2e_ms"] = results["naive_xpu_avg_ms"]
        else:
            # Use CPU timing (less precise but works without events)
            dispatch_times = []
            combine_times = []
            
            for i in range(iterations):
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                gathered = dispatcher_naive.dispatch(input_tensor, sizes)
                torch.xpu.synchronize()
                t1 = time.perf_counter()
                output_naive = dispatcher_naive.combine(gathered, sizes)
                torch.xpu.synchronize()
                t2 = time.perf_counter()
                
                dispatch_times.append((t1 - t0) * 1000)
                combine_times.append((t2 - t1) * 1000)
            
            dist.barrier()
            
            results["naive_dispatch_avg_ms"] = sum(dispatch_times) / len(dispatch_times)
            results["naive_combine_avg_ms"] = sum(combine_times) / len(combine_times)
            results["naive_xpu_avg_ms"] = results["naive_dispatch_avg_ms"] + results["naive_combine_avg_ms"]
            results["naive_e2e_ms"] = results["naive_xpu_avg_ms"]
            results["naive_xpu_min_ms"] = float('nan')
            results["naive_xpu_max_ms"] = float('nan')
    except Exception as e:
        results["naive_error"] = str(e)
        results["naive_e2e_ms"] = float('inf')
        results["naive_dispatch_avg_ms"] = float('inf')
        results["naive_combine_avg_ms"] = float('inf')
        results["naive_xpu_avg_ms"] = float('inf')
    
    dist.barrier()
    
    # =====================================================
    # Benchmark AgRsAll2AllManager (AllGather + ReduceScatter)
    # =====================================================
    dispatcher_agrs = AgRsAll2AllManager(group=dist.group.WORLD)
    
    if use_xpu_events:
        agrs_dispatch_begin = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        agrs_dispatch_end = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        agrs_combine_begin = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        agrs_combine_end = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    
    try:
        for _ in range(warmup):
            gathered = dispatcher_agrs.dispatch(input_tensor, sizes)
            output = dispatcher_agrs.combine(gathered, sizes)
        
        torch.xpu.synchronize()
        dist.barrier()
        
        if use_xpu_events:
            # Use XPU events for precise GPU timing
            for i in range(iterations):
                agrs_dispatch_begin[i].record()
                gathered = dispatcher_agrs.dispatch(input_tensor, sizes)
                agrs_dispatch_end[i].record()
                agrs_combine_begin[i].record()
                output_agrs = dispatcher_agrs.combine(gathered, sizes)
                agrs_combine_end[i].record()
            
            torch.xpu.synchronize()
            dist.barrier()
            
            agrs_dispatch_times = [agrs_dispatch_begin[i].elapsed_time(agrs_dispatch_end[i]) for i in range(iterations)]
            agrs_combine_times = [agrs_combine_begin[i].elapsed_time(agrs_combine_end[i]) for i in range(iterations)]
            agrs_total_times = [d + c for d, c in zip(agrs_dispatch_times, agrs_combine_times)]
            
            results["agrs_dispatch_avg_ms"] = sum(agrs_dispatch_times) / len(agrs_dispatch_times)
            results["agrs_combine_avg_ms"] = sum(agrs_combine_times) / len(agrs_combine_times)
            results["agrs_xpu_avg_ms"] = sum(agrs_total_times) / len(agrs_total_times)
            results["agrs_xpu_min_ms"] = min(agrs_total_times)
            results["agrs_xpu_max_ms"] = max(agrs_total_times)
            results["agrs_e2e_ms"] = results["agrs_xpu_avg_ms"]
        else:
            # Use CPU timing
            dispatch_times = []
            combine_times = []
            
            for i in range(iterations):
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                gathered = dispatcher_agrs.dispatch(input_tensor, sizes)
                torch.xpu.synchronize()
                t1 = time.perf_counter()
                output_agrs = dispatcher_agrs.combine(gathered, sizes)
                torch.xpu.synchronize()
                t2 = time.perf_counter()
                
                dispatch_times.append((t1 - t0) * 1000)
                combine_times.append((t2 - t1) * 1000)
            
            dist.barrier()
            
            results["agrs_dispatch_avg_ms"] = sum(dispatch_times) / len(dispatch_times)
            results["agrs_combine_avg_ms"] = sum(combine_times) / len(combine_times)
            results["agrs_xpu_avg_ms"] = results["agrs_dispatch_avg_ms"] + results["agrs_combine_avg_ms"]
            results["agrs_e2e_ms"] = results["agrs_xpu_avg_ms"]
            results["agrs_xpu_min_ms"] = float('nan')
            results["agrs_xpu_max_ms"] = float('nan')
    except Exception as e:
        results["agrs_error"] = str(e)
        results["agrs_e2e_ms"] = float('inf')
        results["agrs_dispatch_avg_ms"] = float('inf')
        results["agrs_combine_avg_ms"] = float('inf')
        results["agrs_xpu_avg_ms"] = float('inf')
    
    dist.barrier()
    
    # =====================================================
    # Benchmark ISHMEMDispatcher
    # =====================================================
    if use_xpu_events:
        ishmem_dispatch_begin = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        ishmem_dispatch_end = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        ishmem_combine_begin = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
        ishmem_combine_end = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    
    try:
        dispatcher_ishmem = ISHMEMDispatcher(
            group_name=group_name,
            num_experts=num_experts,
            max_tokens=max_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
            device=device
        )
        
        for _ in range(warmup):
            gathered = dispatcher_ishmem.dispatch(input_tensor, splits)
            output = dispatcher_ishmem.combine()
        
        torch.xpu.synchronize()
        dist.barrier()
        
        if use_xpu_events:
            # Use XPU events for precise GPU timing
            for i in range(iterations):
                ishmem_dispatch_begin[i].record()
                gathered = dispatcher_ishmem.dispatch(input_tensor, splits)
                ishmem_dispatch_end[i].record()
                ishmem_combine_begin[i].record()
                output_ishmem = dispatcher_ishmem.combine()
                ishmem_combine_end[i].record()
            
            torch.xpu.synchronize()
            dist.barrier()
            
            ishmem_dispatch_times = [ishmem_dispatch_begin[i].elapsed_time(ishmem_dispatch_end[i]) for i in range(iterations)]
            ishmem_combine_times = [ishmem_combine_begin[i].elapsed_time(ishmem_combine_end[i]) for i in range(iterations)]
            ishmem_total_times = [d + c for d, c in zip(ishmem_dispatch_times, ishmem_combine_times)]
            
            results["ishmem_dispatch_avg_ms"] = sum(ishmem_dispatch_times) / len(ishmem_dispatch_times)
            results["ishmem_combine_avg_ms"] = sum(ishmem_combine_times) / len(ishmem_combine_times)
            results["ishmem_xpu_avg_ms"] = sum(ishmem_total_times) / len(ishmem_total_times)
            results["ishmem_xpu_min_ms"] = min(ishmem_total_times)
            results["ishmem_xpu_max_ms"] = max(ishmem_total_times)
            results["ishmem_e2e_ms"] = results["ishmem_xpu_avg_ms"]
        else:
            # Use CPU timing
            dispatch_times = []
            combine_times = []
            
            for i in range(iterations):
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                gathered = dispatcher_ishmem.dispatch(input_tensor, splits)
                torch.xpu.synchronize()
                t1 = time.perf_counter()
                output_ishmem = dispatcher_ishmem.combine()
                torch.xpu.synchronize()
                t2 = time.perf_counter()
                
                dispatch_times.append((t1 - t0) * 1000)
                combine_times.append((t2 - t1) * 1000)
            
            dist.barrier()
            
            results["ishmem_dispatch_avg_ms"] = sum(dispatch_times) / len(dispatch_times)
            results["ishmem_combine_avg_ms"] = sum(combine_times) / len(combine_times)
            results["ishmem_xpu_avg_ms"] = results["ishmem_dispatch_avg_ms"] + results["ishmem_combine_avg_ms"]
            results["ishmem_e2e_ms"] = results["ishmem_xpu_avg_ms"]
            results["ishmem_xpu_min_ms"] = float('nan')
            results["ishmem_xpu_max_ms"] = float('nan')
    except Exception as e:
        results["ishmem_error"] = str(e)
        results["ishmem_e2e_ms"] = float('inf')
        results["ishmem_dispatch_avg_ms"] = float('inf')
        results["ishmem_combine_avg_ms"] = float('inf')
        results["ishmem_xpu_avg_ms"] = float('inf')
    
    dist.barrier()
    
    # Calculate speedups (vs ISHMEM as baseline)
    ishmem_avg = results.get("ishmem_xpu_avg_ms", float('inf'))
    naive_avg = results.get("naive_xpu_avg_ms", float('inf'))
    agrs_avg = results.get("agrs_xpu_avg_ms", float('inf'))
    
    if ishmem_avg < float('inf') and naive_avg < float('inf'):
        results["naive_vs_ishmem"] = naive_avg / ishmem_avg
    else:
        results["naive_vs_ishmem"] = float('nan')
    
    if ishmem_avg < float('inf') and agrs_avg < float('inf'):
        results["agrs_vs_ishmem"] = agrs_avg / ishmem_avg
    else:
        results["agrs_vs_ishmem"] = float('nan')
    
    # Legacy speedup for backwards compatibility
    results["e2e_speedup"] = results.get("naive_vs_ishmem", float('nan'))
    results["xpu_speedup"] = results.get("naive_vs_ishmem", float('nan'))
    
    return results


def generate_markdown_table(all_results):
    """Generate a Markdown table from results."""
    lines = []
    lines.append("")
    lines.append(f"## Benchmark Results (world_size={world_size})")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Main comparison table with total times
    lines.append("### Total Time Comparison (ms)")
    lines.append("")
    lines.append("| num_tokens | hidden_dim | Naive | AgRs | ISHMEM | Naive/ISHMEM | AgRs/ISHMEM |")
    lines.append("|------------|------------|-------|------|--------|--------------|-------------|")
    
    for r in all_results:
        naive = r.get("naive_xpu_avg_ms", float('inf'))
        agrs = r.get("agrs_xpu_avg_ms", float('inf'))
        ishmem = r.get("ishmem_xpu_avg_ms", float('inf'))
        naive_vs_ishmem = r.get("naive_vs_ishmem", float('nan'))
        agrs_vs_ishmem = r.get("agrs_vs_ishmem", float('nan'))
        
        naive_str = f"{naive:.3f}" if naive < float('inf') else "FAIL"
        agrs_str = f"{agrs:.3f}" if agrs < float('inf') else "FAIL"
        ishmem_str = f"{ishmem:.3f}" if ishmem < float('inf') else "FAIL"
        naive_ratio = f"{naive_vs_ishmem:.2f}x" if not (naive_vs_ishmem != naive_vs_ishmem) else "N/A"
        agrs_ratio = f"{agrs_vs_ishmem:.2f}x" if not (agrs_vs_ishmem != agrs_vs_ishmem) else "N/A"
        
        lines.append(f"| {r['num_tokens']:>10} | {r['hidden_dim']:>10} | {naive_str:>5} | {agrs_str:>4} | {ishmem_str:>6} | {naive_ratio:>12} | {agrs_ratio:>11} |")
    
    lines.append("")
    
    # Dispatch time breakdown
    lines.append("### Dispatch Time Breakdown (ms)")
    lines.append("")
    lines.append("| num_tokens | hidden_dim | Naive | AgRs | ISHMEM |")
    lines.append("|------------|------------|-------|------|--------|")
    
    for r in all_results:
        naive_disp = r.get("naive_dispatch_avg_ms", float('inf'))
        agrs_disp = r.get("agrs_dispatch_avg_ms", float('inf'))
        ishmem_disp = r.get("ishmem_dispatch_avg_ms", float('inf'))
        
        naive_str = f"{naive_disp:.3f}" if naive_disp < float('inf') else "FAIL"
        agrs_str = f"{agrs_disp:.3f}" if agrs_disp < float('inf') else "FAIL"
        ishmem_str = f"{ishmem_disp:.3f}" if ishmem_disp < float('inf') else "FAIL"
        
        lines.append(f"| {r['num_tokens']:>10} | {r['hidden_dim']:>10} | {naive_str:>5} | {agrs_str:>4} | {ishmem_str:>6} |")
    
    lines.append("")
    
    # Combine time breakdown
    lines.append("### Combine Time Breakdown (ms)")
    lines.append("")
    lines.append("| num_tokens | hidden_dim | Naive | AgRs | ISHMEM |")
    lines.append("|------------|------------|-------|------|--------|")
    
    for r in all_results:
        naive_comb = r.get("naive_combine_avg_ms", float('inf'))
        agrs_comb = r.get("agrs_combine_avg_ms", float('inf'))
        ishmem_comb = r.get("ishmem_combine_avg_ms", float('inf'))
        
        naive_str = f"{naive_comb:.3f}" if naive_comb < float('inf') else "FAIL"
        agrs_str = f"{agrs_comb:.3f}" if agrs_comb < float('inf') else "FAIL"
        ishmem_str = f"{ishmem_comb:.3f}" if ishmem_comb < float('inf') else "FAIL"
        
        lines.append(f"| {r['num_tokens']:>10} | {r['hidden_dim']:>10} | {naive_str:>5} | {agrs_str:>4} | {ishmem_str:>6} |")
    
    lines.append("")
    
    # Legend
    lines.append("**Legend:**")
    lines.append("- **Naive**: Broadcast (dispatch) + AllReduce (combine)")
    lines.append("- **AgRs**: AllGather (dispatch) + ReduceScatter (combine)")
    lines.append("- **ISHMEM**: all_to_all_vdev_2d (symmetric memory)")
    lines.append("- **Ratio**: Higher = ISHMEM is faster by that factor")
    lines.append("")
    
    return "\n".join(lines)


def generate_csv(all_results):
    """Generate CSV output from results."""
    lines = []
    headers = [
        "num_tokens", "hidden_dim", "num_experts", "local_tokens", "world_size",
        "naive_e2e_ms", "naive_dispatch_avg_ms", "naive_combine_avg_ms", "naive_xpu_avg_ms",
        "agrs_e2e_ms", "agrs_dispatch_avg_ms", "agrs_combine_avg_ms", "agrs_xpu_avg_ms",
        "ishmem_e2e_ms", "ishmem_dispatch_avg_ms", "ishmem_combine_avg_ms", "ishmem_xpu_avg_ms",
        "naive_vs_ishmem", "agrs_vs_ishmem"
    ]
    lines.append(",".join(headers))
    
    for r in all_results:
        values = []
        for h in headers:
            v = r.get(h, "")
            if isinstance(v, float):
                if v == float('inf') or v != v:  # inf or nan
                    values.append("")
                else:
                    values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append(",".join(values))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark matrix")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for results")
    parser.add_argument("--preset", type=str, default="medium", 
                        choices=["small", "medium", "large", "full"],
                        help="Preset configuration set")
    parser.add_argument("--no-xpu-events", action="store_true",
                        help="Disable XPU event timing (use this if hanging with many ranks)")
    args = parser.parse_args()
    
    use_xpu_events = not args.no_xpu_events
    
    # Define benchmark configurations
    if args.preset == "small":
        configs = [
            {"num_tokens": 1024, "hidden_dim": 2048, "num_experts": 8},
            {"num_tokens": 2048, "hidden_dim": 2048, "num_experts": 8},
            {"num_tokens": 4096, "hidden_dim": 2048, "num_experts": 8},
        ]
    elif args.preset == "medium":
        configs = [
            {"num_tokens": 1024, "hidden_dim": 4096, "num_experts": 8},
            {"num_tokens": 2048, "hidden_dim": 4096, "num_experts": 8},
            {"num_tokens": 4096, "hidden_dim": 4096, "num_experts": 8},
        ]
    elif args.preset == "large":
        configs = [
            {"num_tokens": 8192, "hidden_dim": 4096, "num_experts": 8},
            {"num_tokens": 2048, "hidden_dim": 5120, "num_experts": 16},
            {"num_tokens": 4096, "hidden_dim": 5120, "num_experts": 16},
            {"num_tokens": 8192, "hidden_dim": 5120, "num_experts": 16},
            {"num_tokens": 4096, "hidden_dim": 5120, "num_experts": 64},
        ]
    else:  # full
        configs = [
            # Vary num_tokens
            {"num_tokens": 512,  "hidden_dim": 4096, "num_experts": 8},
            {"num_tokens": 1024, "hidden_dim": 4096, "num_experts": 8},
            {"num_tokens": 2048, "hidden_dim": 4096, "num_experts": 8},
            {"num_tokens": 4096, "hidden_dim": 4096, "num_experts": 8},
            # {"num_tokens": 8192, "hidden_dim": 4096, "num_experts": 8},
            # # Vary hidden_dim
            # {"num_tokens": 4096, "hidden_dim": 2048, "num_experts": 8},
            # {"num_tokens": 4096, "hidden_dim": 5120, "num_experts": 8},
            # {"num_tokens": 4096, "hidden_dim": 6144, "num_experts": 8},
            # # Vary num_experts
            # {"num_tokens": 4096, "hidden_dim": 4096, "num_experts": 4},
            # {"num_tokens": 4096, "hidden_dim": 4096, "num_experts": 16},
            # {"num_tokens": 4096, "hidden_dim": 4096, "num_experts": 32},
        ]
    
    device = setup_distributed()
    group_name = dist.group.WORLD.group_name
    symm_mem.enable_symm_mem_for_group(group_name)
    
    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print(f"ISHMEM vs NaiveAll2All Benchmark Matrix", flush=True)
        print(f"World size: {world_size}", flush=True)
        print(f"Preset: {args.preset} ({len(configs)} configurations)", flush=True)
        print(f"Warmup: {args.warmup}, Iterations: {args.iterations}", flush=True)
        print(f"XPU Events: {'enabled' if use_xpu_events else 'disabled'}", flush=True)
        print(f"{'='*70}\n", flush=True)
    
    all_results = []
    
    for i, config in enumerate(configs):
        if rank == 0:
            print(f"[{i+1}/{len(configs)}] Running: num_tokens={config['num_tokens']}, "
                  f"hidden_dim={config['hidden_dim']}, num_experts={config['num_experts']}...", 
                  end=" ", flush=True)
        
        try:
            results = run_single_benchmark(
                num_tokens=config["num_tokens"],
                hidden_dim=config["hidden_dim"],
                num_experts=config["num_experts"],
                device=device,
                group_name=group_name,
                warmup=args.warmup,
                iterations=args.iterations,
                use_xpu_events=use_xpu_events
            )
            all_results.append(results)
            
            if rank == 0:
                speedup = results.get("xpu_speedup", float('nan'))
                if speedup == speedup:  # not nan
                    print(f"Done! Speedup: {speedup:.2f}x", flush=True)
                else:
                    print("Done! (speedup N/A)", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"FAILED: {e}", flush=True)
            all_results.append({
                **config,
                "error": str(e),
                "world_size": world_size,
            })
        
        dist.barrier()
    
    # Generate and save results
    if rank == 0:
        print(f"\n{'='*70}", flush=True)
        print("Generating results...", flush=True)
        
        # Generate Markdown
        md_content = generate_markdown_table(all_results)
        md_file = os.path.join(args.output_dir, f"benchmark_results_n{world_size}.md")
        with open(md_file, "w") as f:
            f.write(md_content)
        print(f"Markdown saved: {md_file}", flush=True)
        
        # Generate CSV
        csv_content = generate_csv(all_results)
        csv_file = os.path.join(args.output_dir, f"benchmark_results_n{world_size}.csv")
        with open(csv_file, "w") as f:
            f.write(csv_content)
        print(f"CSV saved: {csv_file}", flush=True)
        
        # Generate JSON
        json_file = os.path.join(args.output_dir, f"benchmark_results_n{world_size}.json")
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"JSON saved: {json_file}", flush=True)
        
        # Print summary table
        print("\n" + md_content, flush=True)
        
        print(f"{'='*70}\n", flush=True)
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
