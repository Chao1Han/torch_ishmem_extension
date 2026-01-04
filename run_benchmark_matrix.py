#!/usr/bin/env python3
"""
Benchmark Matrix Runner for ISHMEM vs NaiveAll2All

This script runs benchmarks across multiple configurations and generates
a summary table in Markdown and CSV formats.

Usage:
    cd /home/sdp/hanchao/symm
    source env.sh
    mpirun -n 2 python torch_ishmem_extension/run_benchmark_matrix.py
    mpirun -n 4 python torch_ishmem_extension/run_benchmark_matrix.py
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
                         warmup=5, iterations=20):
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
    
    # Create XPU events
    naive_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    naive_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ishmem_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    ishmem_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(iterations)]
    
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
    
    try:
        for _ in range(warmup):
            gathered = dispatcher_naive.dispatch(input_tensor, sizes)
            output = dispatcher_naive.combine(gathered, sizes)
        
        torch.xpu.synchronize()
        dist.barrier()
        
        start_time = time.perf_counter()
        for i in range(iterations):
            naive_begin_events[i].record()
            gathered = dispatcher_naive.dispatch(input_tensor, sizes)
            output_naive = dispatcher_naive.combine(gathered, sizes)
            naive_end_events[i].record()
        
        torch.xpu.synchronize()
        dist.barrier()
        end_time = time.perf_counter()
        
        naive_e2e = (end_time - start_time) / iterations * 1000
        naive_xpu_times = [naive_begin_events[i].elapsed_time(naive_end_events[i]) for i in range(iterations)]
        naive_xpu_avg = sum(naive_xpu_times) / len(naive_xpu_times)
        naive_xpu_min = min(naive_xpu_times)
        naive_xpu_max = max(naive_xpu_times)
        
        results["naive_e2e_ms"] = naive_e2e
        results["naive_xpu_avg_ms"] = naive_xpu_avg
        results["naive_xpu_min_ms"] = naive_xpu_min
        results["naive_xpu_max_ms"] = naive_xpu_max
    except Exception as e:
        results["naive_error"] = str(e)
        results["naive_e2e_ms"] = float('inf')
        results["naive_xpu_avg_ms"] = float('inf')
    
    dist.barrier()
    
    # =====================================================
    # Benchmark ISHMEMDispatcher
    # =====================================================
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
        
        start_time = time.perf_counter()
        for i in range(iterations):
            ishmem_begin_events[i].record()
            gathered = dispatcher_ishmem.dispatch(input_tensor, splits)
            output_ishmem = dispatcher_ishmem.combine()
            ishmem_end_events[i].record()
        
        torch.xpu.synchronize()
        dist.barrier()
        end_time = time.perf_counter()
        
        ishmem_e2e = (end_time - start_time) / iterations * 1000
        ishmem_xpu_times = [ishmem_begin_events[i].elapsed_time(ishmem_end_events[i]) for i in range(iterations)]
        ishmem_xpu_avg = sum(ishmem_xpu_times) / len(ishmem_xpu_times)
        ishmem_xpu_min = min(ishmem_xpu_times)
        ishmem_xpu_max = max(ishmem_xpu_times)
        
        results["ishmem_e2e_ms"] = ishmem_e2e
        results["ishmem_xpu_avg_ms"] = ishmem_xpu_avg
        results["ishmem_xpu_min_ms"] = ishmem_xpu_min
        results["ishmem_xpu_max_ms"] = ishmem_xpu_max
    except Exception as e:
        results["ishmem_error"] = str(e)
        results["ishmem_e2e_ms"] = float('inf')
        results["ishmem_xpu_avg_ms"] = float('inf')
    
    dist.barrier()
    
    # Calculate speedups
    if results.get("naive_xpu_avg_ms", float('inf')) < float('inf') and \
       results.get("ishmem_xpu_avg_ms", float('inf')) < float('inf'):
        results["e2e_speedup"] = results["naive_e2e_ms"] / results["ishmem_e2e_ms"]
        results["xpu_speedup"] = results["naive_xpu_avg_ms"] / results["ishmem_xpu_avg_ms"]
    else:
        results["e2e_speedup"] = float('nan')
        results["xpu_speedup"] = float('nan')
    
    return results


def generate_markdown_table(all_results):
    """Generate a Markdown table from results."""
    lines = []
    lines.append("")
    lines.append(f"## Benchmark Results (world_size={world_size})")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Main comparison table
    lines.append("### Performance Comparison (XPU avg time)")
    lines.append("")
    lines.append("| num_tokens | hidden_dim | num_experts | Naive (ms) | ISHMEM (ms) | Speedup |")
    lines.append("|------------|------------|-------------|------------|-------------|---------|")
    
    for r in all_results:
        naive = r.get("naive_xpu_avg_ms", float('inf'))
        ishmem = r.get("ishmem_xpu_avg_ms", float('inf'))
        speedup = r.get("xpu_speedup", float('nan'))
        
        naive_str = f"{naive:.3f}" if naive < float('inf') else "FAIL"
        ishmem_str = f"{ishmem:.3f}" if ishmem < float('inf') else "FAIL"
        speedup_str = f"{speedup:.2f}x" if not (speedup != speedup) else "N/A"  # nan check
        
        lines.append(f"| {r['num_tokens']:>10} | {r['hidden_dim']:>10} | {r['num_experts']:>11} | {naive_str:>10} | {ishmem_str:>11} | {speedup_str:>7} |")
    
    lines.append("")
    
    # Detailed table
    lines.append("### Detailed Results")
    lines.append("")
    lines.append("| Config | Naive E2E | Naive XPU (avg/min/max) | ISHMEM E2E | ISHMEM XPU (avg/min/max) |")
    lines.append("|--------|-----------|-------------------------|------------|--------------------------|")
    
    for r in all_results:
        config = f"{r['num_tokens']}x{r['hidden_dim']}x{r['num_experts']}"
        
        if r.get("naive_xpu_avg_ms", float('inf')) < float('inf'):
            naive_e2e = f"{r['naive_e2e_ms']:.3f}"
            naive_xpu = f"{r['naive_xpu_avg_ms']:.3f}/{r['naive_xpu_min_ms']:.3f}/{r['naive_xpu_max_ms']:.3f}"
        else:
            naive_e2e = "FAIL"
            naive_xpu = "FAIL"
        
        if r.get("ishmem_xpu_avg_ms", float('inf')) < float('inf'):
            ishmem_e2e = f"{r['ishmem_e2e_ms']:.3f}"
            ishmem_xpu = f"{r['ishmem_xpu_avg_ms']:.3f}/{r['ishmem_xpu_min_ms']:.3f}/{r['ishmem_xpu_max_ms']:.3f}"
        else:
            ishmem_e2e = "FAIL"
            ishmem_xpu = "FAIL"
        
        lines.append(f"| {config} | {naive_e2e} | {naive_xpu} | {ishmem_e2e} | {ishmem_xpu} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_csv(all_results):
    """Generate CSV output from results."""
    lines = []
    headers = [
        "num_tokens", "hidden_dim", "num_experts", "local_tokens", "world_size",
        "naive_e2e_ms", "naive_xpu_avg_ms", "naive_xpu_min_ms", "naive_xpu_max_ms",
        "ishmem_e2e_ms", "ishmem_xpu_avg_ms", "ishmem_xpu_min_ms", "ishmem_xpu_max_ms",
        "e2e_speedup", "xpu_speedup"
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
    args = parser.parse_args()
    
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
        print(f"\n{'='*70}")
        print(f"ISHMEM vs NaiveAll2All Benchmark Matrix")
        print(f"World size: {world_size}")
        print(f"Preset: {args.preset} ({len(configs)} configurations)")
        print(f"Warmup: {args.warmup}, Iterations: {args.iterations}")
        print(f"{'='*70}\n")
    
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
                iterations=args.iterations
            )
            all_results.append(results)
            
            if rank == 0:
                speedup = results.get("xpu_speedup", float('nan'))
                if speedup == speedup:  # not nan
                    print(f"Done! Speedup: {speedup:.2f}x")
                else:
                    print("Done! (speedup N/A)")
        except Exception as e:
            if rank == 0:
                print(f"FAILED: {e}")
            all_results.append({
                **config,
                "error": str(e),
                "world_size": world_size,
            })
        
        dist.barrier()
    
    # Generate and save results
    if rank == 0:
        print(f"\n{'='*70}")
        print("Generating results...")
        
        # Generate Markdown
        md_content = generate_markdown_table(all_results)
        md_file = os.path.join(args.output_dir, f"benchmark_results_n{world_size}.md")
        with open(md_file, "w") as f:
            f.write(md_content)
        print(f"Markdown saved: {md_file}")
        
        # Generate CSV
        csv_content = generate_csv(all_results)
        csv_file = os.path.join(args.output_dir, f"benchmark_results_n{world_size}.csv")
        with open(csv_file, "w") as f:
            f.write(csv_content)
        print(f"CSV saved: {csv_file}")
        
        # Generate JSON
        json_file = os.path.join(args.output_dir, f"benchmark_results_n{world_size}.json")
        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"JSON saved: {json_file}")
        
        # Print summary table
        print("\n" + md_content)
        
        print(f"{'='*70}\n")
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
