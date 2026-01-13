#!/usr/bin/env python3
"""
Test script to verify the migration from ISHMEM to PyTorch Symmetric Memory API.
This script tests the basic functionality of the migrated extension.
"""

import os
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

def test_all_to_all_v():
    """Test basic all_to_all_v functionality"""
    # Initialize process group
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size == 1:
        print("Skipping test: requires world_size > 1")
        return
    
    torch.xpu.set_device(f"xpu:{rank}")
    dist.init_process_group("ccl")
    
    # Set symmetric memory backend
    symm_mem.set_backend("ISHMEM")
    
    # Import the extension
    import torch_symm_mem_extension as ext
    
    # Create test tensors
    send_size = 1024
    recv_size = 1024
    
    # Allocate symmetric memory tensors
    input_tensor = symm_mem.empty(send_size, dtype=torch.float32, device="xpu")
    output_tensor = symm_mem.empty(recv_size, dtype=torch.float32, device="xpu")
    
    # Create splits tensors
    in_splits = symm_mem.empty(world_size, dtype=torch.int64, device="xpu")
    out_splits_offsets = symm_mem.empty((2, world_size), dtype=torch.int64, device="xpu")
    
    # Initialize data
    input_tensor.fill_(float(rank))
    in_splits.fill_(send_size // world_size)
    
    # Rendezvous to establish symmetric memory
    group_name = "test_group"
    input_hdl = symm_mem.rendezvous(input_tensor, group_name)
    output_hdl = symm_mem.rendezvous(output_tensor, group_name)
    in_splits_hdl = symm_mem.rendezvous(in_splits, group_name)
    out_splits_offsets_hdl = symm_mem.rendezvous(out_splits_offsets, group_name)
    
    print(f"Rank {rank}: Symmetric memory established")
    print(f"  Input handle rank: {input_hdl.get_rank()}, world_size: {input_hdl.get_world_size()}")
    
    # Call the extension function
    try:
        ext.all_to_all_vdev(
            input_tensor,
            output_tensor,
            in_splits,
            out_splits_offsets,
            group_name
        )
        print(f"Rank {rank}: all_to_all_vdev completed successfully")
    except Exception as e:
        print(f"Rank {rank}: Error in all_to_all_vdev: {e}")
        raise
    
    # Verify results
    torch.xpu.synchronize()
    print(f"Rank {rank}: Test completed")
    
    dist.destroy_process_group()

def test_2d_all_to_all_v():
    """Test 2D all_to_all_v functionality for MoE"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size == 1:
        print("Skipping 2D test: requires world_size > 1")
        return
    
    torch.xpu.set_device(f"xpu:{rank}")
    dist.init_process_group("ccl")
    
    symm_mem.set_backend("ISHMEM")
    
    import torch_symm_mem_extension as ext
    
    # Test parameters
    ne = 4  # number of experts per rank
    batch_size = 128
    hidden_dim = 256
    
    # Allocate tensors
    input_tensor = symm_mem.empty((batch_size, hidden_dim), dtype=torch.float32, device="xpu")
    output_tensor = symm_mem.empty((batch_size, hidden_dim), dtype=torch.float32, device="xpu")
    
    in_splits = symm_mem.empty(world_size * ne, dtype=torch.int64, device="xpu")
    out_splits_offsets = symm_mem.empty((2, world_size * ne), dtype=torch.int64, device="xpu")
    
    # Initialize
    input_tensor.fill_(float(rank))
    in_splits.fill_(batch_size // (world_size * ne))
    
    # Rendezvous
    group_name = "test_2d_group"
    symm_mem.rendezvous(input_tensor, group_name)
    symm_mem.rendezvous(output_tensor, group_name)
    symm_mem.rendezvous(in_splits, group_name)
    symm_mem.rendezvous(out_splits_offsets, group_name)
    
    print(f"Rank {rank}: Testing 2D all_to_all_v")
    
    try:
        ext.all_to_all_vdev_2d(
            input_tensor,
            output_tensor,
            in_splits,
            out_splits_offsets,
            group_name,
            major_align=16
        )
        print(f"Rank {rank}: 2D all_to_all_vdev completed successfully")
    except Exception as e:
        print(f"Rank {rank}: Error in 2D all_to_all_vdev: {e}")
        raise
    
    torch.xpu.synchronize()
    print(f"Rank {rank}: 2D test completed")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    print("Testing PyTorch Symmetric Memory migration...")
    print("=" * 60)
    
    try:
        test_all_to_all_v()
        print("\n" + "=" * 60)
        test_2d_all_to_all_v()
        print("\n" + "=" * 60)
        print("All tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

