#!/usr/bin/env python3
"""Test script for torch_ishmem_extension"""
import torch
import torch_ishmem_extension as ishmem_ext

print("Successfully imported torch_ishmem_extension")
print(f"Has all_to_all_vdev: {hasattr(ishmem_ext, 'all_to_all_vdev')}")
print(f"Torch ops registered: {hasattr(torch.ops, 'ishmem_ext')}")
