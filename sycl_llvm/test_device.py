import os
import sys
import torch
import triton
import triton.language as tl
from triton.language import core

@core.extern
def copy_from_src_to_dst_wrapper(dest, source, size_bytes, _semantic=None):  # type: ignore[no-untyped-def]
    return core.extern_elementwise(
        "",
        "",
        [dest, source, size_bytes],
        {
            (
                core.dtype("int64"),  # dest ptr
                core.dtype("int64"),  # source ptr
                core.dtype("int64"),  # size in bytes
            ): ("copy_from_src_to_dst", core.dtype("int32"))
        },
        is_pure=False,
        _semantic=_semantic,
    )

@triton.jit
def my_get_kernel(
    dest,
    src,
    nelems
):
    tl.static_assert(dest.type == src.type)
    nbytes = nelems * dest.type.element_ty.itemsize
    a = copy_from_src_to_dst_wrapper(
        dest.to(tl.int64), src.to(tl.int64), nbytes.to(tl.int64)
    )
    # tl.device_print("hello world ", a)

@triton.jit
def my_get_kernel_triton(
    dest,
    src,
    nelems
):
    tl.static_assert(dest.type == src.type)
    nbytes = nelems * dest.type.element_ty.itemsize

if __name__ == "__main__":

    src = torch.arange(10, dtype=torch.float32).to("xpu")
    dst = torch.empty_like(src).to("xpu")

    my_get_kernel[(1,)](
                dst,
                src,
                10, extern_libs={"libdevice":"/home/sdp/cherry/torch_ishmem_extension/sycl_llvm/libdevice.bc"}
            )
    
    torch.xpu.synchronize()
    print(dst)
    

