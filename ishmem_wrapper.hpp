#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#ifdef __cplusplus
extern "C" {
#endif

SYCL_EXTERNAL int32_t ishmemx_putmem_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe);

SYCL_EXTERNAL int32_t ishmemx_getmem_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe);

SYCL_EXTERNAL int32_t ishmemx_getmem_nbi_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe);

SYCL_EXTERNAL int32_t ishmemx_putmem_signal_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int64_t signal,
    uint64_t sig_val,
    int32_t sig_op,
    int32_t pe);

SYCL_EXTERNAL int32_t ishmemx_signal_op_wrapper(
    int64_t sig_addr,
    int64_t signal,
    int32_t sig_op,
    int32_t pe);

SYCL_EXTERNAL int32_t ishmemx_alltoallmem_work_group_wrapper(
    int32_t team,
    int64_t dest,
    int64_t source,
    int64_t size_bytes);

SYCL_EXTERNAL int32_t ishmemx_broadcastmem_work_group_wrapper(
    int32_t team,
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe_root);

#ifdef __cplusplus
}
#endif