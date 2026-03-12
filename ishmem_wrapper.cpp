#include "ishmem_wrapper.hpp"

#include <cstddef>
#include <cstdint>
#include <ishmem.h>
#include <ishmemx.h>

namespace {

constexpr int32_t kSignalSet = 0;
constexpr int32_t kSignalAdd = 5;

inline auto current_work_group() {
  return sycl::ext::oneapi::this_work_item::get_work_group<1>();
}

} // namespace

extern "C" {

SYCL_EXTERNAL int32_t ishmemx_putmem_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe) {
  auto grp = current_work_group();
  ishmemx_putmem_work_group(
      reinterpret_cast<void*>(dest),
      reinterpret_cast<const void*>(source),
      static_cast<size_t>(size_bytes),
      pe,
      grp);
  return 0;
}

SYCL_EXTERNAL int32_t ishmemx_getmem_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe) {
  auto grp = current_work_group();
  auto dst = reinterpret_cast<uint8_t*>(dest);
  auto src = reinterpret_cast<const uint8_t*>(source);

  for (int64_t i = 0; i < size_bytes; i++) {
    dst[i] = src[i];
  }

  // ishmemx_getmem_work_group(
  //     reinterpret_cast<void*>(dest),
  //     reinterpret_cast<const void*>(source),
  //     static_cast<size_t>(size_bytes),
  //     pe,
  //     grp);
  return 0;
}

SYCL_EXTERNAL int32_t ishmemx_getmem_nbi_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe) {
  auto grp = current_work_group();
  ishmemx_getmem_nbi_work_group(
      reinterpret_cast<void*>(dest),
      reinterpret_cast<const void*>(source),
      static_cast<size_t>(size_bytes),
      pe,
      grp);
  return 0;
}

SYCL_EXTERNAL int32_t ishmemx_putmem_signal_work_group_wrapper(
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int64_t signal,
    uint64_t sig_val,
    int32_t sig_op,
    int32_t pe) {
  auto grp = current_work_group();
  ishmemx_putmem_signal_work_group(
      reinterpret_cast<void*>(dest),
      reinterpret_cast<const void*>(source),
      static_cast<size_t>(size_bytes),
      reinterpret_cast<uint64_t*>(signal),
      sig_val,
      sig_op,
      pe,
      grp);
  return 0;
}

SYCL_EXTERNAL int32_t ishmemx_signal_op_wrapper(
    int64_t sig_addr,
    int64_t signal,
    int32_t sig_op,
    int32_t pe) {
  auto* remote_signal = reinterpret_cast<uint64_t*>(sig_addr);
  auto value = static_cast<uint64_t>(signal);

  switch (sig_op) {
    case kSignalSet:
      ishmem_uint64_atomic_set(remote_signal, value, pe);
      return 0;
    case kSignalAdd:
      ishmem_uint64_atomic_add(remote_signal, value, pe);
      return 0;
    default:
      return -1;
  }
}

SYCL_EXTERNAL int32_t ishmemx_alltoallmem_work_group_wrapper(
    int32_t team,
    int64_t dest,
    int64_t source,
    int64_t size_bytes) {
  auto grp = current_work_group();
  return ishmemx_alltoallmem_work_group(
      static_cast<ishmem_team_t>(team),
      reinterpret_cast<void*>(dest),
      reinterpret_cast<const void*>(source),
      static_cast<size_t>(size_bytes),
      grp);
}

SYCL_EXTERNAL int32_t ishmemx_broadcastmem_work_group_wrapper(
    int32_t team,
    int64_t dest,
    int64_t source,
    int64_t size_bytes,
    int32_t pe_root) {
  auto grp = current_work_group();
  return ishmemx_broadcastmem_work_group(
      static_cast<ishmem_team_t>(team),
      reinterpret_cast<void*>(dest),
      reinterpret_cast<const void*>(source),
      static_cast<size_t>(size_bytes),
      pe_root,
      grp);
}

} // extern "C"