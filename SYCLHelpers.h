#pragma once

#include <sycl/sycl.hpp>
#include <type_traits>

namespace xpu {

// SYCL memory scopes
static constexpr auto sycl_mem_odr_rlx = sycl::memory_order::relaxed;
static constexpr auto sycl_mem_scp_dev = sycl::memory_scope::device;
static constexpr auto sycl_global_space = sycl::access::address_space::global_space;

template <typename scalar_t, int dims = 1>
using sycl_local_acc_t = sycl::local_accessor<scalar_t, dims>;

template <typename T>
using sycl_global_ptr = typename sycl::global_ptr<T>;

// Kernel configuration convention marker
struct __SYCL_KER_CONFIG_CONVENTION__ {};

// SYCL kernel submission helpers
template <typename ker_t>
static inline typename std::enable_if<
    std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    ker.sycl_ker_config_convention(cgh);
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}

template <typename ker_t>
static inline typename std::enable_if<
    !std::is_base_of_v<__SYCL_KER_CONFIG_CONVENTION__, ker_t>,
    void>::type
sycl_kernel_submit(
    int64_t global_range,
    int64_t local_range,
    ::sycl::queue q,
    ker_t ker) {
  auto cgf = [&](::sycl::handler& cgh) {
    cgh.parallel_for<ker_t>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        ker);
  };
  q.submit(cgf);
}

} // namespace xpu
