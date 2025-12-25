#pragma once

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <sycl/sycl.hpp>
#include "SYCLHelpers.h"
#include <ishmem.h>
#include <ishmemx.h>

using namespace xpu;

#define ISHMEM_CHECK(stmt, msg)                                              \
  do {                                                                       \
    int result = (stmt);                                                     \
    TORCH_CHECK(                                                             \
        result == 0,                                                         \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg + \
            ". Error code: " + std::to_string(result));                      \
  } while (0)

namespace c10d::ishmem_extension {

// Constants aligned with NVSHMEM
constexpr int WORK_GROUP_SIZE = 256;

// Kernel for exchanging splits and calculating offsets
struct ExchangeSplitsKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const;
  void sycl_ker_config_convention(sycl::handler& cgh);

  ExchangeSplitsKernel(
      int64_t* in_splits_,
      int64_t* out_splits_offsets_,
      ishmem_team_t team_);

 private:
  int64_t* in_splits; // input splits
  int64_t* out_splits_offsets; // [output_splits | output_offsets]
  ishmem_team_t team;
  sycl_local_acc_t<int64_t, 1> shared_offsets;
};

// Kernel for AllToAllv data transfer
struct AllToAllVKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const;
  void sycl_ker_config_convention(sycl::handler& cgh);

  AllToAllVKernel(
      void* send_data_,
      void* recv_data_,
      int64_t* out_splits_offsets_,
      size_t stride_,
      ishmem_team_t team_);

 private:
  void* send_data;
  void* recv_data;
  int64_t* out_splits_offsets; // [output_splits | output_offsets]
  size_t stride; // Element size in bytes
  ishmem_team_t team;
  sycl_local_acc_t<int64_t, 1> peer_offsets;
};

void all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name);

// ==================== 2D AllToAllV Kernels (for MoE) ====================

// Constants for 2D AllToAllV
constexpr int WARP_SIZE = 32;  // SYCL sub-group size
constexpr int A2AV_TILE_SIZE = WARP_SIZE;
constexpr int NUM_TILES = WORK_GROUP_SIZE / A2AV_TILE_SIZE;

// Kernel for exchanging splits and calculating offsets for 2D AllToAllV
// Template parameter HAS_IN_OFFSETS indicates if input offsets are provided
template <bool HAS_IN_OFFSETS>
struct ExchangeSplitsKernel2D : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const;
  void sycl_ker_config_convention(sycl::handler& cgh);

  ExchangeSplitsKernel2D(
      int64_t* in_splits_offsets_,
      int64_t* out_splits_offsets_,
      ishmem_team_t team_,
      int ne_,
      size_t input_dim0_,
      bool rank_is_row_in_);

 private:
  int64_t* in_splits_offsets;   // input splits (and optionally offsets)
  int64_t* out_splits_offsets;  // [output_splits | source_offsets]
  ishmem_team_t team;
  int ne;                       // number of experts per rank
  size_t input_dim0;            // size of dim 0 of input tensor
  bool rank_is_row_in;          // true: rank-major input, false: expert-major input
  sycl_local_acc_t<int64_t, 1> shared_offsets;
};

// Kernel for 2D AllToAllV data transfer
struct AllToAllVKernel2D : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const;
  void sycl_ker_config_convention(sycl::handler& cgh);

  AllToAllVKernel2D(
      void* send_data_,
      void* recv_data_,
      int64_t* in_splits_,
      int64_t* out_splits_offsets_,
      size_t stride_,
      int minor_size_,
      int major_size_,
      int64_t major_align_,
      bool rank_is_row_out_,
      ishmem_team_t team_);

 private:
  void* send_data;
  void* recv_data;
  int64_t* in_splits;
  int64_t* out_splits_offsets;  // [output_splits | source_offsets]
  size_t stride;                // Element size in bytes
  int minor_size;               // npes for dispatch, ne for combine
  int major_size;               // ne for dispatch, npes for combine
  int64_t major_align;          // alignment for major dimension
  bool rank_is_row_out;         // output layout
  ishmem_team_t team;
  sycl_local_acc_t<int64_t, 1> tile_prefix_sums;  // [NUM_TILES][A2AV_TILE_SIZE]
  sycl_local_acc_t<int64_t, 1> len_per_tile;       // [NUM_TILES]
  sycl_local_acc_t<int64_t, 1> start_offset_per_tile;  // [NUM_TILES]
};

// 2D AllToAllV for MoE dispatch (rank-major to expert-major)
void all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name,
    std::optional<int64_t> major_align);

// 2D AllToAllV for MoE combine (expert-major to rank-major, with input offsets)
void all_to_all_vdev_2d_offset(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    std::string group_name);

} // namespace c10d::ishmem_extension
