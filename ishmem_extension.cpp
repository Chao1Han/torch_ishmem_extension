#include <ATen/ceil_div.h>
#include <ATen/xpu/XPUContext.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <sycl/sycl.hpp>
#include "ishmem_extension.hpp"

namespace c10d::ishmem_extension {

// ==================== ExchangeSplitsKernel Implementation ====================

void ExchangeSplitsKernel::operator()(sycl::nd_item<1> item) const {
  int my_pe = ishmem_team_my_pe(team);
  int n_pes = ishmem_team_n_pes(team);
  int tid = item.get_global_linear_id();
  auto grp = item.get_group();

  int64_t* input_splits = in_splits;
  int64_t* output_splits = out_splits_offsets;
  int64_t* source_offsets = out_splits_offsets + n_pes;

  // Shared memory for prefix sum results
  auto shared_ptr =
      shared_offsets.get_multi_ptr<sycl::access::decorated::no>().get();

  // Calculate source offsets (prefix sum of input_splits)
  if (tid < n_pes) {
    shared_ptr[tid] = (tid == 0) ? 0 : input_splits[tid - 1];
  }
  sycl::group_barrier(grp);

  // Simple prefix sum for source offsets
  if (tid == 0) {
    int64_t sum = 0;
    for (int i = 0; i < n_pes; ++i) {
      int64_t val = shared_ptr[i];
      shared_ptr[i] = sum;
      sum += val;
    }
  }
  sycl::group_barrier(grp);

  // Exchange splits with remote PEs using device-side RMA
  if (tid < n_pes) {
    int target_global = ishmem_team_translate_pe(team, tid, ISHMEM_TEAM_WORLD);
    // Get values from local/shared memory
    int64_t split_val = input_splits[tid];
    int64_t offset_val = shared_ptr[tid];
    
    // Send my split for this peer to target PE's output_splits[my_pe]
    // This tells target PE how much data I will send to it
    // Use ishmem_int64_p (put scalar value) because source may be in shared memory
    ishmem_int64_p(&output_splits[my_pe], split_val, target_global);
    
    // Send my source offset for this peer to target PE's source_offsets[my_pe]
    // This tells target PE where to read data from my buffer
    ishmem_int64_p(&source_offsets[my_pe], offset_val, target_global);
  }

  ishmemx_barrier_all_work_group(grp);
}

void ExchangeSplitsKernel::sycl_ker_config_convention(sycl::handler& cgh) {
  shared_offsets = sycl_local_acc_t<int64_t, 1>(512, cgh);
}

ExchangeSplitsKernel::ExchangeSplitsKernel(
    int64_t* in_splits_,
    int64_t* out_splits_offsets_,
    ishmem_team_t team_)
    : in_splits(in_splits_),
      out_splits_offsets(out_splits_offsets_),
      team(team_),
      shared_offsets() {}

// ==================== AllToAllVKernel Implementation ====================

void AllToAllVKernel::operator()(sycl::nd_item<1> item) const {
  auto grp = item.get_group();
  int group_id = item.get_group_linear_id();
  int num_groups = item.get_group_range(0);

  int my_pe = ishmem_team_my_pe(team);
  int n_pes = ishmem_team_n_pes(team);

  int64_t* output_splits = out_splits_offsets;
  int64_t* source_offsets = out_splits_offsets + n_pes;

  int groups_per_peer = sycl::max(num_groups / n_pes, 1);

  // Shared memory for output offsets (prefix sum of output_splits)
  auto peer_ptr =
      peer_offsets.get_multi_ptr<sycl::access::decorated::no>().get();

  // Calculate output offsets (prefix sum)
  if (item.get_local_linear_id() == 0) {
    peer_ptr[0] = 0;
    for (int i = 1; i < n_pes; ++i) {
      peer_ptr[i] = peer_ptr[i - 1] + output_splits[i - 1];
    }
  }
  sycl::group_barrier(grp);

  // Each work-group handles data transfer to one or more peers
  for (int i = group_id / groups_per_peer; i < n_pes;
       i += num_groups / groups_per_peer) {
    int peer = (my_pe + i) % n_pes; // Round-robin to avoid hotspots
    int peer_global = ishmem_team_translate_pe(team, peer, ISHMEM_TEAM_WORLD);

    // Total bytes to receive from peer
    size_t peer_size = output_splits[peer] * stride;
    if (peer_size == 0)
      continue;

    // Divide work among groups handling this peer
    size_t block_size = peer_size / groups_per_peer;
    size_t block_offset = block_size * (group_id % groups_per_peer);
    size_t source_offset = source_offsets[peer] * stride + block_offset;
    size_t write_offset = peer_ptr[peer] * stride + block_offset;

    // Use ISHMEM work-group RMA to get data
    ishmemx_getmem_nbi_work_group(
        (char*)recv_data + write_offset,
        (char*)send_data + source_offset,
        block_size,
        peer_global,
        grp);
  }

  // Write output offsets back (for compatibility)
  if (group_id == 0 && item.get_local_linear_id() < n_pes) {
    source_offsets[item.get_local_linear_id()] =
        peer_ptr[item.get_local_linear_id()];
  }

  // Ensure all RMA operations complete
  if (grp.leader()) {
    ishmem_quiet();
  }
}

void AllToAllVKernel::sycl_ker_config_convention(sycl::handler& cgh) {
  peer_offsets = sycl_local_acc_t<int64_t, 1>(512, cgh);
}

AllToAllVKernel::AllToAllVKernel(
    void* send_data_,
    void* recv_data_,
    int64_t* out_splits_offsets_,
    size_t stride_,
    ishmem_team_t team_)
    : send_data(send_data_),
      recv_data(recv_data_),
      out_splits_offsets(out_splits_offsets_),
      stride(stride_),
      team(team_),
      peer_offsets() {}

void all_to_all_vdev(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name) {

  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto in_splits_hdl =
      c10d::symmetric_memory::rendezvous(in_splits, group_name);
  auto out_splits_offsets_hdl =
      c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);

  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();

  // Get ISHMEM team (for now use WORLD, can extend to support sub-teams)
  ishmem_team_t team = ISHMEM_TEAM_WORLD;

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* in_splits_ptr = (int64_t*)(in_splits.const_data_ptr());
  int64_t* out_splits_offsets_ptr =
      out_splits_offsets.mutable_data_ptr<int64_t>();

  TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(in_splits.is_contiguous() && out_splits_offsets.is_contiguous());
  TORCH_CHECK_EQ(input.dtype(), out.dtype());
  TORCH_CHECK_EQ(in_splits.dtype(), at::kLong);
  TORCH_CHECK_EQ(out_splits_offsets.dtype(), at::kLong);
  TORCH_CHECK_EQ(in_splits.dim(), 1);
  TORCH_CHECK_EQ(out_splits_offsets.dim(), 2);
  TORCH_CHECK_EQ(in_splits.size(0), world_size);
  TORCH_CHECK_EQ(out_splits_offsets.size(0), 2);
  TORCH_CHECK_EQ(out_splits_offsets.size(1), world_size);

  size_t stride = input.element_size();
  auto queue = at::xpu::getCurrentXPUStream(input.device().index());

  auto global_range_1 = WORK_GROUP_SIZE;
  auto local_range_1 = WORK_GROUP_SIZE;
  auto kfn_1 =
      ExchangeSplitsKernel(in_splits_ptr, out_splits_offsets_ptr, team);

  sycl_kernel_submit(global_range_1, local_range_1, queue, kfn_1);
  
  // CRITICAL: Wait for ExchangeSplitsKernel to complete before starting AllToAllVKernel
  // The AllToAllVKernel needs the splits/offsets data from ExchangeSplitsKernel
  c10::xpu::syncStreamsOnDevice(input.device().index());

  size_t size = input.nbytes();
  int groups_per_pe = 4;
  if (size < 1024 * 1024) { // < 1MB
    groups_per_pe = 1;
  } else if (size < 16 * 1024 * 1024) { // < 16MB
    groups_per_pe = 2;
  }
  int num_work_groups = world_size * groups_per_pe;

  auto global_range_2 = num_work_groups * WORK_GROUP_SIZE;
  auto local_range_2 = WORK_GROUP_SIZE;
  auto kfn_2 = AllToAllVKernel(
      input_ptr, output_ptr, out_splits_offsets_ptr, stride, team);

  sycl_kernel_submit(global_range_2, local_range_2, queue, kfn_2);
}

// ==================== 2D AllToAllV Implementation (for MoE) ====================

// Helper function: Simple prefix sum for a sub-group
// Returns the sum of all elements
template <int SIZE>
SYCL_EXTERNAL int64_t prefix_sum_subgroup(
    int64_t* odata,
    int64_t* idata,
    int n,
    sycl::nd_item<1>& item,
    int64_t* shared_tmp) {
  static_assert(SIZE <= A2AV_TILE_SIZE, "SIZE must be <= A2AV_TILE_SIZE");
  
  int lane_id = item.get_local_linear_id() % A2AV_TILE_SIZE;
  
  // Load input
  int64_t val = (lane_id < n) ? idata[lane_id] : 0;
  
  // Exclusive prefix sum using sequential algorithm in shared memory
  if (lane_id < n) {
    shared_tmp[lane_id] = val;
  }
  sycl::group_barrier(item.get_group());
  
  // First thread computes prefix sum
  int64_t total_sum = 0;
  if (lane_id == 0) {
    int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
      int64_t v = shared_tmp[i];
      shared_tmp[i] = sum;
      sum += v;
    }
    total_sum = sum;
  }
  sycl::group_barrier(item.get_group());
  
  // Write output
  if (lane_id < n) {
    odata[lane_id] = shared_tmp[lane_id];
  }
  
  // Broadcast total sum
  total_sum = sycl::group_broadcast(item.get_group(), shared_tmp[n-1] + idata[n-1], 0);
  
  return total_sum;
}

// ExchangeSplitsKernel2D Implementation
template <bool HAS_IN_OFFSETS>
void ExchangeSplitsKernel2D<HAS_IN_OFFSETS>::operator()(sycl::nd_item<1> item) const {
  int my_pe = ishmem_team_my_pe(team);
  int n_pes = ishmem_team_n_pes(team);
  int nsplits = n_pes * ne;
  int tid = item.get_local_linear_id();
  auto grp = item.get_group();

  int64_t* input_splits = in_splits_offsets;
  int64_t* output_splits = out_splits_offsets;
  int64_t* source_offsets = out_splits_offsets + nsplits;

  auto shared_ptr = shared_offsets.get_multi_ptr<sycl::access::decorated::no>().get();

  int64_t* input_offsets = nullptr;
  if constexpr (HAS_IN_OFFSETS) {
    // Input offsets are provided
    input_offsets = in_splits_offsets + nsplits;
  } else {
    // Calculate input offsets via prefix sum
    // Simple prefix sum in shared memory
    if (tid < nsplits) {
      shared_ptr[tid] = input_splits[tid];
    }
    sycl::group_barrier(grp);

    if (tid == 0) {
      int64_t sum = 0;
      for (int i = 0; i < nsplits; ++i) {
        int64_t val = shared_ptr[i];
        shared_ptr[i] = sum;
        sum += val;
      }
    }
    sycl::group_barrier(grp);

    input_offsets = shared_ptr;
  }

  // Exchange splits and offsets
  if (tid < nsplits) {
    int peer, e, dst_offset;
    if (rank_is_row_in) {
      peer = tid / ne;
      e = tid % ne;
      dst_offset = e * n_pes + my_pe;
    } else {
      peer = tid % n_pes;
      e = tid / n_pes;
      dst_offset = my_pe * ne + e;
    }

    int64_t split_val = input_splits[tid];
    int64_t offset_val = input_offsets[tid];
    int peer_global = ishmem_team_translate_pe(team, peer, ISHMEM_TEAM_WORLD);

    // Send offset and split to peer using ishmem_int64_p (put scalar value)
    // This is needed because input_offsets may point to shared memory, not symmetric memory
    ishmem_int64_p(&source_offsets[dst_offset], offset_val, peer_global);
    ishmem_int64_p(&output_splits[dst_offset], split_val, peer_global);
  }

  ishmemx_barrier_all_work_group(grp);
}

template <bool HAS_IN_OFFSETS>
void ExchangeSplitsKernel2D<HAS_IN_OFFSETS>::sycl_ker_config_convention(sycl::handler& cgh) {
  shared_offsets = sycl_local_acc_t<int64_t, 1>(512, cgh);
}

template <bool HAS_IN_OFFSETS>
ExchangeSplitsKernel2D<HAS_IN_OFFSETS>::ExchangeSplitsKernel2D(
    int64_t* in_splits_offsets_,
    int64_t* out_splits_offsets_,
    ishmem_team_t team_,
    int ne_,
    size_t input_dim0_,
    bool rank_is_row_in_)
    : in_splits_offsets(in_splits_offsets_),
      out_splits_offsets(out_splits_offsets_),
      team(team_),
      ne(ne_),
      input_dim0(input_dim0_),
      rank_is_row_in(rank_is_row_in_),
      shared_offsets() {}

// Explicit template instantiations
template struct ExchangeSplitsKernel2D<true>;
template struct ExchangeSplitsKernel2D<false>;

// AllToAllVKernel2D Implementation
void AllToAllVKernel2D::operator()(sycl::nd_item<1> item) const {
  auto grp = item.get_group();
  int group_id = item.get_group_linear_id();
  int num_groups = item.get_group_range(0);
  int tid = item.get_local_linear_id();

  int nsplits = minor_size * major_size;
  int64_t* output_splits = out_splits_offsets;
  int64_t* source_offsets = out_splits_offsets + nsplits;

  // Get shared memory pointers
  auto tile_sums_ptr = tile_prefix_sums.get_multi_ptr<sycl::access::decorated::no>().get();
  auto len_ptr = len_per_tile.get_multi_ptr<sycl::access::decorated::no>().get();
  auto start_ptr = start_offset_per_tile.get_multi_ptr<sycl::access::decorated::no>().get();

  // Each "tile" handles one row of the 2D split matrix
  int tile_id = tid / A2AV_TILE_SIZE;
  int lane_id = tid % A2AV_TILE_SIZE;

  // Calculate how many splits this tile handles
  int nsplits_per_tile = (tile_id < major_size) ? minor_size : 0;

  // Step 1: Each tile calculates prefix sum for its row
  if (tile_id < major_size && lane_id < minor_size) {
    // Read output splits for this tile's row
    tile_sums_ptr[tile_id * A2AV_TILE_SIZE + lane_id] = output_splits[tile_id * minor_size + lane_id];
  }
  sycl::group_barrier(grp);

  // Compute prefix sum for each tile
  if (tile_id < major_size) {
    int64_t* row_data = tile_sums_ptr + tile_id * A2AV_TILE_SIZE;
    
    // First lane of each tile computes prefix sum
    if (lane_id == 0) {
      int64_t sum = 0;
      int64_t total = 0;
      for (int i = 0; i < minor_size; ++i) {
        int64_t val = row_data[i];
        row_data[i] = sum;
        sum += val;
        total = sum;
      }
      
      // Apply major alignment
      if (major_align != 0) {
        int64_t aligned_len = (total + major_align - 1) / major_align * major_align;
        len_ptr[tile_id] = (aligned_len == 0) ? major_align : aligned_len;
      } else {
        len_ptr[tile_id] = total;
      }
    }
  }
  sycl::group_barrier(grp);

  // Step 2: Calculate starting offset for each tile
  if (tid == 0) {
    int64_t sum = 0;
    for (int i = 0; i < major_size; ++i) {
      start_ptr[i] = sum;
      sum += len_ptr[i];
    }
  }
  sycl::group_barrier(grp);

  // Add tile offset to each element
  if (tile_id < major_size && lane_id < minor_size) {
    tile_sums_ptr[tile_id * A2AV_TILE_SIZE + lane_id] += start_ptr[tile_id];
  }
  sycl::group_barrier(grp);

  // Step 3: Perform data transfer
  int n_pes = ishmem_team_n_pes(team);
  
  for (int eid = group_id; eid < nsplits; eid += num_groups) {
    int row = eid / minor_size;
    int col = eid % minor_size;

    int64_t peer_size = output_splits[eid] * stride;
    if (peer_size == 0) continue;

    int64_t src_offset = source_offsets[eid] * stride;
    int64_t e_offset = tile_sums_ptr[row * A2AV_TILE_SIZE + col];
    int64_t write_offset = e_offset * stride;

    int peer_idx = rank_is_row_out ? row : col;
    int peer_global = ishmem_team_translate_pe(team, peer_idx, ISHMEM_TEAM_WORLD);

    ishmemx_getmem_nbi_work_group(
        (char*)recv_data + write_offset,
        (char*)send_data + src_offset,
        peer_size,
        peer_global,
        grp);
  }

  // Write output offsets back
  if (group_id == 0 && tid < nsplits) {
    int row = tid / minor_size;
    int col = tid % minor_size;
    source_offsets[tid] = tile_sums_ptr[row * A2AV_TILE_SIZE + col];
  }

  // Ensure all RMA operations complete
  if (grp.leader()) {
    ishmem_quiet();
  }
}

void AllToAllVKernel2D::sycl_ker_config_convention(sycl::handler& cgh) {
  tile_prefix_sums = sycl_local_acc_t<int64_t, 1>(NUM_TILES * A2AV_TILE_SIZE, cgh);
  len_per_tile = sycl_local_acc_t<int64_t, 1>(NUM_TILES, cgh);
  start_offset_per_tile = sycl_local_acc_t<int64_t, 1>(NUM_TILES, cgh);
}

AllToAllVKernel2D::AllToAllVKernel2D(
    void* send_data_,
    void* recv_data_,
    int64_t* in_splits_,
    int64_t* out_splits_offsets_,
    size_t stride_,
    int minor_size_,
    int major_size_,
    int64_t major_align_,
    bool rank_is_row_out_,
    ishmem_team_t team_)
    : send_data(send_data_),
      recv_data(recv_data_),
      in_splits(in_splits_),
      out_splits_offsets(out_splits_offsets_),
      stride(stride_),
      minor_size(minor_size_),
      major_size(major_size_),
      major_align(major_align_),
      rank_is_row_out(rank_is_row_out_),
      team(team_),
      tile_prefix_sums(),
      len_per_tile(),
      start_offset_per_tile() {}

void all_to_all_vdev_2d(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    std::string group_name,
    std::optional<int64_t> major_align) {
  /* Perform a 2D AllToAllv shuffle operation using ISHMEM.
   * Arguments:
   *  - `input` is the input tensor
   *  - `out` is the output tensor
   *  - `in_splits` is a 1D tensor of size (`world_size` * `ne`)
   *  - `out_splits_offsets` is a 2D tensor of size (2, `world_size` * `ne`)
   *  - `group_name` is the group name for symmetric memory
   *  - `major_align` is the alignment of the major dimension
   *
   *  2D AllToAllv shuffle transposes from rank-major to expert-major order:
   *    Source: | Rank 0 | Rank 1 |  (rank-major: c0,c1,c2,c3 on R0)
   *    Dest:   | Rank 0 | Rank 1 |  (expert-major: c0,d0 on R0)
   */
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto in_splits_hdl = c10d::symmetric_memory::rendezvous(in_splits, group_name);
  auto out_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);

  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();

  TORCH_CHECK(world_size <= A2AV_TILE_SIZE, 
      "world_size must be smaller than A2AV_TILE_SIZE (", A2AV_TILE_SIZE, ")");

  int64_t major_align_val = major_align.value_or(1);
  TORCH_CHECK(major_align_val > 0, "major_align must be positive");

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* in_splits_ptr = in_splits.mutable_data_ptr<int64_t>();
  int64_t* out_splits_offsets_ptr = out_splits_offsets.mutable_data_ptr<int64_t>();

  // Shape checks
  TORCH_CHECK(in_splits.is_contiguous() && out_splits_offsets.is_contiguous()
      && input.is_contiguous() && out.is_contiguous(),
      "All tensors must be contiguous");
  
  auto in_split_shape = in_splits.sizes();
  auto out_split_shape = out_splits_offsets.sizes();
  TORCH_CHECK(out_split_shape.size() == 2 && out_split_shape[0] == 2
      && out_split_shape[1] == in_split_shape[0]
      && in_split_shape[0] % world_size == 0,
      "out_splits_offsets must be 2D (2, nsplits), nsplits must be multiple of world_size");

  // Consistency checks
  TORCH_CHECK(input.dtype() == out.dtype() && input.stride(0) == out.stride(0),
      "input and out must have same dtype and stride at dim 0");
  TORCH_CHECK(in_splits.scalar_type() == at::kLong && out_splits_offsets.scalar_type() == at::kLong,
      "splits and offsets must be int64");

  int ne = in_split_shape[0] / world_size;
  TORCH_CHECK(ne <= NUM_TILES, "Number of experts must be smaller than NUM_TILES (", NUM_TILES, ")");

  ishmem_team_t team = ISHMEM_TEAM_WORLD;
  auto queue = at::xpu::getCurrentXPUStream(input.device().index());

  // Kernel 1: Exchange splits and offsets
  size_t input_dim0 = input.size(0);
  bool rank_is_row_in = true;

  auto global_range_1 = WORK_GROUP_SIZE;
  auto local_range_1 = WORK_GROUP_SIZE;
  auto kfn_1 = ExchangeSplitsKernel2D<false>(
      in_splits_ptr, out_splits_offsets_ptr, team, ne, input_dim0, rank_is_row_in);

  sycl_kernel_submit(global_range_1, local_range_1, queue, kfn_1);

  // Sync between kernels
  c10::xpu::syncStreamsOnDevice(input.device().index());

  // Kernel 2: Perform data transfer
  int num_work_groups = std::min(world_size * ne, world_size > 8 ? 8 : 64);

  size_t stride_bytes = input.stride(0) * input.element_size();
  bool rank_is_row_out = !rank_is_row_in;

  auto global_range_2 = num_work_groups * WORK_GROUP_SIZE;
  auto local_range_2 = WORK_GROUP_SIZE;
  auto kfn_2 = AllToAllVKernel2D(
      input_ptr, output_ptr, in_splits_ptr, out_splits_offsets_ptr,
      stride_bytes, world_size, ne, major_align_val, rank_is_row_out, team);

  sycl_kernel_submit(global_range_2, local_range_2, queue, kfn_2);
}

void all_to_all_vdev_2d_offset(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    std::string group_name) {
  /* Perform a 2D AllToAllv shuffle with input offsets (for MoE combine).
   * This is the reverse operation of all_to_all_vdev_2d.
   * 
   * Arguments:
   *  - `input` is the input tensor (expert-major)
   *  - `out` is the output tensor (rank-major)
   *  - `in_splits_offsets` is a 2D tensor of size (2, `ne` * `world_size`)
   *    Row 0: input splits, Row 1: input offsets
   *  - `out_splits_offsets` is a 2D tensor of size (2, `world_size` * `ne`)
   *    Row 0: output splits (OUT), Row 1: output offsets (OUT)
   */
  auto input_hdl = c10d::symmetric_memory::rendezvous(input, group_name);
  auto out_hdl = c10d::symmetric_memory::rendezvous(out, group_name);
  auto in_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(in_splits_offsets, group_name);
  auto out_splits_offsets_hdl = c10d::symmetric_memory::rendezvous(out_splits_offsets, group_name);

  int rank = input_hdl->get_rank();
  int world_size = input_hdl->get_world_size();

  TORCH_CHECK(world_size <= NUM_TILES, 
      "world_size must be smaller than NUM_TILES (", NUM_TILES, ")");

  int64_t major_align_val = 0;  // No alignment for combine

  void* input_ptr = input.data_ptr();
  void* output_ptr = out.mutable_data_ptr();
  int64_t* in_splits_offsets_ptr = in_splits_offsets.mutable_data_ptr<int64_t>();
  int64_t* out_splits_offsets_ptr = out_splits_offsets.mutable_data_ptr<int64_t>();

  // Shape checks
  TORCH_CHECK(in_splits_offsets.is_contiguous() && out_splits_offsets.is_contiguous()
      && input.is_contiguous() && out.is_contiguous(),
      "All tensors must be contiguous");

  auto in_split_shape = in_splits_offsets.sizes();
  auto out_split_shape = out_splits_offsets.sizes();
  TORCH_CHECK(in_split_shape.size() == 2 && in_split_shape[0] == 2
      && in_split_shape[1] % world_size == 0,
      "in_splits_offsets must be 2D (2, nsplits), nsplits must be multiple of world_size");

  // Consistency checks
  TORCH_CHECK(input.dtype() == out.dtype() && input.stride(0) == out.stride(0),
      "input and out must have same dtype and stride at dim 0");
  TORCH_CHECK(in_splits_offsets.scalar_type() == at::kLong && out_splits_offsets.scalar_type() == at::kLong,
      "splits and offsets must be int64");

  int ne = in_split_shape[1] / world_size;
  TORCH_CHECK(ne <= A2AV_TILE_SIZE, "Number of experts must be smaller than A2AV_TILE_SIZE (", A2AV_TILE_SIZE, ")");

  ishmem_team_t team = ISHMEM_TEAM_WORLD;
  auto queue = at::xpu::getCurrentXPUStream(input.device().index());

  // Kernel 1: Exchange splits and offsets (with input offsets provided)
  size_t input_dim0 = input.size(0);
  bool rank_is_row_in = false;  // Expert-major input

  auto global_range_1 = WORK_GROUP_SIZE;
  auto local_range_1 = WORK_GROUP_SIZE;
  auto kfn_1 = ExchangeSplitsKernel2D<true>(
      in_splits_offsets_ptr, out_splits_offsets_ptr, team, ne, input_dim0, rank_is_row_in);

  sycl_kernel_submit(global_range_1, local_range_1, queue, kfn_1);

  // Sync between kernels
  c10::xpu::syncStreamsOnDevice(input.device().index());

  // Kernel 2: Perform data transfer
  int num_work_groups = std::min(world_size * ne, world_size > 8 ? 8 : 64);

  size_t stride_bytes = input.stride(0) * input.element_size();
  bool rank_is_row_out = !rank_is_row_in;  // Rank-major output

  auto global_range_2 = num_work_groups * WORK_GROUP_SIZE;
  auto local_range_2 = WORK_GROUP_SIZE;
  auto kfn_2 = AllToAllVKernel2D(
      input_ptr, output_ptr, in_splits_offsets_ptr, out_splits_offsets_ptr,
      stride_bytes, ne, world_size, major_align_val, rank_is_row_out, team);

  sycl_kernel_submit(global_range_2, local_range_2, queue, kfn_2);
}

} // namespace c10d::ishmem_extension
