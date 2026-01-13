#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include "ishmem_extension.hpp"

namespace py = pybind11;

// Python binding for all_to_all_vdev
void all_to_all_vdev_binding(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    const std::string& group_name) {
  c10d::ishmem_extension::all_to_all_vdev(
      input, out, in_splits, out_splits_offsets, group_name);
}

// Python binding for all_to_all_vdev_2d (MoE dispatch)
void all_to_all_vdev_2d_binding(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits,
    at::Tensor& out_splits_offsets,
    const std::string& group_name,
    std::optional<int64_t> major_align) {
  c10d::ishmem_extension::all_to_all_vdev_2d(
      input, out, in_splits, out_splits_offsets, group_name, major_align);
}

// Python binding for all_to_all_vdev_2d_offset (MoE combine)
void all_to_all_vdev_2d_offset_binding(
    at::Tensor& input,
    at::Tensor& out,
    at::Tensor& in_splits_offsets,
    at::Tensor& out_splits_offsets,
    const std::string& group_name) {
  c10d::ishmem_extension::all_to_all_vdev_2d_offset(
      input, out, in_splits_offsets, out_splits_offsets, group_name);
}

// Register torch ops - only in ishmem_ext namespace
// Note: symm_mem namespace is already registered by PyTorch
TORCH_LIBRARY(ishmem_ext, m) {
  m.def(
      "all_to_all_vdev(Tensor input, Tensor out, Tensor in_splits, "
      "Tensor out_splits_offsets, str group_name) -> ()");
  m.def(
      "all_to_all_vdev_2d(Tensor input, Tensor out, Tensor in_splits, "
      "Tensor out_splits_offsets, str group_name, int? major_align) -> ()");
  m.def(
      "all_to_all_vdev_2d_offset(Tensor input, Tensor out, Tensor in_splits_offsets, "
      "Tensor out_splits_offsets, str group_name) -> ()");
}

TORCH_LIBRARY_IMPL(ishmem_ext, XPU, m) {
  m.impl("all_to_all_vdev", all_to_all_vdev_binding);
  m.impl("all_to_all_vdev_2d", all_to_all_vdev_2d_binding);
  m.impl("all_to_all_vdev_2d_offset", all_to_all_vdev_2d_offset_binding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyTorch Symmetric Memory Extension for XPU";

  m.def(
      "all_to_all_vdev",
      &all_to_all_vdev_binding,
      "Symmetric Memory-based all-to-all-v operation on XPU device",
      py::arg("input"),
      py::arg("out"),
      py::arg("in_splits"),
      py::arg("out_splits_offsets"),
      py::arg("group_name"));

  m.def(
      "all_to_all_vdev_2d",
      &all_to_all_vdev_2d_binding,
      "Symmetric Memory-based 2D all-to-all-v operation for MoE dispatch (rank-major to expert-major)",
      py::arg("input"),
      py::arg("out"),
      py::arg("in_splits"),
      py::arg("out_splits_offsets"),
      py::arg("group_name"),
      py::arg("major_align") = py::none());

  m.def(
      "all_to_all_vdev_2d_offset",
      &all_to_all_vdev_2d_offset_binding,
      "Symmetric Memory-based 2D all-to-all-v operation for MoE combine (expert-major to rank-major)",
      py::arg("input"),
      py::arg("out"),
      py::arg("in_splits_offsets"),
      py::arg("out_splits_offsets"),
      py::arg("group_name"));

  m.def("initialize", []() {
    // No-op: Symmetric memory is initialized via PyTorch's rendezvous
    static bool initialized = false;
    if (!initialized) {
      initialized = true;
    }
  }, "Initialize symmetric memory (no-op, handled by PyTorch)");

  m.def("finalize", []() {
    // No-op: Symmetric memory cleanup is handled by PyTorch
  }, "Finalize symmetric memory (no-op, handled by PyTorch)");
}
