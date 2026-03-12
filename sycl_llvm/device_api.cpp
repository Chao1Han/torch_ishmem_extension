#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>

#ifdef __cplusplus
extern "C" {
#endif

SYCL_EXTERNAL int32_t copy_from_src_to_dst(
    int64_t dest,
    int64_t source,
    int64_t size_bytes) {

  auto dst = reinterpret_cast<uint8_t*>(dest);
  auto src = reinterpret_cast<const uint8_t*>(source);

  for (int64_t i = 0; i < size_bytes; i++) {
    dst[i] = src[i];
  }
  return 0;
}

#ifdef __cplusplus
}
#endif