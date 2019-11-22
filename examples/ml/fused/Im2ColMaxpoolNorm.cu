#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "Im2ColMaxpoolNorm.cuh"

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_stream(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size_maxpool,
    IntArrayRef stride_maxpool,
    IntArrayRef padding_maxpool,
    IntArrayRef dilation_maxpool,
    bool ceil_mode,
    const Tensor& input_batch_norm) {
  // auto r2 = AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_batch_norm.scalar_type(), "batch_norm_stats_cuda", [&] {
  //   return im2col_batch_norm_fused<scalar_t, int32_t>(
  //     input, kernel_size, dilation, padding, stride,
  //     input_batch_norm, 0.1);
  // });
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_batch_norm.scalar_type(), "batch_norm_stats_cuda", [&] {
    return im2col_maxpool_batch_norm_stream<scalar_t, int32_t>(
      input, kernel_size, dilation, padding, stride,
      input_maxpool_,
      kernel_size_maxpool,
      stride_maxpool,
      padding_maxpool,
      dilation_maxpool,
      ceil_mode,
      input_batch_norm, 0.1);
  });
  auto r2 = AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_batch_norm.scalar_type(), "batch_norm_stats_cuda", [&] {
    return im2col_maxpool_batch_norm_fused<scalar_t, int32_t>(
      input, kernel_size, dilation, padding, stride,
      input_maxpool_,
      kernel_size_maxpool,
      stride_maxpool,
      padding_maxpool,
      dilation_maxpool,
      ceil_mode,
      input_batch_norm, 0.1);
  });
  return std::tuple_cat(r2, r2);
}


} // namespace native
} // namespace at
