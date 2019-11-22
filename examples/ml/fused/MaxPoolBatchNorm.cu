#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


#include "MaxPoolBatchNorm.cuh"

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> max_pool2d_batch_norm(
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode,
           const Tensor& input_batch_norm_,
           double epsilon) {

  auto r1 = AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_batch_norm_.scalar_type(), "batch_norm_stats_cuda", [&] {
    return max_pool2d_batch_norm_stream<scalar_t, int32_t>(
              input_,
              kernel_size,
              stride,
              padding,
              dilation,
              ceil_mode,
              input_batch_norm_,
              epsilon);
  });
  auto r2 = AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_batch_norm_.scalar_type(), "batch_norm_stats_cuda", [&] {
    return max_pool2d_batch_norm_fused<scalar_t, int32_t>(
              input_,
              kernel_size,
              stride,
              padding,
              dilation,
              ceil_mode,
              input_batch_norm_,
              epsilon);
  });
  return std::tuple_cat(r1, r2);
}

} }