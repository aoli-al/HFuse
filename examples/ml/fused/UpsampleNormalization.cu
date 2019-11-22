#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../cuda/UpSample.cuh"
#include "../cuda/KernelUtils.cuh"
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../cuda/UpSample.cuh"

#include "UpsampleNormalization.cuh"

namespace at {
namespace native {

std::tuple<Tensor, Tensor, Tensor, Tensor> upsample_batchnorm(
  const Tensor& input,
  IntArrayRef output_size,
  bool align_corners,
  const Tensor& input_bn_, double epsilon) {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_bn_.scalar_type(), "batch_norm_backward_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(input_bn_)) {
        return upsample_batchnorm_stm<scalar_t, int32_t>(
          input,
          output_size,
          align_corners,
          input_bn_,
          epsilon);
      } else {
        return upsample_batchnorm_stm<scalar_t, int64_t>(
          input,
          output_size,
          align_corners,
          input_bn_,
          epsilon);
      }
    });
    auto t1 = AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_bn_.scalar_type(), "batch_norm_backward_cuda", [&] {
      if (cuda::detail::canUse32BitIndexMath(input_bn_)) {
        return upsample_batchnorm_fused<scalar_t, int32_t>(
          input,
          output_size,
          align_corners,
          input_bn_,
          epsilon);
      } else {
        return upsample_batchnorm_fused<scalar_t, int64_t>(
          input,
          output_size,
          align_corners,
          input_bn_,
          epsilon);
      }
    });
    return std::tuple_cat(t1, t1);
}

} // namespace native
} // namespace at
