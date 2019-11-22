#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/WrapDimUtils.h>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>

#include <ATen/AccumulateType.h>
#include <ATen/cuda/NumericLimits.cuh>
#include <type_traits>

#include "../cuda/PersistentSoftmax.cuh"
#include "SoftMaxNomalization.cuh"

namespace at {
namespace native {

namespace {

template<template<typename, typename, typename> class Epilogue, bool is_log_softmax, typename scalar_tt, typename index_tt>
std::tuple<Tensor, Tensor, Tensor> softmax_norm(const Tensor & input_, const int64_t dim_, const bool half_to_float, const Tensor& input_norm_, double epsilon) {
  if (half_to_float) AT_ASSERTM(input_.scalar_type() == ScalarType::Half,"conversion is supported for Half type only");
  auto input = input_.contiguous();
  Tensor output = half_to_float ? at::empty_like(input, input.options().dtype(ScalarType::Float)) : at::empty_like(input);
  static_assert(std::is_same<acc_type<at::Half, true>, float>::value, "accscalar_tt for half should be float");
  if (input.dim() == 0) input = input.view(1);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);

  int64_t inner_size = 1;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i)
    inner_size *= input.size(i);
  uint32_t smem_size;
  dim3 grid, block;
  THCudaCheck(cudaGetLastError());

  using accscalar_tt = at::acc_type<scalar_tt, true>;
  int64_t n_input_norm = input_norm_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_norm_reshaped = input_norm_.reshape({input_norm_.size(0), input_norm_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_norm_reshaped.size(0);
  auto features = input_norm_reshaped.size(2);
  auto input_norm = input_norm_reshaped.packed_accessor<scalar_tt, 3, RestrictPtrTraits, index_tt>();
  auto input_norm_options = input_norm_.options();
  dummy_mean_ = at::empty({0}, input_norm_options);
  dummy_var_ = at::empty({0}, input_norm_options);
  // promote only mean_/invstd_ precision
  if (input_norm_.scalar_type() == at::ScalarType::Half) {
    input_norm_options = input_norm_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input_norm}, input_norm_options);
  invstd_ = at::empty({n_input_norm}, input_norm_options);
  auto mean = packed_accessor_or_dummy<accscalar_tt, 1, RestrictPtrTraits, index_tt>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_tt, 1, RestrictPtrTraits, index_tt>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_tt, 1, RestrictPtrTraits, index_tt>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_tt, 1, RestrictPtrTraits, index_tt>();

  dim3 blocks(input_norm.size(1));
  int tf = getNumThreads(input_norm.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "host_softmax", [&] {
  using accscalar_t = acc_type<scalar_t, true>;
      SpatialSoftMax_getLaunchSizes<accscalar_t>(
          &cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>,
          outer_size, dim_size, inner_size,
          grid, block, smem_size);
    printf("1 %d %d %d %d %d %d %d\n", grid.x, grid.y, grid.z, block.x, block.y, block.z, dim_size);
      cunn_SpatialSoftMaxForward<scalar_t, accscalar_t, scalar_t, Epilogue>
        <<<grid, block, smem_size, stream>>>(
          output.data<scalar_t>(), input.data<scalar_t>(), outer_size, dim_size, inner_size
  );
  });
  batch_norm_collect_statistics_kernel<InvStd, scalar_tt, scalar_tt, accscalar_tt, index_tt> <<<blocks, threads, 0, stream>>>
    (input_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(mean_, invstd_, output);
}

}

Tensor log_softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float){
  return host_softmax<LogSoftMaxForwardEpilogue,true>(input, dim, half_to_float);
}

std::tuple<Tensor, Tensor, Tensor> softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float, const Tensor& self, double epsilon) {
  return AT_DISPATCH_FLOATING_TYPES_AND_HALF(self.scalar_type(), "batch_norm_stats_cuda", [&] {
    return softmax_norm<SoftMaxForwardEpilogue,false, scalar_t, int32_t>(input, dim, half_to_float, self, epsilon);
  });
}

}
}
