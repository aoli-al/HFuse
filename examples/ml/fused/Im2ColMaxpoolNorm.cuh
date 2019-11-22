#pragma once

#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <cuda_profiler_api.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

#include <ATen/native/im2col_shape_check.h>
#include <ATen/AccumulateType.h>
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"

namespace at {
namespace native {

using namespace at::cuda::detail;

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

static const int BACKWARD_THREADS = 256;
// kernels borrowed from Caffe
template <typename scalar_t, typename accscalar_t>
__global__ void MaxPoolForward(const int nthreads, const scalar_t* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while(hstart < 0)
      hstart += dilation_h;
    while(wstart < 0)
      wstart += dilation_w;
    accscalar_t maxval = at::numeric_limits<accscalar_t>::lower_bound(); // -Infinity
    int maxidx = hstart * width + wstart;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
      for (int w = wstart; w < wend; w += dilation_w) {
        scalar_t val = bottom_data[h * width + w];
        if ((ScalarConvert<scalar_t, accscalar_t>::to(val) > maxval) || THCNumerics<scalar_t>::isnan(val)) {
          maxidx = h * width + w;
          maxval = ScalarConvert<scalar_t, accscalar_t>::to(val);
        }
      }
    }
    top_data[index] = ScalarConvert<scalar_t, accscalar_t>::to(maxval);
    top_mask[index] = maxidx;
  }
}

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
// #if defined(__HIP_PLATFORM_HCC__)
// #else
// constexpr int MAX_BLOCK_SIZE = 512;
// #endif
constexpr int MAX_BLOCK_SIZE = 256;

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct SumOp {
  __device__ SumOp(const PTA& t) : tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    return static_cast<accscalar_t>(tensor[batch][plane][n]);
  }
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct VarOp {
  __device__ VarOp(accscalar_t m, const PTA& t) : mean(m), tensor(t) {}
  __device__ __forceinline__ accscalar_t operator()(int batch, int plane, int n) {
    accscalar_t val = tensor[batch][plane][n];
    return (val - mean) * (val - mean);
  }
  const accscalar_t mean;
  const PTA& tensor;
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

template<typename T>
struct InvStd {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / device_sqrt(var + epsilon);
    }
    return invstd;
  }
};

template<typename T>
struct Var {
  __device__ __forceinline__ T operator()(T var, double epsilon) const {
    return var;
  }
};

template <typename dt60, typename scalar_t0, typename accscalar_t1, template <typename T> class VarTransform31, typename input_scalar_t32, typename stat_scalar_t33, typename stat_accscalar_t34, typename index_t35>
void im2col_kernel_MaxPoolForward_batch_norm_collect_statistics_kernel_(const int64_t n61, const dt60 *data_im62, const int64_t height63, const int64_t width64, const int64_t kernel_height65, const int64_t kernel_width66, const int64_t pad_height67, const int64_t pad_width68, const int64_t stride_height69, const int64_t stride_width70, const int64_t dilation_height71, const int64_t dilation_width72, const int64_t height_col73, const int64_t width_col74, dt60 *data_col75, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19, const PackedTensorAccessor<input_scalar_t32, 3, RestrictPtrTraits, index_t35> input36, const stat_accscalar_t34 epsilon37, const stat_accscalar_t34 momentum38, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_mean39, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_var40, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_mean41, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_transformed_var42) __attribute__((global))
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_0;
unsigned int blockDim_x_2;
blockDim_x_2 = 512;
unsigned int threadIdx_x_2;
threadIdx_x_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_2;
blockDim_y_2 = 1;
unsigned int threadIdx_y_2;
threadIdx_y_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_2;
blockDim_z_2 = 1;
unsigned int threadIdx_z_2;
threadIdx_z_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_2 + threadIdx_x_2; index < (n61); index += blockDim_x_2 * gridDim.x) {
    int64_t w_out76;
    w_out76 = index % width_col74;
    index /= width_col74;
    int64_t h_out77;
    h_out77 = index % height_col73;
    int64_t channel_in78;
    channel_in78 = index / height_col73;
    int64_t channel_out79;
    channel_out79 = channel_in78 * kernel_height65 * kernel_width66;
    int64_t h_in80;
    h_in80 = h_out77 * stride_height69 - pad_height67;
    int64_t w_in81;
    w_in81 = w_out76 * stride_width70 - pad_width68;
    data_col75 += (channel_out79 * height_col73 + h_out77) * width_col74 + w_out76;
    data_im62 += (channel_in78 * height63 + h_in80) * width64 + w_in81;
    for (int64_t i = 0; i < kernel_height65; ++i) {
        for (int64_t j = 0; j < kernel_width66; ++j) {
            int64_t h82;
            h82 = h_in80 + i * dilation_height71;
            int64_t w83;
            w83 = w_in81 + j * dilation_width72;
            * data_col75 = (h82 >= 0 && w83 >= 0 && h82 < height63 && w83 < width64) ? data_im62[i * dilation_height71 * width64 + j * dilation_width72] : ScalarConvert<int, dt60>::to(0);
            data_col75 += height_col73 * width_col74;
        }
    }
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_2;
unsigned int blockDim_x_1;
blockDim_x_1 = 32;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 32;
unsigned int blockDim_y_1;
blockDim_y_1 = 8;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 32 % 8;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
static int shared_n43[160] __attribute__((shared));
int plane44;
plane44 = blockIdx.x;
int N45;
N45 = input36.size(0) * input36.size(2);
int tid46;
tid46 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
stat_accscalar_t34 *shared_avg_var47;
shared_avg_var47 = (stat_accscalar_t34 *)&shared_n43[WARP_SIZE];
stat_accscalar_t34 avg48;
avg48 = 0;
stat_accscalar_t34 var_n49;
var_n49 = 0;
int n50;
n50 = 0;
for (int batch = threadIdx_y_1; batch < input36.size(0); batch += blockDim_y_1) {
    for (int x = threadIdx_x_1; x < input36.size(2); x += blockDim_x_1) {
        stat_accscalar_t34 v51;
        v51 = input36[batch][plane44][x];
        stat_accscalar_t34 d152;
        d152 = v51 - avg48;
        n50++;
        avg48 += d152 / n50;
        var_n49 += d152 * (v51 - avg48);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t34 o_avg53;
    o_avg53 = WARP_SHFL_XOR(avg48, 1 << i, WARP_SIZE);
    int o_n54;
    o_n54 = WARP_SHFL_XOR(n50, 1 << i, WARP_SIZE);
    stat_accscalar_t34 factor55;
    factor55 = 1. / fmaxf(1., n50 + o_n54);
    var_n49 += WARP_SHFL_XOR(var_n49, 1 << i, WARP_SIZE) + (avg48 - o_avg53) * (avg48 - o_avg53) * n50 * o_n54 * factor55;
    avg48 = (n50 * avg48 + o_n54 * o_avg53) * factor55;
    n50 += o_n54;
}
label_2:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
if (tid46 % WARP_SIZE == 0) {
    shared_n43[tid46 / WARP_SIZE] = n50;
    shared_avg_var47[tid46 / WARP_SIZE * 2] = avg48;
    shared_avg_var47[tid46 / WARP_SIZE * 2 + 1] = var_n49;
}
label_3:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_1;
unsigned int blockDim_x_0;
blockDim_x_0 = 256;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
    int pw20;
    pw20 = index % pooled_width9;
    int ph21;
    ph21 = (index / pooled_width9) % pooled_height8;
    int c22;
    c22 = (index / pooled_width9 / pooled_height8) % channels5;
    int n23;
    n23 = index / pooled_width9 / pooled_height8 / channels5;
    int hstart24;
    hstart24 = ph21 * stride_h12 - pad_h14;
    int wstart25;
    wstart25 = pw20 * stride_w13 - pad_w15;
    int hend26;
    hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
    int wend27;
    wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
    while (hstart24 < 0)
        hstart24 += dilation_h16;
    while (wstart25 < 0)
        wstart25 += dilation_w17;
    accscalar_t1 maxval28;
    maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
    int maxidx29;
    maxidx29 = hstart24 * width7 + wstart25;
    bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
    for (int h = hstart24; h < hend26; h += dilation_h16) {
        for (int w = wstart25; w < wend27; w += dilation_w17) {
            scalar_t0 val30;
            val30 = bottom_data3[h * width7 + w];
            if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
                maxidx29 = h * width7 + w;
                maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
            }
        }
    }
    top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
    top_mask19[index] = maxidx29;
}
label_1:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_4;
if (tid46 < WARP_SIZE) {
    n50 = (tid46 < blockDim_x_1 * blockDim_y_1 / WARP_SIZE ? shared_n43[tid46] : 0);
    avg48 = (tid46 < blockDim_x_1 * blockDim_y_1 / WARP_SIZE ? shared_avg_var47[2 * tid46] : stat_accscalar_t34(0));
    var_n49 = (tid46 < blockDim_x_1 * blockDim_y_1 / WARP_SIZE ? shared_avg_var47[2 * tid46 + 1] : stat_accscalar_t34(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t34 o_avg56;
    o_avg56 = WARP_SHFL_XOR(avg48, 1 << i, WARP_SIZE);
    int o_n57;
    o_n57 = WARP_SHFL_XOR(n50, 1 << i, WARP_SIZE);
    stat_accscalar_t34 factor58;
    factor58 = 1. / fmaxf(1., n50 + o_n57);
    var_n49 += WARP_SHFL_XOR(var_n49, 1 << i, WARP_SIZE) + (avg48 - o_avg56) * (avg48 - o_avg56) * n50 * o_n57 * factor58;
    avg48 = (n50 * avg48 + o_n57 * o_avg56) * factor58;
    n50 += o_n57;
}
if (tid46 == 0) {
    if (save_mean41.data() != __null) {
        save_mean41[plane44] = avg48;
    }
    if (save_transformed_var42.data() != __null) {
        save_transformed_var42[plane44] = VarTransform31<stat_accscalar_t34>({})(var_n49 / N45, epsilon37);
    }
    if (running_mean39.data() != __null) {
        running_mean39[plane44] = static_cast<stat_scalar_t33>((1 - momentum38) * running_mean39[plane44] + momentum38 * avg48);
    }
    if (running_var40.data() != __null) {
        stat_accscalar_t34 unbiasedVar59;
        unbiasedVar59 = var_n49 / (N45 - 1);
        running_var40[plane44] = static_cast<stat_scalar_t33>((1 - momentum38) * running_var40[plane44] + momentum38 * unbiasedVar59);
    }
}
label_4:;
}

template <template<typename T> class VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {

  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  stat_accscalar_t* shared_avg_var = (stat_accscalar_t*) &shared_n[WARP_SIZE];

  // first the reductions each thread does separately
  stat_accscalar_t avg = 0;
  stat_accscalar_t var_n = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      stat_accscalar_t v = input[batch][plane][x];
      stat_accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean.data() != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var.data() != NULL) {
      save_transformed_var[plane] = VarTransform<stat_accscalar_t>{}(var_n / N, epsilon);
    }
    if (running_mean.data() != NULL) {
      running_mean[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_mean[plane] + momentum * avg);
    }
    if (running_var.data() != NULL) {
      stat_accscalar_t unbiasedVar = var_n / (N - 1);
      running_var[plane] = static_cast<stat_scalar_t>((1 - momentum) * running_var[plane] + momentum * unbiasedVar);
    }
  }

}

template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}

using namespace at::cuda::detail;

template <typename dt>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_kernel(
    const int64_t n,
    const dt* data_im,
    const int64_t height,
    const int64_t width,
    const int64_t kernel_height,
    const int64_t kernel_width,
    const int64_t pad_height,
    const int64_t pad_width,
    const int64_t stride_height,
    const int64_t stride_width,
    const int64_t dilation_height,
    const int64_t dilation_width,
    const int64_t height_col,
    const int64_t width_col,
    dt* data_col) {

  CUDA_KERNEL_LOOP(index, n) {
    int64_t w_out = index % width_col;

    index /= width_col;

    int64_t h_out = index % height_col;
    int64_t channel_in = index / height_col;
    int64_t channel_out = channel_in * kernel_height * kernel_width;
    int64_t h_in = h_out * stride_height - pad_height;
    int64_t w_in = w_out * stride_width - pad_width;

    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;

    for (int64_t i = 0; i < kernel_height; ++i) {
      for (int64_t j = 0; j < kernel_width; ++j) {
        int64_t h = h_in + i * dilation_height;
        int64_t w = w_in + j * dilation_width;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width)
            ? data_im[i * dilation_height * width + j * dilation_width]
            : ScalarConvert<int, dt>::to(0);
        data_col += height_col * width_col;
      }
    }
  }
}

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_stream(
    const Tensor& input_,
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
    const Tensor& input_batch_norm_, double epsilon) {

  Tensor output_maxpool = at::empty({0}, input_maxpool_.options());
  Tensor indices_maxpool = at::empty({0}, input_maxpool_.options().dtype(kLong));
  TensorArg output_maxpool_arg{ output_maxpool, "output_maxpool", 1 };
  TensorArg indices_maxpool_arg{ indices_maxpool, "indices_maxpool", 2 };
  TensorArg input_maxpool_arg{ input_maxpool_ , "input_maxpool_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_maxpool_out_cuda",
                  {output_maxpool_arg, indices_maxpool_arg, input_maxpool_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size_maxpool.size() == 1 || kernel_size_maxpool.size() == 2,
    "max_pool2d: kernel_size_maxpool must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size_maxpool[0]);
  const int kW = kernel_size_maxpool.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size_maxpool[1]);

  // NB: stride_maxpool default is not expressible as an integer constant, so we accept
  // empty stride_maxpool for this case
  TORCH_CHECK(stride_maxpool.size() == 0 || stride_maxpool.size() == 1 || stride_maxpool.size() == 2,
    "max_pool2d: stride_maxpool must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride_maxpool.empty() ? kH : safe_downcast<int, int64_t>(stride_maxpool[0]);
  const int dW = stride_maxpool.empty() ? kW :
                 stride_maxpool.size() == 1 ? dH : safe_downcast<int, int64_t>(stride_maxpool[1]);

  TORCH_CHECK(padding_maxpool.size() == 1 || padding_maxpool.size() == 2,
    "max_pool2d: padding_maxpool must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding_maxpool[0]);
  const int padW = padding_maxpool.size() == 1 ? padH : safe_downcast<int, int64_t>(padding_maxpool[1]);

  TORCH_CHECK(dilation_maxpool.size() == 1 || dilation_maxpool.size() == 2,
    "max_pool2d: dilation_maxpool must be either a single int, or a tuple of two ints");
  const int dilation_maxpoolH = safe_downcast<int, int64_t>(dilation_maxpool[0]);
  const int dilation_maxpoolW = dilation_maxpool.size() == 1 ? dilation_maxpoolH : safe_downcast<int, int64_t>(dilation_maxpool[1]);

  TORCH_CHECK((input_maxpool_.ndimension() == 3 || input_maxpool_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input_maxpool");

  const int64_t nbatch = input_maxpool_.ndimension() == 4 ? input_maxpool_.size(-4) : 1;
  const int64_t nInputPlane = input_maxpool_.size(-3);
  const int64_t input_maxpoolHeight = input_maxpool_.size(-2);
  const int64_t input_maxpoolWidth = input_maxpool_.size(-1);

  const int64_t output_maxpoolWidth = pooling_output_shape<int64_t>(input_maxpoolWidth, kW, padW, dW, dilation_maxpoolW, ceil_mode);
  const int64_t output_maxpoolHeight = pooling_output_shape<int64_t>(input_maxpoolHeight, kH, padH, dH, dilation_maxpoolH, ceil_mode);
  printf("%ld, %ld\n", output_maxpoolWidth, output_maxpoolHeight);

  pool2d_shape_check(
    input_maxpool_,
    kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW,
    nInputPlane,
    input_maxpoolHeight, input_maxpoolWidth,
    output_maxpoolHeight, output_maxpoolWidth);

  Tensor input_maxpool = input_maxpool_.contiguous();

  output_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  indices_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});

  const int count = safe_downcast<int, int64_t>(output_maxpool.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  Tensor output = at::empty_like(input_);
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

  im2col_shape_check(
    input_,
    Tensor(),
    kernel_height,
    kernel_width,
    dilation_height,
    dilation_width,
    pad_height,
    pad_width,
    stride_height,
    stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  // Launch kernel
  using accscalar_t_batch_norm = at::acc_type<scalar_t_batch_norm, true>;
  int64_t n_input = input_batch_norm_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_batch_norm_reshaped = input_batch_norm_.reshape({input_batch_norm_.size(0), input_batch_norm_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_batch_norm_reshaped.size(0);
  auto features = input_batch_norm_reshaped.size(2);
  auto input_batch_norm = input_batch_norm_reshaped.packed_accessor<scalar_t_batch_norm, 3, RestrictPtrTraits, index_t_batch_norm>();
  auto input_batch_norm_options = input_batch_norm_.options();
  dummy_mean_ = at::empty({0}, input_batch_norm_options);
  dummy_var_ = at::empty({0}, input_batch_norm_options);
  // promote only mean_/invstd_ precision
  if (input_batch_norm_.scalar_type() == at::ScalarType::Half) {
    input_batch_norm_options = input_batch_norm_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_batch_norm_options);
  invstd_ = at::empty({n_input}, input_batch_norm_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto stream1 = at::cuda::getStreamFromPool(true);
  auto stream2 = at::cuda::getStreamFromPool(true);

  dim3 blocks(input_batch_norm.size(1));
  int tf = getNumThreads(input_batch_norm.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    input_n = input.select(0, 0);
    output_n = output.select(0, 0);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    printf("nk: %ld\n", num_kernels);
    const int num_of_threads = 512;
    const int num_of_blocks = (num_kernels + num_of_threads - 1) / num_of_threads;
    cudaProfilerStart();
    im2col_kernel<scalar_t><<<num_of_blocks, num_of_threads, 0, stream2>>>(
        num_kernels,
        input_n.data<scalar_t>(),
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output_height,
        output_width,
        output_n.data<scalar_t>());


      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_maxpool_data = output_maxpool.data<scalar_t>();
      scalar_t *input_maxpool_data = input_maxpool.data<scalar_t>();
      int64_t *indices_maxpool_data = indices_maxpool.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads);
      printf("%d %d %d\n", count, blocks, num_threads);
      printf("%d %d\n", threads.x, threads.y);
      printf("%d\n", blocks);
      MaxPoolForward<scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
          count, input_maxpool_data,
          nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
          kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW, output_maxpool_data, indices_maxpool_data);
    batch_norm_collect_statistics_kernel<InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm> <<<blocks, threads, 0, stream1>>>
      (input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);

    cudaDeviceSynchronize();
    cudaProfilerStop();
    AT_CUDA_CHECK(cudaGetLastError());
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  THCudaCheck(cudaGetLastError());
  if(input_maxpool.ndimension() == 3) {
    output_maxpool.resize_({nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  }
  return std::make_tuple(output, mean_, output_maxpool);
}

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_fused(
    const Tensor& input_,
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
    const Tensor& input_batch_norm_, double epsilon) {

  Tensor output_maxpool = at::empty({0}, input_maxpool_.options());
  Tensor indices_maxpool = at::empty({0}, input_maxpool_.options().dtype(kLong));
  TensorArg output_maxpool_arg{ output_maxpool, "output_maxpool", 1 };
  TensorArg indices_maxpool_arg{ indices_maxpool, "indices_maxpool", 2 };
  TensorArg input_maxpool_arg{ input_maxpool_ , "input_maxpool_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_maxpool_out_cuda",
                  {output_maxpool_arg, indices_maxpool_arg, input_maxpool_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size_maxpool.size() == 1 || kernel_size_maxpool.size() == 2,
    "max_pool2d: kernel_size_maxpool must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size_maxpool[0]);
  const int kW = kernel_size_maxpool.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size_maxpool[1]);

  // NB: stride_maxpool default is not expressible as an integer constant, so we accept
  // empty stride_maxpool for this case
  TORCH_CHECK(stride_maxpool.size() == 0 || stride_maxpool.size() == 1 || stride_maxpool.size() == 2,
    "max_pool2d: stride_maxpool must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride_maxpool.empty() ? kH : safe_downcast<int, int64_t>(stride_maxpool[0]);
  const int dW = stride_maxpool.empty() ? kW :
                 stride_maxpool.size() == 1 ? dH : safe_downcast<int, int64_t>(stride_maxpool[1]);

  TORCH_CHECK(padding_maxpool.size() == 1 || padding_maxpool.size() == 2,
    "max_pool2d: padding_maxpool must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding_maxpool[0]);
  const int padW = padding_maxpool.size() == 1 ? padH : safe_downcast<int, int64_t>(padding_maxpool[1]);

  TORCH_CHECK(dilation_maxpool.size() == 1 || dilation_maxpool.size() == 2,
    "max_pool2d: dilation_maxpool must be either a single int, or a tuple of two ints");
  const int dilation_maxpoolH = safe_downcast<int, int64_t>(dilation_maxpool[0]);
  const int dilation_maxpoolW = dilation_maxpool.size() == 1 ? dilation_maxpoolH : safe_downcast<int, int64_t>(dilation_maxpool[1]);

  TORCH_CHECK((input_maxpool_.ndimension() == 3 || input_maxpool_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input_maxpool");

  const int64_t nbatch = input_maxpool_.ndimension() == 4 ? input_maxpool_.size(-4) : 1;
  const int64_t nInputPlane = input_maxpool_.size(-3);
  const int64_t input_maxpoolHeight = input_maxpool_.size(-2);
  const int64_t input_maxpoolWidth = input_maxpool_.size(-1);

  const int64_t output_maxpoolWidth = pooling_output_shape<int64_t>(input_maxpoolWidth, kW, padW, dW, dilation_maxpoolW, ceil_mode);
  const int64_t output_maxpoolHeight = pooling_output_shape<int64_t>(input_maxpoolHeight, kH, padH, dH, dilation_maxpoolH, ceil_mode);
  printf("%ld, %ld\n", output_maxpoolWidth, output_maxpoolHeight);

  pool2d_shape_check(
    input_maxpool_,
    kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW,
    nInputPlane,
    input_maxpoolHeight, input_maxpoolWidth,
    output_maxpoolHeight, output_maxpoolWidth);

  Tensor input_maxpool = input_maxpool_.contiguous();

  output_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  indices_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});

  const int count = safe_downcast<int, int64_t>(output_maxpool.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  Tensor output = at::empty_like(input_);
  int64_t kernel_height = kernel_size[0];
  int64_t kernel_width = kernel_size[1];
  int64_t dilation_height = dilation[0];
  int64_t dilation_width = dilation[1];
  int64_t pad_height = padding[0];
  int64_t pad_width = padding[1];
  int64_t stride_height = stride[0];
  int64_t stride_width = stride[1];

  TensorArg input_arg{input_, "input", 1};
  TensorArg output_arg{output, "output", 2};
  checkAllSameGPU("im2col_cuda", {input_arg, output_arg});

  im2col_shape_check(
    input_,
    Tensor(),
    kernel_height,
    kernel_width,
    dilation_height,
    dilation_width,
    pad_height,
    pad_width,
    stride_height,
    stride_width);

  Tensor input = input_.contiguous();

  bool batched_input = true;

  if (input.dim() == 3) {
    batched_input = false;
    input.resize_({1, input.size(0), input.size(1), input.size(2)});
  }

  int64_t batch_size = input.size(0);
  int64_t n_input_plane = input.size(1);
  int64_t input_height = input.size(2);
  int64_t input_width = input.size(3);

  int64_t output_height = (input_height + 2 * pad_height -
                           (dilation_height * (kernel_height - 1) + 1)) /
          stride_height +
      1;
  int64_t output_width = (input_width + 2 * pad_width -
                          (dilation_width * (kernel_width - 1) + 1)) /
          stride_width +
      1;
  int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
  int64_t output_length = output_height * output_width;

  output.resize_({batch_size, n_output_plane, output_length});
  output.zero_();

  // Launch kernel
  using accscalar_t_batch_norm = at::acc_type<scalar_t_batch_norm, true>;
  int64_t n_input = input_batch_norm_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_batch_norm_reshaped = input_batch_norm_.reshape({input_batch_norm_.size(0), input_batch_norm_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_batch_norm_reshaped.size(0);
  auto features = input_batch_norm_reshaped.size(2);
  auto input_batch_norm = input_batch_norm_reshaped.packed_accessor<scalar_t_batch_norm, 3, RestrictPtrTraits, index_t_batch_norm>();
  auto input_batch_norm_options = input_batch_norm_.options();
  dummy_mean_ = at::empty({0}, input_batch_norm_options);
  dummy_var_ = at::empty({0}, input_batch_norm_options);
  // promote only mean_/invstd_ precision
  if (input_batch_norm_.scalar_type() == at::ScalarType::Half) {
    input_batch_norm_options = input_batch_norm_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_batch_norm_options);
  invstd_ = at::empty({n_input}, input_batch_norm_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_batch_norm, 1, RestrictPtrTraits, index_t_batch_norm>();
  auto stream1 = at::cuda::getStreamFromPool(true);
  auto stream2 = at::cuda::getStreamFromPool(true);

  dim3 blocks(input_batch_norm.size(1));
  int tf = getNumThreads(input_batch_norm.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    input_n = input.select(0, 0);
    output_n = output.select(0, 0);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    printf("nk: %ld\n", num_kernels);
    const int num_of_threads = 1024;
    const int num_of_blocks = (num_kernels + num_of_threads - 1) / num_of_threads;
    cudaProfilerStart();
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_maxpool_data = output_maxpool.data<scalar_t>();
      scalar_t *input_maxpool_data = input_maxpool.data<scalar_t>();
      int64_t *indices_maxpool_data = indices_maxpool.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads);
      printf("%d %d %d\n", count, blocks, num_threads);
      // printf("%d %d\n", threads.x, threads.y);
      // printf("%d %d\n", blocks.x, blocks.y);
      cudaDeviceSynchronize();
    im2col_kernel_MaxPoolForward_batch_norm_collect_statistics_kernel_<scalar_t,scalar_t, scalar_t,InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm>
    <<<blocks, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_kernels,
        input_n.data<scalar_t>(),
        input_height,
        input_width,
        kernel_height,
        kernel_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width,
        dilation_height,
        dilation_width,
        output_height,
        output_width,
        output_n.data<scalar_t>(),
        count, input_maxpool_data,
        nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
        kH, kW, dH, dW, padH, padW, dilation_maxpoolH, dilation_maxpoolW, output_maxpool_data, indices_maxpool_data,
        input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd
    );
    cudaDeviceSynchronize();
    cudaProfilerStop();
    AT_CUDA_CHECK(cudaGetLastError());
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  THCudaCheck(cudaGetLastError());
  if(input_maxpool.ndimension() == 3) {
    output_maxpool.resize_({nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  }
  return std::make_tuple(output, mean_, output_maxpool);
}

} // namespace native
} // namespace at
