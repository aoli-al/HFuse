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

#include <ATen/native/im2col_shape_check.h>
#include <ATen/AccumulateType.h>
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"

namespace at {
namespace native {

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

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

template <typename dt2935, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
void im2col_kernel_batch_norm_collect_statistics_kernel_100(const int64_t n3036, const dt2935 *data_im3137, const int64_t height3238, const int64_t width3339, const int64_t kernel_height3440, const int64_t kernel_width3541, const int64_t pad_height3642, const int64_t pad_width3743, const int64_t stride_height3844, const int64_t stride_width3945, const int64_t dilation_height4046, const int64_t dilation_width4147, const int64_t height_col4248, const int64_t width_col4349, dt2935 *data_col4450, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111) __attribute__((global))
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 512;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_151;
    blockDim_x_151 = 512;
    unsigned int threadIdx_x_152;
    threadIdx_x_152 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_153;
    blockDim_y_153 = 1;
    unsigned int threadIdx_y_154;
    threadIdx_y_154 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_155;
    blockDim_z_155 = 1;
    unsigned int threadIdx_z_156;
    threadIdx_z_156 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_151 + threadIdx_x_152; index < (n3036); index += blockDim_x_151 * gridDim.x) {
        int64_t w_out4557;
        w_out4557 = index % width_col4349;
        index /= width_col4349;
        int64_t h_out4658;
        h_out4658 = index % height_col4248;
        int64_t channel_in4759;
        channel_in4759 = index / height_col4248;
        int64_t channel_out4860;
        channel_out4860 = channel_in4759 * kernel_height3440 * kernel_width3541;
        int64_t h_in4961;
        h_in4961 = h_out4658 * stride_height3844 - pad_height3642;
        int64_t w_in5062;
        w_in5062 = w_out4557 * stride_width3945 - pad_width3743;
        data_col4450 += (channel_out4860 * height_col4248 + h_out4658) * width_col4349 + w_out4557;
        data_im3137 += (channel_in4759 * height3238 + h_in4961) * width3339 + w_in5062;
        for (int64_t i = 0; i < kernel_height3440; ++i) {
            for (int64_t j = 0; j < kernel_width3541; ++j) {
                int64_t h5163;
                h5163 = h_in4961 + i * dilation_height4046;
                int64_t w5264;
                w5264 = w_in5062 + j * dilation_width4147;
                * data_col4450 = (h5163 >= 0 && w5264 >= 0 && h5163 < height3238 && w5264 < width3339) ? data_im3137[i * dilation_height4046 * width3339 + j * dilation_width4147] : ScalarConvert<int, dt2935>::to(0);
                data_col4450 += height_col4248 * width_col4349;
            }
        }
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 32;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 16;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    unsigned int blockDim_x_012;
    blockDim_x_012 = 32;
    unsigned int threadIdx_x_013;
    threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 512) % 32;
    unsigned int blockDim_y_014;
    blockDim_y_014 = 16;
    unsigned int threadIdx_y_015;
    threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 512) / 32 % 16;
    unsigned int blockDim_z_016;
    blockDim_z_016 = 1;
    unsigned int threadIdx_z_017;
    threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 512) / 512;
    static int shared_n1218[160] __attribute__((shared));
    int plane1319;
    plane1319 = blockIdx.x;
    int N1420;
    N1420 = input55.size(0) * input55.size(2);
    int tid1521;
    tid1521 = threadIdx_x_013 + threadIdx_y_015 * blockDim_x_012;
    stat_accscalar_t33 *shared_avg_var1622;
    shared_avg_var1622 = (stat_accscalar_t33 *)&shared_n1218[WARP_SIZE];
    stat_accscalar_t33 avg1723;
    avg1723 = 0;
    stat_accscalar_t33 var_n1824;
    var_n1824 = 0;
    int n1925;
    n1925 = 0;
    for (int batch = threadIdx_y_015; batch < input55.size(0); batch += blockDim_y_014) {
        for (int x = threadIdx_x_013; x < input55.size(2); x += blockDim_x_012) {
            stat_accscalar_t33 v2026;
            v2026 = input55[batch][plane1319][x];
            stat_accscalar_t33 d12127;
            d12127 = v2026 - avg1723;
            n1925++;
            avg1723 += d12127 / n1925;
            var_n1824 += d12127 * (v2026 - avg1723);
        }
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2228;
        o_avg2228 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2329;
        o_n2329 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2430;
        factor2430 = 1. / fmaxf(1., n1925 + o_n2329);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2228) * (avg1723 - o_avg2228) * n1925 * o_n2329 * factor2430;
        avg1723 = (n1925 * avg1723 + o_n2329 * o_avg2228) * factor2430;
        n1925 += o_n2329;
    }
    __syncthreads();
    if (tid1521 % WARP_SIZE == 0) {
        shared_n1218[tid1521 / WARP_SIZE] = n1925;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2] = avg1723;
        shared_avg_var1622[tid1521 / WARP_SIZE * 2 + 1] = var_n1824;
    }
    __syncthreads();
    if (tid1521 < WARP_SIZE) {
        n1925 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_n1218[tid1521] : 0);
        avg1723 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521] : stat_accscalar_t33(0));
        var_n1824 = (tid1521 < blockDim_x_012 * blockDim_y_014 / WARP_SIZE ? shared_avg_var1622[2 * tid1521 + 1] : stat_accscalar_t33(0));
    }
    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
        stat_accscalar_t33 o_avg2531;
        o_avg2531 = WARP_SHFL_XOR(avg1723, 1 << i, WARP_SIZE);
        int o_n2632;
        o_n2632 = WARP_SHFL_XOR(n1925, 1 << i, WARP_SIZE);
        stat_accscalar_t33 factor2733;
        factor2733 = 1. / fmaxf(1., n1925 + o_n2632);
        var_n1824 += WARP_SHFL_XOR(var_n1824, 1 << i, WARP_SIZE) + (avg1723 - o_avg2531) * (avg1723 - o_avg2531) * n1925 * o_n2632 * factor2733;
        avg1723 = (n1925 * avg1723 + o_n2632 * o_avg2531) * factor2733;
        n1925 += o_n2632;
    }
    if (tid1521 == 0) {
        if (save_mean1010.data() != __null) {
            save_mean1010[plane1319] = avg1723;
        }
        if (save_transformed_var1111.data() != __null) {
            save_transformed_var1111[plane1319] = VarTransform00<stat_accscalar_t33>({})(var_n1824 / N1420, epsilon66);
        }
        if (running_mean88.data() != __null) {
            running_mean88[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_mean88[plane1319] + momentum77 * avg1723);
        }
        if (running_var99.data() != __null) {
            stat_accscalar_t33 unbiasedVar2834;
            unbiasedVar2834 = var_n1824 / (N1420 - 1);
            running_var99[plane1319] = static_cast<stat_scalar_t22>((1 - momentum77) * running_var99[plane1319] + momentum77 * unbiasedVar2834);
        }
    }
}
}

template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void im2col_kernel_batch_norm_collect_statistics_kernel_0(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_0;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    index /= width_col43;
    int64_t h_out46;
    h_out46 = index % height_col42;
    int64_t channel_in47;
    channel_in47 = index / height_col42;
    int64_t channel_out48;
    channel_out48 = channel_in47 * kernel_height34 * kernel_width35;
    int64_t h_in49;
    h_in49 = h_out46 * stride_height38 - pad_height36;
    int64_t w_in50;
    w_in50 = w_out45 * stride_width39 - pad_width37;
    data_col44 += (channel_out48 * height_col42 + h_out46) * width_col43 + w_out45;
    data_im31 += (channel_in47 * height32 + h_in49) * width33 + w_in50;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h51;
            h51 = h_in49 + i * dilation_height40;
            int64_t w52;
            w52 = w_in50 + j * dilation_width41;
            * data_col44 = (h51 >= 0 && w52 >= 0 && h51 < height32 && w52 < width33) ? data_im31[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            data_col44 += height_col42 * width_col43;
        }
    }
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_1;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_2;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_2:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_3:;
}

template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void im2col_kernel_batch_norm_collect_statistics_kernel2_(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_0;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
int index;
index = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
int64_t w_out45;
w_out45 = index % width_col43;
index /= width_col43;
int64_t h_out46;
h_out46 = index % height_col42;
int64_t channel_in47;
channel_in47 = index / height_col42;
int64_t channel_out48;
channel_out48 = channel_in47 * kernel_height34 * kernel_width35;
int64_t h_in49;
h_in49 = h_out46 * stride_height38 - pad_height36;
int64_t w_in50;
w_in50 = w_out45 * stride_width39 - pad_width37;
data_col44 += (channel_out48 * height_col42 + h_out46) * width_col43 + w_out45;
data_im31 += (channel_in47 * height32 + h_in49) * width33 + w_in50;
for (int64_t i = 0; i < kernel_height34-100; ++i) {
    for (int64_t j = 0; j < kernel_width35; ++j) {
        int64_t h51;
        h51 = h_in49 + i * dilation_height40;
        int64_t w52;
        w52 = w_in50 + j * dilation_width41;
        * data_col44 = (h51 >= 0 && w52 >= 0 && h51 < height32 && w52 < width33) ? data_im31[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
        data_col44 += height_col42 * width_col43;
    }
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_1;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_2;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_2:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_7;
for (int64_t i = kernel_height34 - 100; i < kernel_height34; ++i) {
    for (int64_t j = 0; j < kernel_width35; ++j) {
        int64_t h51;
        h51 = h_in49 + i * dilation_height40;
        int64_t w52;
        w52 = w_in50 + j * dilation_width41;
        * data_col44 = (h51 >= 0 && w52 >= 0 && h51 < height32 && w52 < width33) ? data_im31[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
        data_col44 += height_col42 * width_col43;
    }
}
label_7:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_3:;
}


template <typename dt29, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void im2col_kernel_batch_norm_collect_statistics_kernel3_(const int64_t n30, const dt29 *data_im31, const int64_t height32, const int64_t width33, const int64_t kernel_height34, const int64_t kernel_width35, const int64_t pad_height36, const int64_t pad_width37, const int64_t stride_height38, const int64_t stride_width39, const int64_t dilation_height40, const int64_t dilation_width41, const int64_t height_col42, const int64_t width_col43, dt29 *data_col44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_1;
unsigned int blockDim_x_0;
blockDim_x_0 = 32;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 32;
unsigned int blockDim_y_0;
blockDim_y_0 = 16;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 32 % 16;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
static int shared_n12[160] __attribute__((shared));
int plane13;
plane13 = blockIdx.x;
int N14;
N14 = input5.size(0) * input5.size(2);
int tid15;
tid15 = threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0;
stat_accscalar_t3 *shared_avg_var16;
shared_avg_var16 = (stat_accscalar_t3 *)&shared_n12[WARP_SIZE];
stat_accscalar_t3 avg17;
avg17 = 0;
stat_accscalar_t3 var_n18;
var_n18 = 0;
int n19;
n19 = 0;
for (int batch = threadIdx_y_0; batch < input5.size(0); batch += blockDim_y_0) {
    for (int x = threadIdx_x_0; x < input5.size(2); x += blockDim_x_0) {
        stat_accscalar_t3 v20;
        v20 = input5[batch][plane13][x];
        stat_accscalar_t3 d121;
        d121 = v20 - avg17;
        n19++;
        avg17 += d121 / n19;
        var_n18 += d121 * (v20 - avg17);
    }
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg22;
    o_avg22 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n23;
    o_n23 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor24;
    factor24 = 1. / fmaxf(1., n19 + o_n23);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg22) * (avg17 - o_avg22) * n19 * o_n23 * factor24;
    avg17 = (n19 * avg17 + o_n23 * o_avg22) * factor24;
    n19 += o_n23;
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_2;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_2:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_0;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n30); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out45;
    w_out45 = index % width_col43;
    index /= width_col43;
    int64_t h_out46;
    h_out46 = index % height_col42;
    int64_t channel_in47;
    channel_in47 = index / height_col42;
    int64_t channel_out48;
    channel_out48 = channel_in47 * kernel_height34 * kernel_width35;
    int64_t h_in49;
    h_in49 = h_out46 * stride_height38 - pad_height36;
    int64_t w_in50;
    w_in50 = w_out45 * stride_width39 - pad_width37;
    data_col44 += (channel_out48 * height_col42 + h_out46) * width_col43 + w_out45;
    data_im31 += (channel_in47 * height32 + h_in49) * width33 + w_in50;
    for (int64_t i = 0; i < kernel_height34; ++i) {
        for (int64_t j = 0; j < kernel_width35; ++j) {
            int64_t h51;
            h51 = h_in49 + i * dilation_height40;
            int64_t w52;
            w52 = w_in50 + j * dilation_width41;
            * data_col44 = (h51 >= 0 && w52 >= 0 && h51 < height32 && w52 < width33) ? data_im31[i * dilation_height40 * width33 + j * dilation_width41] : ScalarConvert<int, dt29>::to(0);
            data_col44 += height_col42 * width_col43;
        }
    }
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
if (tid15 < WARP_SIZE) {
    n19 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_n12[tid15] : 0);
    avg17 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15] : stat_accscalar_t3(0));
    var_n18 = (tid15 < blockDim_x_0 * blockDim_y_0 / WARP_SIZE ? shared_avg_var16[2 * tid15 + 1] : stat_accscalar_t3(0));
}
for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t3 o_avg25;
    o_avg25 = WARP_SHFL_XOR(avg17, 1 << i, WARP_SIZE);
    int o_n26;
    o_n26 = WARP_SHFL_XOR(n19, 1 << i, WARP_SIZE);
    stat_accscalar_t3 factor27;
    factor27 = 1. / fmaxf(1., n19 + o_n26);
    var_n18 += WARP_SHFL_XOR(var_n18, 1 << i, WARP_SIZE) + (avg17 - o_avg25) * (avg17 - o_avg25) * n19 * o_n26 * factor27;
    avg17 = (n19 * avg17 + o_n26 * o_avg25) * factor27;
    n19 += o_n26;
}
if (tid15 == 0) {
    if (save_mean10.data() != __null) {
        save_mean10[plane13] = avg17;
    }
    if (save_transformed_var11.data() != __null) {
        save_transformed_var11[plane13] = VarTransform0<stat_accscalar_t3>({})(var_n18 / N14, epsilon6);
    }
    if (running_mean8.data() != __null) {
        running_mean8[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_mean8[plane13] + momentum7 * avg17);
    }
    if (running_var9.data() != __null) {
        stat_accscalar_t3 unbiasedVar28;
        unbiasedVar28 = var_n18 / (N14 - 1);
        running_var9[plane13] = static_cast<stat_scalar_t2>((1 - momentum7) * running_var9[plane13] + momentum7 * unbiasedVar28);
    }
}
label_3:;
}


#define CUDA_KERNEL_LOOP_C(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * 512 + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=512 * 10000, i=_i_n_d_e_x)


template <template<typename T> class VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_normalization_kernel_fused(
    const int64_t n,
    const input_scalar_t* data_im,
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
    input_scalar_t* data_col,
    const PackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {
  if (threadIdx.x < 512) {
    CUDA_KERNEL_LOOP_C(index, n) {
      const int64_t w_out = index % width_col;

      index /= width_col;

      const int64_t h_out = index % height_col;
      const int64_t channel_in = index / height_col;
      const int64_t channel_out = channel_in * kernel_height * kernel_width;
      const int64_t h_in = h_out * stride_height - pad_height;
      const int64_t w_in = w_out * stride_width - pad_width;

      data_col += (channel_out * height_col + h_out) * width_col + w_out;
      data_im += (channel_in * height + h_in) * width + w_in;

      for (int64_t i = 0; i < kernel_height; ++i) {
        for (int64_t j = 0; j < kernel_width; ++j) {
          const int64_t h = h_in + i * dilation_height;
          const int64_t w = w_in + j * dilation_width;
          *data_col = (h >= 0 && w >= 0 && h < height && w < width)
              ? data_im[i * dilation_height * width + j * dilation_width]
              : ScalarConvert<int, input_scalar_t>::to(0);
          data_col += height_col * width_col;
        }
      }
      __syncthreads();
      __syncthreads();
    }
  } else {
    __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

    const int plane = blockIdx.x;
    const int N = input.size(0) * input.size(2);
    const int tid = threadIdx.x - 512;
    const int blockDimx = 32;
    const int blockDimy = 16;
    const int threadIdxy = tid / 32;
    const int threadIdxx = tid % 32;

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
    for (int batch = threadIdxy; batch < input.size(0); batch += blockDimy) {
      for (int x = threadIdxx; x < input.size(2); x += blockDimx) {
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
    // __threadfence();
    if (tid % WARP_SIZE == 0) {
      shared_n[tid / WARP_SIZE] = n;
      shared_avg_var[tid / WARP_SIZE * 2] = avg;
      shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
    }
    __syncthreads();
    // __threadfence();
    // now have a second warpSum to reduce the intermediate values
    // from shared memory to a single number. The very first
    // thread writes it to shared memory.

    if (tid < WARP_SIZE) {
      n = (tid < blockDimx * blockDimy / WARP_SIZE ? shared_n[tid] : 0);
      avg = (tid < blockDimx * blockDimy  / WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
      var_n = (tid < blockDimx * blockDimy  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
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
}

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor> im2col_batch_norm_stream(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_batch_norm_, double epsilon) {

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


    AT_CUDA_CHECK(cudaGetLastError());
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  batch_norm_collect_statistics_kernel<InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm> <<<blocks, threads, 0, stream1>>>
    (input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  cudaProfilerStop();
  cudaDeviceSynchronize();
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output);
}

template<typename scalar_t_batch_norm, typename index_t_batch_norm>
std::tuple<Tensor> im2col_batch_norm_fused(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_batch_norm_, double epsilon) {

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

  dim3 blocks(input_batch_norm.size(1));
  int tf = getNumThreads(input_batch_norm.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  Tensor input_n;
  Tensor output_n;

  input_n = input.select(0, 0);
  output_n = output.select(0, 0);
  int64_t num_kernels = n_input_plane * output_height * output_width;
  printf("nk: %ld\n", num_kernels);
  const int num_of_threads = 512;
  const int num_of_blocks = (num_kernels + num_of_threads - 1) / num_of_threads;

  printf("kh: %d, kw: %d\n", kernel_height, kernel_width);
  printf("nb: %ld\n", num_of_blocks);

  cudaProfilerStart();
      cudaDeviceSynchronize();
  im2col_kernel_batch_norm_collect_statistics_kernel_0<scalar_t_batch_norm, InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm>
    <<<num_of_blocks, 512 + 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      num_kernels,
      input_n.data<scalar_t_batch_norm>(),
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
      output_n.data<scalar_t_batch_norm>(),
      input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
    im2col_kernel_batch_norm_collect_statistics_kernel_100<scalar_t_batch_norm, InvStd, scalar_t_batch_norm, scalar_t_batch_norm, accscalar_t_batch_norm, index_t_batch_norm>
    <<<num_of_blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      num_kernels,
      input_n.data<scalar_t_batch_norm>(),
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
      output_n.data<scalar_t_batch_norm>(),
      input_batch_norm, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
  cudaProfilerStop();

  AT_CUDA_CHECK(cudaGetLastError());
  if (!batched_input) {
    output.resize_({n_output_plane, output_length});
  }
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output);
}
} // namespace native
} // namespace at
