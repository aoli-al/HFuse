#pragma once

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCGeneral.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"
#include <cuda_profiler_api.h>

namespace at { namespace native {

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

template <typename scalar_t2935, typename accscalar_t3036, template <typename T> class VarTransform00, typename input_scalar_t11, typename stat_scalar_t22, typename stat_accscalar_t33, typename index_t44>
void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_100(const int n3137, const accscalar_t3036 rheight3238, const accscalar_t3036 rwidth3339, const bool align_corners3440, const PackedTensorAccessor<scalar_t2935, 4> idata3541, PackedTensorAccessor<scalar_t2935, 4> odata3642, const PackedTensorAccessor<input_scalar_t11, 3, RestrictPtrTraits, index_t44> input55, const stat_accscalar_t33 epsilon66, const stat_accscalar_t33 momentum77, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_mean88, PackedTensorAccessor<stat_scalar_t22, 1, RestrictPtrTraits, index_t44> running_var99, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_mean1010, PackedTensorAccessor<stat_accscalar_t33, 1, RestrictPtrTraits, index_t44> save_transformed_var1111) __attribute__((global))
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
    unsigned int blockDim_x_143;
    blockDim_x_143 = 512;
    unsigned int threadIdx_x_144;
    threadIdx_x_144 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
    unsigned int blockDim_y_145;
    blockDim_y_145 = 1;
    unsigned int threadIdx_y_146;
    threadIdx_y_146 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
    unsigned int blockDim_z_147;
    blockDim_z_147 = 1;
    unsigned int threadIdx_z_148;
    threadIdx_z_148 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
    int index3749;
    index3749 = threadIdx_x_144 + blockIdx.x * blockDim_x_143;
    int batchsize3850;
    batchsize3850 = idata3541.size(0);
    int channels3951;
    channels3951 = idata3541.size(1);
    int height14052;
    height14052 = idata3541.size(2);
    int width14153;
    width14153 = idata3541.size(3);
    int height24254;
    height24254 = odata3642.size(2);
    int width24355;
    width24355 = odata3642.size(3);
    if (index3749 < n3137) {
        int w24456;
        w24456 = index3749 % width24355;
        int h24557;
        h24557 = index3749 / width24355;
        if (height14052 == height24254 && width14153 == width24355) {
            int h15668;
            h15668 = h24557;
            int w15769;
            w15769 = w24456;
            for (int n = 0; n < batchsize3850; n++) {
                for (int c = 0; c < channels3951; ++c) {
                    scalar_t2935 val5870;
                    val5870 = idata3541[n][c][h15668][w15769];
                    odata3642[n][c][h24557][w24456] = val5870;
                }
            }
            return;
        }
        accscalar_t3036 h1r4658;
        h1r4658 = area_pixel_compute_source_index<accscalar_t3036>(rheight3238, h24557, align_corners3440, false);
        int h14759;
        h14759 = h1r4658;
        int h1p4860;
        h1p4860 = (h14759 < height14052 - 1) ? 1 : 0;
        accscalar_t3036 h1lambda4961;
        h1lambda4961 = h1r4658 - h14759;
        accscalar_t3036 h0lambda5062;
        h0lambda5062 = static_cast<accscalar_t3036>(1) - h1lambda4961;
        accscalar_t3036 w1r5163;
        w1r5163 = area_pixel_compute_source_index<accscalar_t3036>(rwidth3339, w24456, align_corners3440, false);
        int w15264;
        w15264 = w1r5163;
        int w1p5365;
        w1p5365 = (w15264 < width14153 - 1) ? 1 : 0;
        accscalar_t3036 w1lambda5466;
        w1lambda5466 = w1r5163 - w15264;
        accscalar_t3036 w0lambda5567;
        w0lambda5567 = static_cast<accscalar_t3036>(1) - w1lambda5466;
        for (int n = 0; n < batchsize3850 - 1; n++) {
            for (int c = 0; c < channels3951; ++c) {
                accscalar_t3036 val5971;
                val5971 = h0lambda5062 * (w0lambda5567 * idata3541[n][c][h14759][w15264] + w1lambda5466 * idata3541[n][c][h14759][w15264 + w1p5365]) + h1lambda4961 * (w0lambda5567 * idata3541[n][c][h14759 + h1p4860][w15264] + w1lambda5466 * idata3541[n][c][h14759 + h1p4860][w15264 + w1p5365]);
                odata3642[n][c][h24557][w24456] = static_cast<scalar_t2935>(val5971);
            }
        }
        for (int n = batchsize3850 - 1; n < batchsize3850; n++) {
            for (int c = 0; c < channels3951; ++c) {
                accscalar_t3036 val6072;
                val6072 = h0lambda5062 * (w0lambda5567 * idata3541[n][c][h14759][w15264] + w1lambda5466 * idata3541[n][c][h14759][w15264 + w1p5365]) + h1lambda4961 * (w0lambda5567 * idata3541[n][c][h14759 + h1p4860][w15264] + w1lambda5466 * idata3541[n][c][h14759 + h1p4860][w15264 + w1p5365]);
                odata3642[n][c][h24557][w24456] = static_cast<scalar_t2935>(val6072);
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
    threadIdx_x_013 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 32;
    unsigned int blockDim_y_014;
    blockDim_y_014 = 16;
    unsigned int threadIdx_y_015;
    threadIdx_y_015 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 32 % 16;
    unsigned int blockDim_z_016;
    blockDim_z_016 = 1;
    unsigned int threadIdx_z_017;
    threadIdx_z_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
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


__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_bilinear2d_out_frame(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int height1 = idata.size(2);
  const int width1 = idata.size(3);
  const int height2 = odata.size(2);
  const int width2 = odata.size(3);

  if (index < n) {
    const int w2 = index % width2; // 0:width2-1
    const int h2 = index / width2; // 0:height2-1
    // special case: just copy
    if (height1 == height2 && width1 == width2) {
      const int h1 = h2;
      const int w1 = w2;
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][h1][w1];
          odata[n][c][h2][w2] = val;
        }
      }
      return;
    }
    //
    const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
        rheight, h2, align_corners, /*cubic=*/false);
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    //
    const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
        rwidth, w2, align_corners, /*cubic=*/false);
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    //
    for (int n = 0; n < batchsize-1; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = h0lambda *
                (w0lambda * idata[n][c][h1][w1] +
                 w1lambda * idata[n][c][h1][w1 + w1p]) +
            h1lambda *
                (w0lambda * idata[n][c][h1 + h1p][w1] +
                 w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
        odata[n][c][h2][w2] = static_cast<scalar_t>(val);
      }
    }
    for (int n = batchsize-1; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = h0lambda *
                (w0lambda * idata[n][c][h1][w1] +
                 w1lambda * idata[n][c][h1][w1 + w1p]) +
            h1lambda *
                (w0lambda * idata[n][c][h1 + h1p][w1] +
                 w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
        odata[n][c][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}

template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel3_(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
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


template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel2_(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_6;
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
accscalar_t30 w0lambda55;
accscalar_t30 h0lambda50;
accscalar_t30 w1lambda54;
accscalar_t30 h1lambda49;
int h1p48;
int h147;
int w1p53;
int w152;
int w244;
int h245;
accscalar_t30 h1r46;
accscalar_t30 w1r51;
if (index37 < n31) {
    w244 = index37 % width243;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    h147 = h1r46;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    h1lambda49 = h1r46 - h147;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    w152 = w1r51;
    w1p53 = (w152 < width141 - 2) ? 1 : 0;
    w1lambda54 = w1r51 - w152;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38 - 4; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
    }
}
label_6:;
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
if (index37 < n31) {
    for (int n = batchsize38 - 4; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
        }
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


template <typename scalar_t29, typename accscalar_t30, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_0(const int n31, const accscalar_t30 rheight32, const accscalar_t30 rwidth33, const bool align_corners34, const PackedTensorAccessor<scalar_t29, 4> idata35, PackedTensorAccessor<scalar_t29, 4> odata36, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
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
int index37;
index37 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize38;
batchsize38 = idata35.size(0);
int channels39;
channels39 = idata35.size(1);
int height140;
height140 = idata35.size(2);
int width141;
width141 = idata35.size(3);
int height242;
height242 = odata36.size(2);
int width243;
width243 = odata36.size(3);
if (index37 < n31) {
    int w244;
    w244 = index37 % width243;
    int h245;
    h245 = index37 / width243;
    if (height140 == height242 && width141 == width243) {
        int h156;
        h156 = h245;
        int w157;
        w157 = w244;
        for (int n = 0; n < batchsize38; n++) {
            for (int c = 0; c < channels39; ++c) {
                scalar_t29 val58;
                val58 = idata35[n][c][h156][w157];
                odata36[n][c][h245][w244] = val58;
            }
        }
        return;
    }
    accscalar_t30 h1r46;
    h1r46 = area_pixel_compute_source_index<accscalar_t30>(rheight32, h245, align_corners34, false);
    int h147;
    h147 = h1r46;
    int h1p48;
    h1p48 = (h147 < height140 - 1) ? 1 : 0;
    accscalar_t30 h1lambda49;
    h1lambda49 = h1r46 - h147;
    accscalar_t30 h0lambda50;
    h0lambda50 = static_cast<accscalar_t30>(1) - h1lambda49;
    accscalar_t30 w1r51;
    w1r51 = area_pixel_compute_source_index<accscalar_t30>(rwidth33, w244, align_corners34, false);
    int w152;
    w152 = w1r51;
    int w1p53;
    w1p53 = (w152 < width141 - 1) ? 1 : 0;
    accscalar_t30 w1lambda54;
    w1lambda54 = w1r51 - w152;
    accscalar_t30 w0lambda55;
    w0lambda55 = static_cast<accscalar_t30>(1) - w1lambda54;
    for (int n = 0; n < batchsize38; n++) {
        for (int c = 0; c < channels39; ++c) {
            accscalar_t30 val59;
            val59 = h0lambda50 * (w0lambda55 * idata35[n][c][h147][w152] + w1lambda54 * idata35[n][c][h147][w152 + w1p53]) + h1lambda49 * (w0lambda55 * idata35[n][c][h147 + h1p48][w152] + w1lambda54 * idata35[n][c][h147 + h1p48][w152 + w1p53]);
            odata36[n][c][h245][w244] = static_cast<scalar_t29>(val59);
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


template <typename scalar_t, typename accscalar_t, template<typename T> class VarTransform, typename input_scalar_t, typename stat_scalar_t, typename stat_accscalar_t, typename index_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void upsample_batchnorm_kernel(
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata,
    const PackedTensorAccessor<input_scalar_t, 3, RestrictPtrTraits, index_t> input,
    const stat_accscalar_t epsilon,
    const stat_accscalar_t momentum,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_mean,
    PackedTensorAccessor<stat_scalar_t, 1, RestrictPtrTraits, index_t> running_var,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_mean,
    PackedTensorAccessor<stat_accscalar_t, 1, RestrictPtrTraits, index_t> save_transformed_var) {
  if (threadIdx.x < 512) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    const int batchsize = idata.size(0);
    const int channels = idata.size(1);
    const int height1 = idata.size(2);
    const int width1 = idata.size(3);
    const int height2 = odata.size(2);
    const int width2 = odata.size(3);

    if (index < n) {
      const int w2 = index % width2; // 0:width2-1
      const int h2 = index / width2; // 0:height2-1
      // special case: just copy
      if (height1 == height2 && width1 == width2) {
        const int h1 = h2;
        const int w1 = w2;
        for (int n = 0; n < batchsize; n++) {
          for (int c = 0; c < channels; ++c) {
            const scalar_t val = idata[n][c][h1][w1];
            odata[n][c][h2][w2] = val;
          }
        }
        return;
      }
      //
      const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
          rheight, h2, align_corners, /*cubic=*/false);
      const int h1 = h1r;
      const int h1p = (h1 < height1 - 1) ? 1 : 0;
      const accscalar_t h1lambda = h1r - h1;
      const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
      //
      const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
          rwidth, w2, align_corners, /*cubic=*/false);
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const accscalar_t w1lambda = w1r - w1;
      const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
      //
      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const accscalar_t val = h0lambda *
                  (w0lambda * idata[n][c][h1][w1] +
                  w1lambda * idata[n][c][h1][w1 + w1p]) +
              h1lambda *
                  (w0lambda * idata[n][c][h1 + h1p][w1] +
                  w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
          odata[n][c][h2][w2] = static_cast<scalar_t>(val);
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

template<typename scalar_t_bn, typename index_t_bn>
std::tuple<Tensor, Tensor> upsample_batchnorm_stm(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
  const Tensor& input_bn_, double epsilon) {
  Tensor output = at::empty_like(input);
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_bilinear2d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({input.size(0), input.size(1), output_height, output_width});

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

  printf("%d %d\n", num_kernels, num_threads);
  cudaStream_t stream = at::cuda::getStreamFromPool(true);

  using accscalar_t_bn = at::acc_type<scalar_t_bn, true>;
  int64_t n_input_bn = input_bn_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_bn_reshaped = input_bn_.reshape({input_bn_.size(0), input_bn_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_bn_reshaped.size(0);
  auto features = input_bn_reshaped.size(2);
  auto input_bn = input_bn_reshaped.packed_accessor<scalar_t_bn, 3, RestrictPtrTraits, index_t_bn>();
  auto input_bn_options = input_bn_.options();
  dummy_mean_ = at::empty({0}, input_bn_options);
  dummy_var_ = at::empty({0}, input_bn_options);
  // promote only mean_/invstd_ precision
  if (input_bn_.scalar_type() == at::ScalarType::Half) {
    input_bn_options = input_bn_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input_bn}, input_bn_options);
  invstd_ = at::empty({n_input_bn}, input_bn_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto stream1 = at::cuda::getStreamFromPool();

  dim3 blocks(input_bn.size(1));
  int tf = getNumThreads(input_bn.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
        printf("%d\n", num_blocks);
        cudaProfilerStart();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
            <<<num_blocks,
               num_threads,
               0,
               stream>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
        batch_norm_collect_statistics_kernel<InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn> <<<blocks, threads, 0, stream1>>>
          (input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("time: %f\n", milliseconds);
        cudaProfilerStop();
      });

  AT_CUDA_CHECK(cudaGetLastError());
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, mean_);
}

template<typename scalar_t_bn, typename index_t_bn>
std::tuple<Tensor, Tensor> upsample_batchnorm_fused(
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
  const Tensor& input_bn_, double epsilon) {
  Tensor output = at::empty_like(input);
  TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_bilinear2d_out_cuda", {input_arg, output_arg});

  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int nbatch = input.size(0);
  int channels = input.size(1);
  int input_height = input.size(2);
  int input_width = input.size(3);

  upsample_2d_shape_check(
      input,
      Tensor(),
      nbatch,
      channels,
      input_height,
      input_width,
      output_height,
      output_width);

  output.resize_({input.size(0), input.size(1), output_height, output_width});

  AT_ASSERT(
      input_height > 0 && input_width > 0 && output_height > 0 &&
      output_width > 0);

  const int num_kernels = output_height * output_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 512);

  printf("%d %d\n", num_kernels, num_threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using accscalar_t_bn = at::acc_type<scalar_t_bn, true>;
  int64_t n_input_bn = input_bn_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_bn_reshaped = input_bn_.reshape({input_bn_.size(0), input_bn_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_bn_reshaped.size(0);
  auto features = input_bn_reshaped.size(2);
  auto input_bn = input_bn_reshaped.packed_accessor<scalar_t_bn, 3, RestrictPtrTraits, index_t_bn>();
  auto input_bn_options = input_bn_.options();
  dummy_mean_ = at::empty({0}, input_bn_options);
  dummy_var_ = at::empty({0}, input_bn_options);
  // promote only mean_/invstd_ precision
  if (input_bn_.scalar_type() == at::ScalarType::Half) {
    input_bn_options = input_bn_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input_bn}, input_bn_options);
  invstd_ = at::empty({n_input_bn}, input_bn_options);
  auto mean = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t_bn, 1, RestrictPtrTraits, index_t_bn>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t_bn, 1, RestrictPtrTraits, index_t_bn>();
  auto stream1 = at::cuda::getCurrentCUDAStream();

  dim3 blocks(input_bn.size(1));
  int tf = getNumThreads(input_bn.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor<scalar_t, 4>();
        auto odata = output.packed_accessor<scalar_t, 4>();

        const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners);
        const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners);

        const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
        printf("%d\n", num_blocks);
        printf("%d %d\n", idata.size(0), idata.size(1));
        cudaProfilerStart();

        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start);
        cudaDeviceSynchronize();
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_0<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 1024, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
      cudaDeviceSynchronize();
        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);
        // float milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("time: %f\n", milliseconds);
        upsample_bilinear2d_out_frame_batch_norm_collect_statistics_kernel_100<scalar_t, accscalar_t, InvStd, scalar_t_bn, scalar_t_bn, accscalar_t_bn, index_t_bn>
            <<<blocks, 512, 0, stream1>>>(
              num_kernels, rheight, rwidth, align_corners, idata, odata,
              input_bn, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
        cudaDeviceSynchronize();
        cudaProfilerStop();
      });

  AT_CUDA_CHECK(cudaGetLastError());
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, mean_);
}


} } // namespace at::native
