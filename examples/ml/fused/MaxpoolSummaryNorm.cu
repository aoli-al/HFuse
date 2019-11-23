#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
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
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"
#include <c10/macros/Macros.h>

namespace at {
namespace native {

__device__ inline int min(int a, int b) {
  return a <= b ? a : b;
}

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

static const int BACKWARD_THREADS = 256;


template <typename scalar_t, int64_t dim, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
static PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t> packed_accessor_or_dummy(const Tensor& t) {
  if (! t.defined()) {
    const std::vector<index_t> zeros(dim);
    return PackedTensorAccessor<scalar_t, dim, PtrTraits, index_t>(nullptr, zeros.data(), zeros.data());
  }
  return t.packed_accessor<scalar_t, dim, PtrTraits, index_t>();
}


using namespace at::cuda;
using namespace at::cuda::detail;

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
// constexpr int MAX_BLOCK_SIZE = 256;

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

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % WARP_SIZE == 0) {
    shared[tid / WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template <typename scalar_t, typename accscalar_t, bool train, typename index_t>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> input,
    PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, index_t> output,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> mean_,
    const PackedTensorAccessor<typename std::conditional<train, accscalar_t, scalar_t>::type, 1, RestrictPtrTraits, index_t> var_or_invstd,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> weight,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, index_t> bias,
    accscalar_t epsilon) {

  index_t plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd;
  if (train) {
    invstd = var_or_invstd[plane];
  } else {
    invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(var_or_invstd[plane]) + epsilon);
  }

  index_t bs = input.size(0);
  index_t fs = input.size(2);

  index_t bstep  = blockDim.y * gridDim.y;
  for (index_t batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (index_t feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
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


#define THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM 100
#define THRESH_NUMBER_BINS_FOR_GLOBAL_MEM 1000
#define FOR_KERNEL_LOOP(i, lim)                                      \
  for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < lim; \
       i += gridDim.x * blockDim.x)

/*
  Memory types used for the 3 histogram implementations.
  See `CUDA_tensor_histogram` below.
 */
enum class CUDAHistogramMemoryType { SHARED, MULTI_BLOCK, GLOBAL };
namespace {
  template<typename input_t, typename IndexType>
  __device__ static IndexType getBin(input_t bVal, input_t minvalue, input_t maxvalue, int nbins) {
    IndexType bin = (int)((bVal - minvalue) * nbins / (maxvalue - minvalue));
    // (only applicable for histc)
    // while each bin is inclusive at the lower end and exclusive at the higher, i.e. [start, end)
    // the last bin is inclusive at both, i.e. [start, end], in order to include maxvalue if exists
    // therefore when bin == nbins, adjust bin to the last bin
    if (bin == nbins) bin -= 1;
    return bin;
  }
}

/*
  Kernel for computing the histogram of the input.
 */
template <
    typename output_t,
    typename input_t,
    typename IndexType,
    int ADims,
    int PDims,
    int BDims,
    CUDAHistogramMemoryType MemoryType = CUDAHistogramMemoryType::MULTI_BLOCK,
    typename Op>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void kernelHistogram1D(
    TensorInfo<output_t, IndexType> a, /* output */
    TensorInfo<output_t, IndexType> p, /* partial output */
    TensorInfo<input_t, IndexType> b, /* input */
    int nbins,
    input_t minvalue,
    input_t maxvalue,
    IndexType totalElements,
    Op getOp) {
  extern __shared__ unsigned char my_smem[];
  output_t* smem = nullptr;

    ////////////////////////// Shared memory //////////////////////////
    // atomically add to block specific shared memory
    // then atomically add to the global output tensor
    smem = reinterpret_cast<output_t*>(my_smem);
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      smem[i] = 0;
    }
    __syncthreads();
    FOR_KERNEL_LOOP(linearIndex, totalElements) {
      // Convert `linearIndex` into an offset of `b`
      const IndexType bOffset =
          IndexToOffset<input_t, IndexType, BDims>::get(linearIndex, b);
      const input_t bVal = b.data[bOffset];
      if (bVal >= minvalue && bVal <= maxvalue) {
        // Use value at `b` as an offset of `smem`
        const IndexType bin = getBin<input_t, IndexType>(bVal, minvalue, maxvalue, nbins);
        atomicAdd(&smem[bin], getOp(linearIndex));
      }
    }
    __syncthreads();
    // NOTE: atomically update output bin count.
    //   Atomic update is imp since __syncthread() will only synchronize threads
    //   in a given block, not across blocks.
    for (IndexType i = threadIdx.x; i < a.sizes[0]; i += blockDim.x) {
      const IndexType aOffset =
          IndexToOffset<output_t, IndexType, ADims>::get(i, a);
      atomicAdd(&a.data[aOffset], smem[i]);
    }

}

template <typename scalar_t0, typename accscalar_t1, typename output_t60, typename input_t61, typename IndexType62, int ADims63, int PDims64, int BDims65, at::native::CUDAHistogramMemoryType MemoryType66 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op67, template <typename T> class VarTransform31, typename input_scalar_t32, typename stat_scalar_t33, typename stat_accscalar_t34, typename index_t35>
 __attribute__((global)) void FUNC2(const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19, TensorInfo<output_t60, IndexType62> a68, TensorInfo<output_t60, IndexType62> p69, TensorInfo<input_t61, IndexType62> b70, int nbins71, input_t61 minvalue72, input_t61 maxvalue73, IndexType62 totalElements74, Op67 getOp75, const PackedTensorAccessor<input_scalar_t32, 3, RestrictPtrTraits, index_t35> input36, const stat_accscalar_t34 epsilon37, const stat_accscalar_t34 momentum38, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_mean39, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_var40, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_mean41, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_transformed_var42)
 {
   if (threadIdx.x < 256) {
    unsigned int blockDim_x_0;
    blockDim_x_0 = 256;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
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
}
{
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
    extern unsigned char my_smem76[] __attribute__((shared));
    output_t60 *smem77;
    smem77 = nullptr;
    smem77 = reinterpret_cast<output_t60 *>(my_smem76);
    for (IndexType62 i = threadIdx_x_2; i < a68.sizes[0]; i += blockDim_x_2) {
        smem77[i] = 0;
    }
    __syncthreads();
    for (IndexType62 linearIndex = blockIdx.x * blockDim_x_2 + threadIdx_x_2; linearIndex < totalElements74; linearIndex += gridDim.x * blockDim_x_2) {
        IndexType62 bOffset78;
        bOffset78 = IndexToOffset<input_t61, IndexType62, BDims65>::get(linearIndex, b70);
        input_t61 bVal79;
        bVal79 = b70.data[bOffset78];
        if (bVal79 >= minvalue72 && bVal79 <= maxvalue73) {
            IndexType62 bin80;
            bin80 = getBin<input_t61, IndexType62>(bVal79, minvalue72, maxvalue73, nbins71);
            atomicAdd(&smem77[bin80], getOp75(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType62 i = threadIdx_x_2; i < a68.sizes[0]; i += blockDim_x_2) {
        IndexType62 aOffset81;
        aOffset81 = IndexToOffset<output_t60, IndexType62, ADims63>::get(i, a68);
        atomicAdd(&a68.data[aOffset81], smem77[i]);
    }
}
{
    unsigned int blockDim_x_1;
    blockDim_x_1 = 32;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 32;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 16;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 32 % 16;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
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
    __syncthreads();
    if (tid46 % WARP_SIZE == 0) {
        shared_n43[tid46 / WARP_SIZE] = n50;
        shared_avg_var47[tid46 / WARP_SIZE * 2] = avg48;
        shared_avg_var47[tid46 / WARP_SIZE * 2 + 1] = var_n49;
    }
    __syncthreads();
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
}
}



template <typename scalar_t0, typename accscalar_t1, typename output_t60, typename input_t61, typename IndexType62, int ADims63, int PDims64, int BDims65, at::native::CUDAHistogramMemoryType MemoryType66 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op67, template <typename T> class VarTransform31, typename input_scalar_t32, typename stat_scalar_t33, typename stat_accscalar_t34, typename index_t35>
 __attribute__((global)) void FUNC(const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19, TensorInfo<output_t60, IndexType62> a68, TensorInfo<output_t60, IndexType62> p69, TensorInfo<input_t61, IndexType62> b70, int nbins71, input_t61 minvalue72, input_t61 maxvalue73, IndexType62 totalElements74, Op67 getOp75, const PackedTensorAccessor<input_scalar_t32, 3, RestrictPtrTraits, index_t35> input36, const stat_accscalar_t34 epsilon37, const stat_accscalar_t34 momentum38, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_mean39, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_var40, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_mean41, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_transformed_var42)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_1;
unsigned int blockDim_x_2;
blockDim_x_2 = 512;
unsigned int threadIdx_x_2;
threadIdx_x_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
unsigned int blockDim_y_2;
blockDim_y_2 = 1;
unsigned int threadIdx_y_2;
threadIdx_y_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
unsigned int blockDim_z_2;
blockDim_z_2 = 1;
unsigned int threadIdx_z_2;
threadIdx_z_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
extern unsigned char my_smem76[] __attribute__((shared));
output_t60 *smem77;
smem77 = nullptr;
smem77 = reinterpret_cast<output_t60 *>(my_smem76);
for (IndexType62 i = threadIdx_x_2; i < a68.sizes[0]; i += blockDim_x_2) {
    smem77[i] = 0;
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_2;
for (IndexType62 linearIndex = blockIdx.x * blockDim_x_2 + threadIdx_x_2; linearIndex < totalElements74; linearIndex += gridDim.x * blockDim_x_2) {
    IndexType62 bOffset78;
    bOffset78 = IndexToOffset<input_t61, IndexType62, BDims65>::get(linearIndex, b70);
    input_t61 bVal79;
    bVal79 = b70.data[bOffset78];
    if (bVal79 >= minvalue72 && bVal79 <= maxvalue73) {
        IndexType62 bin80;
        bin80 = getBin<input_t61, IndexType62>(bVal79, minvalue72, maxvalue73, nbins71);
        atomicAdd(&smem77[bin80], getOp75(linearIndex));
    }
}
label_2:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_0;
unsigned int blockDim_x_0;
blockDim_x_0 = 256;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
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
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_4;
unsigned int blockDim_x_1;
blockDim_x_1 = 16;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
unsigned int blockDim_y_1;
blockDim_y_1 = 16;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
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
label_4:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_3;
for (IndexType62 i = threadIdx_x_2; i < a68.sizes[0]; i += blockDim_x_2) {
    IndexType62 aOffset81;
    aOffset81 = IndexToOffset<output_t60, IndexType62, ADims63>::get(i, a68);
    atomicAdd(&a68.data[aOffset81], smem77[i]);
}
label_3:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_5;
if (tid46 % WARP_SIZE == 0) {
    shared_n43[tid46 / WARP_SIZE] = n50;
    shared_avg_var47[tid46 / WARP_SIZE * 2] = avg48;
    shared_avg_var47[tid46 / WARP_SIZE * 2 + 1] = var_n49;
}
label_5:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_6;
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
label_6:;
}



inline int64_t getFreeGlobalMemory() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  AT_ASSERTM(
      cudaGetLastError() == cudaSuccess,
      "CUDA_tensor_histogram failed to get free global memory");
  return static_cast<int64_t>(free_mem);
}





template <typename input_hist_t, typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_, double epsilon,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode
    ) {
  printf("2\n");
  if (nbins <= 0) {
    AT_ERROR("bins must be > 0");
  }
  Tensor output_hist = native::zeros({nbins}, device(DeviceType::CUDA).dtype(self_hist.scalar_type()));
  input_hist_t minvalue = min;
  input_hist_t maxvalue = max;
  if (min == max) {
    minvalue = *self_hist.min().cpu().data<input_hist_t>();
    maxvalue = *self_hist.max().cpu().data<input_hist_t>();
  }
  if (minvalue == maxvalue) {
    minvalue = minvalue - 1;
    maxvalue = maxvalue + 1;
  }

  printf("3\n");
  checkBackend("CUDA_tensor_histogram", {output_hist, self_hist}, Backend::CUDA);
  auto totalElements = self_hist.numel();

  const dim3 block = getApplyBlock();
  dim3 grid;
  int64_t curDevice = current_device();

  grid.x = 10000;

  CUDAHistogramMemoryType memType = CUDAHistogramMemoryType::GLOBAL;
  auto maxSharedMem = getCurrentDeviceProperties()->sharedMemPerBlock;
  auto sharedMem = nbins * sizeof(input_hist_t) + 8; // 8 guard bytes
  auto maxGlobalMem = getFreeGlobalMemory();
  auto multiBlockMem = nbins * grid.x * sizeof(input_hist_t) + 8; // 8 guard bytes
  // determine memory type to use in the kernel
    printf("6\n");
  if (nbins < THRESH_NUMBER_BINS_FOR_MULTI_BLOCK_MEM &&
      sharedMem < maxSharedMem) {
    printf("shared\n");
    memType = CUDAHistogramMemoryType::SHARED;
  } else if (
      nbins < THRESH_NUMBER_BINS_FOR_GLOBAL_MEM &&
      multiBlockMem < (maxGlobalMem / 2)) {
    // check against half of free mem to be extra safe
    // due to cached allocator, we may anyway have slightly more free mem
    printf("mb\n");
    memType = CUDAHistogramMemoryType::MULTI_BLOCK;
  }

  // alloc memory for MULTI_BLOCK
  using IndexType = int64_t;
  auto aInfo = getTensorInfo<input_hist_t, IndexType>(output_hist);
  auto bInfo = getTensorInfo<input_hist_t, IndexType>(self_hist);
  TensorInfo<input_hist_t, IndexType> pInfo(nullptr, 0, {}, {});
  Tensor partial_output_hist;
  if (memType == CUDAHistogramMemoryType::MULTI_BLOCK) {
    partial_output_hist = native::zeros({grid.x, nbins}, output_hist.options());
    pInfo = getTensorInfo<input_hist_t, IndexType>(partial_output_hist);
  }

  printf("7\n");
  printf("10\n");
  // Launch kernel
  using accscalar_t = at::acc_type<scalar_t, true>;
  int64_t n_input = input_.size(1);
  Tensor dummy_mean_;
  Tensor dummy_var_;
  Tensor mean_;
  Tensor invstd_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

  auto bs = input_reshaped.size(0);
  auto features = input_reshaped.size(2);
  auto input = input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, index_t>();
  auto input_options = input_.options();
  dummy_mean_ = at::empty({0}, input_options);
  dummy_var_ = at::empty({0}, input_options);
  // promote only mean_/invstd_ precision
  if (input_.scalar_type() == at::ScalarType::Half) {
    input_options = input_options.dtype(ScalarType::Float);
  }
  mean_ = at::empty({n_input}, input_options);
  invstd_ = at::empty({n_input}, input_options);
  auto mean = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(mean_);
  auto invstd = packed_accessor_or_dummy<accscalar_t, 1, RestrictPtrTraits, index_t>(invstd_);
  auto dummy_mean = dummy_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto dummy_invstd = dummy_var_.packed_accessor<scalar_t, 1, RestrictPtrTraits, index_t>();
  auto stream = at::cuda::getCurrentCUDAStream();

  Tensor output_maxpool = at::empty({0}, input_maxpool_.options());
  Tensor indices = at::empty({0}, input_maxpool_.options().dtype(kLong));
  TensorArg output_maxpool_arg{ output_maxpool, "output_maxpool", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_maxpool_arg{ input_maxpool_, "input_maxpool_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {output_maxpool_arg, indices_arg, input_maxpool_arg});

  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
    "max_pool2d: kernel_size must either be a single int, or a tuple of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  // NB: stride default is not expressible as an integer constant, so we accept
  // empty stride for this case
  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
    "max_pool2d: stride must either be omitted, a single int, or a tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 2,
    "max_pool2d: padding must be either be a single int, or a tuple of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 2,
    "max_pool2d: dilation must be either a single int, or a tuple of two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK((input_maxpool_.ndimension() == 3 || input_maxpool_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input_maxpool");

  const int64_t nbatch = input_maxpool_.ndimension() == 4 ? input_maxpool_.size(-4) : 1;
  const int64_t nInputPlane = input_maxpool_.size(-3);
  const int64_t input_maxpoolHeight = input_maxpool_.size(-2);
  const int64_t input_maxpoolWidth = input_maxpool_.size(-1);

  const int64_t output_maxpoolWidth = pooling_output_shape<int64_t>(input_maxpoolWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t output_maxpoolHeight = pooling_output_shape<int64_t>(input_maxpoolHeight, kH, padH, dH, dilationH, ceil_mode);
  printf("%ld, %ld\n", output_maxpoolWidth, output_maxpoolHeight);

  pool2d_shape_check(
    input_maxpool_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    input_maxpoolHeight, input_maxpoolWidth,
    output_maxpoolHeight, output_maxpoolWidth);

  Tensor input_maxpool = input_maxpool_.contiguous();

  output_maxpool.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});
  indices.resize_({nbatch, nInputPlane, output_maxpoolHeight, output_maxpoolWidth});

  const int count = safe_downcast<int, int64_t>(output_maxpool.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
  using accscalar_t = acc_type<scalar_t, true>;

  scalar_t *output_maxpool_data = output_maxpool.data<scalar_t>();
  scalar_t *input_maxpool_data = input_maxpool.data<scalar_t>();
  int64_t *indices_data = indices.data<int64_t>();

  const int blocks_maxpool = cuda::ATenCeilDiv(count, num_threads);
  printf("%d %d %d\n", count, blocks_maxpool, num_threads);
  printf("%d %d %d\n", input.size(0), input.size(1), input.size(2));
  cudaProfilerStart();
  batch_norm_collect_statistics_kernel<InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, getStreamFromPool(true)>>>
    (input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
      <<<grid,
        block,
        sharedMem,
        getStreamFromPool(true)>>>(
          aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
  MaxPoolForward<scalar_t, scalar_t>
    <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, at::cuda::getStreamFromPool(true)>>>(
      count, input_maxpool_data,
      nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_maxpool_data, indices_data);
  cudaDeviceSynchronize();
  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  FUNC<scalar_t, scalar_t,input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
  InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, 1024, sharedMem>>>(
      count, input_maxpool_data,
      nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_maxpool_data, indices_data,
          aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  cudaDeviceSynchronize();
  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  FUNC2<scalar_t, scalar_t,input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
  InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, 512, sharedMem>>>(
      count, input_maxpool_data,
      nbatch, nInputPlane, input_maxpoolHeight, input_maxpoolWidth, output_maxpoolHeight, output_maxpoolWidth,
      kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_maxpool_data, indices_data,
          aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);


  cudaDeviceSynchronize();
  cudaProfilerStop();
  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, mean_);
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> max_hist_norm(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
    Tensor& input_,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode
) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template<scalar_t, scalar_t, int32_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>()
    , input_, 0.2,
    input_maxpool_,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode
  );
  });
}

} // namespace native
} // namespace at
