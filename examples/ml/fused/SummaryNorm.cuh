#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <THC/THCGeneral.h>
#include <THC/THCDeviceUtils.cuh>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

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
#include <ATen/native/im2col_shape_check.h>

namespace at {
namespace native {

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

template <typename output_t29, typename input_t30, typename IndexType31, int ADims32, int PDims33, int BDims34, at::native::CUDAHistogramMemoryType MemoryType35 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op36, template <typename T> class VarTransform0, typename input_scalar_t1, typename stat_scalar_t2, typename stat_accscalar_t3, typename index_t4>
void kernelHistogram1D_batch_norm_collect_statistics_kernel_(TensorInfo<output_t29, IndexType31> a37, TensorInfo<output_t29, IndexType31> p38, TensorInfo<input_t30, IndexType31> b39, int nbins40, input_t30 minvalue41, input_t30 maxvalue42, IndexType31 totalElements43, Op36 getOp44, const PackedTensorAccessor<input_scalar_t1, 3, RestrictPtrTraits, index_t4> input5, const stat_accscalar_t3 epsilon6, const stat_accscalar_t3 momentum7, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_mean8, PackedTensorAccessor<stat_scalar_t2, 1, RestrictPtrTraits, index_t4> running_var9, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_mean10, PackedTensorAccessor<stat_accscalar_t3, 1, RestrictPtrTraits, index_t4> save_transformed_var11) __attribute__((global))
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
extern unsigned char my_smem45[] __attribute__((shared));
output_t29 *smem46;
smem46 = nullptr;
smem46 = reinterpret_cast<output_t29 *>(my_smem45);
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    smem46[i] = 0;
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
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_2;
for (IndexType31 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements43; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType31 bOffset47;
    bOffset47 = IndexToOffset<input_t30, IndexType31, BDims34>::get(linearIndex, b39);
    input_t30 bVal48;
    bVal48 = b39.data[bOffset47];
    if (bVal48 >= minvalue41 && bVal48 <= maxvalue42) {
        IndexType31 bin49;
        bin49 = getBin<input_t30, IndexType31>(bVal48, minvalue41, maxvalue42, nbins40);
        atomicAdd(&smem46[bin49], getOp44(linearIndex));
    }
}
label_2:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
if (tid15 % WARP_SIZE == 0) {
    shared_n12[tid15 / WARP_SIZE] = n19;
    shared_avg_var16[tid15 / WARP_SIZE * 2] = avg17;
    shared_avg_var16[tid15 / WARP_SIZE * 2 + 1] = var_n18;
}
label_3:;
__syncthreads();
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_4;
for (IndexType31 i = threadIdx_x_1; i < a37.sizes[0]; i += blockDim_x_1) {
    IndexType31 aOffset50;
    aOffset50 = IndexToOffset<output_t29, IndexType31, ADims32>::get(i, a37);
    atomicAdd(&a37.data[aOffset50], smem46[i]);
}
label_4:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_5;
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
label_5:;
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
    const Tensor& input_, double epsilon
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
  {
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

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
  batch_norm_collect_statistics_kernel<InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<blocks, threads, 0, stream>>>
    (input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
  kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
      <<<grid,
        block,
        sharedMem,
        getCurrentCUDAStream()>>>(
          aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);        \
  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, mean_);
}
}
template <typename input_hist_t, typename scalar_t, typename index_t>
std::tuple<Tensor, Tensor> _histc_cuda_fused(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_, double epsilon
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
  {
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

  dim3 blocks(input.size(1));
  int tf = getNumThreads(input.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  printf("%d %d %d\n", blocks.x, blocks.y, blocks.z);
  THCudaCheck(cudaGetLastError());

    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    kernelHistogram1D_batch_norm_collect_statistics_kernel_<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED,
  InvStd, scalar_t, scalar_t, accscalar_t, index_t> <<<10000, 1024, sharedMem, stream>>>
         (
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    input, epsilon, 0.0, dummy_mean, dummy_invstd, mean, invstd);
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, mean_);
}
}
} // namespace

namespace native {

} // namespace native
} // namespace at
