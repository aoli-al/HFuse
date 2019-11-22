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
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>
#include <c10/macros/Macros.h>

#include <c10/macros/Macros.h>
#include <ATen/native/im2col_shape_check.h>

#include <cuda_profiler_api.h>
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


using namespace at::cuda;
using namespace at::cuda::detail;

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

template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
void kernelHistogram1D_MaxPoolForward_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((global))
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_0:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_1;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_2;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_2:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_3;
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
label_3:;
}

template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
void kernelHistogram1D_MaxPoolForward_100(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((global))
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
    extern unsigned char my_smem47[] __attribute__((shared));
    output_t31 *smem48;
    smem48 = nullptr;
    smem48 = reinterpret_cast<output_t31 *>(my_smem47);
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        smem48[i] = 0;
    }
    __syncthreads();
    for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType33 bOffset49;
        bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
        input_t32 bVal50;
        bVal50 = b41.data[bOffset49];
        if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
            IndexType33 bin51;
            bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
            atomicAdd(&smem48[bin51], getOp46(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
        IndexType33 aOffset52;
        aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
        atomicAdd(&a39.data[aOffset52], smem48[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
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
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
void kernelHistogram1D_MaxPoolForward_11(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((global))
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_0:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_1;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_1:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_3;
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
label_3:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_2;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_2:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
void kernelHistogram1D_MaxPoolForward_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((global))
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
extern unsigned char my_smem47[] __attribute__((shared));
output_t31 *smem48;
smem48 = nullptr;
smem48 = reinterpret_cast<output_t31 *>(my_smem47);
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    smem48[i] = 0;
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_3;
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
label_3:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_1;
for (IndexType33 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements45; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType33 bOffset49;
    bOffset49 = IndexToOffset<input_t32, IndexType33, BDims36>::get(linearIndex, b41);
    input_t32 bVal50;
    bVal50 = b41.data[bOffset49];
    if (bVal50 >= minvalue43 && bVal50 <= maxvalue44) {
        IndexType33 bin51;
        bin51 = getBin<input_t32, IndexType33>(bVal50, minvalue43, maxvalue44, nbins42);
        atomicAdd(&smem48[bin51], getOp46(linearIndex));
    }
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_2;
for (IndexType33 i = threadIdx_x_1; i < a39.sizes[0]; i += blockDim_x_1) {
    IndexType33 aOffset52;
    aOffset52 = IndexToOffset<output_t31, IndexType33, ADims34>::get(i, a39);
    atomicAdd(&a39.data[aOffset52], smem48[i]);
}
label_2:;
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

inline int64_t getFreeGlobalMemory() {
  // no need to use `cudaSetDevice`
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);
  AT_ASSERTM(
      cudaGetLastError() == cudaSuccess,
      "CUDA_tensor_histogram failed to get free global memory");
  return static_cast<int64_t>(free_mem);
}


template <typename input_hist_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode)
  {
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

  Tensor output = at::empty({0}, input_.options());
  Tensor indices = at::empty({0}, input_.options().dtype(kLong));

  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  checkAllSameGPU("max_pool2d_with_indices_out_cuda",
                  {output_arg, indices_arg, input_arg});

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

  TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  printf("%ld, %ld\n", outputWidth, outputHeight);

  pool2d_shape_check(
    input_,
    kH, kW, dH, dW, padH, padW, dilationH, dilationW,
    nInputPlane,
    inputHeight, inputWidth,
    outputHeight, outputWidth);

  Tensor input = input_.contiguous();

  output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch, nInputPlane, outputHeight, outputWidth});

  const int count = safe_downcast<int, int64_t>(output.numel());
  const int num_threads = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_data = output.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      int64_t *indices_data = indices.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads);
      printf("%d %d %d\n", count, blocks, num_threads);
      cudaProfilerStart();
            cudaDeviceSynchronize();
      // MaxPoolForward<scalar_t, scalar_t>
      //   <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, at::cuda::getStreamFromPool(true)>>>(
      //     count, input_data,
      //     nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
      //     kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
    // Launch kernel
    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    // kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    //     <<<grid,
    //       block,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    //         cudaDeviceSynchronize();
            kernelHistogram1D_MaxPoolForward_0
    <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
      scalar_t, scalar_t>
        <<<grid,
          block.x + num_threads,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
            cudaDeviceSynchronize();
    //   kernelHistogram1D_MaxPoolForward_100
    // <input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp),
    //   scalar_t, scalar_t>
    //     <<<grid,
    //       512,
    //       sharedMem,
    //       getStreamFromPool(true)>>>(
    //         aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
    //       count, input_data,
    //       nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
    //       kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
      cudaProfilerStop();
    });


  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, output);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> _histc_maxpool(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,

           const Tensor& input_,
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
    return native::_histc_cuda_template<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
           input_,
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
