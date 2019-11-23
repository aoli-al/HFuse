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

#include <c10/macros/Macros.h>
#include <ATen/native/im2col_shape_check.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "../cuda/UpSample.cuh"
#include "../cuda/DeviceSqrt.cuh"
#include "../cuda/LaunchUtils.h"

#include <cuda_profiler_api.h>
namespace at {
namespace native {


__device__ __forceinline__ size_t
idx(const size_t nc,
    const size_t height,
    const size_t width,
    const size_t y,
    const size_t x) {
  return (nc * height + y) * width + x;
}

template <typename scalar_t, typename accscalar_t>
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
  }
}


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
void kernelHistogram1D_upsample_bilinear2d_out_frame_0(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7) __attribute__((global))
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
}
label_3:;
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
void kernelHistogram1D_upsample_bilinear2d_out_frame_11(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7) __attribute__((global))
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
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
void kernelHistogram1D_upsample_bilinear2d_out_frame_100(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7) __attribute__((global))
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
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0;
    blockDim_x_0 = 512;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    int index8;
    index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
    int batchsize9;
    batchsize9 = idata6.size(0);
    int channels10;
    channels10 = idata6.size(1);
    int height111;
    height111 = idata6.size(2);
    int width112;
    width112 = idata6.size(3);
    int height213;
    height213 = odata7.size(2);
    int width214;
    width214 = odata7.size(3);
    if (index8 < n2) {
        int w215;
        w215 = index8 % width214;
        int h216;
        h216 = index8 / width214;
        if (height111 == height213 && width112 == width214) {
            int h127;
            h127 = h216;
            int w128;
            w128 = w215;
            for (int n = 0; n < batchsize9; n++) {
                for (int c = 0; c < channels10; ++c) {
                    scalar_t0 val29;
                    val29 = idata6[n][c][h127][w128];
                    odata7[n][c][h216][w215] = val29;
                }
            }
            return;
        }
        accscalar_t1 h1r17;
        h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
        int h118;
        h118 = h1r17;
        int h1p19;
        h1p19 = (h118 < height111 - 1) ? 1 : 0;
        accscalar_t1 h1lambda20;
        h1lambda20 = h1r17 - h118;
        accscalar_t1 h0lambda21;
        h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
        accscalar_t1 w1r22;
        w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
        int w123;
        w123 = w1r22;
        int w1p24;
        w1p24 = (w123 < width112 - 1) ? 1 : 0;
        accscalar_t1 w1lambda25;
        w1lambda25 = w1r22 - w123;
        accscalar_t1 w0lambda26;
        w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                accscalar_t1 val30;
                val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
                odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
            }
        }
    }
}
}
template <typename output_t31, typename input_t32, typename IndexType33, int ADims34, int PDims35, int BDims36, at::native::CUDAHistogramMemoryType MemoryType37 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op38, typename scalar_t0, typename accscalar_t1>
void kernelHistogram1D_upsample_bilinear2d_out_frame_2(TensorInfo<output_t31, IndexType33> a39, TensorInfo<output_t31, IndexType33> p40, TensorInfo<input_t32, IndexType33> b41, int nbins42, input_t32 minvalue43, input_t32 maxvalue44, IndexType33 totalElements45, Op38 getOp46, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7) __attribute__((global))
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
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
unsigned int blockDim_x_0;
blockDim_x_0 = 512;
unsigned int threadIdx_x_0;
threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_0;
blockDim_y_0 = 1;
unsigned int threadIdx_y_0;
threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_0;
blockDim_z_0 = 1;
unsigned int threadIdx_z_0;
threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
int index8;
index8 = threadIdx_x_0 + blockIdx.x * blockDim_x_0;
int batchsize9;
batchsize9 = idata6.size(0);
int channels10;
channels10 = idata6.size(1);
int height111;
height111 = idata6.size(2);
int width112;
width112 = idata6.size(3);
int height213;
height213 = odata7.size(2);
int width214;
width214 = odata7.size(3);
if (index8 < n2) {
    int w215;
    w215 = index8 % width214;
    int h216;
    h216 = index8 / width214;
    if (height111 == height213 && width112 == width214) {
        int h127;
        h127 = h216;
        int w128;
        w128 = w215;
        for (int n = 0; n < batchsize9; n++) {
            for (int c = 0; c < channels10; ++c) {
                scalar_t0 val29;
                val29 = idata6[n][c][h127][w128];
                odata7[n][c][h216][w215] = val29;
            }
        }
        return;
    }
    accscalar_t1 h1r17;
    h1r17 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h216, align_corners5, false);
    int h118;
    h118 = h1r17;
    int h1p19;
    h1p19 = (h118 < height111 - 1) ? 1 : 0;
    accscalar_t1 h1lambda20;
    h1lambda20 = h1r17 - h118;
    accscalar_t1 h0lambda21;
    h0lambda21 = static_cast<accscalar_t1>(1) - h1lambda20;
    accscalar_t1 w1r22;
    w1r22 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w215, align_corners5, false);
    int w123;
    w123 = w1r22;
    int w1p24;
    w1p24 = (w123 < width112 - 1) ? 1 : 0;
    accscalar_t1 w1lambda25;
    w1lambda25 = w1r22 - w123;
    accscalar_t1 w0lambda26;
    w0lambda26 = static_cast<accscalar_t1>(1) - w1lambda25;
    for (int n = 0; n < batchsize9; n++) {
        for (int c = 0; c < channels10; ++c) {
            accscalar_t1 val30;
            val30 = h0lambda21 * (w0lambda26 * idata6[n][c][h118][w123] + w1lambda25 * idata6[n][c][h118][w123 + w1p24]) + h1lambda20 * (w0lambda26 * idata6[n][c][h118 + h1p19][w123] + w1lambda25 * idata6[n][c][h118 + h1p19][w123 + w1p24]);
            odata7[n][c][h216][w215] = static_cast<scalar_t0>(val30);
        }
    }
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
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
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
        cudaDeviceSynchronize();
        cudaProfilerStart();
        upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
            <<<num_blocks,
               num_threads,
               0,
               getStreamFromPool(true)>>>(
                num_kernels, rheight, rwidth, align_corners, idata, odata);
    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
        <<<grid,
          block,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
        cudaDeviceSynchronize();
      kernelHistogram1D_upsample_bilinear2d_out_frame_11<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp), scalar_t, accscalar_t>
        <<<grid,
          block.x + 512,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
                num_kernels, rheight, rwidth, align_corners, idata, odata
          );

      kernelHistogram1D_upsample_bilinear2d_out_frame_0<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp), scalar_t, accscalar_t>
        <<<grid,
          block.x + 512,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
                num_kernels, rheight, rwidth, align_corners, idata, odata
          );
      kernelHistogram1D_upsample_bilinear2d_out_frame_100<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED, decltype(getDummyOp), scalar_t, accscalar_t>
        <<<grid,
          512,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp,
                num_kernels, rheight, rwidth, align_corners, idata, odata
          );
        cudaDeviceSynchronize();
        cudaProfilerStop();
      });

  AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
  return std::make_tuple(output_hist, output_hist);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> _histc_upsample(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
  ) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
    input,
    output_size,
    align_corners
  );
  });
}

} // namespace native
} // namespace at
