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

#include <cuda_profiler_api.h>
namespace at {
namespace native {


using namespace at::cuda;
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

template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
void im2col_kernel_kernelHistogram1D_100(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39) __attribute__((launch_bounds(0x5c82668, 0x0))) __attribute__((global))
 {
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
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
        int64_t w_out16;
        w_out16 = index % width_col14;
        index /= width_col14;
        int64_t h_out17;
        h_out17 = index % height_col13;
        int64_t channel_in18;
        channel_in18 = index / height_col13;
        int64_t channel_out19;
        channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
        int64_t h_in20;
        h_in20 = h_out17 * stride_height9 - pad_height7;
        int64_t w_in21;
        w_in21 = w_out16 * stride_width10 - pad_width8;
        data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
        data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
        for (int64_t i = 0; i < kernel_height5; ++i) {
            for (int64_t j = 0; j < kernel_width6; ++j) {
                int64_t h22;
                h22 = h_in20 + i * dilation_height11;
                int64_t w23;
                w23 = w_in21 + j * dilation_width12;
                * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                data_col15 += height_col13 * width_col14;
            }
        }
    }
}
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
    extern unsigned char my_smem40[] __attribute__((shared));
    output_t24 *smem41;
    smem41 = nullptr;
    smem41 = reinterpret_cast<output_t24 *>(my_smem40);
    for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
        smem41[i] = 0;
    }
    __syncthreads();
    for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
        IndexType26 bOffset42;
        bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
        input_t25 bVal43;
        bVal43 = b34.data[bOffset42];
        if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
            IndexType26 bin44;
            bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
            atomicAdd(&smem41[bin44], getOp39(linearIndex));
        }
    }
    __syncthreads();
    for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
        IndexType26 aOffset45;
        aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
        atomicAdd(&a32.data[aOffset45], smem41[i]);
    }
}
}

template <typename dt00, typename output_t2430, typename input_t2531, typename IndexType2632, int ADims2733, int PDims2834, int BDims2935, at::native::CUDAHistogramMemoryType MemoryType3036 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op3137>
 __attribute__((launch_bounds(0xede6358, 0x0))) __attribute__((global)) void im2col_kernel_kernelHistogram1D_1x(const int64_t n11, const dt00 *data_im22, const int64_t height33, const int64_t width44, const int64_t kernel_height55, const int64_t kernel_width66, const int64_t pad_height77, const int64_t pad_width88, const int64_t stride_height99, const int64_t stride_width1010, const int64_t dilation_height1111, const int64_t dilation_width1212, const int64_t height_col1313, const int64_t width_col1414, dt00 *data_col1515, TensorInfo<output_t2430, IndexType2632> a3238, TensorInfo<output_t2430, IndexType2632> p3339, TensorInfo<input_t2531, IndexType2632> b3440, int nbins3541, input_t2531 minvalue3642, input_t2531 maxvalue3743, IndexType2632 totalElements3844, Op3137 getOp3945)
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_0;
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
unsigned int blockDim_x_016;
blockDim_x_016 = 512;
unsigned int threadIdx_x_017;
threadIdx_x_017 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) % 512;
unsigned int blockDim_y_018;
blockDim_y_018 = 1;
unsigned int threadIdx_y_019;
threadIdx_y_019 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512 % 1;
unsigned int blockDim_z_020;
blockDim_z_020 = 1;
unsigned int threadIdx_z_021;
threadIdx_z_021 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 0) / 512;
for (int index = blockIdx.x * blockDim_x_016 + threadIdx_x_017; index < (n11); index += blockDim_x_016 * gridDim.x) {
    int64_t w_out1622;
    w_out1622 = index % width_col1414;
    index /= width_col1414;
    int64_t h_out1723;
    h_out1723 = index % height_col1313;
    int64_t channel_in1824;
    channel_in1824 = index / height_col1313;
    int64_t channel_out1925;
    channel_out1925 = channel_in1824 * kernel_height55 * kernel_width66;
    int64_t h_in2026;
    h_in2026 = h_out1723 * stride_height99 - pad_height77;
    int64_t w_in2127;
    w_in2127 = w_out1622 * stride_width1010 - pad_width88;
    data_col1515 += (channel_out1925 * height_col1313 + h_out1723) * width_col1414 + w_out1622;
    data_im22 += (channel_in1824 * height33 + h_in2026) * width44 + w_in2127;
    for (int64_t i = 0; i < kernel_height55; ++i) {
        for (int64_t j = 0; j < kernel_width66; ++j) {
            int64_t h2228;
            h2228 = h_in2026 + i * dilation_height1111;
            int64_t w2329;
            w2329 = w_in2127 + j * dilation_width1212;
            * data_col1515 = (h2228 >= 0 && w2329 >= 0 && h2228 < height33 && w2329 < width44) ? data_im22[i * dilation_height1111 * width44 + j * dilation_width1212] : ScalarConvert<int, dt00>::to(0);
            data_col1515 += height_col1313 * width_col1414;
        }
    }
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_1;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
unsigned int blockDim_x_146;
blockDim_x_146 = 512;
unsigned int threadIdx_x_147;
threadIdx_x_147 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 512;
unsigned int blockDim_y_148;
blockDim_y_148 = 1;
unsigned int threadIdx_y_149;
threadIdx_y_149 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512 % 1;
unsigned int blockDim_z_150;
blockDim_z_150 = 1;
unsigned int threadIdx_z_151;
threadIdx_z_151 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 512;
extern unsigned char my_smem4052[] __attribute__((shared));
output_t2430 *smem4153;
smem4153 = nullptr;
smem4153 = reinterpret_cast<output_t2430 *>(my_smem4052);
for (IndexType2632 i = threadIdx_x_147; i < a3238.sizes[0]; i += blockDim_x_146) {
    smem4153[i] = 0;
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_2;
for (IndexType2632 linearIndex = blockIdx.x * blockDim_x_146 + threadIdx_x_147; linearIndex < totalElements3844; linearIndex += gridDim.x * blockDim_x_146) {
    IndexType2632 bOffset4254;
    bOffset4254 = IndexToOffset<input_t2531, IndexType2632, BDims2935>::get(linearIndex, b3440);
    input_t2531 bVal4355;
    bVal4355 = b3440.data[bOffset4254];
    if (bVal4355 >= minvalue3642 && bVal4355 <= maxvalue3743) {
        IndexType2632 bin4456;
        bin4456 = getBin<input_t2531, IndexType2632>(bVal4355, minvalue3642, maxvalue3743, nbins3541);
        atomicAdd(&smem4153[bin4456], getOp3945(linearIndex));
    }
}
label_2:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
for (IndexType2632 i = threadIdx_x_147; i < a3238.sizes[0]; i += blockDim_x_146) {
    IndexType2632 aOffset4557;
    aOffset4557 = IndexToOffset<output_t2430, IndexType2632, ADims2733>::get(i, a3238);
    atomicAdd(&a3238.data[aOffset4557], smem4153[i]);
}
label_3:;
}

template <typename dt0, typename output_t24, typename input_t25, typename IndexType26, int ADims27, int PDims28, int BDims29, at::native::CUDAHistogramMemoryType MemoryType30 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op31>
void im2col_kernel_kernelHistogram1D_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, TensorInfo<output_t24, IndexType26> a32, TensorInfo<output_t24, IndexType26> p33, TensorInfo<input_t25, IndexType26> b34, int nbins35, input_t25 minvalue36, input_t25 maxvalue37, IndexType26 totalElements38, Op31 getOp39) __attribute__((launch_bounds(0xc4fb4b0, 0xc4fb4d0))) __attribute__((global))
 {
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_1;
unsigned int blockDim_x_1;
blockDim_x_1 = 512;
unsigned int threadIdx_x_1;
threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
unsigned int blockDim_y_1;
blockDim_y_1 = 1;
unsigned int threadIdx_y_1;
threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
unsigned int blockDim_z_1;
blockDim_z_1 = 1;
unsigned int threadIdx_z_1;
threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
extern unsigned char my_smem40[] __attribute__((shared));
output_t24 *smem41;
smem41 = nullptr;
smem41 = reinterpret_cast<output_t24 *>(my_smem40);
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    smem41[i] = 0;
}
label_1:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_0;
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
for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (n1); index += blockDim_x_0 * gridDim.x) {
    int64_t w_out16;
    w_out16 = index % width_col14;
    index /= width_col14;
    int64_t h_out17;
    h_out17 = index % height_col13;
    int64_t channel_in18;
    channel_in18 = index / height_col13;
    int64_t channel_out19;
    channel_out19 = channel_in18 * kernel_height5 * kernel_width6;
    int64_t h_in20;
    h_in20 = h_out17 * stride_height9 - pad_height7;
    int64_t w_in21;
    w_in21 = w_out16 * stride_width10 - pad_width8;
    data_col15 += (channel_out19 * height_col13 + h_out17) * width_col14 + w_out16;
    data_im2 += (channel_in18 * height3 + h_in20) * width4 + w_in21;
    for (int64_t i = 0; i < kernel_height5; ++i) {
        for (int64_t j = 0; j < kernel_width6; ++j) {
            int64_t h22;
            h22 = h_in20 + i * dilation_height11;
            int64_t w23;
            w23 = w_in21 + j * dilation_width12;
            * data_col15 = (h22 >= 0 && w23 >= 0 && h22 < height3 && w23 < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
            data_col15 += height_col13 * width_col14;
        }
    }
}
label_0:;
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_2;
for (IndexType26 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements38; linearIndex += gridDim.x * blockDim_x_1) {
    IndexType26 bOffset42;
    bOffset42 = IndexToOffset<input_t25, IndexType26, BDims29>::get(linearIndex, b34);
    input_t25 bVal43;
    bVal43 = b34.data[bOffset42];
    if (bVal43 >= minvalue36 && bVal43 <= maxvalue37) {
        IndexType26 bin44;
        bin44 = getBin<input_t25, IndexType26>(bVal43, minvalue36, maxvalue37, nbins35);
        atomicAdd(&smem41[bin44], getOp39(linearIndex));
    }
}
label_2:;
__syncthreads();
if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_3;
for (IndexType26 i = threadIdx_x_1; i < a32.sizes[0]; i += blockDim_x_1) {
    IndexType26 aOffset45;
    aOffset45 = IndexToOffset<output_t24, IndexType26, ADims27>::get(i, a32);
    atomicAdd(&a32.data[aOffset45], smem41[i]);
}
label_3:;
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
std::tuple<Tensor, Tensor> _histc_cuda_template_fused(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    int64_t elt = 0;
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };

  cudaProfilerStart();
    im2col_kernel_kernelHistogram1D_1x
    <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
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
        aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);
    // im2col_kernel_kernelHistogram1D_100
    // <scalar_t, input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
    // <<<10000, 1024, sharedMem, at::cuda::getCurrentCUDAStream()>>>(
    //     num_kernels,
    //     input_n.data<scalar_t>(),
    //     input_height,
    //     input_width,
    //     kernel_height,
    //     kernel_width,
    //     pad_height,
    //     pad_width,
    //     stride_height,
    //     stride_width,
    //     dilation_height,
    //     dilation_width,
    //     output_height,
    //     output_width,
    //     output_n.data<scalar_t>(),
    //     aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);

  cudaProfilerStop();
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  cudaDeviceSynchronize();
  return std::make_tuple(output_hist, output);
}
}

template <typename input_hist_t>
std::tuple<Tensor, Tensor> _histc_cuda_template(
    const Tensor& self_hist,
    int64_t nbins,
    input_hist_t min,
    input_hist_t max,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride) {
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
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "im2col_out_cuda", [&] {
    Tensor input_n;
    Tensor output_n;

    int64_t elt = 0;
    input_n = input.select(0, elt);
    output_n = output.select(0, elt);
    int64_t num_kernels = n_input_plane * output_height * output_width;
    cudaProfilerStart();
    im2col_kernel<<<10000, 512, 0, at::cuda::getStreamFromPool(true)>>>(
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

    static const auto getDummyOp = [] __device__(IndexType) { return 1L; };
    kernelHistogram1D<input_hist_t, input_hist_t, IndexType, 1, 2, -1, CUDAHistogramMemoryType::SHARED>
        <<<grid,
          block,
          sharedMem,
          getStreamFromPool(true)>>>(
            aInfo, pInfo, bInfo, nbins, minvalue, maxvalue, totalElements, getDummyOp);        \
    cudaProfilerStop();
    cudaDeviceSynchronize();
    AT_ASSERTM(cudaGetLastError() == cudaSuccess, "kernelHistogram1D failed");
    if (!batched_input) {
      output.resize_({n_output_plane, output_length});
    }
  });
  return std::make_tuple(output_hist, output);
}
}
} // namespace

namespace native {

std::tuple<Tensor, Tensor> _histc_cuda2(
  const Tensor& input_im2col_,
  IntArrayRef kernel_size_im2col,
  IntArrayRef dilation_im2col,
  IntArrayRef pad_im2colding_im2col,
  IntArrayRef stride_im2col,
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max) {
  if (self.scalar_type() == ScalarType::Half) {
    AT_ERROR("HalfTensor is not supported");
  }
    printf("0\n");
  // AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
  //   printf("1\n");
  //   return native::_histc_cuda_template<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
  //   input_im2col_,
  //   kernel_size_im2col,
  //   dilation_im2col,
  //   pad_im2colding_im2col,
  //   stride_im2col
  // );
  // });
  return AT_DISPATCH_ALL_TYPES(self.scalar_type(), "histc", [&] {
    printf("1\n");
    return native::_histc_cuda_template_fused<scalar_t>(self, nbins, min.to<scalar_t>(), max.to<scalar_t>(),
    input_im2col_,
    kernel_size_im2col,
    dilation_im2col,
    pad_im2colding_im2col,
    stride_im2col
  );
  });
}

} // namespace native
} // namespace at
