#pragma once

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

#include <c10/macros/Macros.h>
#include "../cuda/UpSample.cuh"
#include "../cuda/KernelUtils.cuh"
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

#include <c10/macros/Macros.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/div_rtn.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>


#include <ATen/native/im2col_shape_check.h>
#include <ATen/AccumulateType.h>

namespace at {
namespace native {

using namespace at::cuda::detail;

#define CUDA_KERNEL_LOOP_C(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * 512 + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=512 * 10000, i=_i_n_d_e_x)
// Kernel for fast unfold+copy
// (borrowed from Caffe:
// https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
// CUDA_NUM_THREADS = 1024

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
    const int ns,
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

  if (index < ns) {
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

template <typename dt0, typename scalar_t24, typename accscalar_t25>
void im2col_kernel_upsample_bilinear2d_out_frame_100(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns26, const accscalar_t25 rheight27, const accscalar_t25 rwidth28, const bool align_corners29, const PackedTensorAccessor<scalar_t24, 4> idata30, PackedTensorAccessor<scalar_t24, 4> odata31) __attribute__((launch_bounds(0xfcc3640, 0xfcc3660))) __attribute__((global))
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
    int index32;
    index32 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
    int batchsize33;
    batchsize33 = idata30.size(0);
    int channels34;
    channels34 = idata30.size(1);
    int height135;
    height135 = idata30.size(2);
    int width136;
    width136 = idata30.size(3);
    int height237;
    height237 = odata31.size(2);
    int width238;
    width238 = odata31.size(3);
    if (index32 < ns26) {
        int w239;
        w239 = index32 % width238;
        int h240;
        h240 = index32 / width238;
        if (height135 == height237 && width136 == width238) {
            int h151;
            h151 = h240;
            int w152;
            w152 = w239;
            for (int n = 0; n < batchsize33; n++) {
                for (int c = 0; c < channels34; ++c) {
                    scalar_t24 val53;
                    val53 = idata30[n][c][h151][w152];
                    odata31[n][c][h240][w239] = val53;
                }
            }
            return;
        }
        accscalar_t25 h1r41;
        h1r41 = area_pixel_compute_source_index<accscalar_t25>(rheight27, h240, align_corners29, false);
        int h142;
        h142 = h1r41;
        int h1p43;
        h1p43 = (h142 < height135 - 1) ? 1 : 0;
        accscalar_t25 h1lambda44;
        h1lambda44 = h1r41 - h142;
        accscalar_t25 h0lambda45;
        h0lambda45 = static_cast<accscalar_t25>(1) - h1lambda44;
        accscalar_t25 w1r46;
        w1r46 = area_pixel_compute_source_index<accscalar_t25>(rwidth28, w239, align_corners29, false);
        int w147;
        w147 = w1r46;
        int w1p48;
        w1p48 = (w147 < width136 - 1) ? 1 : 0;
        accscalar_t25 w1lambda49;
        w1lambda49 = w1r46 - w147;
        accscalar_t25 w0lambda50;
        w0lambda50 = static_cast<accscalar_t25>(1) - w1lambda49;
        for (int n = 0; n < batchsize33; n++) {
            for (int c = 0; c < channels34; ++c) {
                accscalar_t25 val54;
                val54 = h0lambda45 * (w0lambda50 * idata30[n][c][h142][w147] + w1lambda49 * idata30[n][c][h142][w147 + w1p48]) + h1lambda44 * (w0lambda50 * idata30[n][c][h142 + h1p43][w147] + w1lambda49 * idata30[n][c][h142 + h1p43][w147 + w1p48]);
                odata31[n][c][h240][w239] = static_cast<scalar_t24>(val54);
            }
        }
    }
}
}

template <typename dt0, typename scalar_t24, typename accscalar_t25>
void im2col_kernel_upsample_bilinear2d_out_frame_0(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns26, const accscalar_t25 rheight27, const accscalar_t25 rwidth28, const bool align_corners29, const PackedTensorAccessor<scalar_t24, 4> idata30, PackedTensorAccessor<scalar_t24, 4> odata31) __attribute__((launch_bounds(0xbf4db48, 0x0))) __attribute__((global))
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
int index32;
index32 = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
int batchsize33;
batchsize33 = idata30.size(0);
int channels34;
channels34 = idata30.size(1);
int height135;
height135 = idata30.size(2);
int width136;
width136 = idata30.size(3);
int height237;
height237 = odata31.size(2);
int width238;
width238 = odata31.size(3);
if (index32 < ns26) {
    int w239;
    w239 = index32 % width238;
    int h240;
    h240 = index32 / width238;
    if (height135 == height237 && width136 == width238) {
        int h151;
        h151 = h240;
        int w152;
        w152 = w239;
        for (int n = 0; n < batchsize33; n++) {
            for (int c = 0; c < channels34; ++c) {
                scalar_t24 val53;
                val53 = idata30[n][c][h151][w152];
                odata31[n][c][h240][w239] = val53;
            }
        }
        return;
    }
    accscalar_t25 h1r41;
    h1r41 = area_pixel_compute_source_index<accscalar_t25>(rheight27, h240, align_corners29, false);
    int h142;
    h142 = h1r41;
    int h1p43;
    h1p43 = (h142 < height135 - 1) ? 1 : 0;
    accscalar_t25 h1lambda44;
    h1lambda44 = h1r41 - h142;
    accscalar_t25 h0lambda45;
    h0lambda45 = static_cast<accscalar_t25>(1) - h1lambda44;
    accscalar_t25 w1r46;
    w1r46 = area_pixel_compute_source_index<accscalar_t25>(rwidth28, w239, align_corners29, false);
    int w147;
    w147 = w1r46;
    int w1p48;
    w1p48 = (w147 < width136 - 1) ? 1 : 0;
    accscalar_t25 w1lambda49;
    w1lambda49 = w1r46 - w147;
    accscalar_t25 w0lambda50;
    w0lambda50 = static_cast<accscalar_t25>(1) - w1lambda49;
    for (int n = 0; n < batchsize33; n++) {
        for (int c = 0; c < channels34; ++c) {
            accscalar_t25 val54;
            val54 = h0lambda45 * (w0lambda50 * idata30[n][c][h142][w147] + w1lambda49 * idata30[n][c][h142][w147 + w1p48]) + h1lambda44 * (w0lambda50 * idata30[n][c][h142 + h1p43][w147] + w1lambda49 * idata30[n][c][h142 + h1p43][w147 + w1p48]);
            odata31[n][c][h240][w239] = static_cast<scalar_t24>(val54);
        }
    }
}
label_1:;
}


template <typename dt0, typename scalar_t16, typename accscalar_t17>
void im2col_kernel_upsample_bilinear2d_out_frame_(const int64_t n1, const dt0 *data_im2, const int64_t height3, const int64_t width4, const int64_t kernel_height5, const int64_t kernel_width6, const int64_t pad_height7, const int64_t pad_width8, const int64_t stride_height9, const int64_t stride_width10, const int64_t dilation_height11, const int64_t dilation_width12, const int64_t height_col13, const int64_t width_col14, dt0 *data_col15, const int ns18, const accscalar_t17 rheight19, const accscalar_t17 rwidth20, const bool align_corners21, const PackedTensorAccessor<scalar_t16, 4> idata22, PackedTensorAccessor<scalar_t16, 4> odata23) __attribute__((launch_bounds(0xc5fdab0, 0xc5fdad0))) __attribute__((global)) {
  if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512))  {
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
          int64_t w_out;
          w_out = index % width_col14;
          index /= width_col14;
          int64_t h_out;
          h_out = index % height_col13;
          int64_t channel_in;
          channel_in = index / height_col13;
          int64_t channel_out;
          channel_out = channel_in * kernel_height5 * kernel_width6;
          int64_t h_in;
          h_in = h_out * stride_height9 - pad_height7;
          int64_t w_in;
          w_in = w_out * stride_width10 - pad_width8;
          data_col15 += (channel_out * height_col13 + h_out) * width_col14 + w_out;
          data_im2 += (channel_in * height3 + h_in) * width4 + w_in;
          for (int64_t i = 0; i < kernel_height5; ++i) {
              for (int64_t j = 0; j < kernel_width6; ++j) {
                  int64_t h;
                  h = h_in + i * dilation_height11;
                  int64_t w;
                  w = w_in + j * dilation_width12;
                  * data_col15 = (h >= 0 && w >= 0 && h < height3 && w < width4) ? data_im2[i * dilation_height11 * width4 + j * dilation_width12] : ScalarConvert<int, dt0>::to(0);
                  data_col15 += height_col13 * width_col14;
              }
          }
      }
  }
  if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024))  {
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
      int index;
      index = threadIdx_x_1 + blockIdx.x * blockDim_x_1;
      int batchsize;
      batchsize = idata22.size(0);
      int channels;
      channels = idata22.size(1);
      int height1;
      height1 = idata22.size(2);
      int width1;
      width1 = idata22.size(3);
      int height2;
      height2 = odata23.size(2);
      int width2;
      width2 = odata23.size(3);
      if (index < ns18) {
          int w2;
          w2 = index % width2;
          int h2;
          h2 = index / width2;
          if (height1 == height2 && width1 == width2) {
              int h1;
              h1 = h2;
              int w1;
              w1 = w2;
              for (int n = 0; n < batchsize; n++) {
                  for (int c = 0; c < channels; ++c) {
                      scalar_t16 val;
                      val = idata22[n][c][h1][w1];
                      odata23[n][c][h2][w2] = val;
                  }
              }
              return;
          }
          accscalar_t17 h1r;
          h1r = area_pixel_compute_source_index<accscalar_t17>(rheight19, h2, align_corners21, false);
          int h1;
          h1 = h1r;
          int h1p;
          h1p = (h1 < height1 - 1) ? 1 : 0;
          accscalar_t17 h1lambda;
          h1lambda = h1r - h1;
          accscalar_t17 h0lambda;
          h0lambda = static_cast<accscalar_t17>(1) - h1lambda;
          accscalar_t17 w1r;
          w1r = area_pixel_compute_source_index<accscalar_t17>(rwidth20, w2, align_corners21, false);
          int w1;
          w1 = w1r;
          int w1p;
          w1p = (w1 < width1 - 1) ? 1 : 0;
          accscalar_t17 w1lambda;
          w1lambda = w1r - w1;
          accscalar_t17 w0lambda;
          w0lambda = static_cast<accscalar_t17>(1) - w1lambda;
          for (int n = 0; n < batchsize; n++) {
              for (int c = 0; c < channels; ++c) {
                  accscalar_t17 val;
                  val = h0lambda * (w0lambda * idata22[n][c][h1][w1] + w1lambda * idata22[n][c][h1][w1 + w1p]) + h1lambda * (w0lambda * idata22[n][c][h1 + h1p][w1] + w1lambda * idata22[n][c][h1 + h1p][w1 + w1p]);
                  odata23[n][c][h2][w2] = static_cast<scalar_t16>(val);
              }
          }
      }
  }
}



template <typename dt, typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_upsample(
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
    dt* data_col,
    const int ns,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata) {
  if (threadIdx.y == 0) {
    CUDA_KERNEL_LOOP_C(index, n) {
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
  } else {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    const int batchsize = idata.size(0);
    const int channels = idata.size(1);
    const int height1 = idata.size(2);
    const int width1 = idata.size(3);
    const int height2 = odata.size(2);
    const int width2 = odata.size(3);

    if (index < ns) {
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
}

} // namespace native
} // namespace at
