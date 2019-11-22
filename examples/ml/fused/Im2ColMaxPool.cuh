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

namespace at {
namespace native {

using namespace at::cuda::detail;

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


#define CUDA_KERNEL_LOOP_C(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * 512 + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=512 * 10000, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP_M(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * 256 + threadIdx.x - 512;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=256 * gridDim.x, i=_i_n_d_e_x)
template <typename dt, typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void im2col_maxpool(
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
    const int nthreads, const scalar_t* bottom_data,
    const int num, const int channels, const int height_maxp,
    const int width_maxp, const int pooled_height_maxp, const int pooled_width_maxp,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask) {
  if (threadIdx.x < 512) {
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
    CUDA_KERNEL_LOOP_M(index, nthreads) {
      int pw = index % pooled_width_maxp;
      int ph = (index / pooled_width_maxp) % pooled_height_maxp;
      int c = (index / pooled_width_maxp / pooled_height_maxp) % channels;
      int n = index / pooled_width_maxp / pooled_height_maxp / channels;
      int hstart = ph * stride_h - pad_h;
      int wstart = pw * stride_w - pad_w;
      int hend = min(hstart + (kernel_h - 1) * dilation_h + 1, height_maxp);
      int wend = min(wstart + (kernel_w - 1) * dilation_w + 1, width_maxp);
      while(hstart < 0)
        hstart += dilation_h;
      while(wstart < 0)
        wstart += dilation_w;
      accscalar_t maxval = at::numeric_limits<accscalar_t>::lower_bound(); // -Infinity
      int maxidx = hstart * width_maxp + wstart;
      bottom_data += (n * channels + c) * height_maxp * width_maxp;
      for (int h = hstart; h < hend; h += dilation_h) {
        for (int w = wstart; w < wend; w += dilation_w) {
          scalar_t val = bottom_data[h * width_maxp + w];
          if ((ScalarConvert<scalar_t, accscalar_t>::to(val) > maxval) || THCNumerics<scalar_t>::isnan(val)) {
            maxidx = h * width_maxp + w;
            maxval = ScalarConvert<scalar_t, accscalar_t>::to(val);
          }
        }
      }
      top_data[index] = ScalarConvert<scalar_t, accscalar_t>::to(maxval);
      top_mask[index] = maxidx;
    }
  }
}

template <typename dt31, typename scalar_t0, typename accscalar_t1>
void im2col_kernel_MaxPoolForward_100(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((global))
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
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
        int64_t w_out47;
        w_out47 = index % width_col45;
        index /= width_col45;
        int64_t h_out48;
        h_out48 = index % height_col44;
        int64_t channel_in49;
        channel_in49 = index / height_col44;
        int64_t channel_out50;
        channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
        int64_t h_in51;
        h_in51 = h_out48 * stride_height40 - pad_height38;
        int64_t w_in52;
        w_in52 = w_out47 * stride_width41 - pad_width39;
        data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
        data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
        for (int64_t i = 0; i < kernel_height36; ++i) {
            for (int64_t j = 0; j < kernel_width37; ++j) {
                int64_t h53;
                h53 = h_in51 + i * dilation_height42;
                int64_t w54;
                w54 = w_in52 + j * dilation_width43;
                * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
                data_col46 += height_col44 * width_col45;
            }
        }
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


template <typename dt31, typename scalar_t0, typename accscalar_t1>
void im2col_kernel_MaxPoolForward_0(const int64_t n32, const dt31 *data_im33, const int64_t height34, const int64_t width35, const int64_t kernel_height36, const int64_t kernel_width37, const int64_t pad_height38, const int64_t pad_width39, const int64_t stride_height40, const int64_t stride_width41, const int64_t dilation_height42, const int64_t dilation_width43, const int64_t height_col44, const int64_t width_col45, dt31 *data_col46, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((global))
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
for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n32); index += blockDim_x_1 * gridDim.x) {
    int64_t w_out47;
    w_out47 = index % width_col45;
    index /= width_col45;
    int64_t h_out48;
    h_out48 = index % height_col44;
    int64_t channel_in49;
    channel_in49 = index / height_col44;
    int64_t channel_out50;
    channel_out50 = channel_in49 * kernel_height36 * kernel_width37;
    int64_t h_in51;
    h_in51 = h_out48 * stride_height40 - pad_height38;
    int64_t w_in52;
    w_in52 = w_out47 * stride_width41 - pad_width39;
    data_col46 += (channel_out50 * height_col44 + h_out48) * width_col45 + w_out47;
    data_im33 += (channel_in49 * height34 + h_in51) * width35 + w_in52;
    for (int64_t i = 0; i < kernel_height36; ++i) {
        for (int64_t j = 0; j < kernel_width37; ++j) {
            int64_t h53;
            h53 = h_in51 + i * dilation_height42;
            int64_t w54;
            w54 = w_in52 + j * dilation_width43;
            * data_col46 = (h53 >= 0 && w54 >= 0 && h53 < height34 && w54 < width35) ? data_im33[i * dilation_height42 * width35 + j * dilation_width43] : ScalarConvert<int, dt31>::to(0);
            data_col46 += height_col44 * width_col45;
        }
    }
}
label_0:;
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
}

template <typename dt, typename scalar_t0, typename accscalar_t1>
void im2col_kernel_MaxPoolForward(const int64_t n, const dt *data_im, const int64_t height, const int64_t width, const int64_t kernel_height, const int64_t kernel_width, const int64_t pad_height, const int64_t pad_width, const int64_t stride_height, const int64_t stride_width, const int64_t dilation_height, const int64_t dilation_width, const int64_t height_col, const int64_t width_col, dt *data_col, const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19) __attribute__((launch_bounds(0xde53490, 0xde534b0))) __attribute__((global)) {
  if ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)  {
      unsigned int blockDim_x_1;
      blockDim_x_1 = 512;
      unsigned int threadIdx_x_1;
      threadIdx_x_1 = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % 512;
      unsigned int blockDim_y_1;
      blockDim_y_1 = 1;
      unsigned int threadIdx_y_1;
      threadIdx_y_1 = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / 512 % 1;
      unsigned int blockDim_z_1;
      blockDim_z_1 = 1;
      unsigned int threadIdx_z_1;
      threadIdx_z_1 = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / 512;
      for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (n); index += blockDim_x_1 * gridDim.x) {
          int64_t w_out;
          w_out = index % width_col;
          index /= width_col;
          int64_t h_out;
          h_out = index % height_col;
          int64_t channel_in;
          channel_in = index / height_col;
          int64_t channel_out;
          channel_out = channel_in * kernel_height * kernel_width;
          int64_t h_in;
          h_in = h_out * stride_height - pad_height;
          int64_t w_in;
          w_in = w_out * stride_width - pad_width;
          data_col += (channel_out * height_col + h_out) * width_col + w_out;
          data_im += (channel_in * height + h_in) * width + w_in;
          for (int64_t i = 0; i < kernel_height; ++i) {
              for (int64_t j = 0; j < kernel_width; ++j) {
                  int64_t h;
                  h = h_in + i * dilation_height;
                  int64_t w;
                  w = w_in + j * dilation_width;
                  * data_col = (h >= 0 && w >= 0 && h < height && w < width) ? data_im[i * dilation_height * width + j * dilation_width] : ScalarConvert<int, dt>::to(0);
                  data_col += height_col * width_col;
              }
          }
      }
  }
  if ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256)  {
      unsigned int blockDim_x_0;
      blockDim_x_0 = 256;
      unsigned int threadIdx_x_0;
      threadIdx_x_0 = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) % 256;
      unsigned int blockDim_y_0;
      blockDim_y_0 = 1;
      unsigned int threadIdx_y_0;
      threadIdx_y_0 = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / 256 % 1;
      unsigned int blockDim_z_0;
      blockDim_z_0 = 1;
      unsigned int threadIdx_z_0;
      threadIdx_z_0 = (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) / 256;
      for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
          int pw;
          pw = index % pooled_width9;
          int ph;
          ph = (index / pooled_width9) % pooled_height8;
          int c;
          c = (index / pooled_width9 / pooled_height8) % channels5;
          int n;
          n = index / pooled_width9 / pooled_height8 / channels5;
          int hstart;
          hstart = ph * stride_h12 - pad_h14;
          int wstart;
          wstart = pw * stride_w13 - pad_w15;
          int hend;
          hend = min(hstart + (kernel_h10 - 1) * dilation_h16 + 1, height6);
          int wend;
          wend = min(wstart + (kernel_w11 - 1) * dilation_w17 + 1, width7);
          while (hstart < 0)
              hstart += dilation_h16;
          while (wstart < 0)
              wstart += dilation_w17;
          accscalar_t1 maxval;
          maxval = at::numeric_limits<accscalar_t1>::lower_bound();
          int maxidx;
          maxidx = hstart * width7 + wstart;
          bottom_data3 += (n * channels5 + c) * height6 * width7;
          for (int h = hstart; h < hend; h += dilation_h16) {
              for (int w = wstart; w < wend; w += dilation_w17) {
                  scalar_t0 val;
                  val = bottom_data3[h * width7 + w];
                  if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val) > maxval) || THCNumerics<scalar_t0>::isnan(val)) {
                      maxidx = h * width7 + w;
                      maxval = ScalarConvert<scalar_t0, accscalar_t1>::to(val);
                  }
              }
          }
          top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval);
          top_mask19[index] = maxidx;
      }
  }
}

} // namespace native
} // namespace at
