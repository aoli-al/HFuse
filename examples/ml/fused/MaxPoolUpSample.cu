#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/native/Pool.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <THC/THCNumerics.cuh>
#include <c10/macros/Macros.h>

#include <cuda_profiler_api.h>
#include "../cuda/UpSample.cuh"
#include "../cuda/KernelUtils.cuh"


namespace at {
namespace native {
namespace {

#define CUDA_KERNEL_LOOP_C(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * 256 + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=256 * gridDim.x, i=_i_n_d_e_x)
static const int BACKWARD_THREADS = 256;

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

template <typename scalar_t31, typename accscalar_t32, typename scalar_t0, typename accscalar_t1>
void MaxPoolForward_upsample_bilinear2d_out_frame_100(const int nthreads33, const scalar_t31 *bottom_data34, const int num35, const int channels36, const int height37, const int width38, const int pooled_height39, const int pooled_width40, const int kernel_h41, const int kernel_w42, const int stride_h43, const int stride_w44, const int pad_h45, const int pad_w46, const int dilation_h47, const int dilation_w48, scalar_t31 *top_data49, int64_t *top_mask50, const int n2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7) __attribute__((global))
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1;
    blockDim_x_1 = 256;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads33); index += blockDim_x_1 * gridDim.x) {
        int pw51;
        pw51 = index % pooled_width40;
        int ph52;
        ph52 = (index / pooled_width40) % pooled_height39;
        int c53;
        c53 = (index / pooled_width40 / pooled_height39) % channels36;
        int n54;
        n54 = index / pooled_width40 / pooled_height39 / channels36;
        int hstart55;
        hstart55 = ph52 * stride_h43 - pad_h45;
        int wstart56;
        wstart56 = pw51 * stride_w44 - pad_w46;
        int hend57;
        hend57 = min(hstart55 + (kernel_h41 - 1) * dilation_h47 + 1, height37);
        int wend58;
        wend58 = min(wstart56 + (kernel_w42 - 1) * dilation_w48 + 1, width38);
        while (hstart55 < 0)
            hstart55 += dilation_h47;
        while (wstart56 < 0)
            wstart56 += dilation_w48;
        accscalar_t32 maxval59;
        maxval59 = at::numeric_limits<accscalar_t32>::lower_bound();
        int maxidx60;
        maxidx60 = hstart55 * width38 + wstart56;
        bottom_data34 += (n54 * channels36 + c53) * height37 * width38;
        for (int h = hstart55; h < hend57; h += dilation_h47) {
            for (int w = wstart56; w < wend58; w += dilation_w48) {
                scalar_t31 val61;
                val61 = bottom_data34[h * width38 + w];
                if ((ScalarConvert<scalar_t31, accscalar_t32>::to(val61) > maxval59) || THCNumerics<scalar_t31>::isnan(val61)) {
                    maxidx60 = h * width38 + w;
                    maxval59 = ScalarConvert<scalar_t31, accscalar_t32>::to(val61);
                }
            }
        }
        top_data49[index] = ScalarConvert<scalar_t31, accscalar_t32>::to(maxval59);
        top_mask50[index] = maxidx60;
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

template <typename scalar_t, typename accscalar_t>
__global__ void max_pool_upsample_kernel(const int nthreads, const scalar_t* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, scalar_t* top_data,
    int64_t* top_mask,
    const int n,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const bool align_corners,
    const PackedTensorAccessor<scalar_t, 4> idata,
    PackedTensorAccessor<scalar_t, 4> odata)
{
  if (threadIdx.x < 256) {
    CUDA_KERNEL_LOOP_C(index, nthreads) {
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
  } else {
    int index = threadIdx.x + blockIdx.x * blockDim.x - 256;

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
}

void max_pool2d_upsample_fused(
           Tensor& output,
           Tensor& indices,
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode,
           Tensor& output_upsample,
           const Tensor& input_upsample,
           IntArrayRef output_upsample_size,
           bool align_corners)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const int64_t nbatch_max_pool = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  printf("%ld, %ld\n", outputWidth, outputHeight);

  Tensor input = input_.contiguous();

  output.resize_({nbatch_max_pool, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch_max_pool, nInputPlane, outputHeight, outputWidth});

  const int count = safe_downcast<int, int64_t>(output.numel());
  const int num_threads_max_pool = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }

  TensorArg input_upsample_arg{input_upsample, "input_upsample", 1}, output_upsample_arg{output_upsample, "output_upsample", 2};

  int output_upsample_height = output_upsample_size[0];
  int output_upsample_width = output_upsample_size[1];

  int nbatch = input_upsample.size(0);
  int channels = input_upsample.size(1);
  int input_upsample_height = input_upsample.size(2);
  int input_upsample_width = input_upsample.size(3);

  output_upsample.resize_({input_upsample.size(0), input_upsample.size(1), output_upsample_height, output_upsample_width});

  const int num_kernels = output_upsample_height * output_upsample_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 512);

  printf("%d %d\n", num_kernels, num_threads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_data = output.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      int64_t *indices_data = indices.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads_max_pool);
      using accscalar_t = at::acc_type<scalar_t, true>;

      auto idata = input_upsample.packed_accessor<scalar_t, 4>();
      auto odata = output_upsample.packed_accessor<scalar_t, 4>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_upsample_height, output_upsample_height, align_corners);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_upsample_width, output_upsample_width, align_corners);

      const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
      printf("%d\n", num_blocks);
      printf("%d\n", num_threads_max_pool);
      printf("%d\n", num_threads_max_pool + num_threads);
      printf("%d\n", at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock);
      // dim3 threads(1024, 2);
      cudaProfilerStart();
      cudaDeviceSynchronize();
      MaxPoolForward_upsample_bilinear2d_out_frame_100<scalar_t,scalar_t,scalar_t,accscalar_t>
      <<<num_blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>
      (
        count, input_data,
        nbatch_max_pool, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data,
        num_kernels, rheight, rwidth, align_corners, idata, odata);

      cudaDeviceSynchronize();
      max_pool_upsample_kernel<<<num_blocks, num_threads_max_pool + num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        count, input_data,
        nbatch_max_pool, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data,
        num_kernels, rheight, rwidth, align_corners, idata, odata);
      cudaDeviceSynchronize();
      cudaProfilerStop();
    });
  AT_CUDA_CHECK(cudaGetLastError());
}

void max_pool2d_upsample_stream(
           Tensor& output,
           Tensor& indices,
           const Tensor& input_,
           IntArrayRef kernel_size,
           IntArrayRef stride,
           IntArrayRef padding,
           IntArrayRef dilation,
           bool ceil_mode,
           Tensor& output_upsample,
           const Tensor& input_upsample,
           IntArrayRef output_upsample_size,
           bool align_corners)
{
  TensorArg output_arg{ output, "output", 1 };
  TensorArg indices_arg{ indices, "indices", 2 };
  TensorArg input_arg{ input_, "input_", 3 };

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1 ? kH : safe_downcast<int, int64_t>(kernel_size[1]);

  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dH : safe_downcast<int, int64_t>(stride[1]);

  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1 ? dilationH : safe_downcast<int, int64_t>(dilation[1]);

  const int64_t nbatch_max_pool = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, dilationW, ceil_mode);
  const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, dilationH, ceil_mode);
  printf("%ld, %ld\n", outputWidth, outputHeight);

  Tensor input = input_.contiguous();

  output.resize_({nbatch_max_pool, nInputPlane, outputHeight, outputWidth});
  indices.resize_({nbatch_max_pool, nInputPlane, outputHeight, outputWidth});

  const int count = safe_downcast<int, int64_t>(output.numel());
  const int num_threads_max_pool = std::min(at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
                                   BACKWARD_THREADS);

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }

  TensorArg input_upsample_arg{input_upsample, "input_upsample", 1}, output_upsample_arg{output_upsample, "output_upsample", 2};

  int output_upsample_height = output_upsample_size[0];
  int output_upsample_width = output_upsample_size[1];

  int nbatch = input_upsample.size(0);
  int channels = input_upsample.size(1);
  int input_upsample_height = input_upsample.size(2);
  int input_upsample_width = input_upsample.size(3);

  output_upsample.resize_({input_upsample.size(0), input_upsample.size(1), output_upsample_height, output_upsample_width});

  const int num_kernels = output_upsample_height * output_upsample_width;
  const int num_threads = std::min(
      at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

  printf("%d %d\n", num_kernels, num_threads);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
    "max_pool2d_with_indices_out_cuda_frame",
    [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      scalar_t *output_data = output.data<scalar_t>();
      scalar_t *input_data = input.data<scalar_t>();
      int64_t *indices_data = indices.data<int64_t>();

      const int blocks = cuda::ATenCeilDiv(count, num_threads_max_pool);
      using accscalar_t = at::acc_type<scalar_t, true>;

      auto idata = input_upsample.packed_accessor<scalar_t, 4>();
      auto odata = output_upsample.packed_accessor<scalar_t, 4>();

      const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
          input_upsample_height, output_upsample_height, align_corners);
      const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
          input_upsample_width, output_upsample_width, align_corners);

      const int num_blocks = cuda::ATenCeilDiv(num_kernels, num_threads);
      printf("%d\n", num_blocks);
      printf("%d\n", num_threads_max_pool);
      printf("%d\n", num_threads_max_pool + num_threads);
      printf("%d\n", at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock);
      // dim3 threads(1024, 2);
      cudaProfilerStart();
      upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
          <<<num_blocks,
             num_threads,
             0,
             at::cuda::getStreamFromPool(true)>>>(
             num_kernels, rheight, rwidth, align_corners, idata, odata);
      MaxPoolForward<scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads_max_pool), num_threads_max_pool, 0, at::cuda::getStreamFromPool(true)>>>(
          count, input_data,
          nbatch_max_pool, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
      cudaProfilerStop();
      cudaDeviceSynchronize();
    });
  AT_CUDA_CHECK(cudaGetLastError());
}
} // namespace

std::tuple<Tensor, Tensor, Tensor> max_pool_upsample_stream(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor &input_upsample,
  const IntArrayRef output_size,
  bool align_corners)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  Tensor output_upsample = at::empty_like(input_upsample);
  max_pool2d_upsample_stream(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    output_upsample, input_upsample, output_size, align_corners);
  max_pool2d_upsample_fused(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
    output_upsample, input_upsample, output_size, align_corners);
  return std::tuple<Tensor, Tensor, Tensor>(output, indices, output_upsample);
}

} // at::native
} // at
