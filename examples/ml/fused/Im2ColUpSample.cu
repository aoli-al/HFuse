#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
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

#include <cuda_profiler_api.h>
#include <c10/macros/Macros.h>
#include "Im2ColUpSample.cuh"

namespace at { namespace native {


std::tuple<Tensor, Tensor> im2col_upsample_stream(
      const Tensor& input_im2col_,
      IntArrayRef kernel_size,
      IntArrayRef dilation,
      IntArrayRef padding,
      IntArrayRef stride,
      const Tensor& input,
      IntArrayRef output_size,
      bool align_corners) {
    Tensor output_im2col = at::empty_like(input_im2col_);
    TORCH_CHECK(
        kernel_size.size() == 2,
        "It is expected kernel_size equals to 2, but got size ",
        kernel_size.size());

    TORCH_CHECK(
        dilation.size() == 2,
        "It is expected dilation equals to 2, but got size ",
        dilation.size());

    TORCH_CHECK(
        padding.size() == 2,
        "It is expected padding equals to 2, but got size ",
        padding.size());

    TORCH_CHECK(
        stride.size() == 2,
        "It is expected stride equals to 2, but got size ",
        stride.size());

    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];

    TensorArg input_im2col_arg{input_im2col_, "input_im2col", 1};
    TensorArg output_im2col_arg{output_im2col, "output_im2col", 2};
    checkAllSameGPU("im2col_cuda", {input_im2col_arg, output_im2col_arg});

    im2col_shape_check(
        input_im2col_,
        Tensor(),
        kernel_height,
        kernel_width,
        dilation_height,
        dilation_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width);

    Tensor input_im2col = input_im2col_.contiguous();

    bool batched_input_im2col = true;

    if (input_im2col.dim() == 3) {
      batched_input_im2col = false;
      input_im2col.resize_({1, input_im2col.size(0), input_im2col.size(1), input_im2col.size(2)});
    }

    int64_t batch_size = input_im2col.size(0);
    int64_t n_input_im2col_plane = input_im2col.size(1);
    int64_t input_im2col_height = input_im2col.size(2);
    int64_t input_im2col_width = input_im2col.size(3);

    int64_t output_im2col_height = (input_im2col_height + 2 * pad_height -
                            (dilation_height * (kernel_height - 1) + 1)) /
            stride_height +
        1;
    int64_t output_im2col_width = (input_im2col_width + 2 * pad_width -
                            (dilation_width * (kernel_width - 1) + 1)) /
            stride_width +
        1;
    int64_t n_output_im2col_plane = n_input_im2col_plane * kernel_width * kernel_height;
    int64_t output_im2col_length = output_im2col_height * output_im2col_width;

    output_im2col.resize_({batch_size, n_output_im2col_plane, output_im2col_length});
    output_im2col.zero_();

    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_im2col.scalar_type(), "im2col_out_cuda", [&] {
    });
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

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
          Tensor input_im2col_n;
          Tensor output_im2col_n;

          int64_t elt = 0;
          input_im2col_n = input_im2col.select(0, elt);
          output_im2col_n = output_im2col.select(0, elt);

          int64_t num_kernels_im2col = n_input_im2col_plane * output_im2col_height * output_im2col_width;
          // Launch CUDA_NUM_THREADS = 1024
          printf("num_kernels %ld, %ld, %ld\n", num_kernels_im2col, n_input_im2col_plane, output_im2col_height);
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
          upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
              <<<num_blocks,
                num_threads,
                0,
                stream>>>(
                  num_kernels, rheight, rwidth, align_corners, idata, odata);
          im2col_kernel<scalar_t><<<GET_BLOCKS(num_kernels_im2col), 1024, 0, at::cuda::getStreamFromPool(true)>>>(
              num_kernels_im2col,
              input_im2col_n.data<scalar_t>(),
              input_im2col_height,
              input_im2col_width,
              kernel_height,
              kernel_width,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              output_im2col_height,
              output_im2col_width,
              output_im2col_n.data<scalar_t>());
          if (!batched_input_im2col) {
            output_im2col.resize_({n_output_im2col_plane, output_im2col_length});
          }
        cudaProfilerStop();
          AT_CUDA_CHECK(cudaGetLastError());
        });

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, output_im2col);
  }

std::tuple<Tensor, Tensor> im2col_upsample_fused(
      const Tensor& input_im2col_,
      IntArrayRef kernel_size,
      IntArrayRef dilation,
      IntArrayRef padding,
      IntArrayRef stride,
      const Tensor& input,
      IntArrayRef output_size,
      bool align_corners) {
    Tensor output_im2col = at::empty_like(input_im2col_);
    TORCH_CHECK(
        kernel_size.size() == 2,
        "It is expected kernel_size equals to 2, but got size ",
        kernel_size.size());

    TORCH_CHECK(
        dilation.size() == 2,
        "It is expected dilation equals to 2, but got size ",
        dilation.size());

    TORCH_CHECK(
        padding.size() == 2,
        "It is expected padding equals to 2, but got size ",
        padding.size());

    TORCH_CHECK(
        stride.size() == 2,
        "It is expected stride equals to 2, but got size ",
        stride.size());

    int64_t kernel_height = kernel_size[0];
    int64_t kernel_width = kernel_size[1];
    int64_t dilation_height = dilation[0];
    int64_t dilation_width = dilation[1];
    int64_t pad_height = padding[0];
    int64_t pad_width = padding[1];
    int64_t stride_height = stride[0];
    int64_t stride_width = stride[1];

    TensorArg input_im2col_arg{input_im2col_, "input_im2col", 1};
    TensorArg output_im2col_arg{output_im2col, "output_im2col", 2};
    checkAllSameGPU("im2col_cuda", {input_im2col_arg, output_im2col_arg});

    im2col_shape_check(
        input_im2col_,
        Tensor(),
        kernel_height,
        kernel_width,
        dilation_height,
        dilation_width,
        pad_height,
        pad_width,
        stride_height,
        stride_width);

    Tensor input_im2col = input_im2col_.contiguous();

    bool batched_input_im2col = true;

    if (input_im2col.dim() == 3) {
      batched_input_im2col = false;
      input_im2col.resize_({1, input_im2col.size(0), input_im2col.size(1), input_im2col.size(2)});
    }

    int64_t batch_size = input_im2col.size(0);
    int64_t n_input_im2col_plane = input_im2col.size(1);
    int64_t input_im2col_height = input_im2col.size(2);
    int64_t input_im2col_width = input_im2col.size(3);

    int64_t output_im2col_height = (input_im2col_height + 2 * pad_height -
                            (dilation_height * (kernel_height - 1) + 1)) /
            stride_height +
        1;
    int64_t output_im2col_width = (input_im2col_width + 2 * pad_width -
                            (dilation_width * (kernel_width - 1) + 1)) /
            stride_width +
        1;
    int64_t n_output_im2col_plane = n_input_im2col_plane * kernel_width * kernel_height;
    int64_t output_im2col_length = output_im2col_height * output_im2col_width;

    output_im2col.resize_({batch_size, n_output_im2col_plane, output_im2col_length});
    output_im2col.zero_();

    // Launch kernel
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
          Tensor input_im2col_n;
          Tensor output_im2col_n;

          int64_t elt = 0;
          input_im2col_n = input_im2col.select(0, elt);
          output_im2col_n = output_im2col.select(0, elt);

          int64_t num_kernels_im2col = n_input_im2col_plane * output_im2col_height * output_im2col_width;
          // Launch CUDA_NUM_THREADS = 1024
          printf("num_kernels %ld, %ld, %ld\n", num_kernels_im2col, n_input_im2col_plane, output_im2col_height);
          if (!batched_input_im2col) {
            output_im2col.resize_({n_output_im2col_plane, output_im2col_length});
          }
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
      cudaDeviceSynchronize();
        im2col_kernel_upsample_bilinear2d_out_frame_0<scalar_t, scalar_t, accscalar_t>
            <<<num_blocks, dim3(512, 2), 0, at::cuda::getCurrentCUDAStream()>>>(
              num_kernels_im2col,
              input_im2col_n.data<scalar_t>(),
              input_im2col_height,
              input_im2col_width,
              kernel_height,
              kernel_width,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              output_im2col_height,
              output_im2col_width,
              output_im2col_n.data<scalar_t>(),
              num_kernels, rheight, rwidth, align_corners, idata, odata);
      cudaDeviceSynchronize();
        im2col_kernel_upsample_bilinear2d_out_frame_100<scalar_t, scalar_t, accscalar_t>
            <<<num_blocks, dim3(512, 2), 0, at::cuda::getCurrentCUDAStream()>>>(
              num_kernels_im2col,
              input_im2col_n.data<scalar_t>(),
              input_im2col_height,
              input_im2col_width,
              kernel_height,
              kernel_width,
              pad_height,
              pad_width,
              stride_height,
              stride_width,
              dilation_height,
              dilation_width,
              output_im2col_height,
              output_im2col_width,
              output_im2col_n.data<scalar_t>(),
              num_kernels, rheight, rwidth, align_corners, idata, odata);
      cudaDeviceSynchronize();
        cudaProfilerStop();
          AT_CUDA_CHECK(cudaGetLastError());
        });

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, output_im2col);
  }


  std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_upsample(
    const Tensor& input_im2col_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners) {
    im2col_upsample_stream(
      input_im2col_,
      kernel_size,
      dilation,
      padding,
      stride,
      input,
      output_size,
      align_corners);
    im2col_upsample_fused(
    input_im2col_,
    kernel_size,
    dilation,
    padding,
    stride,
    input,
    output_size,
    align_corners);
    return std::make_tuple(input, input, input, input);
  }
}
}