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
#include "Im2ColMaxPool.cuh"
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>

#include <cuda_profiler_api.h>
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

std::tuple<Tensor, Tensor> im2col_maxpool_stream(
    const Tensor& input_im2col_,
    IntArrayRef kernel_size_im2col,
    IntArrayRef dilation_im2col,
    IntArrayRef pad_im2colding_im2col,
    IntArrayRef stride_im2col,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output_im2col = at::empty_like(input_im2col_);
  TORCH_CHECK(
      kernel_size_im2col.size() == 2,
      "It is expected kernel_size_im2col equals to 2, but got size ",
      kernel_size_im2col.size());

  TORCH_CHECK(
      dilation_im2col.size() == 2,
      "It is expected dilation_im2col equals to 2, but got size ",
      dilation_im2col.size());

  TORCH_CHECK(
      pad_im2colding_im2col.size() == 2,
      "It is expected pad_im2colding_im2col equals to 2, but got size ",
      pad_im2colding_im2col.size());

  TORCH_CHECK(
      stride_im2col.size() == 2,
      "It is expected stride_im2col equals to 2, but got size ",
      stride_im2col.size());

  int64_t kernel_height = kernel_size_im2col[0];
  int64_t kernel_width = kernel_size_im2col[1];
  int64_t dilation_im2col_height = dilation_im2col[0];
  int64_t dilation_im2col_width = dilation_im2col[1];
  int64_t pad_im2col_height = pad_im2colding_im2col[0];
  int64_t pad_im2col_width = pad_im2colding_im2col[1];
  int64_t stride_im2col_height = stride_im2col[0];
  int64_t stride_im2col_width = stride_im2col[1];

  TensorArg input_im2col_arg{input_im2col_, "input_im2col", 1};
  TensorArg output_im2col_arg{output_im2col, "output_im2col", 2};
  checkAllSameGPU("im2col_cuda", {input_im2col_arg, output_im2col_arg});

  im2col_shape_check(
      input_im2col_,
      Tensor(),
      kernel_height,
      kernel_width,
      dilation_im2col_height,
      dilation_im2col_width,
      pad_im2col_height,
      pad_im2col_width,
      stride_im2col_height,
      stride_im2col_width);

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

  int64_t output_im2col_height = (input_im2col_height + 2 * pad_im2col_height -
                           (dilation_im2col_height * (kernel_height - 1) + 1)) /
          stride_im2col_height +
      1;
  int64_t output_im2col_width = (input_im2col_width + 2 * pad_im2col_width -
                          (dilation_im2col_width * (kernel_width - 1) + 1)) /
          stride_im2col_width +
      1;
  int64_t n_output_im2col_plane = n_input_im2col_plane * kernel_width * kernel_height;
  int64_t output_im2col_length = output_im2col_height * output_im2col_width;

  output_im2col.resize_({batch_size, n_output_im2col_plane, output_im2col_length});
  output_im2col.zero_();

  // Launch kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_im2col.scalar_type(), "im2col_out_cuda", [&] {
  });

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
      Tensor input_im2col_im2col_n;
      Tensor output_im2col_im2col_n;

      int64_t elt = 0;
      input_im2col_im2col_n = input_im2col.select(0, elt);
      output_im2col_im2col_n = output_im2col.select(0, elt);

      int64_t num_kernels_im2col = n_input_im2col_plane * output_im2col_height * output_im2col_width;
      // Launch CUDA_NUM_THREADS = 1024
      printf("num_kernels %ld, %ld, %ld\n", num_kernels_im2col, n_input_im2col_plane, output_im2col_height);
    cudaProfilerStart();
      auto s1 = at::cuda::getStreamFromPool(true);
      auto s2 = at::cuda::getStreamFromPool(true);
      im2col_kernel<scalar_t><<<GET_BLOCKS(num_kernels_im2col), 1024, 0, s1>>>(
        num_kernels_im2col,
        input_im2col_im2col_n.data<scalar_t>(),
        input_im2col_height,
        input_im2col_width,
        kernel_height,
        kernel_width,
        pad_im2col_height,
        pad_im2col_width,
        stride_im2col_height,
        stride_im2col_width,
        dilation_im2col_height,
        dilation_im2col_width,
        output_im2col_height,
        output_im2col_width,
        output_im2col_im2col_n.data<scalar_t>());
      MaxPoolForward<scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads), num_threads, 0, s2>>>(
          count, input_data,
          nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
          kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
      s1.synchronize();
      s2.synchronize();
    cudaProfilerStop();

      if (!batched_input_im2col) {
        output_im2col.resize_({n_output_im2col_plane, output_im2col_length});
      }
      AT_CUDA_CHECK(cudaGetLastError());
    });

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "max_pool2d_with_indices_out_cuda_frame failed with error code ",
     cudaGetLastError());

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
  return std::make_tuple(output_im2col, output);
}

std::tuple<Tensor, Tensor> im2col_maxpool_fused(
    const Tensor& input_im2col_,
    IntArrayRef kernel_size_im2col,
    IntArrayRef dilation_im2col,
    IntArrayRef pad_im2colding_im2col,
    IntArrayRef stride_im2col,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  Tensor output_im2col = at::empty_like(input_im2col_);
  TORCH_CHECK(
      kernel_size_im2col.size() == 2,
      "It is expected kernel_size_im2col equals to 2, but got size ",
      kernel_size_im2col.size());

  TORCH_CHECK(
      dilation_im2col.size() == 2,
      "It is expected dilation_im2col equals to 2, but got size ",
      dilation_im2col.size());

  TORCH_CHECK(
      pad_im2colding_im2col.size() == 2,
      "It is expected pad_im2colding_im2col equals to 2, but got size ",
      pad_im2colding_im2col.size());

  TORCH_CHECK(
      stride_im2col.size() == 2,
      "It is expected stride_im2col equals to 2, but got size ",
      stride_im2col.size());

  int64_t kernel_height = kernel_size_im2col[0];
  int64_t kernel_width = kernel_size_im2col[1];
  int64_t dilation_im2col_height = dilation_im2col[0];
  int64_t dilation_im2col_width = dilation_im2col[1];
  int64_t pad_im2col_height = pad_im2colding_im2col[0];
  int64_t pad_im2col_width = pad_im2colding_im2col[1];
  int64_t stride_im2col_height = stride_im2col[0];
  int64_t stride_im2col_width = stride_im2col[1];

  TensorArg input_im2col_arg{input_im2col_, "input_im2col", 1};
  TensorArg output_im2col_arg{output_im2col, "output_im2col", 2};
  checkAllSameGPU("im2col_cuda", {input_im2col_arg, output_im2col_arg});

  im2col_shape_check(
      input_im2col_,
      Tensor(),
      kernel_height,
      kernel_width,
      dilation_im2col_height,
      dilation_im2col_width,
      pad_im2col_height,
      pad_im2col_width,
      stride_im2col_height,
      stride_im2col_width);

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

  int64_t output_im2col_height = (input_im2col_height + 2 * pad_im2col_height -
                           (dilation_im2col_height * (kernel_height - 1) + 1)) /
          stride_im2col_height +
      1;
  int64_t output_im2col_width = (input_im2col_width + 2 * pad_im2col_width -
                          (dilation_im2col_width * (kernel_width - 1) + 1)) /
          stride_im2col_width +
      1;
  int64_t n_output_im2col_plane = n_input_im2col_plane * kernel_width * kernel_height;
  int64_t output_im2col_length = output_im2col_height * output_im2col_width;

  output_im2col.resize_({batch_size, n_output_im2col_plane, output_im2col_length});
  output_im2col.zero_();

  // Launch kernel
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_im2col.scalar_type(), "im2col_out_cuda", [&] {
  });

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
      Tensor input_im2col_im2col_n;
      Tensor output_im2col_im2col_n;

      int64_t elt = 0;
      input_im2col_im2col_n = input_im2col.select(0, elt);
      output_im2col_im2col_n = output_im2col.select(0, elt);

      int64_t num_kernels_im2col = n_input_im2col_plane * output_im2col_height * output_im2col_width;
      // Launch CUDA_NUM_THREADS = 1024
      printf("num_kernels %ld, %ld, %ld\n", num_kernels_im2col, n_input_im2col_plane, output_im2col_height);
        cudaProfilerStart();
      cudaDeviceSynchronize();
      im2col_kernel_MaxPoolForward_0<scalar_t, scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads), num_threads + 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_kernels_im2col,
        input_im2col_im2col_n.data<scalar_t>(),
        input_im2col_height,
        input_im2col_width,
        kernel_height,
        kernel_width,
        pad_im2col_height,
        pad_im2col_width,
        stride_im2col_height,
        stride_im2col_width,
        dilation_im2col_height,
        dilation_im2col_width,
        output_im2col_height,
        output_im2col_width,
        output_im2col_im2col_n.data<scalar_t>(),
        count, input_data,
        nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
      cudaDeviceSynchronize();
      im2col_kernel_MaxPoolForward_100<scalar_t, scalar_t, scalar_t>
        <<<cuda::ATenCeilDiv(count, num_threads), 512, 0, at::cuda::getCurrentCUDAStream()>>>(
        num_kernels_im2col,
        input_im2col_im2col_n.data<scalar_t>(),
        input_im2col_height,
        input_im2col_width,
        kernel_height,
        kernel_width,
        pad_im2col_height,
        pad_im2col_width,
        stride_im2col_height,
        stride_im2col_width,
        dilation_im2col_height,
        dilation_im2col_width,
        output_im2col_height,
        output_im2col_width,
        output_im2col_im2col_n.data<scalar_t>(),
        count, input_data,
        nbatch, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth,
        kH, kW, dH, dW, padH, padW, dilationH, dilationW, output_data, indices_data);
      cudaDeviceSynchronize();

        cudaProfilerStop();
      if (!batched_input_im2col) {
        output_im2col.resize_({n_output_im2col_plane, output_im2col_length});
      }
      AT_CUDA_CHECK(cudaGetLastError());
    });

  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
     "max_pool2d_with_indices_out_cuda_frame failed with error code ",
     cudaGetLastError());

  if(input.ndimension() == 3) {
    output.resize_({nInputPlane, outputHeight, outputWidth});
  }
  return std::make_tuple(output_im2col, output);
}


std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_maxpool(
  const Tensor& input_im2col_,
  IntArrayRef kernel_size_im2col,
  IntArrayRef dilation_im2col,
  IntArrayRef pad_im2colding_im2col,
  IntArrayRef stride_im2col,
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode) {
    im2col_maxpool_fused(
      input_im2col_,
      kernel_size_im2col,
      dilation_im2col,
      pad_im2colding_im2col,
      stride_im2col,
      input_,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode);
    return std::tuple_cat(
      im2col_maxpool_stream(
        input_im2col_,
        kernel_size_im2col,
        dilation_im2col,
        pad_im2colding_im2col,
        stride_im2col,
        input_,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode),
      std::make_tuple(input_im2col_,
        input_im2col_));
}

} // at::native
} // at
