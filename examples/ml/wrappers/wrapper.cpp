#include <torch/extension.h>

#include <cuda.h>
#include <vector>

using at::IntArrayRef;
using at::TensorList;
using torch::Tensor;

namespace at
{
namespace native
{

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
  );
std::tuple<Tensor, Tensor> _histc_upsample(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners
  );
std::tuple<Tensor, Tensor> _histc_cuda2(
  const Tensor& input_im2col_,
  IntArrayRef kernel_size_im2col,
  IntArrayRef dilation_im2col,
  IntArrayRef pad_im2colding_im2col,
  IntArrayRef stride_im2col,
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max);
std::tuple<Tensor, Tensor> hist_norm(
    const Tensor& self,
    int64_t nbins,
    Scalar min,
    Scalar max,
  Tensor& input_);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> im2col_maxpool_batch_norm_stream(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor& input_maxpool_,
    IntArrayRef kernel_size_maxpool,
    IntArrayRef stride_maxpool,
    IntArrayRef padding_maxpool,
    IntArrayRef dilation_maxpool,
    bool ceil_mode,
    const Tensor& input_batch_norm) ;
std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_upsample(
    const Tensor &input_im2col_,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor &input,
    IntArrayRef output_size,
    bool align_corners);
// std::tuple<Tensor, Tensor, Tensor> softmax_cuda(const Tensor &input, const int64_t dim, const bool half_to_float, const Tensor& self, double epsilon);
// Tensor im2col_cuda(
//     const Tensor &input,
//     IntArrayRef kernel_size,
//     IntArrayRef dilation,
//     IntArrayRef padding,
//     IntArrayRef stride);
std::tuple<Tensor, Tensor, Tensor> max_pool_upsample_stream(
    const Tensor &input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor &input_upsample,
    const IntArrayRef output_size,
    bool align_corners);
std::tuple<Tensor, Tensor> im2col_batchnorm_cuda(
    const Tensor &input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    const Tensor &input_batch_norm);
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> max_pool2d_batch_norm(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& input_batch_norm_,
    double epsilon);
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
  bool ceil_mode);
std::tuple<Tensor, Tensor, Tensor, Tensor> upsample_batchnorm(
  const Tensor& input,
  IntArrayRef output_size,
  bool align_corners,
  const Tensor& input_bn_, double epsilon);
} // namespace native
} // namespace at

const auto defaultOptions = torch::TensorOptions({at::kCUDA}).dtype(at::kFloat).requires_grad(true);

std::tuple<Tensor, Tensor, Tensor> call_max_pool_upsample_fused()
{
  auto input_max_pool = torch::randn({4, 4, 3210, 5010}, defaultOptions);
  auto input_upsample = torch::randn({20, 20, 256, 100}, defaultOptions);
  return at::native::max_pool_upsample_stream(input_max_pool, {20, 20}, {10, 10}, 0, 1, false,
                                              input_upsample, {2000, 2560}, true);
}
//
std::tuple<Tensor, Tensor> im2col_batchnorm()
{
  auto im2col_input = torch::randn({1, 1, 2700, 2048}, defaultOptions);
  // auto r = at::native::im2col_cuda(im2col_input, {251, 1}, {1, 1}, {0, 0}, {1, 1});
  // return std::make_tuple(r, r, r, r, r, r);
  auto batch_norm_input = torch::randn({10000, 10000}, defaultOptions);
  return at::native::im2col_batchnorm_cuda(im2col_input, {201, 1}, {1, 1}, {0, 0}, {1, 1},
                                           batch_norm_input);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> im2col_maxpool_batchnorm()
{
  auto im2col_input = torch::randn({1, 1, 2700, 2048}, defaultOptions);
  auto input_max_pool = torch::randn({4, 4, 3210, 5010}, defaultOptions);
  // auto r = at::native::im2col_cuda(im2col_input, {251, 1}, {1, 1}, {0, 0}, {1, 1});
  // return std::make_tuple(r, r, r, r, r, r);
  auto batch_norm_input = torch::randn({10000, 10000}, defaultOptions);
  return at::native::im2col_maxpool_batch_norm_stream(im2col_input, {201, 1}, {1, 1}, {0, 0}, {1, 1},
                                          input_max_pool, {20, 20}, {10, 10}, 0, 1, false,
                                           batch_norm_input);
}

Tensor histc(Tensor t, uint64_t y)
{
  auto im2col_input = torch::randn({1, 1, 2700, 2048}, defaultOptions);
  auto hist_input = torch::randn({100000000}, defaultOptions);
  // auto x = torch::randn({1000, 1000}, defaultOptions);
  at::native::_histc_cuda2(im2col_input,{201, 1}, {1, 1}, {0, 0}, {1, 1},
   hist_input, 20, c10::Scalar(0.f), c10::Scalar(0.f));
  return torch::randn({100, 100});
}


std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> max_pool_batch_norm() {
  auto batch_norm_input = torch::randn({10000, 10000}, defaultOptions);
  auto input_max_pool = torch::randn({4, 4, 3210, 5010}, defaultOptions);
  return at::native::max_pool2d_batch_norm(input_max_pool, {20, 20}, {10, 10}, 0, 1, false,
    batch_norm_input, 0.1);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_upsample() {
  auto im2col_input = torch::randn({1, 1, 2700, 2048}, defaultOptions);
  // auto input_upsample = torch::randn({17, 16, 256, 100}, defaultOptions);
  auto input_upsample = torch::randn({20, 15, 256, 100}, defaultOptions);
  return at::native::im2col_upsample(im2col_input, {201, 1}, {1, 1}, {0, 0}, {1, 1},
                                     input_upsample, {2000, 2560}, true);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> im2col_maxpool() {
  auto im2col_input = torch::randn({1, 1, 2750, 2048}, defaultOptions);
  auto input_max_pool = torch::randn({4, 4, 3210, 5010}, defaultOptions);
  return at::native::im2col_maxpool(
    im2col_input, {251, 1}, {1, 1}, {0, 0}, {1, 1},
    input_max_pool, {20, 20}, {10, 10}, 0, 1, false
  );
}
//
std::tuple<Tensor, Tensor, Tensor, Tensor> upsample_batchnorm() {
  auto input_upsample = torch::randn({20, 20, 256, 100}, defaultOptions);
  auto batch_norm_input = torch::randn({10000, 10000}, defaultOptions);
  return at::native::upsample_batchnorm(input_upsample, {2000, 2560}, true,
                                        batch_norm_input, 0.2);
}
Tensor hist_norm() {
  auto batch_norm_input = torch::randn({10000, 10000}, defaultOptions);
  auto hist_input = torch::randn({100000000}, defaultOptions);
  at::native::hist_norm(hist_input, 20, 0.f, 0.f, batch_norm_input);
 return torch::randn({100, 100});
}

Tensor histc_maxpool()
{
  auto hist_input = torch::randn({100000000}, defaultOptions);
  auto input_max_pool = torch::randn({4, 4, 3210, 5010}, defaultOptions);
  at::native::_histc_maxpool(hist_input, 20, 0.f, 0.f,
    input_max_pool, {20, 20}, {10, 10}, 0, 1, false
  );
 return torch::randn({100, 100});
}

Tensor histc_upsample()
{
  auto hist_input = torch::randn({100000000}, defaultOptions);
  auto input_upsample = torch::randn({20, 20, 256, 100}, defaultOptions);
  at::native::_histc_upsample(hist_input, 20, 0.f, 0.f,
                              input_upsample, {2000, 2560}, true);
 return torch::randn({100, 100});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("call_max_pool_upsample_fused", &call_max_pool_upsample_fused, "LLTM forward (CUDA)");
  m.def("im2col_batchnorm", &im2col_batchnorm, "LLTM forward (CUDA)");
  m.def("max_pool_batch_norm", &max_pool_batch_norm, "LLTM forward (CUDA)");
  m.def("im2col_upsample", &im2col_upsample, "LLTM forward (CUDA)");
  m.def("im2col_maxpool", &im2col_maxpool, "LLTM forward (CUDA)");
  m.def("upsample_batchnorm", &upsample_batchnorm, "LLTM forward (CUDA)");
  m.def("im2col_maxpool_batchnorm", &im2col_maxpool_batchnorm, "LLTM forward (CUDA)");
  m.def("histc", &histc, "LLTM forward (CUDA)");
  m.def("hist_norm", &hist_norm, "LLTM forward (CUDA)");
  m.def("histc_maxpool", &histc_maxpool, "LLTM forward (CUDA)");
  m.def("histc_upsample", &histc_upsample, "LLTM forward (CUDA)");
}
