#include <ATen/ATen.h>
#include <ATen/native/Pool.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#include <jitify.hpp>

#include "TorchKernels/Utils.h"
#include "TorchKernels/LaunchParams.h"

using namespace at;
using namespace at::native;

namespace kernel_fusion {

#define CUDA_MAX_THREADS 1024
#define BLOCK_STRIDE 2


void MaxPool2dWithIndices(Tensor &Output, Tensor &Indices, const Tensor &Input,
                          IntArrayRef KernelSize, IntArrayRef Stride,
                          IntArrayRef Padding, IntArrayRef Dilation,
                          bool CeilMode) {
  TensorArg OutputArg{ Output, "output", 1 };
  TensorArg IndicesArg{ Indices, "indices", 2 };
  TensorArg InputArg{ Input, "input", 3 };

  checkAllSameGPU("MaxPool2dWithIndices", {OutputArg, IndicesArg, InputArg});

  TORCH_CHECK(KernelSize.size() == 1 || KernelSize.size() == 2,
              "MaxPool2d: KernelSize must either be a single int, or a tuple of"
              " two ints")
  auto *KernelHeight = new int(safe_downcast<int, int64_t>(KernelSize[0]));
  auto *KernelWidth = new int(KernelSize.size() == 1
                              ? *KernelHeight
                              : safe_downcast<int, int64_t>(KernelSize[1]));

  TORCH_CHECK(Stride.size() == 0 || Stride.size() == 1 || Stride.size() == 2,
              "MaxPool2d: stride must either be omitted, a single int, "
              "or a tuple of two ints")
  auto *StrideHeight = new int(Stride.empty()
                               ? *KernelHeight
                               : safe_downcast<int, int64_t>(Stride[0]));
  auto *StrideWidth = new int(Stride.empty()
                              ? *KernelWidth
                              : Stride.size() == 1
                                ? *StrideHeight
                                : safe_downcast<int, int64_t>(Stride[1]));

  TORCH_CHECK(Padding.size() == 1 || Padding.size() == 2,
              "MaxPool2d: padding must be either be a single int, or a tuple "
              "of two ints");
  auto *PadHeight = new int(safe_downcast<int, int64_t>(Padding[0]));
  auto *PadWidth = new int(Padding.size() == 1
                           ? *PadHeight
                           : safe_downcast<int, int64_t>(Padding[1]));

  TORCH_CHECK(Dilation.size() == 1 || Dilation.size() == 2,
              "MaxPool2d: dilation must be either a single int, or a tuple of "
              "two ints");
  auto *DilationHeight = new int(safe_downcast<int, int64_t>(Dilation[0]));
  auto *DilationWidth = new int(Dilation.size() == 1
                                ? *DilationHeight
                                : safe_downcast<int, int64_t>(Dilation[1]));


  const auto MemoryFormat = Input.suggest_memory_format();
  if (MemoryFormat == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(Input.ndimension() == 4,
                "non-empty 4D (batch mode) tensor expected for input with "
                "channels_last layout")
  } else {
    TORCH_CHECK((Input.ndimension() == 3 || Input.ndimension() == 4),
                "non-empty 3D or 4D (batch mode) tensor expected for input")
  }

  auto *NumOfBatch = new int64_t(
      Input.ndimension() == 4 ? Input.size(-4) : 1);
  auto *NumOfInputPlane = new int64_t(Input.size(-3));
  auto *InputHeight = new int64_t(Input.size(-2));
  auto *InputWidth = new int64_t(Input.size(-1));

  auto *OutputWidth = new int64_t(pooling_output_shape<int64_t>(
      *InputWidth, *KernelWidth, *PadWidth, *StrideWidth, *DilationWidth,
      CeilMode));
  auto *OutputHeight = new int64_t(pooling_output_shape<int64_t>(
      *InputHeight, *KernelHeight, *PadHeight, *StrideHeight, *DilationHeight,
      CeilMode));

  pool2d_shape_check(
      Input, *KernelHeight, *KernelWidth, *StrideHeight, *StrideWidth,
      *PadHeight, *PadWidth, *DilationHeight, *DilationWidth, *NumOfInputPlane,
      *InputHeight, *InputWidth, *OutputHeight, *OutputWidth);

  auto *InputContiguous =
      new Tensor(std::move(Input.contiguous(MemoryFormat)));
//  Tensor InputContiguous = Input.contiguous(MemoryFormat);

  auto *InStrideN = new int64_t(
      Input.ndimension() == 4 ? Input.stride(-4) : 0);
  auto *InStrideC = new int64_t(InputContiguous->stride(-3));
  auto *InStrideH = new int64_t(InputContiguous->stride(-2));
  auto *InStrideW = new int64_t(InputContiguous->stride(-1));

  Output.resize_({*NumOfBatch, *NumOfInputPlane, *OutputHeight, *OutputWidth});
  Indices.resize_({*NumOfBatch, *NumOfInputPlane, *OutputHeight, *OutputWidth});

  Output.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat);
  Indices.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, InputContiguous->scalar_type(),
      "MaxPool2dWithIndices", [&] {
    AT_SKIP_BFLOAT16_IF_NOT_ROCM(scalar_t, "MaxPool2dWithIndices", [&] {
      using accscalar_t = acc_type<scalar_t, true>;

      auto *OutputData = Output.data_ptr<scalar_t>();
      auto *InputData = InputContiguous->data_ptr<scalar_t>();
      auto *IndicesData = Indices.data_ptr<int64_t>();

      switch (MemoryFormat) {
      case MemoryFormat::ChannelsLast: {
        const int MaxThreads = std::min<int>(
            at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock,
            CUDA_MAX_THREADS);
        int* MaxThreadsDim =
            at::cuda::getCurrentDeviceProperties()->maxThreadsDim;
        int BlockX = std::min<int>(
            MaxThreadsDim[0], std::min<int>(lastPow2(*NumOfInputPlane),
                                            at::cuda::warp_size()));
        int BlockY = std::min<int>(
            MaxThreadsDim[1], std::min<int>(lastPow2(*OutputWidth),
                                            MaxThreads / BlockX));
        int BlockZ = std::min<int>(
            MaxThreadsDim[2], std::min<int>(lastPow2(*OutputHeight),
                                            MaxThreads / BlockX / BlockY));
        BlockX = std::min<int>(
            MaxThreadsDim[0], std::min<int>(lastPow2(*NumOfInputPlane),
                                            MaxThreads / BlockY / BlockZ));
        const dim3 Block(BlockX, BlockY, BlockZ);

        int KernelStrideC = at::cuda::ATenCeilDiv(
            safe_downcast<int, int64_t>(*NumOfInputPlane), BlockX * 4);
        int KernelSizeC = at::cuda::ATenCeilDiv(
            safe_downcast<int, int64_t>(*NumOfInputPlane),
                BlockX * KernelStrideC);

        int GridX = *NumOfBatch * KernelStrideC;
        int GridY = std::min<int>(
            at::cuda::getCurrentDeviceProperties()->maxGridSize[1],
            at::cuda::ATenCeilDiv(safe_downcast<int, int64_t>(*OutputWidth),
                BlockY * BLOCK_STRIDE));
        int GridZ = std::min<int>(
            at::cuda::getCurrentDeviceProperties()->maxGridSize[2],
            at::cuda::ATenCeilDiv(safe_downcast<int, int64_t>(*OutputHeight),
                BlockZ * BLOCK_STRIDE));
        const dim3 Grid(GridX, GridY, GridZ);

        size_t SharedMemSize =
            (KernelSizeC * BlockX * BlockY * BlockZ) *
                (sizeof(int) + sizeof(scalar_t));
        AT_ASSERT(SharedMemSize <=
            at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock);

        LaunchParams Params {
            Block,
            Grid,
            std::vector<string>({jitify::reflection::reflect<scalar_t>(),
                                 jitify::reflection::reflect<scalar_t>()}),
            std::vector<void *>({})
        };


        break;
      }
      case MemoryFormat::Contiguous: {
        break;
      }
      default: TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
    });

  });


}

}