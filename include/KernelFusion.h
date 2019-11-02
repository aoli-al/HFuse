//
// Created by Leo Li on 2019-10-29.
//

#ifndef SMART_FUSER_INCLUDE_KERNELFUSION_H
#define SMART_FUSER_INCLUDE_KERNELFUSION_H

#include <string>
#include <map>
#include <llvm/Support/YAMLTraits.h>



namespace kernel_fusion {

struct BlockDim {
  unsigned X;
  unsigned Y;
  unsigned Z;

  [[nodiscard]] unsigned size() const {
    return X * Y * Z;
  }
};

struct KernelInfo {
  std::string KernelName;
  bool HasBarriers;
  BlockDim BlockDim;
};


struct Context {
  std::pair<KernelInfo, KernelInfo> Kernels;

  [[nodiscard]] bool isFirstKernel(const std::string &KName) const {
    return Kernels.first.KernelName == KName;
  }

  [[nodiscard]] bool isSecondKernel(const std::string &KName) const {
    return Kernels.second.KernelName == KName;
  }

  [[nodiscard]] KernelInfo &getKernelWithName(const std::string &KName) {
    return isFirstKernel(KName) ? Kernels.first : Kernels.second;
  }

  [[nodiscard]] const KernelInfo &getKernelWithName(const std::string &KName) const {
    return isFirstKernel(KName) ? Kernels.first : Kernels.second;
  }
};

std::string branchingStatement(const Context &C, const std::string &FName);

std::string generateNewVarName(const std::string &Base);

const static std::string CurrentTid = "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";

}

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::KernelInfo> {
  static void mapping(IO &io, kernel_fusion::KernelInfo &Info) {
    io.mapRequired("KernelName", Info.KernelName);
    io.mapRequired("HasBarriers", Info.HasBarriers);
    io.mapRequired("BlockDim", Info.BlockDim);
  }
};

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::BlockDim> {
  static void mapping(IO &io, kernel_fusion::BlockDim &Dim) {
    io.mapRequired("X", Dim.X);
    io.mapRequired("Y", Dim.Y);
    io.mapRequired("Z", Dim.Z);
  }
};


LLVM_YAML_IS_SEQUENCE_VECTOR(kernel_fusion::KernelInfo)

#endif //SMART_FUSER_INCLUDE_KERNELFUSION_H
