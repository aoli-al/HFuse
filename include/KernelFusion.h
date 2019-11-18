//
// Created by Leo Li on 2019-10-29.
//

#ifndef SMART_FUSER_INCLUDE_KERNELFUSION_H
#define SMART_FUSER_INCLUDE_KERNELFUSION_H

#include <string>
#include <map>
#include <llvm/Support/YAMLTraits.h>
#include <clang/AST/ASTContext.h>

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
  std::map<std::string, KernelInfo> Kernels;
  std::vector<std::string> Order;
  std::map<std::string, std::pair<unsigned, unsigned>> Bounds;
  bool BaseLine;

  explicit Context(std::vector<KernelInfo> &Infos, bool BaseLine): BaseLine(BaseLine) {
    unsigned B = 0;
    for (auto &Info: Infos) {
      Kernels[Info.KernelName] = Info;
      Order.push_back(Info.KernelName);
      Bounds[Info.KernelName] = std::make_pair(B, B + Info.BlockDim.size());
      if (!BaseLine) {
        B += Info.BlockDim.size();
      }
    }
  }

  [[nodiscard]] bool hasKernel(const std::string &KName) const {
    return Kernels.find(KName) != Kernels.end();
  }

};

std::string branchingStatement(const Context &C, const std::string &FName, bool Inverse=false);

std::string generateNewVarName(const std::string &Base);

template <typename NodeT>
std::string getDeclFunctionName(const clang::ASTContext *C, const NodeT &N) {
  auto NodeLists = C->getParents(N);

  while (NodeLists.size()) {
    if (auto N = NodeLists[0].template get<clang::FunctionDecl>()) {
      return N->getName();
    }
    NodeLists = C->getParents(NodeLists[0]);
  }
  return "";
}

const static std::string CurrentTid = "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";

}

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::KernelInfo> {
  static void mapping(IO &Io, kernel_fusion::KernelInfo &Info) {
    Io.mapRequired("KernelName", Info.KernelName);
    Io.mapRequired("HasBarriers", Info.HasBarriers);
    Io.mapRequired("BlockDim", Info.BlockDim);
  }
};

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::BlockDim> {
  static void mapping(IO &Io, kernel_fusion::BlockDim &Dim) {
    Io.mapRequired("X", Dim.X);
    Io.mapRequired("Y", Dim.Y);
    Io.mapRequired("Z", Dim.Z);
  }
};


LLVM_YAML_IS_SEQUENCE_VECTOR(kernel_fusion::KernelInfo)

#endif //SMART_FUSER_INCLUDE_KERNELFUSION_H
