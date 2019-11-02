//
// Created by Leo Li on 2019-10-31.
//

#include "KernelFusion.h"

namespace kernel_fusion {

std::string branchingStatement(const Context &C, const std::string &FName) {
  const auto &Kernel = FName == C.Kernels.first.KernelName ? C.Kernels.first : C.Kernels.second;
  return std::string("if (" + CurrentTid)
      + (FName == C.Kernels.first.KernelName ? " < " : ">=") +
      std::to_string(Kernel.BlockDim.size()) + ")";
}


static unsigned Count = 0;

std::string generateNewVarName(const std::string &Base) {
  return Base + "_" + std::to_string(Count++);
}


}
