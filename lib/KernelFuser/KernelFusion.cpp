//
// Created by Leo Li on 2019-10-31.
//

#include "KernelFusion.h"

namespace kernel_fusion {

std::string branchingStatement(const Context &C, const std::string &FName) {
  return "if (threadIdx." + C.Dimension  +
      (FName == C.Kernels.first ? " < " : ">=") +
      std::to_string(C.Offset) + ")";
}


static unsigned Count = 0;

std::string generateNewVarName(const std::string &Base) {
  return Base + "_" + std::to_string(Count++);
}


}
