//
// Created by Leo Li on 2019-10-31.
//

#include "KernelFusion.h"

namespace kernel_fusion {

std::string branchingStatement(const Context &C, std::string FName) {
  return "if (threadIdx." + C.Dimension  +
      (FName == C.Kernels.first ? " < " : ">=") +
      std::to_string(C.Offset) + ")";
}


}
