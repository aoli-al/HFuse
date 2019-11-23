#include "KernelFusion.h"

namespace kernel_fusion {

std::string branchingStatement(const Context &C, const std::string &FName, bool Inverse) {
  const auto &Bound = C.Bounds.find(FName)->second;
  return std::string("if (") + (Inverse ? "!" : "") + "(" + CurrentTid
      + ">=" + std::to_string(Bound.first) + " && " + CurrentTid
      + " < " + std::to_string(Bound.second) + "))";
}


static unsigned Count = 0;

std::string generateNewVarName(const std::string &Base) {
  return Base + "_" + std::to_string(Count++);
}

}
