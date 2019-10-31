//
// Created by Leo Li on 2019-10-29.
//

#ifndef SMART_FUSER_INCLUDE_KERNELFUSION_H
#define SMART_FUSER_INCLUDE_KERNELFUSION_H

#include <string>
#include <map>

namespace kernel_fusion {

struct KernelInfo {
  bool HasBarriers;
};

struct Context {
  std::pair<std::string, std::string> Kernels;
  std::string Dimension;
  unsigned Offset;
  std::map<std::string, KernelInfo> Info;
};

std::string branchingStatement(const Context &C, std::string FName);

}

#endif //SMART_FUSER_INCLUDE_KERNELFUSION_H
