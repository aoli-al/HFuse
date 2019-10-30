//
// Created by Leo Li on 2019-10-29.
//

#ifndef SMART_FUSER_INCLUDE_KERNELFUSION_H
#define SMART_FUSER_INCLUDE_KERNELFUSION_H

#include <string>

namespace kernel_fusion {
struct Context {
  std::pair<std::string, std::string> Kernels;
  std::string Dimension;
  unsigned Offset;
};

}

#endif //SMART_FUSER_INCLUDE_KERNELFUSION_H
