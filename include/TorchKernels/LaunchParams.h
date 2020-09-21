//
// Created by leele on 9/1/2020.
//

#include <cuda.h>

#ifndef SMART_FUSER_LAUNCHPARAMS_H
#define SMART_FUSER_LAUNCHPARAMS_H

struct LaunchParams {
  dim3 GridSize;
  dim3 BlockSize;
  std::vector<std::string> TemplateArgs;
  std::vector<void *> Args;
};

#endif // SMART_FUSER_LAUNCHPARAMS_H
