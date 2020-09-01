#include <algorithm>

#ifndef SMART_FUSER_UTILS_H
#define SMART_FUSER_UTILS_H


namespace kernel_fusion {

static int lastPow2(unsigned int n) {
  n |= (n >> 1);
  n |= (n >> 2);
  n |= (n >> 4);
  n |= (n >> 8);
  n |= (n >> 16);
  return std::max<int>(1, n - (n >> 1));
}

}

#endif // SMART_FUSER_UTILS_H
