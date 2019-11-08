//
// Created by Leo Li on 2019-11-07.
//

#ifndef SMART_FUSER_INCLUDE_KERNELRECURSIVEFUSER_H
#define SMART_FUSER_INCLUDE_KERNELRECURSIVEFUSER_H

#include <vector>
#include <clang/AST/Stmt.h>
#include "KernelPrinter.h"

using namespace clang;
namespace kernel_fusion {

using StmtPointers = std::vector<std::pair<Stmt **, Stmt **>>;
using FunctionRanges = std::vector<std::list<SourceRange>>;

class KernelRecursiveFuser {

public:
  KernelRecursiveFuser(KernelPrinter &Printer, Context &Context):
      Printer(Printer), Context(Context) {};

  void fuse(StmtPointers Pointer, FunctionRanges Ranges);

private:
  KernelPrinter &Printer;
  Context &Context;
};

}

#endif //SMART_FUSER_INCLUDE_KERNELRECURSIVEFUSER_H
