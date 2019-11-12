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
  KernelRecursiveFuser(std::string FuncStr, Context &Context, const ASTContext *ASTContext):
      Context(Context), ASTContext(ASTContext) {
    Streams.push_back(FuncStr);
  };

  void fuse(StmtPointers &Pointers, FunctionRanges &Ranges);
  void fuseRecursive(std::string Blocks);
  void selectBlockRecursive(const std::string &Blocks, unsigned Start, unsigned NumLeft);
  void generateCodeBlocks(StmtPointers &Pointers, FunctionRanges &Ranges);

private:
  std::vector<std::string> Streams;
  std::vector<unsigned> SelectedBlocks;
  std::vector<unsigned> Progress;
  std::vector<std::vector<std::pair<std::string, std::string>>> CodeBlocks;
  Context &Context;
  unsigned Count = 0;
  const ASTContext *ASTContext;
};

}

#endif //SMART_FUSER_INCLUDE_KERNELRECURSIVEFUSER_H
