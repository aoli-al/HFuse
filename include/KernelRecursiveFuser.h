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
  KernelRecursiveFuser(Context &Context, const ASTContext *ASTContext):
      Context(Context), ASTContext(ASTContext) {}

  void fuse(StmtPointers &Pointers, FunctionRanges &Ranges);
  void fuseRecursive(unsigned CheckStart, uint64_t CurrentHash);
  void generateCodeBlocks(StmtPointers &Pointers, FunctionRanges &Ranges);

  unsigned shift() const {
    return 64u - __builtin_clzll(NumOfBlocks << 1u) << 1u;
  }

  const std::vector<std::string> &getCandidates() const {
    return CandidateSnippets;
  }

private:

  struct Block {
    std::string Code;
    std::string Sync;
    unsigned BlockId;
    unsigned SegId;
    unsigned Id;
  };


  std::map<unsigned, unsigned> LastVisited;
  std::string Code;
  std::vector<unsigned> Progress;
  std::vector<std::vector<Block>> CodeBlocks;
  std::vector<const Block *> SelectedBlocks;
  std::set<uint64_t> Searched;
  std::vector<std::string> CandidateSnippets;
  Context &Context;
  unsigned NumOfBlocks = 0;
  const ASTContext *ASTContext;
};

}

#endif //SMART_FUSER_INCLUDE_KERNELRECURSIVEFUSER_H
