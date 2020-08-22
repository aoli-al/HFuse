//
// Created by Leo Li on 2020-02-11.
//

#include "BarrierRewriter.h"
#include "BarrierAnalyzer.h"
#include <clang/Analysis/CallGraph.h>
#include <sstream>

namespace kernel_fusion {

std::string BuildBarrierAsm(unsigned Barrier, unsigned NumThreads) {
  std::ostringstream Stream;
  Stream << "asm(\"bar.sync " << Barrier << "," << NumThreads << ";\");";
  return Stream.str();
}

void BarrierRewriter::run(const MatchFinder::MatchResult &Result) {
  if (auto *Expr = Result.Nodes.getNodeAs<CallExpr>(BarrierExpressionBindId)) {
    const auto F = Result.Nodes.getNodeAs<FunctionDecl>(ContainingFunction);
    const auto FName = F->getName().str();
    if (!Context.hasKernel(FName)) return;
    if (BarrierIdxMap.find(FName) == BarrierIdxMap.end()) {
      BarrierIdxMap[FName] = BarrierIdx++;
    }
    auto Threads = Context.Kernels[FName].BlockDim.size();
    const tooling::Replacement Replacement(Result.Context->getSourceManager(),
        Expr, BuildBarrierAsm(BarrierIdxMap[FName], Threads));
    if (Replacements[Replacement.getFilePath().str()].add(Replacement)) {
      llvm::outs() << "error?";
    }
  }
}

}

