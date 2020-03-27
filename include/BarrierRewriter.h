//
// Created by Leo Li on 2020-02-11.
//

#ifndef SMART_FUSER_BARRIERREWRITER_H
#define SMART_FUSER_BARRIERREWRITER_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Core/Replacement.h>

#include <string>
#include "KernelFusion.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace kernel_fusion {

class BarrierRewriter: public MatchFinder::MatchCallback {
public:
  explicit BarrierRewriter(
      std::map<std::string, tooling::Replacements> &Replacements,
      Context &Context): Replacements(Replacements),
                         Context(Context) {};
  void run(const MatchFinder::MatchResult &Result) override;
  bool hasBarrier() const {
    return BarrierIdx != 1;
  }
private:
  Context &Context;
  unsigned BarrierIdx = 1;
  std::map<std::string, unsigned> BarrierIdxMap;
  std::map<std::string, tooling::Replacements> &Replacements;
};

}

#endif // SMART_FUSER_BARRIERREWRITER_H
