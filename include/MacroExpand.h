#ifndef SMART_FUSER_INCLUDE_MACROEXPAND_H
#define SMART_FUSER_INCLUDE_MACROEXPAND_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Core/Replacement.h>
#include "KernelFusion.h"

using namespace clang;

namespace kernel_fusion {

class MacroExpand: public ast_matchers::MatchFinder::MatchCallback {
public:
  explicit MacroExpand(std::map<std::string, tooling::Replacements> &Replacements,
      const Context &Context): Context(Context), Replacements(Replacements) {}
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
private:
  const Context &Context;
  std::map<std::string, tooling::Replacements> &Replacements;
};

}

#endif //SMART_FUSER_INCLUDE_MACROEXPAND_H
