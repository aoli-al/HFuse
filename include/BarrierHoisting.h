//
// Created by Leo Li on 2019-10-30.
//

#ifndef SMART_FUSER_INCLUDE_BARRIERHOISTING_H
#define SMART_FUSER_INCLUDE_BARRIERHOISTING_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Core/Replacement.h>

#include <string>
#include "KernelFusion.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace kernel_fusion {

static const std::string BarrierExpressionBindId = "barrier-expression";
static const std::string ContainingFunction = "containing-function";

template <typename... Params>
static StatementMatcher barrierMatcherFactory(Params&&... Args) {
  return callExpr(
      hasDeclaration(functionDecl(hasName("__syncthreads"))),
      hasAncestor(functionDecl(anyOf(std::forward<Params>(Args)...))
                      .bind(ContainingFunction)))
      .bind(BarrierExpressionBindId);
}

class BarrierHoisting: public MatchFinder::MatchCallback {
public:
  explicit BarrierHoisting(std::map<std::string, tooling::Replacements> &Replacements,
                           Context &Context) :
                           Replacements(Replacements), Context(Context) {}
  void run(const MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;

private:
  std::map<std::string, std::map<const CompoundStmt *, std::vector<SourceRange>>> Splits;
  std::map<std::string, tooling::Replacements> &Replacements;
  ASTContext *ASTContext = nullptr;
  Context &Context;

  void createAndInsert(clang::SourceLocation Loc, const std::string &S) {
    const tooling::Replacement Replacement(ASTContext->getSourceManager(), Loc, 0, S);
    if (auto Err =
        Replacements[Replacement.getFilePath().str()].add(Replacement)) {
      llvm::outs() << "error?";
    }
  }
};

}


#endif //SMART_FUSER_INCLUDE_BARRIERHOISTING_H
