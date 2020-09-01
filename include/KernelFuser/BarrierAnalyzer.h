#ifndef SMART_FUSER_INCLUDE_BARRIERANALYZER_H
#define SMART_FUSER_INCLUDE_BARRIERANALYZER_H

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
static const std::string KernelFuseBindId = "kernel-fuse-bind";
static const ast_matchers::DeclarationMatcher KernelFuseMatcher =
    ast_matchers::functionDecl(ast_matchers::hasAttr(attr::CUDAGlobal))
        .bind(KernelFuseBindId);

template <typename... Params>
static StatementMatcher barrierMatcherFactory(Params&&... Args) {
  return callExpr(
      hasDeclaration(functionDecl(hasName("__syncthreads"))),
      hasAncestor(functionDecl(std::forward<Params>(Args)...)
                      .bind(ContainingFunction)))
      .bind(BarrierExpressionBindId);
}

using KFMap = std::map<StringRef, FunctionDecl *>;

class BarrierAnalyzer: public MatchFinder::MatchCallback {
public:
  explicit BarrierAnalyzer(Context &Context) : Context(Context) {}
  void run(const MatchFinder::MatchResult &Result) override;

protected:
  KFMap KernelFunctionMap;
  std::map<std::string, std::map<const CompoundStmt *, std::list<SourceRange>>> Splits;
  std::set<unsigned> VisitedSplits;
  ASTContext *ASTContext = nullptr;
  Context &Context;
};

}


#endif //SMART_FUSER_INCLUDE_BARRIERANALYZER_H
