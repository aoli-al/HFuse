#ifndef SMART_FUSER_THREADINFOREWRITER_H
#define SMART_FUSER_THREADINFOREWRITER_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Core/Replacement.h>
#include <clang/Tooling/Execution.h>
#include <string>
#include <vector>

#include "KernelFusion.h"
using namespace clang;
using namespace ast_matchers;

namespace kernel_fusion {
static const std::string ThreadAccessId = "thread-info";
static const std::string FunctionDeclId = "thread-info-function";
static const std::string ThreadVarDeclId = "thread-info-var-decl";
static const std::string ThreadIdx = "threadIdx";
static const std::string BlockDim = "blockDim";
static const ast_matchers::StatementMatcher ThreadInfoMatcher =
    memberExpr(
        hasObjectExpression(
            opaqueValueExpr(
                hasSourceExpression(
                    declRefExpr(
                        to(
                            varDecl(
                                anyOf(
                                    hasName(ThreadIdx),
                                    hasName(BlockDim)))
                                .bind(ThreadVarDeclId)))))),
        hasAncestor(functionDecl().bind(FunctionDeclId)))
        .bind(ThreadAccessId);

class ThreadInfoRewriter : public ast_matchers::MatchFinder::MatchCallback {
public:
  ThreadInfoRewriter(
      std::map<std::string, tooling::Replacements> &Replacements,
      const Context &Context) :
      Replacements(Replacements), Context(Context) {}
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  const static std::map<std::string, std::string> MemberNameMapping;
private:
  std::map<std::string, std::string> KernelInfoNameMap;
  unsigned Idx = 0;
  std::map<std::string, tooling::Replacements> &Replacements;
  const Context &Context;
};

}


#endif // SMART_FUSER_THREADINFOREWRITER_H
