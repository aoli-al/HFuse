//
// Created by Leo Li on 2019-10-27.
//

#ifndef SMART_FUSER_THREADINFOREWRITER_H
#define SMART_FUSER_THREADINFOREWRITER_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Core/Replacement.h>
#include <clang/Tooling/Execution.h>
#include <string>
#include <vector>

using namespace clang;
using namespace ast_matchers;

namespace kernel_fusion {
static const std::string ThreadInfoAccessId = "thread-info";
static const std::string FunctionDeclId = "thread-info-function";
static const ast_matchers::StatementMatcher ThreadInfoMatcher =
    memberExpr(
        hasObjectExpression(
            opaqueValueExpr(
                hasSourceExpression(
                    declRefExpr(
                        to(
                            varDecl(
                                hasName("threadIdx"))))))),
        hasAncestor(functionDecl().bind(FunctionDeclId)))
        .bind(ThreadInfoAccessId);

class ThreadInfoRewriter : public ast_matchers::MatchFinder::MatchCallback {
public:
  explicit ThreadInfoRewriter(
      std::map<std::string, tooling::Replacements> &Replacements) :
      Replacements(Replacements) {}
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  const static std::map<std::string, std::string> MemberNameMapping;
private:
  std::map<std::string, std::string> ThreadIdxNameMap;
  unsigned Idx = 0;
  std::map<std::string, tooling::Replacements> &Replacements;

};

}


#endif // SMART_FUSER_THREADINFOREWRITER_H
