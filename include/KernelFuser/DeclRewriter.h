#ifndef SMART_FUSER_INCLUDE_DECLREWRITER_H
#define SMART_FUSER_INCLUDE_DECLREWRITER_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Core/Replacement.h>

#include <string>
#include "BarrierHoisting.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace kernel_fusion {

static const std::string DeclStmtBindId = "decl-stmt";

template <typename... Params>
static StatementMatcher declStmtMatcherFactory(Params&&... Args) {
  return declStmt(
      hasAncestor(functionDecl(std::forward<Params>(Args)...)
                      .bind(ContainingFunction)),
      hasParent(compoundStmt())).bind(DeclStmtBindId);
}


class DeclRewriter: public MatchFinder::MatchCallback {
public:
  explicit DeclRewriter(std::map<std::string, tooling::Replacements> &Replacements,
                        Context &Context) :
      Replacements(Replacements), Context(Context) {}
  void run(const MatchFinder::MatchResult &Result) override;
private:
  std::string printVarDecl(VarDecl *D, const PrintingPolicy &Policy);
  void printDeclType(QualType T, StringRef DeclName, llvm::raw_string_ostream &Out,
                     const PrintingPolicy &Policy, unsigned Indentation);
  std::map<std::string, tooling::Replacements> &Replacements;
  std::set<unsigned> VisitedDecl;
  Context &Context;
};


}

#endif //SMART_FUSER_INCLUDE_DECLREWRITER_H
