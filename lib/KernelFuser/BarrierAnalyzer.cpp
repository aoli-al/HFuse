#include "BarrierAnalyzer.h"

namespace kernel_fusion {

void BarrierAnalyzer::run(const MatchFinder::MatchResult &Result) {
  if (!ASTContext) {
    ASTContext = Result.Context;
  }
  if (auto *Expr = Result.Nodes.getNodeAs<CallExpr>(BarrierExpressionBindId)) {
    const auto F = Result.Nodes.getNodeAs<FunctionDecl>(ContainingFunction);
    const auto FName = F->getName().str();
    auto NodeLists = Result.Context->getParents(*Expr);
    auto CurrentSplit = Expr->getSourceRange();
    if (VisitedSplits.find(CurrentSplit.getBegin().getRawEncoding()) != VisitedSplits.end())  return;
    VisitedSplits.insert(CurrentSplit.getBegin().getRawEncoding());

    // implied semicolon of expression statements.
    CurrentSplit.setEnd(CurrentSplit.getEnd().getLocWithOffset(1));

    while (NodeLists.size()) {
      if (auto Node = NodeLists[0].get<CompoundStmt>()) {
        Splits[FName][Node].push_back(CurrentSplit);
      }
      else if (auto Func = NodeLists[0].get<FunctionDecl>()) {
        if (Func->getName().str() == FName) {
          break;
        }
      }
      CurrentSplit = NodeLists[0].getSourceRange();
      NodeLists = Result.Context->getParents(NodeLists[0]);
    }
  }

  if (auto *D = Result.Nodes.getNodeAs<FunctionDecl>(KernelFuseBindId)) {
    if (D->isTemplateInstantiation()) return;
    KernelFunctionMap[D->getName()] = const_cast<FunctionDecl *>(D);
  }
}

}


