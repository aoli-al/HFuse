//
// Created by Leo Li on 2019-10-31.
//

#include "BarrierHoisting.h"

namespace kernel_fusion {

void BarrierHoisting::run(const MatchFinder::MatchResult &Result) {
  if (!ASTContext) {
    ASTContext = Result.Context;
  }
  auto *Expr = Result.Nodes.getNodeAs<CallExpr>(BarrierExpressionBindId);
  const auto FName = Result.Nodes.getNodeAs<FunctionDecl>(ContainingFunction)->getName().str();
  auto NodeLists = Result.Context->getParents(*Expr);
  auto CurrentSplit = Expr->getSourceRange();
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

void BarrierHoisting::onEndOfTranslationUnit() {
  if (!ASTContext) return;
  for (auto &F: Splits) {
    const auto FName = F.first;
    Context.Info[FName].HasBarriers = true;
    const auto BS = branchingStatement(Context, FName) + " {";
    for (auto &T: F.second) {
      const auto Stmt = T.first;
      createAndInsert(Stmt->getBeginLoc().getLocWithOffset(1), BS);
      for (auto &R: T.second) {
        createAndInsert(R.getBegin(), "}");
        createAndInsert(R.getEnd().getLocWithOffset(1), BS);
      }
      createAndInsert(Stmt->getEndLoc(), "}");
    }
  }
}

}

