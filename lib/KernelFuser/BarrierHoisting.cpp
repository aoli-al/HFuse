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

void BarrierHoisting::onEndOfTranslationUnit() {
  if (!ASTContext) return;
  for (auto &F: Splits) {
    const auto FName = F.first;
    const auto &SkipedFName = Context.isFirstKernel(FName)
                              ? Context.Kernels.second.KernelName
                              : Context.Kernels.first.KernelName;
    Context.getKernelWithName(FName).HasBarriers = true;
    const auto BS = branchingStatement(Context, SkipedFName) + " goto ";
    for (auto &T: F.second) {
      const auto Stmt = T.first;
      auto CurrentLabel = generateNewVarName("label");
      createAndInsert(Stmt->getBeginLoc().getLocWithOffset(1),
          BS + CurrentLabel + ";");
      for (auto &R: T.second) {
        createAndInsert(R.getBegin(), " "  + CurrentLabel + ":;");
        CurrentLabel = generateNewVarName("label");
        createAndInsert(R.getEnd().getLocWithOffset(1),
            BS + CurrentLabel + ";");
      }
      createAndInsert(Stmt->getEndLoc(), " "  + CurrentLabel + ":;");
    }
  }
}

}

