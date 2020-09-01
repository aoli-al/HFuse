#include "KernelFuser/BarrierHoisting.h"

namespace kernel_fusion {

void BarrierHoisting::onEndOfTranslationUnit() {
  if (!ASTContext) return;
  for (auto &F: Splits) {
    const auto FName = F.first;
    Context.Kernels[FName].HasBarriers = true;
    const auto BS = branchingStatement(Context, FName, true) + " goto ";
    for (auto &T: F.second) {
      const auto Stmt = T.first;
      if (ASTContext->getParents(*Stmt)[0].get<FunctionDecl>() && !Context.BaseLine) {
        continue;
      }
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

