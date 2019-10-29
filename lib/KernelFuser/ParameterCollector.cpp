//
// Created by Leo Li on 2019-10-26.
//
#include "ParameterCollector.h"

namespace kernel_fusion {

void ParameterCollector::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto *D = Result.Nodes.getNodeAs<FunctionDecl>(ParameterCollectorBindId);
  if (D->getName().empty()
      || Kernels.find(D->getName().str()) == Kernels.end()) {
    return;
  }
  llvm::outs() << D->getName() << "\n";
  llvm::outs().flush();
  if (auto *TF = D->getDescribedFunctionTemplate()) {
    TF->getTemplateParameters();
    for (auto Param: *TF->getTemplateParameters()) {
      if (!Param->getName().empty()) {
        ParameterList.push_back(Param->getLocation().getRawEncoding());
      }
    }
  }

  for (unsigned i = 0, End = D->getNumParams(); i != End; ++i) {
    if (!D->getParamDecl(i)->getName().empty()) {
      ParameterList.push_back(D->getParamDecl(i)->getLocation().getRawEncoding());
    }
  }
}
}
