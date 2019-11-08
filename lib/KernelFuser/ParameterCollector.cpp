//
// Created by Leo Li on 2019-10-26.
//
#include <clang/Tooling/Refactoring/Rename/USRFindingAction.h>
#include <DeclRewriter.h>
#include "ParameterCollector.h"

using namespace clang;

namespace kernel_fusion {

void ParameterCollector::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  if (auto *D = Result.Nodes.getNodeAs<FunctionDecl>(ParameterCollectorBindId)) {
    if (!D->getName().empty()
        && Context.hasKernel(D->getName().str())
        && VisitedFunctions.find(D->getName().str()) == VisitedFunctions.end()) {
      VisitedFunctions.insert(D->getName().str());
      if (auto *TF = D->getDescribedFunctionTemplate()) {
        TF->getTemplateParameters();
        for (auto Param: *TF->getTemplateParameters()) {
          if (!Param->getName().empty()) {
            ParameterList.push_back(Param->getName().str());
            USRList.push_back(
                tooling::getUSRsForDeclaration(Param->getUnderlyingDecl(), *Result.Context));
          }
        }
      }

      for (unsigned i = 0, End = D->getNumParams(); i != End; ++i) {
        if (!D->getParamDecl(i)->getName().empty()) {
          const auto *NamedDecl = D->getParamDecl(i)->getUnderlyingDecl();
          ParameterList.push_back(NamedDecl->getName().str());
          USRList.push_back(
              tooling::getUSRsForDeclaration(NamedDecl, *Result.Context));
        }
      }
    }
  }

  if (auto *Stmt = Result.Nodes.getNodeAs<DeclStmt>(DeclStmtBindId)) {
    if (VisitedDecl.find(Stmt->getBeginLoc().getRawEncoding()) == VisitedDecl.end()) {
      Stmt->dump();
      VisitedDecl.insert(Stmt->getBeginLoc().getRawEncoding());
      for (auto Decl: Stmt->decls()) {
        if (auto VD = dyn_cast<VarDecl>(Decl)) {
          ParameterList.push_back(VD->getName());
          USRList.push_back(
              tooling::getUSRsForDeclaration(VD->getUnderlyingDecl(), *Result.Context)
              );
        }
      }
    }
  }
}
}
