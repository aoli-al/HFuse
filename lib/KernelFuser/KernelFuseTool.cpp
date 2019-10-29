//
// Created by Leo Li on 2019-10-24.
//

#include <clang/AST/Decl.h>

#include "KernelFuseTool.h"
#include "KernelPrinter.h"

using namespace llvm;

namespace kernel_fusion {

void KernelFuseTool::run(const ast_matchers::MatchFinder::MatchResult &Result) {
  auto *D = Result.Nodes.getNodeAs<FunctionDecl>(KernelFuseBindId);
  KernelFunctionMap[D->getName()] = const_cast<FunctionDecl *>(D);
}

void KernelFuseTool::onEndOfTranslationUnit() {
  fuseKernel(KernelFunctionMap["im2col_kernel"],
             KernelFunctionMap["upsample_bilinear2d_out_frame"]);
}

void KernelFuseTool::fuseKernel(FunctionDecl *FunctionA,
                                FunctionDecl *FunctionB) {


  auto *C = &FunctionA->getASTContext();
  KernelPrinter Printer(outs(), C->getPrintingPolicy(), *C);
  Printer.printFusedFunction(FunctionA, FunctionB);
}

}
