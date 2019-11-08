//
// Created by Leo Li on 2019-10-24.
//

#include <clang/AST/Decl.h>

#include "KernelFuseTool.h"
#include "KernelPrinter.h"
#include "KernelRecursiveFuser.h"

using namespace llvm;

namespace kernel_fusion {

void KernelFuseTool::onEndOfTranslationUnit() {
  auto *C = &KernelFunctionMap.begin()->second->getASTContext();
  KernelPrinter Printer(outs(), C->getPrintingPolicy(), *C, Context, KernelFunctionMap);
  Printer.printFusedFunction();
  Printer.Out << "\n {\n";

  StmtPointers Pointers;
  FunctionRanges Ranges;

  for (auto Func: Context.Order) {
    if (Splits.find(Func) != Splits.end()) {
      for (auto &Split: Splits[Func]) {
        if (ASTContext->getParents(*Split.first)[0].get<FunctionDecl>()) {
          Ranges.push_back(Split.second);
        }
      }
    } else {
      Ranges.push_back(std::list<SourceRange>());
    }
    if (auto *F = dyn_cast<CompoundStmt>(KernelFunctionMap[Func]->getBody())) {
      Pointers.push_back(std::make_pair(F->body_begin(), F->body_end()));
    } else {
      Pointers.push_back(std::make_pair(nullptr, nullptr));
    }
//    Pointers.push_back(std::make_pair(KernelFunctionMap[Func].ge))
  }

  KernelRecursiveFuser Fuser(Printer, Context);
  Fuser.fuse(Pointers, Ranges);
  Printer.Out << "}\n";
}

}
