#include <clang/AST/Decl.h>

#include "KernelFuseTool.h"
#include "KernelPrinter.h"
#include "KernelRecursiveFuser.h"
#include "llvm/Support/JSON.h"

using namespace llvm;

namespace kernel_fusion {

void KernelFuseTool::onEndOfTranslationUnit() {
  auto *C = &KernelFunctionMap.begin()->second->getASTContext();

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
  }

  KernelRecursiveFuser Fuser(Context, C);
  Fuser.fuse(Pointers, Ranges);

  std::string FuncStr;
  auto FuncStream = llvm::raw_string_ostream(FuncStr);
  KernelPrinter Printer( FuncStream, C->getPrintingPolicy(), *C, Context);
  std::vector<std::string> Candidates;
  if (!Context.BaseLine) {
    unsigned Idx = 0;
    for (const auto &Snippet: Fuser.getCandidates()) {
      Printer.printFusedFunction(KernelFunctionMap, Idx++);
      FuncStream << "\n {\n";
      FuncStream << Snippet;
      FuncStream << "}\n";
      FuncStream.flush();
      Candidates.push_back(FuncStr);
      Results.push_back(FuncStr);
//      llvm::outs() << FuncStr;
      llvm::outs().flush();
      FuncStr.clear();
    }
  } else {
    Printer.printFusedFunction(KernelFunctionMap, 0);
    FuncStream << "\n {\n";
    for (const auto &FName: Context.Order) {
      FuncStream << branchingStatement(Context, FName);
      KernelFunctionMap[FName]->getBody()->printPretty(FuncStream, nullptr, C->getPrintingPolicy());
    }
    FuncStream << "}\n";
    FuncStream.flush();
    Candidates.push_back(FuncStr);
    Results.push_back(FuncStr);
  }
//  auto R = llvm::json::Value(Candidates);
//  llvm::outs() << R;
}

}
