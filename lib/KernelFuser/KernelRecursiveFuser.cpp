//
// Created by Leo Li on 2019-11-07.
//

#include "KernelRecursiveFuser.h"

namespace kernel_fusion {


void KernelRecursiveFuser::fuse(StmtPointers Pointers, FunctionRanges Ranges) {
  bool UncoverdStmt = false;
  bool PrintBarrier = false;
  for (unsigned I = 0; I < Context.Order.size(); I++) {
    if (Pointers[I].second == Pointers[I].first) {
      continue;
    }
    UncoverdStmt = true;
    const auto BS =
        branchingStatement(Context, Context.Order[I], true) + " goto ";
    auto CurrentLabel = generateNewVarName("label");
    Printer.Out << BS + CurrentLabel + ";\n";
    while ((Ranges[I].size() == 0 ||
        (*Pointers[I].first)->getBeginLoc() < Ranges[I].front().getBegin()) &&
        Pointers[I].first != Pointers[I].second) {
      Printer.printStmt(*(Pointers[I].first++));
    }
    Printer.Out << CurrentLabel + ":;\n";

    if (Ranges[I].size() != 0) {
      if (isa<CallExpr>(*Pointers[I].first) && !PrintBarrier) {
        PrintBarrier = true;
        Printer.printStmt(*(Pointers[I].first++));
      }
      else {
        Printer.printStmt(*(Pointers[I].first++));
      }
      Ranges[I].pop_front();
    }
  }
  if (UncoverdStmt) {
    fuse(Pointers, Ranges);
  }
}

}