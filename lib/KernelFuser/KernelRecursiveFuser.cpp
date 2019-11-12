//
// Created by Leo Li on 2019-11-07.
//

#include "KernelRecursiveFuser.h"

namespace kernel_fusion {

void KernelRecursiveFuser::selectBlockRecursive(const std::string &Blocks, unsigned Start, unsigned NumLeft) {
  if (NumLeft == 0) {
    std::string Sync = "";
    std::string B = Blocks;
    for (const auto &BId: SelectedBlocks) {
      B += CodeBlocks[BId][Progress[BId]].first;
      Sync += CodeBlocks[BId][Progress[BId]++].second;
    }
    auto SB = SelectedBlocks;
    SelectedBlocks.resize(0);
    fuseRecursive(B + Sync);
    SelectedBlocks = std::move(SB);
    for (const auto &BId: SelectedBlocks) {
      Progress[BId]--;
    }
    return;
  }
  if (Start == Progress.size()) return;
  if (Progress[Start] < CodeBlocks[Start].size()) {
    SelectedBlocks.push_back(Start);
    selectBlockRecursive(Blocks, Start + 1, NumLeft - 1);
    SelectedBlocks.pop_back();
  }
  selectBlockRecursive(Blocks, Start + 1, NumLeft);
}

void KernelRecursiveFuser::fuseRecursive(std::string Blocks) {
  auto Num = 0;
  for (unsigned I = 0; I < Context.Order.size(); I++) {
    if (Progress[I] != CodeBlocks[I].size()) Num++;
  }
  if (Num == 0) {
    llvm::outs() << Blocks;
    llvm::outs().flush();
    return;
  }
  for (unsigned I = 1; I <= Num; I++) {
    selectBlockRecursive(Blocks, 0, I);
  }
}

void KernelRecursiveFuser::fuse(StmtPointers &Pointers, FunctionRanges &Ranges) {
  generateCodeBlocks(Pointers, Ranges);
  Progress.resize(Context.Order.size(), 0);
  fuseRecursive("");
}
void KernelRecursiveFuser::generateCodeBlocks(StmtPointers &Pointers,
                                              FunctionRanges &Ranges) {
  CodeBlocks.resize(Context.Order.size());
  for (unsigned I = 0; I < Context.Order.size(); I++) {
    while (Pointers[I].second != Pointers[I].first) {
      std::string Block;
      llvm::raw_string_ostream BlockStream(Block);
      KernelPrinter Printer( BlockStream, ASTContext->getPrintingPolicy(), *ASTContext, Context);
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
      std::string Sync;
      if (Ranges[I].size() != 0) {
        llvm::raw_string_ostream SyncStream(Sync);
        KernelPrinter P2( SyncStream, ASTContext->getPrintingPolicy(), *ASTContext, Context);
        P2.printStmt(*(Pointers[I].first++));
        Ranges[I].pop_front();
        SyncStream.flush();
      }
      BlockStream.flush();
      CodeBlocks[I].push_back(std::make_pair(std::move(Block), std::move(Sync)));
    }
  }
}
}