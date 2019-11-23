#include "KernelRecursiveFuser.h"

namespace kernel_fusion {


void KernelRecursiveFuser::fuseRecursive(unsigned CheckStart, uint64_t CurrentHash) {
  bool Change = false;
  auto UpdateHash = [CheckStart, this](uint64_t NewHash, unsigned  End, auto Success) {
    uint64_t SyncHash = 0;
    uint64_t CodeHash = 0;
    bool Synced = false;
    std::string Sync, Stmt;
    for (auto J = CheckStart; J < End; J++  ) {
      SyncHash ^= SelectedBlocks[J]->Id + NumOfBlocks;
      if (SelectedBlocks[J]->Sync == "__syncthreads();\n") {
        if (!Synced) {
          Synced = true;
          Sync += SelectedBlocks[J]->Sync;
        }
      } else {
        Sync += SelectedBlocks[J]->Sync;
      }
      CodeHash ^= SelectedBlocks[J]->Id;
      Stmt += SelectedBlocks[J]->Code;
    }
    NewHash <<= shift();
    NewHash += CodeHash;
    NewHash <<= shift();
    NewHash += SyncHash;
    if (Searched.find(NewHash) == Searched.end()) {
      unsigned Cut = Code.size();
      Searched.insert(NewHash);
      Code += std::move(Stmt) + std::move(Sync);
      Success(NewHash);
      Code.resize(Cut);
    }
  };
  for (unsigned I = 0; I < Context.Order.size(); I++) {
    if (Progress[I] != CodeBlocks[I].size()) {
      Change = true;
      const auto *Block = &CodeBlocks[I][Progress[I]++];
      SelectedBlocks.push_back(Block);
      auto V = LastVisited[Block->BlockId];
      LastVisited[Block->BlockId] = SelectedBlocks.size() - 1;
      if (V >= CheckStart && Block->SegId != 0) {
        UpdateHash(CurrentHash, SelectedBlocks.size() - 1, [this](uint64_t NewHash) {
          fuseRecursive(SelectedBlocks.size() - 1, NewHash);
        });
      } else {
        fuseRecursive(CheckStart, CurrentHash);
      }
      SelectedBlocks.pop_back();
      LastVisited[Block->BlockId] = V;
      Progress[I]--;
    }
  }
  if (!Change) {
    UpdateHash(CurrentHash, SelectedBlocks.size(), [this](uint64_t) {
      CandidateSnippets.push_back(Code);
    });
  }
}

void KernelRecursiveFuser::fuse(StmtPointers &Pointers, FunctionRanges &Ranges) {
  generateCodeBlocks(Pointers, Ranges);
  Progress.resize(Context.Order.size(), 0);
  fuseRecursive(0, 0);
}
void KernelRecursiveFuser::generateCodeBlocks(StmtPointers &Pointers,
                                              FunctionRanges &Ranges) {
  CodeBlocks.resize(Context.Order.size());
  for (unsigned I = 0; I < Context.Order.size(); I++) {
    unsigned SegId = 0;
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
      CodeBlocks[I].push_back({std::move(Block), std::move(Sync), I, SegId++, NumOfBlocks++});
    }
  }
}
}