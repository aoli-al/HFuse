//
// Created by Leo Li on 2019-10-27.
//
#include "ThreadInfoRewriter.h"
#include <numeric>

using namespace llvm;

namespace kernel_fusion {

const std::map<std::string, std::string>
    ThreadInfoRewriter::MemberNameMapping = {
    {"__fetch_builtin_y", "threadIdx_y"},
    {"__fetch_builtin_x", "threadIdx_x"},
    {"__fetch_builtin_z", "threadIdx_z"},
};

void ThreadInfoRewriter::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  const auto NameBuilder = [](std::string a, std::string b) {
    return a + "_" + b;
  };
  auto *ME = Result.Nodes.getNodeAs<MemberExpr>(ThreadInfoAccessId);
  auto *FDecl = Result.Nodes.getNodeAs<FunctionDecl>(FunctionDeclId);
  auto FName = FDecl->getName().str();
  const auto &SM = Result.Context->getSourceManager();
  if (ME &&
      Result.Context->getSourceManager()
          .isInMainFile(ME->getSourceRange().getBegin())) {
    if (ThreadIdxNameMap.find(FName) == ThreadIdxNameMap.end()) {
      ThreadIdxNameMap[FName] = std::to_string(Idx++);
      const auto Stmts = std::accumulate(MemberNameMapping.begin(),
          MemberNameMapping.end(),
          std::string(""),
          [&NameBuilder, &FName, this](auto a, auto b) {
        const auto NewName = NameBuilder(b.second,
                                         ThreadIdxNameMap[FName]);
        return a + "unsigned int " + NewName + " = threadIdx." +
            b.first[b.first.length() - 1] + ";\n";
      });
      const auto Replacement =
          tooling::Replacement(SM,
                               FDecl->getBody()->getBeginLoc().getLocWithOffset(1),
                               0,
                               Stmts);
      if (auto Err = Replacements[Replacement.getFilePath().str()]
          .add(Replacement)) {
        outs() << "error?";
      }
    }
    const auto MemberName = ME->getMemberNameInfo().getAsString();
    if (MemberNameMapping.find(MemberName) == MemberNameMapping.end()) {
      return;
    }
    const auto NewName = NameBuilder(MemberNameMapping.at(MemberName),
                                     ThreadIdxNameMap[FName]);
    const auto CSR = CharSourceRange::getTokenRange(ME->getSourceRange());
    const auto Replacement =
        tooling::Replacement(SM, CSR, NewName);
    if (auto Err =
        Replacements[Replacement.getFilePath().str()].add(Replacement)) {
      outs() << "error?";
    }
  }
}
}


