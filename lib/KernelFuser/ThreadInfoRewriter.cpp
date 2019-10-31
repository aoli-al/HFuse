//
// Created by Leo Li on 2019-10-27.
//
#include "ThreadInfoRewriter.h"
#include <numeric>

using namespace llvm;

namespace kernel_fusion {

const std::map<std::string, std::string>
    ThreadInfoRewriter::MemberNameMapping = {
    {"__fetch_builtin_y", "y"},
    {"__fetch_builtin_x", "x"},
    {"__fetch_builtin_z", "z"},
};

void ThreadInfoRewriter::run(
    const ast_matchers::MatchFinder::MatchResult &Result) {
  auto *ME = Result.Nodes.getNodeAs<MemberExpr>(ThreadAccessId);
  auto *FDecl = Result.Nodes.getNodeAs<FunctionDecl>(FunctionDeclId);
  auto *VDecl = Result.Nodes.getNodeAs<VarDecl>(ThreadVarDeclId);
  auto ThreadInfoName = VDecl->getName().str();
  auto FName = FDecl->getName().str();
  const auto &SM = Result.Context->getSourceManager();
  const auto NameBuilder = [](const std::string &a,
      const std::string &b, const std::string &ThreadInfoName) {
    return ThreadInfoName + "_" + a + "_" + b;
  };
  if (ME &&
      (Context.Kernels.first == FName || Context.Kernels.second == FName)) {
    if (KernelInfoNameMap.find(FName) == KernelInfoNameMap.end()) {
      KernelInfoNameMap[FName] = std::to_string(Idx++);
      const auto Stmts = std::accumulate(
          MemberNameMapping.begin(),
          MemberNameMapping.end(),
          std::string(""),
          [&NameBuilder, &FName, this]
              (std::string a, std::pair<std::string, std::string> b) {
            const auto NewNameThread =
                NameBuilder(b.second, KernelInfoNameMap[FName], ThreadIdx);
            const auto NewNameBlock =
                NameBuilder(b.second, KernelInfoNameMap[FName], BlockDim);
            auto BaseStrThread = "unsigned int " + NewNameThread + " = ";
            auto BaseStrBlock = "unsigned int " + NewNameBlock + " = ";
            if (b.second == Context.Dimension) {
              if (FName == Context.Kernels.second) {
                BaseStrThread += ThreadIdx + "."
                    + b.second + " - " + std::to_string(Context.Offset);
                BaseStrBlock += BlockDim + "."
                    + b.second + " - " + std::to_string(Context.Offset);
              } else {
                BaseStrBlock += std::to_string(Context.Offset);
                BaseStrThread += ThreadIdx + "." + b.second;
              }
            }
            else {
              BaseStrBlock += BlockDim + "." + b.second;
              BaseStrThread += ThreadIdx + "." + b.second;
            }
            return std::move(a + BaseStrBlock + ";\n" + BaseStrThread + ";\n");
          });
      const tooling::Replacement Replacement(SM,
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
                                     KernelInfoNameMap[FName],
                                     ThreadInfoName);
    const auto CSR = CharSourceRange::getTokenRange(ME->getSourceRange());
    const tooling::Replacement Replacement(SM, CSR, NewName);
    if (auto Err =
        Replacements[Replacement.getFilePath().str()].add(Replacement)) {
      outs() << "error?";
    }
  }
}
}


