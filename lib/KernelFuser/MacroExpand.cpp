//
// Created by Leo Li on 2019-10-29.
//

#include "MacroExpand.h"
#include "KernelFuseTool.h"

namespace kernel_fusion {

void MacroExpand::run(const ast_matchers::MatchFinder::MatchResult &Result) {

  const auto *D = Result.Nodes.getNodeAs<FunctionDecl>(KernelFuseBindId);
  if (D->isTemplateInstantiation() ||
      !Context.hasKernel(D->getName())) return;
  std::string Impl;
  llvm::raw_string_ostream ImplStream(Impl);
  auto *Body = D->getBody();
  Body->printPretty(ImplStream, nullptr, Result.Context->getPrintingPolicy());
  ImplStream.flush();
  const auto Replacement = tooling::Replacement(Result.Context->getSourceManager(),
                                                Body, Impl);
  std::string Path = D->getLocation().printToString(Result.Context->getSourceManager());
  Path = Path.substr(0, Path.find_first_of(":"));
  if (auto Err = Replacements[Path].add(Replacement)) {
    llvm::outs() << "error";
  }
}
}
