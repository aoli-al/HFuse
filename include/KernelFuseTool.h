//
// Created by Leo Li on 2019-10-24.
//

#ifndef SMART_FUSER_KERNELFUSETOOL_H
#define SMART_FUSER_KERNELFUSETOOL_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include "KernelFusion.h"

using namespace clang;

namespace kernel_fusion {

static const std::string KernelFuseBindId = "kernel-fuse-bind";
static const ast_matchers::DeclarationMatcher KernelFuseMatcher =
    ast_matchers::functionDecl(ast_matchers::hasAttr(attr::CUDAGlobal))
        .bind(KernelFuseBindId);

class KernelFuseTool: public ast_matchers::MatchFinder::MatchCallback {
public:
  explicit KernelFuseTool(const Context &Context) : Context(Context) {}
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  void onEndOfTranslationUnit() override;
private:
  void fuseKernel(FunctionDecl *FunctionA, FunctionDecl *FunctionB);

  const Context &Context;
  std::map<StringRef, FunctionDecl *> KernelFunctionMap;
};
}

#endif // SMART_FUSER_KERNELFUSETOOL_H
