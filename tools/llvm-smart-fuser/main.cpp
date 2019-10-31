#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/Refactoring.h>
#include <clang/Tooling/Refactoring/Rename/USRFindingAction.h>
#include <clang/Tooling/Refactoring/Rename/RenamingAction.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>

#include "KernelFusion.h"
#include "KernelFuseTool.h"
#include "ThreadInfoRewriter.h"
#include "ParameterCollector.h"
#include "MacroExpand.h"
#include "BarrierHoisting.h"

#include <set>

using namespace llvm;
using namespace clang;
using namespace kernel_fusion;


static cl::OptionCategory KernelFuseCategory("kernel-fuse options");

namespace  {

void applyRewrites(tooling::RefactoringTool &Tool,
    std::unique_ptr<tooling::FrontendActionFactory> Factory) {
  if (auto Result =
      Tool.run(Factory.get())) {
    exit(Result);
  }
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
  DiagnosticsEngine Diagnostics(
      IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
      new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts), true);
  SourceManager Sources(Diagnostics, Tool.getFiles());

  // Apply all replacements to a rewriter.
  Rewriter Rewrite(Sources, LangOptions());
  Tool.applyAllReplacements(Rewrite);
  Rewrite.overwriteChangedFiles();
}

void expandMacros(tooling::CommonOptionsParser &Op, const Context &Context) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  MacroExpand MacroExpand(Tool.getReplacements(), Context);
  Finder.addMatcher(KernelFuseMatcher, &MacroExpand);
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
}

void renameParameters(tooling::CommonOptionsParser &Op, const Context &Context) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());

  ParameterCollector Collector(Context);

  ast_matchers::MatchFinder ParameterFinder;
  ParameterFinder.addMatcher(ParameterMatcher, &Collector);
  Tool.run(tooling::newFrontendActionFactory(&ParameterFinder).get());

  const std::vector<std::vector<std::string>> &USRList = Collector.USRList;
  const std::vector<std::string> &PrevNames = Collector.ParameterList;
  std::vector<std::string> NewNames(PrevNames.size());

  int i = 0;
  std::transform(PrevNames.begin(), PrevNames.end(), NewNames.begin(),
                 [&i](std::string name) {
                   return std::move(name + std::to_string(i++));
                 });
  i++;
  tooling::RenamingAction Action(NewNames, PrevNames, USRList,
                                 Tool.getReplacements(), false);
  Tool.runAndSave(tooling::newFrontendActionFactory(&Action).get());
}

void fuseKernel(tooling::CommonOptionsParser &Op, Context &Context) {
  KernelFuseTool FuseTool(Context);
  ast_matchers::MatchFinder KernelFinder;
  KernelFinder.addMatcher(KernelFuseMatcher, &FuseTool);
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  Tool.run(tooling::newFrontendActionFactory(&KernelFinder).get());
}

void rewriteThreadInfo(tooling::CommonOptionsParser &Op, const Context &C) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  ThreadInfoRewriter ThreadInfoRewriter(Tool.getReplacements(), C);
  Finder.addMatcher(ThreadInfoMatcher, &ThreadInfoRewriter);
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
}

void barrierAnalyzer(tooling::CommonOptionsParser &Op, Context &C) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  BarrierHoisting Hoisting(Tool.getReplacements(), C);
  Finder.addMatcher(
      barrierMatcherFactory(hasName(C.Kernels.first), hasName(C.Kernels.second)), &Hoisting);
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
}

}

int main(int argc, const char** argv){
  tooling::CommonOptionsParser Op(argc, argv, KernelFuseCategory);
  Context C {
      {"foo", "bar"},
      "x",
      512,
      {}
  };
  expandMacros(Op, C);
  renameParameters(Op, C);
  barrierAnalyzer(Op, C);
  rewriteThreadInfo(Op, C);
  fuseKernel(Op, C);
  return 0;
}

