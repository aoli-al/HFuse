#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <clang/Tooling/Tooling.h>
#include <clang/Tooling/Refactoring.h>
#include <clang/Tooling/Refactoring/Rename/USRFindingAction.h>
#include <clang/Tooling/Refactoring/Rename/RenamingAction.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>

#include "KernelFuseTool.h"
#include "ThreadInfoRewriter.h"
#include "ParameterCollector.h"

using namespace llvm;
using namespace clang;
using namespace kernel_fusion;


static cl::OptionCategory KernelFuseCategory("kernel-fuse options");

namespace  {

void renameParameters(tooling::CommonOptionsParser &Op) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());

  ParameterCollector Collector({"upsample_bilinear2d_out_frame"});

  ast_matchers::MatchFinder ParameterFinder;
  ParameterFinder.addMatcher(ParameterMatcher, &Collector);
  Tool.run(tooling::newFrontendActionFactory(&ParameterFinder).get());

  tooling::USRFindingAction FindingAction(Collector.ParameterList, {}, false);
  Tool.run(tooling::newFrontendActionFactory(&FindingAction).get());
  const std::vector<std::vector<std::string>> &USRList =
      FindingAction.getUSRList();
  const std::vector<std::string> &PrevNames = FindingAction.getUSRSpellings();
  std::vector<std::string> NewNames(PrevNames.size());

  if (FindingAction.errorOccurred()) {
    return;
  }
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

void fuseKernel(tooling::CommonOptionsParser &Op) {
  KernelFuseTool FuseTool;
  ast_matchers::MatchFinder KernelFinder;
  KernelFinder.addMatcher(KernelFuseMatcher, &FuseTool);
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  Tool.run(tooling::newFrontendActionFactory(&KernelFinder).get());
}

void rewriteThreadInfo(tooling::CommonOptionsParser &Op) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  ThreadInfoRewriter ThreadInfoRewriter(Tool.getReplacements(),
                                        "y", "upsample_bilinear2d_out_frame", 1);
  Finder.addMatcher(ThreadInfoMatcher, &ThreadInfoRewriter);
  if (auto Result =
      Tool.run(tooling::newFrontendActionFactory(&Finder).get())) {
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

}

int main(int argc, const char** argv){
  tooling::CommonOptionsParser Op(argc, argv, KernelFuseCategory);
  renameParameters(Op);
  rewriteThreadInfo(Op);
  fuseKernel(Op);
  return 0;
}

