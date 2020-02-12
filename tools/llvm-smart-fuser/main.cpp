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
#include "BarrierRewriter.h"

#include <set>
#include <DeclRewriter.h>

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
  for (const auto &K: Context.Kernels) {
    ParameterFinder.addMatcher(
        declStmtMatcherFactory(hasName(K.first)), &Collector);
  }
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
  for (const auto &K: Context.Kernels) {
    KernelFinder.addMatcher(
        barrierMatcherFactory(hasName(K.first)), &FuseTool);
  }
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
  for (const auto &K: C.Kernels) {
    Finder.addMatcher(
        barrierMatcherFactory(hasName(K.first)), &Hoisting);
  }
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
}

void barrierRewriter(tooling::CommonOptionsParser &Op, Context &C) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  BarrierRewriter Rewriter(Tool.getReplacements(), C);
  for (const auto &K: C.Kernels) {
    Finder.addMatcher(
        barrierMatcherFactory(hasName(K.first)), &Rewriter);
  }
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
}


void declRewriter(tooling::CommonOptionsParser &Op, Context &C) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  DeclRewriter Rewriter(Tool.getReplacements(), C);
  for (const auto &K: C.Kernels) {
    Finder.addMatcher(
        declStmtMatcherFactory(hasName(K.first)), &Rewriter);
  }
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
}

}


static cl::opt<std::string> Config("config", cl::desc("YAML file of the configurations of kernels."),
                                   cl::Required, cl::cat(KernelFuseCategory));

int main(int argc, const char** argv){
  tooling::CommonOptionsParser Op(argc, argv, KernelFuseCategory);

  ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer = llvm::MemoryBuffer::getFile(Config);
  if (!Buffer) {
    llvm::errs() << "failed to read configs.\n";
    return 1;
  }

  llvm::yaml::Input YAML(Buffer.get()->getBuffer());

  std::vector<kernel_fusion::KernelInfo> Infos;
  YAML >> Infos;

  Context C(Infos, false);

  expandMacros(Op, C);
  renameParameters(Op, C);
  rewriteThreadInfo(Op, C);
  declRewriter(Op, C);
  if (C.IsBarSyncEnabled) {
    barrierRewriter(Op, C);
  }
  barrierAnalyzer(Op, C);
//  if (!C.BaseLine) {
//  }
  fuseKernel(Op, C);
  return 0;
}

