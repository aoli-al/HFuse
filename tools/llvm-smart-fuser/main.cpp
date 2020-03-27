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
#include <llvm/Support/FileSystem.h>

#include "KernelFusion.h"
#include "KernelFuseTool.h"
#include "ThreadInfoRewriter.h"
#include "ParameterCollector.h"
#include "MacroExpand.h"
#include "BarrierHoisting.h"
#include "BarrierRewriter.h"

#include <set>
#include <utility>
#include <fstream>
#include <thread>
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

std::vector<std::string> fuseKernel(tooling::CommonOptionsParser &Op,
                                           Context &Context) {
  KernelFuseTool FuseTool(Context);
  ast_matchers::MatchFinder KernelFinder;
  KernelFinder.addMatcher(KernelFuseMatcher, &FuseTool);
  for (const auto &K: Context.Kernels) {
    KernelFinder.addMatcher(
        barrierMatcherFactory(hasName(K.first)), &FuseTool);
  }
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  Tool.run(tooling::newFrontendActionFactory(&KernelFinder).get());
  return FuseTool.GetResults();
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

bool barrierRewriter(tooling::CommonOptionsParser &Op, Context &C) {
  tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
  ast_matchers::MatchFinder Finder;
  BarrierRewriter Rewriter(Tool.getReplacements(), C);
  for (const auto &K: C.Kernels) {
    Finder.addMatcher(
        barrierMatcherFactory(hasName(K.first)), &Rewriter);
  }
  applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
  return Rewriter.hasBarrier();
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


static std::vector<std::pair<std::vector<bool>, std::string>> Presets= {
    std::make_pair(std::vector({true, false, false}), "vfuse"),
    std::make_pair(std::vector({true, false, true}), "vfuse_lb"),
    std::make_pair(std::vector({false, true, false}), "hfuse_bar_sync"),
    std::make_pair(std::vector({false, false, false}), "hfuse"),
    std::make_pair(std::vector({false, false, true}), "hfuse_lb"),
    std::make_pair(std::vector({false, true, true}), "hfuse_lb_bar_sync"),
};

void Fuse(int argc, const char **argv,
    const std::map<std::string, KernelInfo> &KernelInfo,
    const FusionInfo &Fusion) {
  std::string BasePath = argv[3];
  int NArgc = argc - 2;
  const char **NArgv = (const char **)malloc(sizeof(char *) * (NArgc));
  NArgv[0] = argv[0];
  for (unsigned i = 4; i < argc; i++) {
    NArgv[i-2] = argv[i];
  }

  std::string Path = BasePath + "/" + Fusion.File;
  std::vector<kernel_fusion::KernelInfo> Infos;
  for (const auto KName: Fusion.Kernels) {
    Infos.push_back(KernelInfo.at(KName));
  }

  NArgv[1] = Path.c_str();
  tooling::CommonOptionsParser Op(NArgc, NArgv, KernelFuseCategory);
  for (const auto &S : Op.getSourcePathList()) {
    llvm::outs() << S << "\n";
    llvm::sys::fs::copy_file(S, S + ".bak");
  }

  std::vector<std::string> Results;
  for (const auto &Preset : Presets) {
    Context C(Infos, Preset.first, Preset.second);
    expandMacros(Op, C);
    renameParameters(Op, C);
    rewriteThreadInfo(Op, C);
    declRewriter(Op, C);
    if (C.IsBarSyncEnabled) {
      bool HasBarrier = barrierRewriter(Op, C);
      if (!HasBarrier) {
        for (const auto &S : Op.getSourcePathList()) {
          llvm::sys::fs::copy_file(S + ".bak", S);
        }
        continue;
      }
    }
    if (!C.BaseLine) {
      barrierAnalyzer(Op, C);
    }
    const auto &R = fuseKernel(Op, C);
    Results.insert(Results.end(), R.begin(), R.end());
    for (const auto &S : Op.getSourcePathList()) {
      llvm::sys::fs::copy_file(S + ".bak", S);
    }
  }

  std::string FName;
  for (auto &K : Infos) {
    FName += K.KernelName + "_";
  }
  FName += ".inc";
  std::ofstream Of;
  Of.open(FName);
  for (const auto &R : Results) {
    Of << R;
  }
  Of.close();

}


int main(int argc, const char** argv){

  assert(argc >= 4);

  const auto FusionInfo =
      readYAMLInfo<std::vector<kernel_fusion::FusionInfo>>(argv[1]);
  const auto KernelInfo =
      readYAMLInfo<std::map<std::string, kernel_fusion::KernelInfo>>(argv[2]);

  std::vector<std::thread> FusionThreads;

  // We can't do multi-thread here...
  for (const auto &Fusion: FusionInfo) {
    std::thread T(Fuse, argc, argv, KernelInfo, Fusion);
    T.join();
  }
  return 0;
}

