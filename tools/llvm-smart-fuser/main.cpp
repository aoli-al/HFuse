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

#include <unistd.h>
#include <sys/wait.h>
#include <set>
#include <utility>
#include <fstream>
#include <thread>
#include <algorithm>
#include <DeclRewriter.h>

using namespace llvm;
using namespace clang;
using namespace kernel_fusion;



namespace  {

static cl::OptionCategory KernelFuseCategory("kernel-fusion option");

class FuseInstance {
private:
  std::set<std::string> ModifiedFiles;
  tooling::CommonOptionsParser &Op;
  kernel_fusion::Context &Context;

  void backupFile(const std::string &File) {
    if (ModifiedFiles.find(File) == ModifiedFiles.end()) {
      llvm::sys::fs::copy_file(File, File + ".bak");
      ModifiedFiles.insert(File);
    }
  }
public:
  FuseInstance(tooling::CommonOptionsParser &Op, kernel_fusion::Context &Context): Op(Op),
                                                                                   Context(Context) {}

  void recoverFiles() {
    for (const auto &File: ModifiedFiles) {
      llvm::sys::fs::copy_file(File + ".bak", File);
      llvm::sys::fs::remove(File + ".bak");
    }
  }

  void applyRewrites(tooling::RefactoringTool &Tool,
                     std::unique_ptr<tooling::FrontendActionFactory> Factory) {
    if (auto Result = Tool.run(Factory.get())) {
      exit(Result);
    }
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    DiagnosticsEngine Diagnostics(
        IntrusiveRefCntPtr<DiagnosticIDs>(new DiagnosticIDs()), &*DiagOpts,
        new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts), true);
    SourceManager Sources(Diagnostics, Tool.getFiles());

    // Apply all replacements to a rewriter.
    Rewriter Rewrite(Sources, LangOptions());
    for (auto R: Tool.getReplacements()) {
      backupFile(R.first);
    }
    Tool.applyAllReplacements(Rewrite);
    Rewrite.overwriteChangedFiles();
  }

  void expandMacros() {
    tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
    ast_matchers::MatchFinder Finder;
    MacroExpand MacroExpand(Tool.getReplacements(), Context);
    Finder.addMatcher(KernelFuseMatcher, &MacroExpand);
    applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
  }

  void renameParameters() {
    tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());

    ParameterCollector Collector(Context);

    ast_matchers::MatchFinder ParameterFinder;
    ParameterFinder.addMatcher(ParameterMatcher, &Collector);
    for (const auto &K : Context.Kernels) {
      ParameterFinder.addMatcher(declStmtMatcherFactory(hasName(K.first)),
                                 &Collector);
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
    for (auto R: Tool.getReplacements()) {
      backupFile(R.first);
    }
    Tool.runAndSave(tooling::newFrontendActionFactory(&Action).get());
  }

  std::vector<std::string> fuseKernel() {
    KernelFuseTool FuseTool(Context);
    ast_matchers::MatchFinder KernelFinder;
    KernelFinder.addMatcher(KernelFuseMatcher, &FuseTool);
    for (const auto &K : Context.Kernels) {
      KernelFinder.addMatcher(barrierMatcherFactory(hasName(K.first)),
                              &FuseTool);
    }
    tooling::ClangTool Tool(Op.getCompilations(), Op.getSourcePathList());
    Tool.run(tooling::newFrontendActionFactory(&KernelFinder).get());
    return FuseTool.GetResults();
  }

  void rewriteThreadInfo() {
    tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
    ast_matchers::MatchFinder Finder;
    ThreadInfoRewriter ThreadInfoRewriter(Tool.getReplacements(), Context);
    Finder.addMatcher(ThreadInfoMatcher, &ThreadInfoRewriter);
    applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
  }

  void barrierAnalyzer() {
    tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
    ast_matchers::MatchFinder Finder;
    BarrierHoisting Hoisting(Tool.getReplacements(), Context);
    for (const auto &K : Context.Kernels) {
      Finder.addMatcher(barrierMatcherFactory(hasName(K.first)), &Hoisting);
    }
    applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
  }

  bool barrierRewriter() {
    tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
    ast_matchers::MatchFinder Finder;
    BarrierRewriter Rewriter(Tool.getReplacements(), Context);
    for (const auto &K : Context.Kernels) {
      Finder.addMatcher(barrierMatcherFactory(hasName(K.first)), &Rewriter);
    }
    applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
    return Rewriter.hasBarrier();
  }

  void declRewriter() {
    tooling::RefactoringTool Tool(Op.getCompilations(), Op.getSourcePathList());
    ast_matchers::MatchFinder Finder;
    DeclRewriter Rewriter(Tool.getReplacements(), Context);
    for (const auto &K : Context.Kernels) {
      Finder.addMatcher(declStmtMatcherFactory(hasName(K.first)), &Rewriter);
    }
    applyRewrites(Tool, tooling::newFrontendActionFactory(&Finder));
  }
};

}


static std::vector<std::pair<std::vector<bool>, std::string>> Presets= {
    std::make_pair(std::vector({true, false, true, false}), "vfuse_lb"),
    std::make_pair(std::vector({true, false, false, false}), "vfuse"),
//    std::make_pair(std::vector({false, true, false, false}), "hfuse_bar_sync"),
    std::make_pair(std::vector({false, false, false, true}), "hfuse"),
    std::make_pair(std::vector({false, false, true, true}), "hfuse_lb"),
//    std::make_pair(std::vector({false, true, true, false}), "hfuse_lb_bar_sync"),
//    std::make_pair(std::vector({false, true, false, true}), "hfuse_bar_sync_imba"),
//    std::make_pair(std::vector({false, false, false, true}), "hfuse_imba"),
//    std::make_pair(std::vector({false, false, true, true}), "hfuse_lb_imba"),
//    std::make_pair(std::vector({false, true, true, true}), "hfuse_lb_bar_sync_imba"),
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
  for (const auto &KName: Fusion.Kernels) {
    Infos.push_back(KernelInfo.at(KName));
  }

  NArgv[1] = Path.c_str();
  std::vector<std::string> Results;
  tooling::CommonOptionsParser Op(NArgc, NArgv, KernelFuseCategory);
  for (const auto &Preset : Presets) {
    auto SplitStart = 128;
    auto SplitEnd = 0;
    if (Preset.first[0]) {
      SplitEnd = 129;
    } else {
      SplitEnd = Infos[0].BlockDim.size() + Infos[1].BlockDim.size();
    }
    for (unsigned Split = SplitStart; Split < SplitEnd; Split += SplitStart) {
      Infos[0].ExecTime = Split;
      Infos[1].ExecTime = SplitEnd - Split;
      Context C(Infos, Preset.first, Preset.second + "_idx_" +
          std::to_string(Split / SplitStart - 1));
      FuseInstance I(Op, C);
      I.expandMacros();
      I.renameParameters();
      I.rewriteThreadInfo();
      I.barrierRewriter();
      const auto &R = I.fuseKernel();
      Results.insert(Results.end(), R.begin(), R.end());
      I.recoverFiles();
    }
//    if (C.ImBalancedThread) {
//      if (std::any_of(C.Kernels.begin(), C.Kernels.end(), [](auto I) {
//        return I.second.ExecTime < 0;
//      })) {
//        continue;
//      }
//    }

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

  std::vector<int> FusionProcesses;

  // We can't do multi-thread here...
  for (const auto &Fusion: FusionInfo) {
    int id = fork();
    if (id == 0) {
      Fuse(argc, argv, KernelInfo, Fusion);
      break;
    } else {
      int status;
      waitpid(id, &status, 0);
      FusionProcesses.push_back(id);
    }
  }
  for (auto &P: FusionProcesses) {
    int status;
    waitpid(P, &status, 0);
  }
  return 0;
}

