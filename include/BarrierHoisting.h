#ifndef SMART_FUSER_INCLUDE_BARRIERHOISTING_H
#define SMART_FUSER_INCLUDE_BARRIERHOISTING_H

#include "BarrierAnalyzer.h"

using namespace clang;
using namespace clang::ast_matchers;

namespace kernel_fusion {

class BarrierHoisting: public BarrierAnalyzer {
public:
  explicit BarrierHoisting(std::map<std::string, tooling::Replacements> &Replacements,
                           struct Context &Context) :
                           Replacements(Replacements), BarrierAnalyzer(Context) {}
  void onEndOfTranslationUnit() override;

private:
  std::map<std::string, tooling::Replacements> &Replacements;

  void createAndInsert(clang::SourceLocation Loc, const std::string &S) {
    const tooling::Replacement Replacement(ASTContext->getSourceManager(), Loc, 0, S);
    if (auto Err =
        Replacements[Replacement.getFilePath().str()].add(Replacement)) {
      llvm::outs() << "error?";
    }
  }
};

}


#endif //SMART_FUSER_INCLUDE_BARRIERHOISTING_H
