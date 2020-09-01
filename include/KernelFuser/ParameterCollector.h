#ifndef SMART_FUSER_PARAMETERCOLLECTOR_H
#define SMART_FUSER_PARAMETERCOLLECTOR_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <string>
#include <vector>

#include "KernelFusion.h"

using namespace clang;

namespace kernel_fusion {
static const std::string ParameterCollectorBindId = "parameter-collector";
static const ast_matchers::DeclarationMatcher ParameterMatcher =
    ast_matchers::functionDecl(ast_matchers::hasAttr(attr::CUDAGlobal))
        .bind(ParameterCollectorBindId);


class ParameterCollector: public ast_matchers::MatchFinder::MatchCallback {
public:
  explicit ParameterCollector(const Context &Context) :
      Context(Context) {};
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
  std::vector<std::string> ParameterList;
  std::vector<std::vector<std::string>> USRList;

public:
  std::set<std::string> VisitedFunctions;
  std::set<int64_t> VisitedDecl;
  const Context &Context;
};

}

#endif // SMART_FUSER_PARAMETERCOLLECTOR_H
