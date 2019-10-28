//
// Created by Leo Li on 2019-10-26.
//

#ifndef SMART_FUSER_PARAMETERCOLLECTOR_H
#define SMART_FUSER_PARAMETERCOLLECTOR_H

#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/Tooling/Execution.h>
#include <string>
#include <vector>

using namespace clang;

namespace kernel_fusion {
static const std::string ParameterCollectorBindId = "parameter-collector";
static const ast_matchers::DeclarationMatcher ParameterMatcher =
    ast_matchers::functionDecl(ast_matchers::hasAttr(attr::CUDAGlobal))
        .bind(ParameterCollectorBindId);


class ParameterCollector: public ast_matchers::MatchFinder::MatchCallback {
public:
  ParameterCollector() = default;
  void run(const ast_matchers::MatchFinder::MatchResult &Result) override;
public:
  std::map<std::string, std::vector<unsigned>> ParameterMap;
  std::vector<unsigned> ParameterList;
};

}

#endif // SMART_FUSER_PARAMETERCOLLECTOR_H
