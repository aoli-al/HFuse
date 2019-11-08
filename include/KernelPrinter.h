//
// Created by Leo Li on 2019-10-24.
//

#ifndef SMART_FUSER_KERNELPRINTER_H
#define SMART_FUSER_KERNELPRINTER_H

#include <clang/AST/ASTContext.h>
#include <clang/AST/Attr.h>
#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclObjC.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/DeclVisitor.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/PrettyPrinter.h>
#include <clang/Basic/Module.h>
#include <llvm/Support/raw_ostream.h>

#include "KernelFusion.h"

using namespace clang;

namespace kernel_fusion {
class KernelPrinter {
  PrintingPolicy Policy;
  const ASTContext &Context;
  struct Context &KFContext;
  unsigned Indentation;
  bool PrintInstantiation;
  std::map<StringRef, FunctionDecl *> &KernelFunctionMap;
  raw_ostream& Indent(unsigned Indentation);

  void printFusedTemplateDecl();
  void printFusedFunctionSignature();

public:
  KernelPrinter(raw_ostream &Out, const PrintingPolicy &Policy,
                const ASTContext &Context, struct Context &KFContext,
                std::map<StringRef, FunctionDecl *> &KernelFunctionMap)
      : Out(Out), Policy(Policy), Context(Context), KFContext(KFContext),
        Indentation(0), PrintInstantiation(false), KernelFunctionMap(KernelFunctionMap) {}
  void printFusedFunction();
  void printTemplateParameters(const TemplateParameterList *Params);
  void prettyPrintAttributes(Decl *D);
  void printDeclType(QualType T, StringRef DeclName, bool Pack);
  void printStmt(Stmt *S);

  raw_ostream &Out;
};
}

#endif // SMART_FUSER_KERNELPRINTER_H
