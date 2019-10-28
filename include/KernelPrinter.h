//
// Created by Leo Li on 2019-10-24.
//

#ifndef SMART_FUSER_KERNELPRINTER_H
#define SMART_FUSER_KERNELPRINTER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Module.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace kernel_fusion {
class KernelPrinter {
  raw_ostream &Out;
  PrintingPolicy Policy;
  const ASTContext &Context;
  unsigned Indentation;
  bool PrintInstantiation;

  raw_ostream& Indent() { return Indent(Indentation); }
  raw_ostream& Indent(unsigned Indentation);

  void printFusedTemplateDecl(FunctionDecl *FA, FunctionDecl *FB);
  void printFusedFunctionSignature(FunctionDecl *FA, FunctionDecl *FB);

public:
  KernelPrinter(raw_ostream &Out, const PrintingPolicy &Policy,
                const ASTContext &Context, unsigned Indentation = 0,
                bool PrintInstantiation = false)
      : Out(Out), Policy(Policy), Context(Context), Indentation(Indentation),
        PrintInstantiation(PrintInstantiation) {}
  void printFusedFunction(FunctionDecl *FA, FunctionDecl *FB);
  void printTemplateParameters(const TemplateParameterList *Params);
  void prettyPrintAttributes(Decl *D);
  void printDeclType(QualType T, StringRef DeclName, bool Pack);
};
}

#endif // SMART_FUSER_KERNELPRINTER_H
