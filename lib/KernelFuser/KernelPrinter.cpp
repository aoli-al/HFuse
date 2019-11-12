#include "KernelPrinter.h"

using namespace clang;

namespace kernel_fusion {

raw_ostream &KernelPrinter::Indent(unsigned Indentation) {
  for (unsigned i = 0; i != Indentation; ++i)
    Out << "  ";
  return Out;
}

void KernelPrinter::printFusedTemplateDecl(KFMap &KernelFunctionMap) {
  bool TemplatePrinted = false;
  for (auto &FName: KFContext.Order) {
    const auto *F = KernelFunctionMap[FName];
    if (const auto *TF = F->getDescribedFunctionTemplate()) {
      if (!TemplatePrinted) {
        TemplatePrinted = true;
        Out << "template <";
      } else {
        Out << ", ";
      }
      printTemplateParameters(TF->getTemplateParameters());
    }
  }
  if (TemplatePrinted) Out << ">\n";
}

void KernelPrinter::printFusedFunction(KFMap &KernelFunctionMap) {
  printFusedTemplateDecl(KernelFunctionMap);
  printFusedFunctionSignature(KernelFunctionMap);

}

void KernelPrinter::printFusedFunctionSignature(KFMap &KernelFunctionMap) {
  std::string Proto = "void ";
  for (auto &FName: KFContext.Order) {
    Proto += FName + "_";
  }

  PrintingPolicy SubPolicy(Policy);
  SubPolicy.SuppressSpecifiers = false;
  const auto PrintParameters = [&SubPolicy, &Proto, this](FunctionDecl *D) {
    QualType Ty = D->getType();

    if (const auto *AFT = Ty->getAs<FunctionType>()) {
      const FunctionProtoType *FT = nullptr;
      if (D->hasWrittenPrototype())
        FT = dyn_cast<FunctionProtoType>(AFT);

      if (FT) {
        llvm::raw_string_ostream POut(Proto);
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i)
            POut << ", ";
          D->getParamDecl(i)->print(POut, Policy);
        }

        if (FT->isVariadic()) {
          if (D->getNumParams())
            POut << ", ";
          POut << "...";
        }
      } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i)
            Proto += ", ";
          Proto += D->getParamDecl(i)->getNameAsString();
        }
      }
    }
  };
  Proto += "(";
  for (auto &FName: KFContext.Order) {
    PrintParameters(KernelFunctionMap[FName]);
    if (FName != *KFContext.Order.rbegin()) {
      Proto += ", ";
    }
  }
  Proto += ")";
  Out << Proto;

  prettyPrintAttributes(KernelFunctionMap.begin()->second);

//  Out << " {\n";
//  for (const auto &K: KFContext.Kernels) {
//    if (!K.second.HasBarriers) {
//      Indent(1);
//      Out << branchingStatement(KFContext, K.first);
//    }
//    KernelFunctionMap[K.first]->getBody()->printPretty(
//        Out, nullptr, SubPolicy, Indentation+1);
//  }
//  Out << "}\n";
}

void KernelPrinter::prettyPrintAttributes(Decl *D) {
  if (Policy.PolishForDeclaration)
    return;

  if (D->hasAttrs()) {
    AttrVec &Attrs = D->getAttrs();
    for (auto *A : Attrs) {
      if (A->isInherited() || A->isImplicit())
        continue;
      switch (A->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case attr::X:
#include "clang/Basic/AttrList.inc"
        break;
      default:
        A->printPretty(Out, Policy);
        break;
      }
    }
  }
}

void KernelPrinter::printDeclType(QualType T, StringRef DeclName, bool Pack) {
  // Normally, a PackExpansionType is written as T[3]... (for instance, as a
  // template argument), but if it is the type of a declaration, the ellipsis
  // is placed before the name being declared.
  if (auto *PET = T->getAs<PackExpansionType>()) {
    Pack = true;
    T = PET->getPattern();
  }
  T.print(Out, Policy, (Pack ? "..." : "") + DeclName, Indentation);
}

void KernelPrinter::printStmt(Stmt *S) {
  S->printPretty(Out, nullptr, Policy);
  if (isa<Expr>(S)) {
    Out << ";\n" ;
  }
}


void KernelPrinter::printTemplateParameters(const TemplateParameterList *Params) {
  assert(Params);

  bool NeedComma = false;
  for (const Decl *Param : *Params) {
    if (Param->isImplicit())
      continue;

    if (NeedComma)
      Out << ", ";
    else
      NeedComma = true;

    if (auto TTP = dyn_cast<TemplateTypeParmDecl>(Param)) {

      if (TTP->wasDeclaredWithTypename())
        Out << "typename";
      else
        Out << "class";

      if (TTP->isParameterPack())
        Out << " ...";
      else if (!TTP->getName().empty())
        Out << ' ';

      Out << *TTP;

      if (TTP->hasDefaultArgument()) {
        Out << " = ";
        Out << TTP->getDefaultArgument().getAsString(Policy);
      };
    } else if (auto NTTP = dyn_cast<NonTypeTemplateParmDecl>(Param)) {
      StringRef Name;
      if (IdentifierInfo *II = NTTP->getIdentifier())
        Name = II->getName();
      printDeclType(NTTP->getType(), Name, NTTP->isParameterPack());

      if (NTTP->hasDefaultArgument()) {
        Out << " = ";
        NTTP->getDefaultArgument()->printPretty(Out, nullptr, Policy,
                                                Indentation);
      }
    } else if (auto TTPD = dyn_cast<TemplateTemplateParmDecl>(Param)) {
      TTPD->print(Out, Policy);
      // FIXME: print the default argument, if present.
    }
  }
}


}
