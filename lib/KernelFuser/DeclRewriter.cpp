//
// Created by Leo Li on 2019-11-01.
//

#include "DeclRewriter.h"

namespace kernel_fusion {


void DeclRewriter::run(const MatchFinder::MatchResult &Result) {
  auto Stmt = Result.Nodes.getNodeAs<DeclStmt>(DeclStmtBindId);
  std::string DeclString;
  llvm::raw_string_ostream DeclStream(DeclString);
  if (VisitedDecl.find(Stmt->getBeginLoc().getRawEncoding()) != VisitedDecl.end()) return;
  VisitedDecl.insert(Stmt->getBeginLoc().getRawEncoding());
  for (auto Decl: Stmt->decls()) {
    if (auto VD = dyn_cast<VarDecl>(Decl)) {
      DeclStream << printVarDecl(VD, Result.Context->getPrintingPolicy());
    }
    else {
      Decl->print(DeclStream);
    }
    DeclStream << ";";
  }
  DeclStream.flush();
  const tooling::Replacement R(
      Result.Context->getSourceManager(),
      Stmt,
      DeclString);
  if (Replacements[R.getFilePath().str()].add(R)) {
    llvm::outs() << "error?";
  }
}

std::string DeclRewriter::printVarDecl(VarDecl *D, const PrintingPolicy &Policy) {
  std::string OutStr;
  llvm::raw_string_ostream Out(OutStr);
  if (D->hasAttrs()) {
    AttrVec &Attrs = D->getAttrs();
    for (auto *A : Attrs) {
      switch (A->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case attr::X:
#include "clang/Basic/AttrList.inc"
        A->printPretty(Out, Policy);
        break;
      default:
        break;
      }
    }
  }

  QualType T = D->getType();

  StorageClass SC = D->getStorageClass();
  if (SC != SC_None)
    Out << VarDecl::getStorageClassSpecifierString(SC) << " ";

  switch (D->getTSCSpec()) {
  case TSCS_unspecified:
    break;
  case TSCS___thread:
    Out << "__thread ";
    break;
  case TSCS__Thread_local:
    Out << "_Thread_local ";
    break;
  case TSCS_thread_local:
    Out << "thread_local ";
    break;
  }

  if (D->isModulePrivate())
    Out << "__module_private__ ";

  if (D->isConstexpr()) {
    Out << "constexpr ";
    T.removeLocalConst();
  }
  T.removeLocalConst();
  printDeclType(T, D->getName(), Out, Policy, Policy.Indentation);
  if (Expr *Init = D->getInit()) {
    bool ImplicitInit = false;
    if (CXXConstructExpr *Construct =
        dyn_cast<CXXConstructExpr>(Init->IgnoreImplicit())) {
      if (D->getInitStyle() == VarDecl::CallInit &&
          !Construct->isListInitialization()) {
        ImplicitInit = Construct->getNumArgs() == 0 ||
            Construct->getArg(0)->isDefaultArgument();
      }
    }
    if (!ImplicitInit) {
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << "(";
      else if (D->getInitStyle() == VarDecl::CInit) {
        if (isa<ConstantArrayType>(T.getTypePtr())) {
          Out << " = ";
        } else {
          Out << "; " + D->getName() + " = ";
        }
      }
      PrintingPolicy SubPolicy(Policy);
      SubPolicy.SuppressSpecifiers = false;
      SubPolicy.IncludeTagDefinition = false;
      Init->printPretty(Out, nullptr, SubPolicy, Policy.Indentation);
      if ((D->getInitStyle() == VarDecl::CallInit) && !isa<ParenListExpr>(Init))
        Out << ")";
    }
  }

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
  Out.flush();
  return std::move(OutStr);
}

void DeclRewriter::printDeclType(QualType T, StringRef DeclName, llvm::raw_string_ostream &Out,
    const PrintingPolicy &Policy, unsigned Indentation) {
  bool Pack = false;
  if (auto *PET = T->getAs<PackExpansionType>()) {
    Pack = true;
    T = PET->getPattern();
  }
  if (T->getAs<AutoType>()) {
    auto VT = T.getCanonicalType();
    VT.print(Out, Policy, (Pack ? "..." : "") + DeclName, Indentation);
    return;
  }
  T.print(Out, Policy, (Pack ? "..." : "") + DeclName, Indentation);
}



}