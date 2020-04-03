#ifndef SMART_FUSER_INCLUDE_KERNELFUSION_H
#define SMART_FUSER_INCLUDE_KERNELFUSION_H

#include <string>
#include <map>
#include <llvm/Support/YAMLTraits.h>
#include <clang/AST/ASTContext.h>

namespace kernel_fusion {

struct BlockDim {
  unsigned X;
  unsigned Y;
  unsigned Z;

  [[nodiscard]] unsigned size() const {
    return X * Y * Z;
  }
};

struct KernelInfo {
  std::string KernelName;
  bool HasBarriers;
  BlockDim BlockDim;
  unsigned Reg;
  double ExecTime;
};

struct FusionInfo {
  std::string File;
  std::vector<std::string> Kernels;
};


struct Context {
  std::map<std::string, KernelInfo> Kernels;
  std::vector<std::string> Order;
  std::map<std::string, std::pair<unsigned, unsigned>> Bounds;
  bool BaseLine;
  bool IsBarSyncEnabled;
  bool LaunchBound;
  bool ImBalancedThread;
  std::string Name;

  explicit Context(const std::vector<KernelInfo> &Infos, bool BaseLine,
                   bool IsBarSyncEnabled, bool LaunchBound,
                   bool ImBalancedThread, std::string Name):
      BaseLine(BaseLine),
      IsBarSyncEnabled(IsBarSyncEnabled),
      LaunchBound(LaunchBound),
      ImBalancedThread(ImBalancedThread),
      Name(std::move(Name)) {
    unsigned B = 0;
    double TotalTime = 0.;
    unsigned TotalThread = 0;
    for (auto &Info: Infos) {
      TotalTime += Info.ExecTime;
      TotalThread += Info.BlockDim.size();
    }
    unsigned AllocatedThreads = 0;
    for (unsigned I = 0; I < Infos.size(); I++) {
      auto Info = Infos[I];
      if (ImBalancedThread) {
        unsigned Thread = 0;
        if (I == Infos.size() - 1) {
          Thread = TotalThread - AllocatedThreads;
        } else {
          Thread = int(Info.ExecTime / TotalTime * TotalThread / 32) * 32;
          AllocatedThreads += Thread;
        }
        Info.BlockDim.X = Thread / Info.BlockDim.Y;
      }
      Kernels[Info.KernelName] = Info;
      Order.push_back(Info.KernelName);
      Bounds[Info.KernelName] = std::make_pair(B, B + Info.BlockDim.size());
      if (!BaseLine) {
        B += Info.BlockDim.size();
      }
    }
  }

  explicit Context(const std::vector<KernelInfo> &Infos,
                   const std::vector<bool> &Configs, const std::string &Name="") :
      Context(Infos,
              Configs[0], Configs[1], Configs[2], Configs[3], Name) {
  }

  [[nodiscard]] bool hasKernel(const std::string &KName) const {
    return Kernels.find(KName) != Kernels.end();
  }

};

std::string branchingStatement(const Context &C, const std::string &FName, bool Inverse=false);

std::string generateNewVarName(const std::string &Base);

template <typename NodeT>
std::string getDeclFunctionName(const clang::ASTContext *C, const NodeT &N) {
  auto NodeLists = C->getParents(N);

  while (NodeLists.size()) {
    if (auto N = NodeLists[0].template get<clang::FunctionDecl>()) {
      return N->getName();
    }
    NodeLists = C->getParents(NodeLists[0]);
  }
  return "";
}

template <typename Info>
Info readYAMLInfo(const char *Path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> Buffer =
      llvm::MemoryBuffer::getFile(Path);
  Info Infos;
  if (!Buffer) {
    llvm::errs() << "failed to read configs.\n";
    return std::move(Infos);
  }
  llvm::yaml::Input YAML(Buffer.get()->getBuffer());
  YAML >> Infos;
  return std::move(Infos);
}

const static std::string CurrentTid = "(threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)";

}

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::KernelInfo> {
  static void mapping(IO &Io, kernel_fusion::KernelInfo &Info) {
    Io.mapRequired("KernelName", Info.KernelName);
    Io.mapRequired("HasBarriers", Info.HasBarriers);
    Io.mapRequired("BlockDim", Info.BlockDim);
    Io.mapRequired("Reg", Info.Reg);
    Io.mapOptional("ExecTime", Info.ExecTime, -1);
  }
};

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::FusionInfo> {
  static void mapping(IO &Io, kernel_fusion::FusionInfo &Info) {
    Io.mapRequired("File", Info.File);
    Io.mapRequired("Kernels", Info.Kernels);
  }
};

template <>
struct llvm::yaml::MappingTraits<kernel_fusion::BlockDim> {
  static void mapping(IO &Io, kernel_fusion::BlockDim &Dim) {
    Io.mapRequired("X", Dim.X);
    Io.mapRequired("Y", Dim.Y);
    Io.mapRequired("Z", Dim.Z);
  }
};


LLVM_YAML_IS_STRING_MAP(kernel_fusion::KernelInfo)
LLVM_YAML_IS_SEQUENCE_VECTOR(kernel_fusion::FusionInfo)

#endif //SMART_FUSER_INCLUDE_KERNELFUSION_H
