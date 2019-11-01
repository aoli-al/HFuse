set(LLVM_LINK_COMPONENTS support)

add_library(LLVMKernelFuser
        KernelFusion.cpp
        BarrierHoisting.cpp
        KernelPrinter.cpp
        KernelFuseTool.cpp
        ParameterCollector.cpp
        ThreadInfoRewriter.cpp
        MacroExpand.cpp
        DeclRewriter.cpp)

target_link_libraries(LLVMKernelFuser
        clangAST
        clangASTMatchers
        clangTooling
        clangToolingCore
        clangToolingRefactoring)
