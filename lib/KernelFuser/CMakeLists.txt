set(LLVM_LINK_COMPONENTS support)

add_library(LLVMKernelFuser
        KernelFusion.cpp
        BarrierHoisting.cpp
        KernelPrinter.cpp
        KernelFuseTool.cpp
        ParameterCollector.cpp
        ThreadInfoRewriter.cpp
        MacroExpand.cpp
        DeclRewriter.cpp
        BarrierAnalyzer.cpp
        KernelRecursiveFuser.cpp
        BarrierRewriter.cpp)

target_include_directories(LLVMKernelFuser PUBLIC
        ${LLVM_INCLUDE_DIRS}
        ${CLANG_INCLUDE_DIRS})

target_link_libraries(LLVMKernelFuser
        clangAST
        clangASTMatchers
        clangTooling
        clangToolingCore
        clangToolingRefactoring)
