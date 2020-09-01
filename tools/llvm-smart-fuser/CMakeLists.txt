llvm_map_components_to_libnames(LLVM_LIBS_
                                core support irreader passes)

set(SOURCE_FILES
        main.cpp ../../include/KernelFuser/BarrierAnalyzer.h ../../include/KernelFuser/KernelRecursiveFuser.h)
add_executable(llvm-smart-fuser ${SOURCE_FILES})
add_dependencies(llvm-smart-fuser LLVMKernelFuser)
target_link_libraries(llvm-smart-fuser
        ${LLVM_LIBS_}
        LLVMKernelFuser
        clangTooling
        clangToolingCore
        )
