set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(TorchKernels
        MaxPoolBatchNormFused.cu
        )

target_compile_definitions(TorchKernels PUBLIC __CUDA_NO_HALF_OPERATORS__)
target_link_libraries(TorchKernels ${TORCH_LIBRARIES})
target_include_directories(TorchKernels PUBLIC
        ${TORCH_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/deps/jitify)