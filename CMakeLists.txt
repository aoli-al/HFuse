cmake_minimum_required(VERSION 3.4.3)

project(smart-fuser LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

set(LLVM_DIST_PATH "${PROJECT_SOURCE_DIR}/deps/llvm-project/build"
        CACHE STRING "LLVM distribution install path")

set(LLVM_DIR ${LLVM_DIST_PATH}/lib/cmake/llvm)
set(Clang_DIR ${LLVM_DIST_PATH}/lib/cmake/clang)

find_package(Torch REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using LLVM version ${LLVM_PACKAGE_VERSION}")
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CLANG_CMAKE_DIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib)
include(HandleLLVMOptions)
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(include)

add_subdirectory(lib)

add_subdirectory(tools/llvm-smart-fuser)