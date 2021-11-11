cmake_minimum_required(VERSION 3.4.3)
project(smart-fuser)

set(CMAKE_CXX_STANDARD 17)

set(LLVM_DIR "/usr/lib/llvm-10/cmake")
set(Clang_DIR "/usr/lib/cmake/clang-10")

find_package(LLVM REQUIRED CONFIG HINTS ${LLVM_DIR})
find_package(Clang REQUIRED CONFIG)
message(STATUS "Using LLVM version ${LLVM_PACKAGE_VERSION}")
list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
list(APPEND CMAKE_MODULE_PATH ${CLANG_CMAKE_DIR})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib)
include(HandleLLVMOptions)
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(include)

add_subdirectory(lib/KernelFuser)

add_subdirectory(tools/llvm-smart-fuser)
