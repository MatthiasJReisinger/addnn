cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
    torch
    URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
)

set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(torch)
set(Torch_DIR ${torch_SOURCE_DIR}/share/cmake/Torch)
message(${Torch_DIR})

project(addnn)

find_package(Torch REQUIRED)

add_executable(torchscript-atomize main.cpp)
target_link_libraries(torchscript-atomize "${TORCH_LIBRARIES}")
set_property(TARGET torchscript-atomize PROPERTY CXX_STANDARD 20)
