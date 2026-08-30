cmake_minimum_required(VERSION 3.24)

project(boost-multi-adaptors-lapack-test-cuda LANGUAGES CXX CUDA)

# GPU (cuSOLVER) tests for the lapack adaptor; unlike the CPU tests, these need neither
# a CPU LAPACK nor ENABLE_CIRCLE, so they are wired directly from the root under ENABLE_CUDA

find_package(Boost REQUIRED COMPONENTS headers)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()
include(CTest)

add_executable(lapack_cusolver.cu.x cusolver.cu)
target_link_libraries(lapack_cusolver.cu.x PRIVATE multi CUDA::cusolver Boost::headers)

add_test(NAME lapack_cusolver.cu.x COMMAND $<TARGET_FILE:lapack_cusolver.cu.x>)
