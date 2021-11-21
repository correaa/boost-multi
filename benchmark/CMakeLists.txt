# CXX=g++-9 cmake .. -D CMAKE_BUILD_TYPE=Release -D CMAKE_CUDA_HOST_COMPILER=g++-9
cmake_minimum_required(VERSION 3.10)

project(
	boost-multi-benchmark
	VERSION 0.1
	LANGUAGES CXX CUDA
)

find_package(Boost REQUIRED COMPONENTS serialization unit_test_framework)
find_package(benchmark REQUIRED)
find_package(BLAS REQUIRED)
#find_package(TBB REQUIRED HINTS "/usr")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61 -std=c++17 --use_fast_math --expt-relaxed-constexpr --extended-lambda") # set(CMAKE_CUDA_STANDARD 17)

enable_testing()
include(CTest)

# file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

add_executable(thrust_assignment.cu.x thrust_assignment.cu)
add_test(NAME thrust_assignment.cu.x COMMAND thrust_assignment.cu.x)
target_link_libraries(thrust_assignment.cu.x PRIVATE benchmark::benchmark)

add_executable(matrix_multiplication.cpp.x matrix_multiplication.cpp)
target_compile_options(matrix_multiplication.cpp.x PRIVATE -Wfatal-errors -Ofast)

target_link_libraries(matrix_multiplication.cpp.x benchmark::benchmark)
target_link_libraries(matrix_multiplication.cpp.x -ltbb) #TBB::tbb)
target_link_libraries(matrix_multiplication.cpp.x BLAS::BLAS)

add_test(NAME matrix_multiplication.cpp.x COMMAND matrix_multiplication.cpp.x)
