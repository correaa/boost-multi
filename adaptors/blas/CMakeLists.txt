cmake_minimum_required(VERSION 3.11)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(boost-multi-adaptors-blas VERSION 0.1 LANGUAGES CXX)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
if(BLAS_FOUND)
	message("MKL environment detected")
	add_definitions(-DRETURN_BY_STACK)
endif()
unset(BLA_VENDOR)

find_package(BLAS REQUIRED)
#find_path(BLAS_INCLUDE_DIRS cblas.h
#  /usr/include
#  /usr/local/include
#  $ENV{BLAS_HOME}/include)


link_libraries(${BLAS_LIBRARIES})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#configure_file("mkl_config.hpp.in" ${CMAKE_BINARY_DIR}/mkl_config.hpp)

include_directories(${CMAKE_BINARY_DIR})

add_subdirectory(test)

