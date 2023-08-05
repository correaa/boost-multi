# -*-indent-tabs-mode:nil;c-basic-offset:2;tab-width:4;autowrap:nil;-*-
#[=[Multi Test suite can be run like this:
  mkdir -p build
  cd build
  cmake .. [-DENABLE_CUDA=1]
  make -j
  ctest -j --output-on-error [-T memcheck]
  exit
#]=]
cmake_minimum_required(VERSION 3.11)

project(multi-adaptors-hipfft-test LANGUAGES CXX)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(rocthrust REQUIRED)

enable_language(HIP)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
#include_directories(PkgConfig::FFTW)
#link_libraries(PkgConfig::FFTW)

enable_testing()
include(CTest)

add_executable(hipfft.cpp.x hipfft.cpp)
set_source_files_properties(hipfft.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(hipfft.cpp.x PRIVATE Boost::unit_test_framework PkgConfig::FFTW multi roc::rocthrust)

add_test(hipfft.cpp.x COMMAND ./hipfft.cpp.x)
