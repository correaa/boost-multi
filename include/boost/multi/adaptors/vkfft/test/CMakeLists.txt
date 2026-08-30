cmake_minimum_required(VERSION 3.24)

project(
	boost-multi-adaptors-vkfft-test
	VERSION 0.1
	LANGUAGES CXX
)

# multi::vkfft (VkFFT headers + backend libs + VKFFT_BACKEND) comes from the parent
# CMakeLists.  Pick the source language from the enabled GPU toolchain.
if(ENABLE_CUDA OR DEFINED CXXCUDA)
	set(_vkfft_lang CUDA)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()
elseif(ENABLE_HIP)
	set(_vkfft_lang HIP)
else()
	return()
endif()
enable_language(${_vkfft_lang})

find_package(Boost CONFIG)

# FFTW: the CPU cross-check reference for the tests, and the flop-count source for
# the benchmark (multi::fftw::plan::flops()).
find_package(PkgConfig)
if(NOT PKG_CONFIG_FOUND)
	message(WARNING "pkg-config not found (needed to locate FFTW); skipping vkfft tests")
	return()
endif()
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)  # sudo apt install libfftw3-dev

enable_testing()
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
include(CTest)

include_directories(${CMAKE_BINARY_DIR})

set(TEST_SRCS vkfft.cpp)

foreach(TEST_FILE ${TEST_SRCS})
	set(TEST_EXE "${TEST_FILE}.x")
	add_executable(${TEST_EXE} ${TEST_FILE})
	set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE ${_vkfft_lang})

	target_compile_definitions(${TEST_EXE} PRIVATE VKFFT_MAX_FFT_DIMENSIONS=4 ${Boost_DEFINITIONS})
	target_include_directories(${TEST_EXE} PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

	target_link_libraries(${TEST_EXE} PRIVATE multi::vkfft PkgConfig::FFTW ${Boost_LIBRARIES})

	add_test(NAME ${TEST_EXE} COMMAND $<TARGET_FILE:${TEST_EXE}>)
endforeach()

# Throughput comparison of cuFFT vs VkFFT through the adaptors (CUDA backend only).
# Links the FFTW CPU adaptor too -- used only to count each transform's flops via
# fftw_flops().  Plain executable, not a ctest -- run it by hand:
#   cmake --build <dir> --target vkfft-benchmark.x
#   <dir>/include/boost/multi/adaptors/vkfft/test/vkfft-benchmark.x > bench.csv
if(_vkfft_lang STREQUAL CUDA)
	find_package(CUDAToolkit REQUIRED)
	add_executable(vkfft-benchmark.x ${CMAKE_CURRENT_SOURCE_DIR}/../benchmark/fft.cpp)
	set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../benchmark/fft.cpp PROPERTIES LANGUAGE CUDA)
	set_target_properties(vkfft-benchmark.x PROPERTIES EXCLUDE_FROM_ALL TRUE)

	target_compile_definitions(vkfft-benchmark.x PRIVATE VKFFT_MAX_FFT_DIMENSIONS=4 NDEBUG)
	target_compile_options(vkfft-benchmark.x PRIVATE $<$<CONFIG:Debug>:-O2>)
	target_include_directories(vkfft-benchmark.x PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_libraries(vkfft-benchmark.x PRIVATE multi::vkfft CUDA::cufft PkgConfig::FFTW)
endif()
