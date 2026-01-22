cmake_minimum_required(VERSION 3.16)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_VERBOSE_MAKEFILE ON)

project(
	boost-multi-adaptors-blis-test
	VERSION 0.1
	LANGUAGES CXX
)

include(FetchContent)

#find_package(Boost REQUIRED COMPONENTS unit_test_framework timer)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

# if(ENABLE_CUDA OR DEFINED CXXCUDA)
# 	enable_language(CUDA)
# 	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# endif()

# find_package(CUDA QUIET)

# if(CUDA_FOUND)
# 	message("CUDA found")
# 	include_directories(${CUDA_INCLUDE_DIRS})
# else()
# 	message("CUDA not found")
# endif()

enable_testing()
# list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure") # needs cmake 3.17
include(CTest)

# configure_file("config.hpp.in" ${CMAKE_BINARY_DIR}/config.hpp)

include_directories(${CMAKE_BINARY_DIR})

include_directories(/home/correaa/include /home/correaa/tblis/src/external/tci)
#link_libraries(/home/correaa/lib/libtblis.so)

set(OpenMP_C_FLAGS "-Xclang;-fopenmp" CACHE STRING "" FORCE)
set(OpenMP_CXX_FLAGS "-Xclang;-fopenmp" CACHE STRING "" FORCE)

set(OpenMP_C_LIB_NAMES "omp" CACHE STRING "" FORCE)
set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)

#find_package(tblis REQUIRED)
FetchContent_Declare(
    tblis
    GIT_REPOSITORY https://github.com/devinamatthews/tblis.git
    GIT_TAG        master   # or a specific commit / tag
)

FetchContent_MakeAvailable(tblis)

target_link_libraries(tblis PRIVATE $<$<CXX_COMPILER_ID:GNU>:atomic>)  # for macos

# /home/correaa/lib/libtci.a libhwloc.a) find_library(TBLIS_LIBRARY NAMES tblis HINTS) ink_libraries(TBLIS_LIBRARY::CC)

# file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
set(TEST_SRCS tensor.cpp)

foreach(TEST_FILE ${TEST_SRCS})
	set(TEST_EXE "${TEST_FILE}.x")
	add_executable(${TEST_EXE} ${TEST_FILE})
	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
		# target_compile_options(${TEST_EXE} PRIVATE -std=c++17)
	endif()
	# target_compile_features   (${TEST_EXE} PUBLIC cxx_std_17)
	# target_compile_definitions(${TEST_EXE} PRIVATE "BOOST_PP_VARIADICS")
	target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS})
	target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

	target_link_libraries(${TEST_EXE} PRIVATE multi)
	target_link_libraries(${TEST_EXE} SYSTEM PRIVATE tblis)

	target_link_libraries(${TEST_EXE} PRIVATE ${Boost_LIBRARIES})
	target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

	if(NOT ENABLE_CUDA)
		target_compile_options(
			${TEST_EXE}
			PRIVATE -Werror
					-fno-common
					$<$<CXX_COMPILER_ID:GNU>:
					-Wformat-truncation
					-fstack-usage> # -Wconversion
					$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
					-Wmove>
					$<$<CXX_COMPILER_ID:Intel>:
					-wd161
					-diag-disable=remark
					-Warray-bounds
					-Wchar-subscripts
					-Wcomment
					-Wenum-compare
					-Wformat
					-Wuninitialized
					-Wmaybe-uninitialized
					-Wmain
					-Wnarrowing
					-Wnonnull
					-Wparentheses
					-Wpointer-sign
					-Wreorder
					-Wno-return-type
					-Wsign-compare
					-Wsequence-point
					-Wtrigraphs
					-Wunused-function
					-Wunused-but-set-variable
					-Wunused-variable
					-Wwrite-strings
					-Werror
					-diag-error:3846
					>
					$<$<CXX_COMPILER_ID:MSVC>:
					#/W4
					>
		)
	endif()
	add_test(NAME ${TEST_EXE} COMMAND $<TARGET_FILE:${TEST_EXE}>)
endforeach()
