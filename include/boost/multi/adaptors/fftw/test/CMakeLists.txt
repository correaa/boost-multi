cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

#project(
#   boost-multi-adaptors-fftw-test
#   VERSION 0.1
#   LANGUAGES CXX
#)

#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA OR DEFINED CXXCUDA)
	enable_language(CUDA)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()
endif()

if(ENABLE_CIRCLE)
find_package(BLAS REQUIRED)
find_path(
   BLAS_INCLUDE_DIRS
   cblas.h
   /usr/include
   /usr/local/include
   $ENV{BLAS_HOME}/include
)
include_directories(${TEST_EXE} PRIVATE ${BLAS_INCLUDE_DIRS})
link_libraries(${BLAS_LIBRARIES})
endif()

enable_testing()

include(CTest)

include_directories(${CMAKE_BINARY_DIR})

# file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
set(TEST_SRCS
  combinations.cpp
# copy.cpp
  core.cpp
  fft.cpp
  moved.cpp
  shift.cpp
  so_shift.cpp
# thrust.cpp
  transpose.cpp
  transpose_square.cpp
)

foreach(TEST_FILE ${TEST_SRCS})

	set(TEST_EXE "${TEST_FILE}.x")
	add_executable(${TEST_EXE} ${TEST_FILE})
	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
		# target_compile_options(${TEST_EXE} PRIVATE -std=c++17)
	endif()

	target_include_directories(${TEST_EXE}        PRIVATE ${PROJECT_SOURCE_DIR}/include)
	target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

	target_link_libraries(${TEST_EXE} PRIVATE ${Boost_LIBRARIES})
	target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

	target_link_libraries(${TEST_EXE} PRIVATE multi multi_fftw3)  # PkgConfig::FFTW)

	if(NOT DEFINED ENABLE_CIRCLE)
		target_compile_options(
			${TEST_EXE}
			PRIVATE
				$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CUDA_COMPILER_ID:NVIDIA>>,$<NOT:$<CUDA_COMPILER_ID:Clang>>>:
					-Werror
					-Wall
					-Wextra
					-fno-common
					-Wpedantic
					-Wformat-truncation
					-Wparentheses
					-fstack-usage
				>
				$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
					-Werror
					-Wall
					-Wextra
					-fno-common
					-Wpedantic
					-Wmove
					-Wno-error=\#warnings
				>
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
					/W4
				>
				$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
					#-Xcompiler=-Werror,-Wall,-Wextra,-Wcast-align,-Wcast-qual,-Wdouble-promotion,-Wduplicated-branches,-Wduplicated-cond,-Wformat-truncation,-Wformat=2,-Wlogical-op,-Wmisleading-indentation,-Wno-missing-include-dirs,-Wnon-virtual-dtor,-Wno-missing-declarations,-Wnon-virtual-dtor,-Wnull-dereference,-Woverloaded-virtual,-Wpointer-arith,-Wno-redundant-decls,-Wno-shadow,-Wno-switch-enum,-Wno-unknown-pragmas,-Wtrampolines,-Wuninitialized,-Wunused,-Wunused-but-set-variable,-Wunused-result,-Wno-zero-as-null-pointer-constant
					#--Werror=cross-execution-space-call
					#--Werror ext-lambda-captures-this
					#--use_fast_math
					#--extra-device-vectorization
					#--restrict
					-Xcudafe=--display_error_number -Xcudafe=--diag_suppress=20011 -Xcudafe=--diag_suppress=20014 -Xcudafe=--diag_suppress=20011 -Xcudafe=--diag_error=incompatible_assignment_operands -Xcudafe=--diag_error=returning_ptr_to_local_variable -Xcudafe=--diag_error=subscript_out_of_range -Xcudafe=--diag_error=used_before_set -Xcudafe=--diag_error=undefined_preproc_id -Xcudafe=--diag_error=implicit_func_decl -Xcudafe=--diag_error=implicit_return_from_non_void_function -Xcudafe=--diag_error=missing_type_specifier
				>
		)
	endif()

	add_test(NAME ${TEST_EXE} COMMAND $<TARGET_FILE:${TEST_EXE}>)
endforeach()
