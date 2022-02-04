# -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
#[=[Multi Test suite can be run like this:
  mkdir -p build
  cd build
  cmake .. [-DENABLE_CUDA=1]
  make -j
  ctest -j --output-on-error [-T memcheck]
  exit
#]=]
cmake_minimum_required(VERSION 3.11)

if(EXISTS "CMakeLists.txt")
	message(FATAL_ERROR "You probably don't want to run CMake in a directory with CMakeLists.txt")
endif()

project(
	boost-multi-test
	VERSION 0.1
	LANGUAGES CXX
)

find_package(Boost REQUIRED COMPONENTS unit_test_framework serialization)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA OR DEFINED CXXCUDA)
	enable_language(CUDA)
#	set_property(GLOBAL PROPERTY CUDA_STANDARD 17)  # with cmake 3.16 this is not enough, -std=c++17 is needed below
#	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")

	include(FindCUDA/select_compute_arch)
	cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(
		REPLACE " "
				";"
				INSTALLED_GPU_CCS_3
				"${INSTALLED_GPU_CCS_2}"
	)
	string(
		REPLACE "."
				""
				CUDA_ARCH_LIST
				"${INSTALLED_GPU_CCS_3}"
	)
	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

endif()

enable_testing()
include(CTest)

file(
	GLOB TEST_SRCS
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	*.cpp
)

foreach(TEST_FILE ${TEST_SRCS})
	set(TEST_EXE "${TEST_FILE}.x")
	add_executable(${TEST_EXE} ${TEST_FILE})
	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set_property(TARGET ${TEST_EXE} PROPERTY "${CUDA_ARCH_LIST}")
		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
		target_compile_options(${TEST_EXE} PRIVATE
			$<$<CUDA_COMPILER_ID:NVIDIA>:
				-std=c++17
			>
		)
	else()
		target_compile_features(${TEST_EXE} PUBLIC cxx_std_17)
	endif()
#	target_compile_definitions(${TEST_EXE} PRIVATE "BOOST_PP_VARIADICS")
	target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS})

	target_include_directories(${TEST_EXE} PRIVATE ${Boost_INCLUDE_DIRS})

	target_link_libraries(${TEST_EXE} PRIVATE ${Boost_LIBRARIES})
	target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

	target_compile_options(
		${TEST_EXE}
		PRIVATE
			$<$<AND:$<CXX_COMPILER_ID:GNU>,$<CUDA_COMPILER_ID:NVIDIA>>:
				-Xcompiler=-Werror,-Wall,-Wextra,-Wcast-align,-Wcast-qual,-Wdouble-promotion,-Wduplicated-branches,-Wduplicated-cond,-Wformat-truncation,-Wformat=2,-Wlogical-op,-Wmisleading-indentation,-Wmissing-include-dirs,-Wnon-virtual-dtor,,-Wno-missing-declarations,-Wnon-virtual-dtor,-Wnull-dereference,-Woverloaded-virtual,-Wpointer-arith,-Wno-redundant-decls,-Wno-shadow,-Wswitch-enum,-Wtrampolines,-Wuninitialized,-Wunused,-Wunused-but-set-variable,-Wunused-result,-Wno-zero-as-null-pointer-constant
				--expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe=--diag_suppress=implicit_return_from_non_void_function,--display_error_number
			>
			$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CUDA_COMPILER_ID:NVIDIA>>>:
				-Werror
				-Wconversion
				-Wall
				-Wextra
				-Wcast-align
				-Wcast-qual
				-Wdouble-promotion
				-Wduplicated-branches
				-Wduplicated-cond
				-Wformat-truncation
				-Wformat=2
				-Wlogical-op
				-Wmisleading-indentation
				-Wmissing-include-dirs
				-Wnon-virtual-dtor
				-Wno-missing-declarations
				-Wnon-virtual-dtor
				-Wnull-dereference
				-Wold-style-cast
				-Woverloaded-virtual
				-Wpointer-arith
				-Wpedantic
				-Wredundant-decls
				-Wshadow
				-Wsign-conversion
				-Wstrict-aliasing
				-Wswitch-enum
				-Wtrampolines
				-Wuninitialized
				-Wunused
				-Wunused-but-set-variable
				-Wunused-result
				-Wuseless-cast
				-Wzero-as-null-pointer-constant
			>
			$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
				-Werror
				-Wall
				-Wextra
				-Wcast-align
				-Wcast-qual
				-Wconversion
				-Wdouble-promotion
				-Wformat=2
				-Wnon-virtual-dtor
				-Wnull-dereference
				-Wmove
				-Wshadow
				-Wold-style-cast
				-Woverloaded-virtual
				-Wpedantic
				-Wsign-conversion
				-Wunused
            >
			$<$<CXX_COMPILER_ID:Intel>:  # also IntelLLVM, XL (ibm), XLClang (ibm)
				-Werror
				-Wall
				-Wextra
				-pedantic-errors
				-diag-disable=remark
				-diag-error:3846
				-wd161
				-Warray-bounds
				-Wcast-qual
				-Wchar-subscripts
				-Wcomment
				-Wenum-compare
				-Wformat=2
				-Wnarrowing
				-Wno-return-type
				-Wnon-virtual-dtor
				-Wnonnull
				-Wmaybe-uninitialized
				-Wmain
				-Woverloaded-virtual
				-Wparentheses
				-Wpointer-sign
				-Wreorder
				-Wsign-compare
				-Wsequence-point
				-Wshadow
				-Wtrigraphs
				-Wuninitialized
				-Wunused
				-Wunused-but-set-variable
				-Wunused-function
				-Wunused-variable
				-Wwrite-strings
			>
			$<$<OR:$<CXX_COMPILER_ID:PGI>,$<CXX_COMPILER_ID:NVHPC>>:
				-Werror
				-Wall
				-Wcast-qual
				-Wformat=2
				-Wshadow
				-Wunused
				--diag_suppress=implicit_return_from_non_void_function
				--display_error_number
			>
			$<$<CXX_COMPILER_ID:MSVC>:
				/WX
				/W4
			>
	)
#	endif()
	add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
endforeach()
