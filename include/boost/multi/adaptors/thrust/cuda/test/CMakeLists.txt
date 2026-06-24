cmake_minimum_required(VERSION 3.16)

if(ENABLE_CUDA OR DEFINED CXXCUDA)
	enable_language(CUDA)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()
	find_package(CUDAToolkit REQUIRED COMPONENTS Thrust)  # requires cmake 3.17

	# Carry Thrust's usage requirements on a target, so anything that includes Thrust inherits them.
	# /wd4668: Thrust/CUB headers reference undefined macros (e.g. __NV_NO_HOST_COMPILER_CHECK),
	# which trips MSVC's C4668; suppress it for the nvcc host compiler.
	if(NOT TARGET MyThrust)
		add_library(MyThrust INTERFACE)
		target_include_directories(MyThrust SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
		target_link_libraries(MyThrust INTERFACE CUDA::cudart)
		target_compile_options(MyThrust INTERFACE
			$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>:-Xcompiler=/wd4668>
		)
	endif()
endif()

find_package(Boost CONFIG REQUIRED)  # tests require Boost.Core LightweightTest

enable_testing()
include(CTest)

include_directories(${CMAKE_BINARY_DIR})

set(TEST_SRCS)
#	neighbor_list.cu)

foreach(TEST_FILE ${TEST_SRCS})
	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set(TEST_EXE "${TEST_FILE}.x")
		add_executable(${TEST_EXE} ${TEST_FILE})

		target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS})

		# target_include_directories(${TEST_EXE}        PRIVATE ${PROJECT_SOURCE_DIR}/include)
		target_include_directories(${TEST_EXE}        PRIVATE ${CUDA_INCLUDE_DIRS})
	
		#target_compile_options(${TEST_EXE} PRIVATE --expt-relaxed-constexpr)
	
		target_link_libraries(${TEST_EXE} PRIVATE ${CUDA_LIBRARIES})
		target_link_libraries(${TEST_EXE} PRIVATE multi)
		target_link_libraries(${TEST_EXE} PRIVATE MyThrust)  # inherits Thrust usage requirements, incl. -Xcompiler=/wd4668 on MSVC+nvcc

		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
		
		add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
	endif()
endforeach()

# target_compile_options(neighbor_list.cu.x PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
