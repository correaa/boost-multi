# -*-indent-tabs-mode:nil;c-basic-offset:2;tab-width:4;autowrap:nil;-*-
cmake_minimum_required(VERSION 3.11)
project(
	boost-multi-adaptor-cuda-thrust-test
	VERSION 0.1
	LANGUAGES CXX HIP
)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe \"--display_error_number\"")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

enable_testing()

include(CTest)

set(TEST_SRCS
#    array.cu
#   memory_resource.cu
#   set_identity_kernel.cu
#    speed.cu
#    speed_algo.cu
#    universal.cu
	vector.hip
)

foreach(TEST_FILE ${TEST_SRCS})
	set(TEST_EXE "${TEST_FILE}.x")
	add_executable(${TEST_EXE} ${TEST_FILE})
	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE HIP)
	endif()

	target_include_directories(${TEST_EXE}        PUBLIC ../../../../../include)

	target_compile_definitions(${TEST_EXE} PRIVATE "BOOST_PP_VARIADICS")
	target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS})
	target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_libraries(${TEST_EXE} PRIVATE ${Boost_LIBRARIES})
	target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

    if(ENABLE_CUDA)
	    add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
    endif()
endforeach()

# set_tests_properties(speed.cu.x PROPERTIES RUN_SERIAL TRUE)
