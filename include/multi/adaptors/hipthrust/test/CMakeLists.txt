# -*-indent-tabs-mode:nil;c-basic-offset:2;tab-width:4;autowrap:nil;-*-
cmake_minimum_required(VERSION 3.11)
project(
	boost-multi-adaptor-cuda-thrust-test
	VERSION 0.1
	LANGUAGES CXX HIP
	#CUDA
)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe \"--display_error_number\"")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

find_package(rocthrust REQUIRED)  # CONFIG PATHS "/opt/rocm-5.6.0/rocthrust")

enable_testing()

include(CTest)

# set(TEST_SRCS
# #    array.cu
# #   memory_resource.cu
# #   set_identity_kernel.cu
# #    speed.cu
# #    speed_algo.cu
# #    universal.cu
# 	vector.cpp
# )

# add_executable(vector.cu.nvidia.x vector.cu)
# set_source_files_properties(vector.cu PROPERTIES LANGUAGE CUDA)

# target_link_libraries(vector.cu.nvidia.x PRIVATE Boost::unit_test_framework)
# add_test(NAME vector.cu.nvidia.x COMMAND vector.cu.nvidia.x)

add_executable(vector.hip.amd.x vector.hip)
set_source_files_properties(vector.hip PROPERTIES LANGUAGE HIP)

target_link_libraries(vector.hip.amd.x PRIVATE roc::rocthrust Boost::unit_test_framework)
add_test(NAME vector.hip.amd.x COMMAND ./vector.hip.amd.x)


# foreach(TEST_FILE ${TEST_SRCS})
# #	set(TEST_EXE "${TEST_FILE}")
# 	add_executable(vector ${TEST_FILE})
# #	if(ENABLE_CUDA OR DEFINED CXXCUDA)
# #		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE HIP)
# #	endif()

# #	target_include_directories(${TEST_EXE}        PRIVATE ../../../../../include)

# 	target_compile_definitions(${TEST_EXE} PRIVATE "BOOST_PP_VARIADICS")
# 	target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS})
# 	target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
# 	target_link_libraries(${TEST_EXE} PRIVATE ${Boost_LIBRARIES})
# 	target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

# #    if(ENABLE_CUDA)
# #	    add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
# #    endif()
# endforeach()

# set_tests_properties(speed.cu.x PROPERTIES RUN_SERIAL TRUE)
