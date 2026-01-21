cmake_minimum_required(VERSION 3.11)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(
	boost-multi-adaptors-cutensor
	VERSION 0.1
	LANGUAGES CXX
)

if(ENABLE_CUDA OR DEFINED CXXCUDA)
	enable_language(CUDA)
	# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
endif()

# for findpackage FindcuTENSOR.cmake
find_path(cuTENSOR_INCLUDE_DIR
    NAMES cutensor.h
    PATHS ${CUDAToolkit_INCLUDE_DIRS}
)

find_library(cuTENSOR_LIBRARY
    NAMES cutensor
    PATHS ${CUDAToolkit_LIBRARY_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuTENSOR
    REQUIRED_VARS cuTENSOR_LIBRARY cuTENSOR_INCLUDE_DIR
)

if (cuTENSOR_FOUND)
    add_library(cuTENSOR::cuTENSOR UNKNOWN IMPORTED)
    set_target_properties(cuTENSOR::cuTENSOR PROPERTIES
        IMPORTED_LOCATION ${cuTENSOR_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${cuTENSOR_INCLUDE_DIR}
    )
endif()
#
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(cuTENSOR)

if (cuTENSOR_FOUND)
	message(STATUS "cuTENSOR found")
	add_subdirectory(test)
else()
	message(STATUS "cuTENSOR NOT found")
endif()
