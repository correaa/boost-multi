cmake_minimum_required(VERSION 3.18)
project(
	boost-multi-adaptor-thrust-omp-test
	VERSION 0.1
	LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED)  # tests require Boost.Core LightweightTest

find_package(OpenMP)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC")
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS_EQUAL "22.11")
        message(WARNING "NVHPC 22.11 produces an internal error when compiling OMP code")
        return()
    endif()
endif()

if(DISABLE_THRUST_OMP)
    message(WARNING "Cannot find Thurst in the system, fetching from CCCL")
else()
    if(OpenMP_FOUND)

        find_package(Thrust CONFIG)
        if(NOT Thrust_FOUND)
            message(WARNING "Cannot find Thurst in the system, fetching from CCCL")

            find_package(Git QUIET)

            if(GIT_FOUND)
                message(STATUS "Git command found: ${GIT_EXECUTABLE}")
            else()
                message(WARNING "Git command not found, won't fetch content.")
                return()
            endif()

            include(FetchContent)
            FetchContent_Declare(
                cccl-cmake
                GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
                GIT_TAG main
                GIT_SHALLOW ON
            )
            FetchContent_MakeAvailable(cccl-cmake)
            
        endif()

        include(CTest)
        enable_testing()

        thrust_create_target(ThrustOMP DEVICE OMP)

        add_executable(omp.cpp.x omp.cpp)

        target_include_directories(omp.cpp.x SYSTEM PRIVATE ${_THRUST_INCLUDE_DIR})
        target_link_libraries(omp.cpp.x PRIVATE multi Boost::boost OpenMP::OpenMP_CXX ThrustOMP)

        add_test(NAME omp.cpp.x COMMAND $<TARGET_FILE:omp.cpp.x>)
    else()
        message(WARNING "Cannot find OpenMP, will not be tested")
    endif()
else()
    message(WARNING "Cannot find OpenMP, will not be tested.\n Install with `sudo apt install libomp-dev`")
endif()