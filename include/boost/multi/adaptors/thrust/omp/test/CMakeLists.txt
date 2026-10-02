cmake_minimum_required(VERSION 3.18)
project(
	boost-multi-adaptor-thrust-omp-test
	VERSION 0.1
	LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost CONFIG REQUIRED)  # tests require Boost.Core LightweightTest

find_package(OpenMP COMPONENTS CXX)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "NVHPC")
    if(${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS_EQUAL "22.11")
        message(WARNING "NVHPC 22.11 produces an internal error when compiling OMP code")
        return()
    endif()
endif()

if(DISABLE_THRUST_OMP)
    message(WARNING "OMP disabled, OMP-through-Thrust capabilities will not be tested")
else()
    if(OpenMP_FOUND)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
            get_filename_component(NVHPC_CUDA_CMAKE_DIR "${CMAKE_CXX_COMPILER}/../../../cuda/lib64/cmake" ABSOLUTE)  # NVHPC ships its own CCCL: compilers/bin/nvc++ -> cuda/lib64/cmake/thrust
        endif()
        find_package(Thrust CONFIG HINTS "${NVHPC_CUDA_CMAKE_DIR}")

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
                GIT_TAG main  # or v3.6.0
                GIT_SHALLOW ON
            )
            FetchContent_MakeAvailable(cccl-cmake)
        endif()

        include(CTest)
        enable_testing()

        thrust_create_target(ThrustOMP DEVICE OMP)

        add_executable(omp.cpp.x omp.cpp)

        target_include_directories(omp.cpp.x SYSTEM PUBLIC
            ${_THRUST_INCLUDE_DIR}  # needs this to include thrust as an -isystem/`/external:I` header
            "${cccl-cmake_SOURCE_DIR}/libcudacxx/include"
            "${cccl-cmake_SOURCE_DIR}/cub"
        )
        target_link_libraries(omp.cpp.x PRIVATE ThrustOMP)  # SYSTEM has no effect here
        target_link_libraries(omp.cpp.x PRIVATE multi Boost::boost)

        add_test(NAME omp.cpp.x COMMAND $<TARGET_FILE:omp.cpp.x>)
    else()
        message(WARNING "Cannot find OpenMP, Thrust-OMP adaptor will not be tested. If you want this feature install the OpenMP runtime, for example please run:"
                        "\n    sudo apt install libomp-dev    # for clang, or libgomp1 for gcc, in Ubuntu"
                        "\n    sudo dnf install libomp-devel  # for clang, or libgomp for gcc, in Fedora"
                        "\n    brew install libomp            # on macOS")
    endif()
endif()
