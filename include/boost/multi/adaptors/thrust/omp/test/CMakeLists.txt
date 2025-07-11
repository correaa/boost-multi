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
find_package(Thrust)

if(OpenMP_FOUND)
    if(Thrust_FOUND)
        include(CTest)
        enable_testing()

        add_executable(omp.cpp.x omp.cpp)

        target_compile_options(omp.cpp.x PUBLIC -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
        target_include_directories(omp.cpp.x SYSTEM PUBLIC /Users/correatedesco1/thrust /Users/correatedesco1/thrust/dependencies/libcudacxx/include)  # /Users/correatedesco1/thrust/dependencies/cub)
        target_link_libraries(omp.cpp.x PRIVATE multi Boost::boost OpenMP::OpenMP_CXX)

        add_test(NAME omp.cpp.x COMMAND $<TARGET_FILE:omp.cpp.x>)
    else()
        message(WARNING "Cannot find Thurst, will not be tested")
    endif()
else()
    message(WARNING "Cannot find OpenMP, will not be tested.\n Install with `sudo apt install libomp-dev`")
endif()
