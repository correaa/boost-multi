cmake_minimum_required(VERSION 3.18)

project(
	boost-multi-examples
	VERSION 0.1
	LANGUAGES CXX
)

include_directories(../include)

#find_package(multi)  # see https://gitlab.com/correaa/boost-multi#using-the-library-installation-and-tests
#...
#target_link_library(my_target PUBLIC multi)

find_package(Boost REQUIRED COMPONENTS iostreams serialization unit_test_framework timer)
add_subdirectory("../" multi-bin)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
include(CTest)

add_executable(serialization.cpp.x serialization.cpp)
target_link_libraries(serialization.cpp.x PUBLIC multi)
target_link_libraries(serialization.cpp.x PRIVATE ${Boost_LIBRARIES})
add_test(NAME serialization.cpp.x COMMAND serialization.cpp.x)

add_executable(gj_solve.cpp.x gj_solve.cpp)
target_link_libraries(gj_solve.cpp.x PUBLIC multi)
target_link_libraries(gj_solve.cpp.x PRIVATE ${Boost_LIBRARIES})
add_test(NAME gj_solve.cpp.x COMMAND gj_solve.cpp.x)

add_executable(redux.cpp.x redux.cpp)
target_link_libraries(redux.cpp.x PUBLIC multi)
target_link_libraries(redux.cpp.x PRIVATE ${Boost_LIBRARIES})

find_package(TBB)

if(TBB_FOUND)
	target_link_libraries     (redux.cpp.x PRIVATE TBB::tbb)
	target_compile_definitions(redux.cpp.x PUBLIC  TBB_FOUND=1)
endif()

add_test(NAME redux.cpp.x COMMAND redux.cpp.x)
