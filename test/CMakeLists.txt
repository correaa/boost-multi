cmake_minimum_required(VERSION 3.16)

project(boost-multi-test VERSION 0.1)

find_package(Boost REQUIRED COMPONENTS unit_test_framework timer)
find_package(Threads REQUIRED)

#find_package(BLAS REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE OFF)

enable_testing()
include(CTest)

if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
  link_directories(${Boost_LIBRARY_DIRS})
  add_definitions(${Boost_DEFINITIONS})
  message(STATUS "Setting Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
endif()

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(TEST_FILE ${TEST_SRCS})
  SET(TEST_EXE "${TEST_FILE}x")
  add_executable(${TEST_EXE} ${TEST_FILE})
  target_link_libraries(${TEST_EXE} ${Boost_LIBRARIES} ${pthread_LIBRARIES} stdc++fs)
  add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
  if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(${TEST_EXE} PUBLIC "-pthread")
  endif()
  if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${TEST_EXE} "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endforeach()

