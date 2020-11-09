#[=[Multi Test suite can be run like this:
  mkdir -p build
  cd build
  cmake ..
  make -j
  make test -j
  exit
#]=]
cmake_minimum_required(VERSION 3.11)

project(boost-multi-test VERSION 0.1 LANGUAGES CXX)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(Threads REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)

enable_testing()
include(CTest)

file(GLOB TEST_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

foreach(TEST_FILE ${TEST_SRCS})
  SET(TEST_EXE "${TEST_FILE}x")
  add_executable            (${TEST_EXE} ${TEST_FILE})
  set_target_properties     (${TEST_EXE} PROPERTIES LANGUAGES CXX)
  target_compile_features   (${TEST_EXE} PRIVATE cxx_std_14)
  target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS} "BOOST_ALL_DYN_LINK")
  target_include_directories(${TEST_EXE} PRIVATE ${Boost_INCLUDE_DIRS})
  target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS} "BOOST_ALL_DYN_LINK")
  target_link_libraries     (${TEST_EXE} ${Boost_LIBRARIES} ${pthread_LIBRARIES} stdc++fs)
  target_link_directories   (${TEST_EXE} PUBLIC ${Boost_LIBRARY_DIRS})
  if(DEFINED ENV{VALGRIND})
    add_test(NAME ${TEST_EXE} COMMAND valgrind --leak-check=full --track-origins=yes --show-leak-kinds=all --error-exitcode=1 ./${TEST_EXE})
  else()
    add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
  endif()
  if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(${TEST_EXE} PUBLIC "-pthread")
  endif()
  if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(${TEST_EXE} "${CMAKE_THREAD_LIBS_INIT}")
  endif()
endforeach()

