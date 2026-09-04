# Copyright 2025 Alfredo A. Correa
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

cmake_minimum_required(VERSION 3.15)

enable_testing()
include(CTest)

find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_FOUND)
  message(WARNING "Python3 interpreter not found, cppyy adaptor will not be tested.")
  return()
endif()

# cppyy provides the automatic C++ bindings; skip gracefully if it is not installed.
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import cppyy"
  RESULT_VARIABLE _cppyy_import_result
  OUTPUT_QUIET ERROR_QUIET
)

if(NOT _cppyy_import_result EQUAL 0)
  message(WARNING "Python module 'cppyy' not found, cppyy adaptor will not be tested. Install it with:"
                  "\n    pip install cppyy")
  return()
endif()

# test/ -> cppyy -> adaptors -> multi -> boost -> (include)
get_filename_component(_multi_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

add_test(
  NAME cppyy_test.py
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cppyy_test.py ${_multi_include_dir}
)
