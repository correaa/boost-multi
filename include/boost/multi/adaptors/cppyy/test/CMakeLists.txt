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

add_test(
  NAME cppyy_pythonize_test.py
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/pythonize_test.py ${_multi_include_dir}
)

# GPU variant: multi.thrust.cuda.array["T", D] over a precompiled set (see
# ../boost_multi_thrust_cuda.py). Skips gracefully without CUDA enabled; this is not part
# of the CPU-only "no glue" cppyy story above, it needs a real nvcc build.
if(ENABLE_CUDA OR DEFINED CXXCUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  find_package(CUDAToolkit REQUIRED)

  add_library(thrust_cuda_array SHARED ${CMAKE_CURRENT_SOURCE_DIR}/../thrust_cuda_array.cu)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../thrust_cuda_array.cu PROPERTIES LANGUAGE CUDA)
  target_include_directories(thrust_cuda_array PRIVATE ${_multi_include_dir})
  target_link_libraries(thrust_cuda_array PRIVATE CUDA::cudart)
  target_compile_features(thrust_cuda_array PRIVATE cxx_std_17)

  add_test(
    NAME cppyy_thrust_cuda_array_test.py
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/thrust_cuda_array_test.py
            $<TARGET_FILE:thrust_cuda_array>
  )
else()
  message(STATUS "ENABLE_CUDA not set, cppyy GPU (multi.thrust.cuda.array) test will not be built.")
endif()
