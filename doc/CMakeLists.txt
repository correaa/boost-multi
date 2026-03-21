# Minimal CMake project used only by MrDocs to generate compile_commands.json.
cmake_minimum_required(VERSION 3.16)
project(boost_multi_docs LANGUAGES CXX)

add_library(mrdocs_driver OBJECT mrdocs_driver.cpp)
target_include_directories(mrdocs_driver PUBLIC ../include)
target_compile_features(mrdocs_driver PUBLIC cxx_std_17)
