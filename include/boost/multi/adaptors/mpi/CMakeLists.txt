cmake_minimum_required(VERSION 3.5)

# project(boost-multi-mpi VERSION 0.1)

find_package(MPI REQUIRED COMPONENTS CXX)

# add_library(boost-multi-mpi INTERFACE)

# target_link_libraries(boost-multi-mpi INTERFACE MPI::MPI)

add_subdirectory(../mpi/test)
