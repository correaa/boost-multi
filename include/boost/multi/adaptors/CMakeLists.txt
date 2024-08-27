cmake_minimum_required(VERSION 3.5)

find_package(MPI COMPONENTS CXX)
if(MPI_FOUND)
  add_subdirectory(./mpi)
else()
  message(WARNING "MPI not found, MPI adaptor will not be tested.")
endif()
