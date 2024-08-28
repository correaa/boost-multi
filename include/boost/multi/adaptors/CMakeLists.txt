cmake_minimum_required(VERSION 3.5)

find_package(MPI)

if(MPI_FOUND AND NOT MPI_DISABLED)
  add_subdirectory(./mpi)
else()
  message(WARNING "MPI not found or disabled, MPI adaptor will not be tested.")
endif()
