cmake_minimum_required(VERSION 3.5)

enable_testing()
include(CTest)

add_executable(mpi.cpp.x mpi.cpp)

# if(ENABLE_CUDA OR DEFINED CXXCUDA)
#     set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
#     target_compile_options(${TEST_EXE} PRIVATE -std=c++17)
# endif()

# target_include_directories(${TEST_EXE}        PRIVATE ${PROJECT_SOURCE_DIR}/include)
# target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})

# target_link_libraries(${TEST_EXE} PRIVATE ${Boost_LIBRARIES})
# target_link_directories(${TEST_EXE} PRIVATE ${Boost_LIBRARY_DIRS})

# target_link_libraries(mpi.cpp.x PRIVATE multi boost-multi-mpi)

target_link_libraries(mpi.cpp.x PRIVATE multi MPI::MPI_CXX)

add_test(NAME mpi.cpp.x COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi.cpp.x> ${MPIEXEC_POSTFLAGS})
