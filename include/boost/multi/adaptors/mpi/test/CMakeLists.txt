cmake_minimum_required(VERSION 3.15)

enable_testing()
include(CTest)

find_package(Boost REQUIRED COMPONENTS)

add_executable(mpi.cpp.x mpi.cpp)

if(APPLE)
    # https://apple.stackexchange.com/a/121010
    # add_custom_command (TARGET mpi.cpp.x POST_BUILD COMMAND codesign --force --deep -s ACTMPI ${CMAKE_CURRENT_BINARY_DIR}/mpi.cpp.x)
endif()

target_link_libraries(mpi.cpp.x PRIVATE multi MPI::MPI_CXX Boost::headers)

if(APPLE)
    add_test(NAME mpi.cpp.x COMMAND ${MPIEXEC_EXECUTABLE} --mca btl ^tcp ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi.cpp.x> ${MPIEXEC_POSTFLAGS})
else()
    add_test(NAME mpi.cpp.x COMMAND ${MPIEXEC_EXECUTABLE}                ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi.cpp.x> ${MPIEXEC_POSTFLAGS})
endif()