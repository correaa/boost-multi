# -*-indent-tabs-mode:nil;c-basic-offset:2;tab-width:4;autowrap:nil;-*-
cmake_minimum_required(VERSION 3.13)

set(CMAKE_VERBOSE_MAKEFILE ON)

#project(
# boost-multi-adaptors-blas
# VERSION 0.1
# LANGUAGES CXX
#)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if((NOT
	CMAKE_CXX_COMPILER_ID
	STREQUAL
	"PGI"
   )
   AND (NOT
   CMAKE_CXX_COMPILER_ID
   STREQUAL
   "NVHPC"
   )
   AND (NOT
		DART_COMPILER_NAME
		STREQUAL
		"nvcc"
	   )
)
	set(BLA_VENDOR Intel10_64lp)
	find_package(BLAS)
	if(BLAS_FOUND) # in some systems with MKL, regular BLAS headers need to be found for it to work
		message("Multi/BLAS: MKL environment detected")
		add_definitions(-DFORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID)
        SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
        SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib64")
        SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
        SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib64")
	else()
		message("Multi/BLAS: MKL environment not detected, looking for other BLAS")
		unset(BLA_VENDOR)
		find_package(BLAS)
		find_path(
			BLAS_INCLUDE_DIRS
			cblas.h
			/usr/include
			/usr/local/include
			$ENV{BLAS_HOME}/include
		)
		if(BLAS_FOUND)
			message("Some BLAS found")
		endif()
	endif()

    link_libraries(${BLAS_LIBRARIES})
    # link_libraries(BLAS::BLAS) // needs 3.18

    # include_directories(BLAS::BLAS)  # does not including a real path so far
	find_path(
		BLAS_INCLUDE_DIRS
		cblas.h
		/usr/include
		/usr/local/include
		$ENV{BLAS_HOME}/include
	)
else()
	link_libraries("-lblas") # cmake cannot detect BLAS with pgi/nvc++ but it ships with its own version
endif()

message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "PGI" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
	add_definitions(-DFORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID)
	add_definitions(-DRETURN_BY_STACK)  # TODO(correaa) check
	message("BLAS (e.g. dot) returns by stack")
endif()

include_directories(${CMAKE_BINARY_DIR})

if(BLAS_FOUND)
  add_subdirectory(test)
else()
  message(WARNING "BLAS not found, BLAS-adaptor tests will not be compiled and run. If you want this feature install BLAS, for example please run:")
  message(WARNING "    sudo apt install libblas-dev")
  message(WARNING "    sudo dnf install blas-devel  # in Fedora")
endif()
