# -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
# Copyright 2018-2023 Alfredo A. Correa

cmake_minimum_required(VERSION 3.13)

# to install this project: cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$HOME && cmake --build . --config Release --target test --target install -- -j $(nproc)
# to use this project do: project("Your project") find_package(boost-multi CONFIG REQUIRED) add_executable(${PROJECT_NAME} src/your_main.cpp) target_link_libraries(${PROJECT_NAME} boost-multi::boost-multi)

message("CMake version: ${CMAKE_VERSION}")
message("current build directory: ${CMAKE_CURRENT_BINARY_DIR}")

if(EXISTS "CMakeLists.txt")
	message(FATAL_ERROR "You probably don't want to run CMake in a directory with CMakeLists.txt")
endif()

project(
	multi
	HOMEPAGE_URL "https://gitlab.com/correaa/boost-multi"
	DESCRIPTION "A header-only modern C++ library that provides access and manipulation of data in multidimensional arrays."
	VERSION 0.80.1
	LANGUAGES CXX)

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(GNUInstallDirs)
include(CTest)

option(MULTI_BUILD_PACKAGE "Build package files as well" ON)

cmake_dependent_option(MULTI_BUILD_TESTS
  "Enable multi tests" ON
  "BUILD_TESTING" OFF)

cmake_dependent_option(MULTI_BUILD_PACKAGE_DEB
  "Create a DEB" ON
  "MULTI_BUILD_PACKAGE" OFF)

add_library(multi INTERFACE)
target_include_directories(multi
	INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if (NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  add_library(correaa::multi ALIAS multi)
endif()

# Installation help
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/multi-config.cmake.in"
  "${PROJECT_BINARY_DIR}/multi-config.cmake"
  INSTALL_DESTINATION "share/cmake/multi")

write_basic_package_version_file(
	"${PROJECT_BINARY_DIR}/multi-config-version.cmake"
	COMPATIBILITY SameMajorVersion
	ARCH_INDEPENDENT)

message("current install prefix directory: ${CMAKE_INSTALL_PREFIX}")

install(TARGETS multi
	EXPORT multi-targets
	INCLUDES DESTINATION "${CMAKE_INSTALL_DATADIR}")

install(EXPORT ${PROJECT_NAME}-targets
	DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
	NAMESPACE multi::
	FILE "${PROJECT_NAME}-targets.cmake")

install(FILES
	"${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
	"${PROJECT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
	DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}")

install(DIRECTORY "include/" TYPE INCLUDE)

if(MULTI_BUILD_TESTS)
	find_package(Boost COMPONENTS unit_test_framework REQUIRED)

	if(NOT Boost_FOUND)
		message (WARNING "Cannot find Boost, library will not be tested. If you want this feature install Boost.Test, for example please run:\n    sudo apt install libboost-test-dev")
	else()
		enable_testing()
		add_subdirectory(test)

		add_subdirectory(include/multi/adaptors/blas  )
		add_subdirectory(include/multi/adaptors/fftw  )
		add_subdirectory(include/multi/adaptors/cuda  )
		add_subdirectory(include/multi/adaptors/thrust)
	endif()
endif()

if (MULTI_BUILD_PACKAGE)
	list(APPEND source-generators TBZ2 TGZ TXZ ZIP)

	if (CMAKE_HOST_WIN32)
	list(APPEND binary-generators "WIX")
	endif()

	if (MULTI_BUILD_PACKAGE_DEB)
	list(APPEND binary-generators "DEB")
	endif()

	if (MULTI_BUILD_RPM)
	list(APPEND binary-generators "RPM")
	endif()

	set(CPACK_SOURCE_GENERATOR ${source-generators})
	set(CPACK_GENERATOR ${binary-generators})

	set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${PROJECT_VERSION}")
	set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}")

	set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Alfredo Correa")

	list(APPEND CPACK_SOURCE_IGNORE_FILES /.git/ /.build*/ /build/ .gitignore .DS_Store)

	include(CPack)
endif()

# https://youtrack.jetbrains.com/issue/CPP-25608
target_compile_features(${PROJECT_NAME} INTERFACE $<$<NOT:$<CUDA_COMPILER_ID:NVIDIA>>:cxx_std_17>)
target_compile_options (${PROJECT_NAME} INTERFACE $<$<CUDA_COMPILER_ID:NVIDIA>:-std=c++17>)

# target_compile_options (
# 	${PROJECT_NAME}
# 	INTERFACE
# 		$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
# 			-Xcompiler=-Werror,-Wall,-Wextra,-Wcast-align,-Wcast-qual,-Wno-double-promotion,-Wduplicated-branches,-Wduplicated-cond,-Wformat-truncation,-Wformat=2,-Wlogical-op,-Wmisleading-indentation,-Wno-missing-include-dirs,-Wnon-virtual-dtor,-Wno-missing-declarations,-Wnon-virtual-dtor,-Wnull-dereference,-Woverloaded-virtual,-Wpointer-arith,-Wno-redundant-decls,-Wno-shadow,-Wno-switch-enum,-Wno-unknown-pragmas,-Wtrampolines,-Wuninitialized,-Wunused,-Wunused-but-set-variable,-Wunused-result,-Wno-zero-as-null-pointer-constant
# 			--expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe=--display_error_number -Xcudafe=--diag_error=incompatible_assignment_operands -Xcudafe=--diag_error=returning_ptr_to_local_variable -Xcudafe=--diag_error=subscript_out_of_range -Xcudafe=--diag_error=used_before_set -Xcudafe=--diag_error=undefined_preproc_id -Xcudafe=--diag_error=implicit_func_decl -Xcudafe=--diag_error=implicit_return_from_non_void_function -Xcudafe=--diag_error=missing_type_specifier
# 		>
# 		$<$<OR:$<CXX_COMPILER_ID:PGI>,$<CXX_COMPILER_ID:NVHPC>>:  # EDG diagnostics list: http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
#         	--display_error_number --diag_error=incompatible_assignment_operands,returning_ptr_to_local_variable,subscript_out_of_range,used_before_set,undefined_preproc_id,implicit_func_decl,implicit_return_from_non_void_function,missing_type_specifier
# 		>
# )

# install(
# 	TARGETS ${PROJECT_NAME}
# 	EXPORT ${PROJECT_NAME}_Targets
# 	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# 	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )

# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
# 	"${PROJECT_NAME}ConfigVersion.cmake"
# 	VERSION ${PROJECT_VERSION}
# 	COMPATIBILITY SameMajorVersion
# )

# install(
# 	EXPORT ${PROJECT_NAME}_Targets
# 	FILE ${PROJECT_NAME}Targets.cmake
# 	NAMESPACE boost::multi::
# 	DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
# )

# if (NOT MSVC)
# 	install(FILES     ${PROJECT_SOURCE_DIR}/include/multi/array_ref.hpp ${PROJECT_SOURCE_DIR}/include/multi/array.hpp DESTINATION include/multi)
# 	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/multi/detail                                                      DESTINATION include/multi)
# 	install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/multi/adaptors                                                    DESTINATION include/multi)
# endif()

find_program(MEMORYCHECK_COMMAND valgrind)
set(VALGRIND_COMMAND_OPTIONS    "-q --tool=memcheck --leak-check=yes --num-callers=51 --trace-children=yes --leak-check=full --track-origins=yes --gen-suppressions=all")
set(MEMORYCHECK_COMMAND_OPTIONS "-q --tool=memcheck --leak-check=yes --num-callers=52 --trace-children=yes --leak-check=full --track-origins=yes --gen-suppressions=all")  # must go before `include(CTest)`
set(MEMORYCHECK_SUPPRESSIONS_FILE "${PROJECT_SOURCE_DIR}/.valgrind-suppressions")  # must go before `include(CTest)`
