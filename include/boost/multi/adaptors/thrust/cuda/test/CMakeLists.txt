cmake_minimum_required(VERSION 3.16)

if(ENABLE_CUDA OR DEFINED CXXCUDA)
	enable_language(CUDA)
	if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES native)
	endif()
	find_package(CUDAToolkit REQUIRED COMPONENTS Thrust)  # requires cmake 3.17
		$<$<CXX_COMPILER_ID:MSVC>:
		$<$<NOT:$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>>:
			/W4  # /volatile:iso /WX /Zc:wchar_t /Zc:forScope /Zc:inline
			/wd4514  # unreferenced inline function has been removed
			/wd4866  # compiler may not enforce left-to-right evaluation order for call to 'boost::multi::subarray<int,1,int *,boost::multi::layout_t<1,__int64> >::operator[]' [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\test\array_ref.cpp.x.vcxproj]
			/wd4868  # compiler may not enforce left-to-right evaluation order in braced initializer list
			/wd5045  # Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\include\boost\multi\adaptors\thrust\omp\test\omp.cpp.x.vcxproj]
			/wd5031  # likely mismatch, popping warning state pushed in different file
		>
		$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
			-Xcompiler=/W4  # /volatile:iso /WX /Zc:wchar_t /Zc:forScope /Zc:inline /W4
			-Xcompiler=/wd4464  # relative include path contains '..'
			-Xcompiler=/wd4514  # unreferenced inline function has been removed
			-Xcompiler=/wd4668  # warning C4668: '__NV_NO_HOST_COMPILER_CHECK' is not defined as a preprocessor macro, replacing with '0' for '#if/#elif'
			-Xcompiler=/wd4866  # compiler may not enforce left-to-right evaluation order for call
			-Xcompiler=/wd4868  # compiler may not enforce left-to-right evaluation order in braced initializer list
			-Xcompiler=/wd5045  # Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\include\boost\multi\adaptors\thrust\omp\test\omp.cpp.x.vcxproj]
		>
	>

endif()

find_package(Boost CONFIG REQUIRED)  # tests require Boost.Core LightweightTest

enable_testing()
include(CTest)

include_directories(${CMAKE_BINARY_DIR})

set(TEST_SRCS)
#	neighbor_list.cu)

foreach(TEST_FILE ${TEST_SRCS})
	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set(TEST_EXE "${TEST_FILE}.x")
		add_executable(${TEST_EXE} ${TEST_FILE})

		target_compile_definitions(${TEST_EXE} PRIVATE ${Boost_DEFINITIONS})

		# target_include_directories(${TEST_EXE}        PRIVATE ${PROJECT_SOURCE_DIR}/include)
		target_include_directories(${TEST_EXE}        PRIVATE ${CUDA_INCLUDE_DIRS})
	
		#target_compile_options(${TEST_EXE} PRIVATE --expt-relaxed-constexpr)
	
		target_link_libraries(${TEST_EXE} PRIVATE ${CUDA_LIBRARIES})
		target_link_libraries(${TEST_EXE} PRIVATE multi)

		target_compile_options(${TEST_EXE} PRIVATE
			$<$<CXX_COMPILER_ID:MSVC>:
			$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
				-Xcompiler=/wd4668  # warning undefined macros TODO(correaa) put this in in Thrust target
			>
			>
		)

		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
		
		add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
	endif()
endforeach()

# target_compile_options(neighbor_list.cu.x PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)
