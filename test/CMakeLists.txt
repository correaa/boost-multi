# Copyright 2018-2024 Alfredo A. Correa
# Copyright 2024 Matt Borland
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

cmake_minimum_required(VERSION 3.16) # 3.16 for precompiled_headers

if (DEFINED BOOST_SUPERPROJECT_VERSION AND NOT DEFINED BOOST_MULTI_STANDALONE AND BUILD_TESTING)

	include(BoostTestJamfile OPTIONAL RESULT_VARIABLE HAVE_BOOST_TEST)

	if(HAVE_BOOST_TEST)

		boost_test_jamfile(FILE Jamfile LINK_LIBRARIES Boost::multi Boost::multi_array Boost::iterator)

	endif()

else()

	enable_testing()

	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_EXTENSIONS OFF)

	find_package(Boost CONFIG)  #  COMPONENTS boost unit_test_framework)  # tests require Boost.Test `sudo apt install libboost-test-dev`

	if(ENABLE_CUDA)
		enable_language(CUDA)
		set(CMAKE_CUDA_EXTENSIONS OFF)
		if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
		  set(CMAKE_CUDA_ARCHITECTURES native)
		endif()
	endif()

	find_package(TBB)

	enable_testing()
	include(CTest)

	set(WARNS
	$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
		-Xcompiler=-Werror,-Wall,-Wextra,-Wcast-align,-Wcast-qual,-Wdouble-promotion,-Wduplicated-branches,-Wduplicated-cond,-Wformat-truncation,-Wformat=2,-Wlogical-op,-Wmisleading-indentation,-Wno-missing-include-dirs,-Wnon-virtual-dtor,-Wno-missing-declarations,-Wnon-virtual-dtor,-Wnull-dereference,-Woverloaded-virtual,-Wpointer-arith,-Wno-redundant-decls,-Wno-shadow,-Wno-switch-enum,-Wno-unknown-pragmas,-Wtrampolines,-Wuninitialized,-Wunused,-Wunused-but-set-variable,-Wunused-result,-Wno-zero-as-null-pointer-constant
		#--Werror=cross-execution-space-call
		--Werror ext-lambda-captures-this
		--use_fast_math
		--extra-device-vectorization
		--restrict
		-Xcudafe=--display_error_number -Xcudafe=--diag_error=incompatible_assignment_operands -Xcudafe=--diag_error=returning_ptr_to_local_variable -Xcudafe=--diag_error=subscript_out_of_range -Xcudafe=--diag_error=used_before_set -Xcudafe=--diag_error=undefined_preproc_id -Xcudafe=--diag_error=implicit_func_decl -Xcudafe=--diag_error=implicit_return_from_non_void_function -Xcudafe=--diag_error=missing_type_specifier
	>
	$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CUDA_COMPILER_ID:NVIDIA>>,$<NOT:$<CUDA_COMPILER_ID:Clang>>>: -Werror
		$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-ftrivial-auto-var-init=pattern>
	# -WNSObject-attribute # (gcc 12, not in 11)
	# -Wabi=13 -Wabi-tag # (maybe important when linking with very old libraries)
	-Waddress
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>:-Waddress-of-packed-member>
	-Wno-aggregate-return # (disallow return classes or structs, seems a C-compatibility warning)
	-Waggressive-loop-optimizations
	-Waligned-new=all # (gcc 12, not in 11)
	# -Walloc-size-larger-than=<bytes> # (gcc 12, not in 11)
	-Walloc-zero
	-Walloca  # -Walloca-larger-than=<number>
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>:-Warith-conversion>
	-Warray-bounds # -Warray-bounds=<0,2>
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-Warray-compare>
	# -Warray-parameter #=<0,2> # (gcc 11, not in gcc 10)
	# -Wattribute-alias #=<0,2> # (gcc 12, not in 11, not in 7)
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,9>:-Wattribute-warning>
	-Wattributes
	-Wbool-compare -Wbool-operation
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-Wbidi-chars -Wbidi-chars=any>
	-Wbuiltin-declaration-mismatch -Wbuiltin-macro-redefined
	#-Wc++-compat
	-Wc++0x-compat -Wc++11-compat -Wc++14-compat -Wc++17-compat
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-Wc++17-extensions>
	-Wc++1z-compat
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>:-Wc++20-compat>
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-Wc++20-extensions -Wc++23-extensions>
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>:-Wc++2a-compat>
	#-Wcannot-profile # (gcc 9, not in gcc 8)
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,8>:-Wcast-align=strict -Wcast-function-type> # (gcc 8, not in gcc 7)
	-Wcast-qual
	#-Wcatch-value  #=<0, 3> # (gcc 8, not in gcc 7)
	-Wchar-subscripts
	#-Wclass-conversion # (gcc 11, not in 8)
	#-Wclass-memaccess # (gcc 12, not in 11, not in 7)
	-Wclobbered
	# -Wcomma-subscript # (gcc 12, not in 11)
	-Wcomment  # (same as -Wcomments)
	-Wconditionally-supported
	-Wconversion -Wconversion-null
	-Wcoverage-mismatch -Wcpp
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-Wctad-maybe-unsupported>
	-Wctor-dtor-privacy
	-Wdangling-else
	# -Wdangling-pointer # (gcc 12, not in 11)
	# -Wdangling-reference # (gcc 13, not in 12)
	-Wdate-time
	-Wdelete-incomplete -Wdelete-non-virtual-dtor
	-Wdeprecated
	#-Wdeprecated-copy -Wdeprecated-copy-dtor # (gcc 11, not in gcc 8)
	-Wdeprecated-declarations
	# -Wdeprecated-enum-enum-conversion -Wdeprecated-enum-float-conversion # (gcc 11, not in gcc 10)
	-Wdisabled-optimization
	-Wdiv-by-zero -Wdouble-promotion
	-Wduplicated-branches -Wduplicated-cond
	# -Weffc++ # (doesn't allow some advanced techniques, such as CRTP)
	-Wempty-body -Wendif-labels
	-Wenum-compare
	# -Wenum-conversion # (gcc 11, not in 10)
	-Wexpansion-to-defined
	# -Wexceptions  # (gcc 11, not in 10)
	# -Wextra-semi # (gcc 8, not in 7)
	-Wfloat-conversion -Wfloat-equal
	-Wformat=2
	-Wformat-contains-nul # (gcc 12, not in 11)
	# -Wformat-diag # (gcc 10, not in 9)
	-Wformat-extra-args -Wformat-nonliteral
	-Wformat-overflow=1
	-Wformat-security -Wformat-signedness -Wformat-truncation -Wformat-y2k -Wformat-zero-length
	-Wframe-address  # -Wframe-larger-than=<byte-size>
	-Wfree-nonheap-object -Whsa
	# -Wif-not-aligned # (gcc 8, not in 7)
	-Wignored-attributes
	-Wignored-qualifiers
	-Wimplicit-fallthrough #=3  # -Wimplicit-fallthrough=<0,5>
	# -Winaccessible-base # (gcc 12, not in 11)
	# -Winfinite-recursion # (gcc 12, not in 9)
	-Winherited-variadic-ctor
	#-Winit-list-lifetime # (gcc 12, not in 11)
	-Winit-self
	-Wno-inline  # not all inline attempts can be successful
	-Wint-in-bool-context -Wint-to-pointer-cast
	# -Winterference-size # (gcc 12, not in 11)
	# -Winvalid-imported-macros # (gcc 11, not in gcc 10)
	-Winvalid-memory-model -Winvalid-offsetof -Winvalid-pch
	# -Wlarger-than=<byte-size>  # (disallow large objects types? in executable)
	-Wliteral-suffix
	-Wlogical-not-parentheses -Wlogical-op
	# -Wlong-long # (C++98 warning)
	-Wlto-type-mismatch -Wmain -Wmaybe-uninitialized
	-Wmemset-elt-size -Wmemset-transposed-args
	-Wmisleading-indentation
	# -Wmismatched-dealloc -Wmismatched-new-delete # (gcc 11, not in gcc 10)
	# -Wmismatched-tags # (gcc 11, not in 9)
	# -Wmissing-attributes # (gcc 8, not in 8)
	-Wmissing-braces -Wmissing-declarations -Wmissing-field-initializers -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn
	#-Wmissing-profile # (gcc 11, not in gcc 8)
	# -Wmissing-requires -Wmissing-template-keyword # (gcc 12, not in 11)
	-Wmultichar
	-Wno-multiple-inheritance # (disallows composition by inheritance)
	# -Wmultistatement-macros # (gcc 8, not in 7)
	-Wno-namespaces # (disallows use of namespaces, seems a C-tool)
	-Wnarrowing
	# -Wno-alloc-size-larger-than=<bytes> -Wframe-larger-than=<bytes> -Wno-larger-than<bytes> -Wstack-usage=<bytes> # (gcc 112, not in 11)
	-Wnoexcept -Wnoexcept-type
	-Wnon-template-friend -Wnon-virtual-dtor
	-Wnonnull -Wnonnull-compare
	-Wnormalized  #=nfc -Wnormalized=[none|id|nfc|nfkc]
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,14>:-Wnrvo>  # (not in gcc 13)
	-Wnull-dereference
	-Wodr
	-Wold-style-cast  # TODO(correaa) fix this
	# -Wopenacc-parallelism # (gcc 12, not in 11)
	-Wopenmp-simd -Woverflow
	-Woverlength-strings -Woverloaded-virtual
	-Wpacked -Wpacked-bitfield-compat
	# -Wpacked-not-aligned # (gcc 8, not 7)
	# -Wpadded # (disallows structs that need padding for alignment)
	-Wparentheses
	#-Wpessimizing-move # (gcc 11, not in gcc 8)
	-Wplacement-new  #=1  -Wplacement-new=<0,2>
	-Wpmf-conversions
	-Wpointer-arith -Wpointer-compare
	-Wpragmas
	#-Wprio-ctor-dtor # (gcc 11, not in gcc 8)
	-Wpsabi
	# -Wrange-loop-construct # (gcc 11, not in gcc 10)
	-Wredundant-decls
	#-Wredundant-move # (gcc 11, not in gcc 8)
	# -Wredundant-tags # (gcc 11, not in gcc 9)
	-Wregister
	-Wreorder # (gcc 12, not in 11)
	-Wreturn-local-addr -Wreturn-type
	-Wrestrict -Wreorder
	-Wscalar-storage-order -Wsequence-point
	-Wshadow -Wshadow-compatible-local -Wshadow-local -Wshadow=compatible-local -Wshadow=local
	-Wshift-count-negative -Wshift-count-overflow -Wshift-negative-value -Wshift-overflow  #=1 -Wshift-overflow=<0,2>
	-Wsign-compare -Wsign-conversion -Wsign-promo
	-Wsized-deallocation
	-Wsizeof-array-argument
	# -Wsizeof-array-div # (gcc 11, not in 10)
	-Wsizeof-pointer-memaccess
	-Wstack-protector  # -Wstack-usage=<byte-size>
	-Wstrict-aliasing #=3 # -Wstrict-aliasing=<0,3>
	-Wstrict-null-sentinel  
	# -Wstrict-overflow=<0,5>
	-Wstrict-overflow=1
	# -Wstrict-overflow=2  # generates false positives when comparing pointers
	# -Wstrict-overflow=3  # incompatible with index_range operator-
	# -Wstrict-overflow=4  # not compatible with stl heap
	# -Wstrict-overflow=5  # not compatible with Boost.Test
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,11>:-Wstring-compare>  # (not in gcc 9)
	-Wstringop-overflow  #=2  -Wstringop-overflow=<0,4>
	# -Wstringop-overread # (gcc 11, not in 10)
	# -Wstringop-truncation # (gcc 8, not in 7)
	-Wsubobject-linkage
	# -Wsuggest-attribute=cold # (gcc 12, not in 11, not in 7)
	-Wsuggest-attribute=const -Wsuggest-attribute=format
	# -Wsuggest-attribute=malloc # (gcc 8, not in 7)
	-Wsuggest-attribute=noreturn # -Wsuggest-attribute=pure # (false positive in gcc 8.5.0 RedHat)
	-Wsuggest-final-methods -Wsuggest-final-types
	-Wsuggest-override # (gcc 12, not in gcc 11)
	-Wswitch -Wswitch-bool
	-Wno-switch-default
	-Wswitch-enum
	# -Wswitch-outside-range # (gcc 11, not in gcc 9)
	-Wswitch-unreachable
	-Wsync-nand -Wsynth
	-Wno-system-headers  # (expects system headers to be warning-compliant which they are not)
	-Wtautological-compare
	-Wno-templates # (disallows templates, C-tool)
	-Wterminate # (gcc 12, not in 11)
	-Wtrampolines -Wtrigraphs
	# -Wtrivial-auto-var-init # (gcc 12, not in 11)
	# -Wtsan # (gcc 11, not in 10)
	-Wtype-limits -Wundef -Wuninitialized
	-Wno-unknown-pragmas  # (see above) -Wunknown-pragmas (other compilers need their own pragmas for their warnings)
	-Wunreachable-code -Wunsafe-loop-optimizations
	-Wunused -Wunused-but-set-parameter -Wunused-but-set-variable
	-Wunused-const-variable  #=2 TODO(correaa) add [[maybe_unused]] to niebloids
	-Wunused-function -Wunused-label -Wunused-local-typedefs -Wunused-macros -Wunused-parameter -Wunused-result -Wunused-value -Wunused-variable
	# -Wuse-after-free  # =<0,3> # (gcc 12, not in 11)
	-Wuseless-cast
	-Wvarargs -Wvariadic-macros -Wvector-operation-performance
	# -Wvexing-parse # (gcc 11, not in gcc 10)
	-Wvirtual-inheritance -Wvirtual-move-assign
	-Wvla # -Wvla-larger-than=<number> # (gcc 12, not in 11)
	# -Wvla-parameter # (gcc 11, not in gcc 10)
	# -Wvolatile # (gcc 11, not in gcc 9)
	-Wvolatile-register-var
	-Wwrite-strings
	-Wzero-as-null-pointer-constant
	# -Wzero-length-bounds # (gcc 12, not in 11)
	>
	$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CUDA_COMPILER_ID:Clang>>:
	$<$<VERSION_GREATER_EQUAL:$<CXX_COMPILER_VERSION>,12>:-ftrivial-auto-var-init=pattern>
	-Werror -Wall -Wextra -Weverything
	-Wno-error=\#warnings  # -Wno-\#warnings disable this working for a bug in rocm ~5.6
	-Wno-c++98-compat -Wno-c++98-compat-bind-to-temporary-copy -Wno-c++98-compat-pedantic
	-Wlarge-by-value-copy  # subarray references can be large
	-Wno-padded  # most classes, e.g. layouts, need padding
	-Wstrict-aliasing -Wstrict-aliasing=0 -Wstrict-aliasing=1 -Wstrict-aliasing=2
	# -Wstrict-overflow -Wstrict-overflow=0 -Wstrict-overflow=1 -Wstrict-overflow=2 -Wstrict-overflow=3 -Wstrict-overflow=4
	-Wstrict-overflow=5
	-Wundefined-reinterpret-cast  # needed to cast to c-array
	-Wunknown-pragmas  # other compilers need their own pragmas
	# -Rpass -Rpass-analysis -Rpass-missed
	-Rsanitize-address # -Rsearch-path-usage
	>
	$<$<CXX_COMPILER_ID:Intel>: -Werror -Wall -Wextra # also IntelLLVM, XL (ibm), XLClang (ibm)
	-diag-disable=remark
	-diag-error:3846
	-diag-disable=1011  # disables warning missing return at the end of non-void function
	-diag-disable=2196  # disables "error #2196: routine is both "inline" and "noinline"" in icpc 2021.5.0
	-wd161
	-Wabi
	-Warray-bounds
	-Wcast-qual
	-Wchar-subscripts
	-Wcomment
	-Wdeprecated
	-Wenum-compare
	-Wextra-tokens
	-Wformat -Wformat=2 -Wformat-security
	-Wic-pointer
	-Wnarrowing
	-Wreturn-type
	-Wnon-virtual-dtor
	-Wnonnull
	-Wmaybe-uninitialized
	-Wmain
	-Wmissing-declarations -Wmissing-prototypes
	-Wmultichar
	-Woverloaded-virtual
	-Woverflow
	-Wp64
	-Wparentheses
	-Wpointer-arith
	-Wpointer-sign
	-Wreorder
	-Wreturn-type
	-Wsequence-point
	-Wshadow
	-Wsign-compare
	-Wshorten-64-to-32
	-Wmissing-prototypes
	-Wstrict-aliasing
	-Wno-unknown-pragmas
	-Wstrict-prototypes
	-Wtrigraphs
	-Wtype-limits
	-Wuninitialized
	-Wunused -Wunused-but-set-variable -Wunused-function -Wunused-parameter -Wunused-variable
	-Wwrite-strings
	>
	# $<$<OR:$<CXX_COMPILER_ID:PGI>,$<CXX_COMPILER_ID:NVHPC>>:
	$<$<CXX_COMPILER_ID:NVHPC>:
	-Werror -Wall
	-Wcast-qual
	-Wformat=2
	-Wshadow
	-Wuninitialized
	--diag_suppress unrecognized_gcc_pragma
	>
	$<$<CXX_COMPILER_ID:MSVC>:
	/GS  # Enables buffer security check (can help detect stack-based buffer overflows)
	/Wall /EHsc  # /volatile:iso /WX /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- /W4
	/wd4371  # layout of class may have changed from a previous version of the compiler due to better packing of member
	/wd4514  # unreferenced inline function has been removed
	/wd4623  # default constructor was implicitly defined as deleted #TODO(correaa) investigate
	/wd4625  # copy constructor was implicitly defined as deleted
	/wd4626  # assignment operator was implicitly defined as deleted #TODO(correaa) investigate
	/wd4710  # function not inlined
	/wd4711  # function selected for automatic inline expansion
	/wd4820  # bytes padding added after base class
	/wd4866  # compiler may not enforce left-to-right evaluation order for call
	/wd4848  # support for attribute [[msvc::no_unique_address]] in C++17 and earlier is a vendor extension
	/wd4868  # compiler may not enforce left-to-right evaluation order in braced initializer list
	/wd5026  # move constructor was implicitly defined as deleted
	/wd5027  # move assignment operator was implicitly defined as deleted
	>
	)

	file(
		GLOB TEST_SRCS
		RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
		*.cpp
	)

	if(NOT Boost_FOUND)
		message(WARNING "Cannot find Boost, Multi library will have a very minimal test. If you want to test the library install Boost.Test, for example please run:\n    sudo apt install libboost-test-dev\n    sudo dnf install boost-devel")

		add_executable(main main.cpp)

		# target_link_libraries(main PRIVATE multi)
		target_compile_features   (main PUBLIC cxx_std_17)
		target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include)

		# target_compile_definitions(main PRIVATE "$<$<CUDA_COMPILER_ID:NVIDIA>:BOOST_PP_VARIADICS>")

		if(NOT DEFINED ENABLE_CIRCLE)
			target_compile_options(main PRIVATE ${WARNS})
		else()
			target_compile_options    (main PRIVATE -Werror -Wall)
		endif()
		add_test(NAME main COMMAND $<TARGET_FILE:main>)

	else()

		foreach(TEST_FILE ${TEST_SRCS})
			set(TEST_EXE "${TEST_FILE}.x")
			add_executable(${TEST_EXE} ${TEST_FILE})

			if(ENABLE_CUDA)
				set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
			endif()

			target_link_libraries(${TEST_EXE} PRIVATE multi)
			target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
			# target_link_libraries     (${TEST_EXE} PRIVATE multi Boost::boost   )
			if(TBB_FOUND)
				target_link_libraries     (${TEST_EXE} PRIVATE TBB::tbb)
				target_compile_definitions(${TEST_EXE} PUBLIC TBB_FOUND=1)
			endif()

			if(NOT DEFINED ENABLE_CIRCLE)
				target_compile_options(${TEST_EXE} PRIVATE ${WARNS})
			else()
				target_compile_options    (${TEST_EXE} PRIVATE -Werror -Wall)
			endif()
			add_test(NAME ${TEST_EXE} COMMAND $<TARGET_FILE:${TEST_EXE}>)
		endforeach()
        
		# find_package(BLAS REQUIRED)
		# target_link_libraries(redux.cpp.x PRIVATE BLAS::BLAS)

	endif()

endif()
