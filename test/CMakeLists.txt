#[=[Multi Test suite can be run like this: -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
  mkdir -p build && cd build
  cmake .. [-DENABLE_CUDA=1]
  make -j
  ctest -j --output-on-error [-T memcheck]
  exit
#]=]
cmake_minimum_required(VERSION 3.11)

if(EXISTS "CMakeLists.txt")
	message(FATAL_ERROR "You probably don't want to run CMake in a directory with CMakeLists.txt")
endif()

project(
	boost-multi-test
	VERSION 0.1
	LANGUAGES CXX
)

find_package(Boost REQUIRED COMPONENTS unit_test_framework serialization)

if((ENABLE_CUDA OR DEFINED CXXCUDA) AND NOT DEFINED FORCED_ARCH)
	enable_language(CUDA)

#	include(FindCUDA/select_compute_arch)
#	cuda_detect_installed_gpus(INSTALLED_GPU_CCS_1)
#	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#	string(
#		REPLACE " "
#				";"
#				INSTALLED_GPU_CCS_3
#				"${INSTALLED_GPU_CCS_2}"
#	)
#	string(
#		REPLACE "."
#				""
#				CUDA_ARCH_LIST
#				"${INSTALLED_GPU_CCS_3}"
#	)
#	set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

enable_testing()
include(CTest)

file(
	GLOB TEST_SRCS
	RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
	*.cpp
)

foreach(TEST_FILE ${TEST_SRCS})
	set(TEST_EXE "${TEST_FILE}.x")
	add_executable(${TEST_EXE} ${TEST_FILE})

	if((ENABLE_CUDA OR DEFINED CXXCUDA) AND NOT DEFINED FORCED_ARCH)
#		set_property(TARGET ${TEST_EXE} PROPERTY "${CUDA_ARCH_LIST}")
	endif()

	if(ENABLE_CUDA OR DEFINED CXXCUDA)
		set_source_files_properties(${TEST_FILE} PROPERTIES LANGUAGE CUDA)
	endif()

	target_link_libraries(${TEST_EXE} PRIVATE multi)

	target_include_directories(${TEST_EXE} SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
	target_link_libraries     (${TEST_EXE}        PRIVATE Boost::unit_test_framework Boost::serialization)

	target_compile_definitions(${TEST_EXE} PRIVATE "$<$<CUDA_COMPILER_ID:NVIDIA>:BOOST_PP_VARIADICS>")

	if(NOT DEFINED ENABLE_CIRCLE)
		target_compile_options(
			${TEST_EXE}
			PRIVATE
				$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:
					-Xcompiler=-Werror,-Wall,-Wextra,-Wcast-align,-Wcast-qual,-Wdouble-promotion,-Wduplicated-branches,-Wduplicated-cond,-Wformat-truncation,-Wformat=2,-Wlogical-op,-Wmisleading-indentation,-Wno-missing-include-dirs,-Wnon-virtual-dtor,-Wno-missing-declarations,-Wnon-virtual-dtor,-Wnull-dereference,-Woverloaded-virtual,-Wpointer-arith,-Wno-redundant-decls,-Wno-shadow,-Wno-switch-enum,-Wno-unknown-pragmas,-Wtrampolines,-Wuninitialized,-Wunused,-Wunused-but-set-variable,-Wunused-result,-Wno-zero-as-null-pointer-constant
					--expt-relaxed-constexpr --extended-lambda --Werror=cross-execution-space-call -Xcudafe=--display_error_number -Xcudafe=--diag_error=incompatible_assignment_operands -Xcudafe=--diag_error=returning_ptr_to_local_variable -Xcudafe=--diag_error=subscript_out_of_range -Xcudafe=--diag_error=used_before_set -Xcudafe=--diag_error=undefined_preproc_id -Xcudafe=--diag_error=implicit_func_decl -Xcudafe=--diag_error=implicit_return_from_non_void_function -Xcudafe=--diag_error=missing_type_specifier
				>
				$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CUDA_COMPILER_ID:NVIDIA>>,$<NOT:$<CUDA_COMPILER_ID:Clang>>>:
					-Werror
					-Wall
					-Wextra  # (activates -Wunknown-pragmas)
					-Wextra-semi
					-Wpedantic
					# -Wabi=13 -Wabi-tag (maybe important when linking with very old libraries)
					-Waddress
					# -Waddress-of-packed-member (gcc 11, not in gcc 8)
					# -Waggregate-return (disallow return classes or structs, seems a C-compatibility warning)
					-Waggressive-loop-optimizations
					-Walloc-zero  # -Walloc-size-larger-than=
					-Walloca  # -Walloca-larger-than=
					# -Wanalyzer-double-fclose -Wanalyzer-double-free -Wanalyzer-exposure-through-output-file -Wanalyzer-file-leak -Wanalyzer-free-of-non-heap -Wanalyzer-malloc-leak (gcc 11, not in gcc 9)
					# -Wanalyzer-mismatching-deallocation (gcc 11, not in gcc 10)
					# -Wanalyzer-null-argument -Wanalyzer-null-dereference -Wanalyzer-possible-null-argument -Wanalyzer-possible-null-dereference (gcc 11, not in gcc 9)
					# -Wanalyzer-shift-count-negative -Wanalyzer-shift-count-overflow (gcc 11, not in gcc 10)
					# -Wanalyzer-stale-setjmp-buffer -Wanalyzer-tainted-array-index -Wanalyzer-too-complex -Wanalyzer-unsafe-call-within-signal-handler -Wanalyzer-use-after-free -Wanalyzer-use-of-pointer-in-stale-stack-frame (gcc 11, not in gcc 9)
					# -Wanalyzer-write-to-const -Wanalyzer-write-to-string-literal (gcc 11, not in gcc 10)
					# -Warith-conversion (gcc 11, not in gcc 9)
					-Warray-bounds 
					# -Warray-parameter #  -Warray-bounds=<0,2> -Warray-parameter=<0,2>  (gcc 11, not in gcc 10)
					# -Wattribute-warning (gcc 11, not in gcc 8)
					-Wattributes
					-Wbool-compare -Wbool-operation
					-Wbuiltin-declaration-mismatch -Wbuiltin-macro-redefined
					-Wc++0x-compat -Wc++14-compat -Wc++17-compat -Wc++1z-compat
					# -Wc++20-compat -Wc++2a-compat (gcc 11, not in gcc 9)
					# -Wcannot-profile (gcc 11, not in gcc 8)
					-Wcast-align -Wcast-align=strict -Wcast-function-type -Wcast-qual
					-Wcatch-value  # -Wcatch-value=<0, 3>
					-Wchar-subscripts -Wclobbered -Wcomment  # (same as -Wcomments)
					-Wconditionally-supported
					-Wconversion -Wconversion-null
					-Wcoverage-mismatch -Wcpp 
					# -Wctad-maybe-unsupported  # TODO(correaa) add ctad explicitly as necessary
					-Wctor-dtor-privacy
					-Wdangling-else -Wdate-time
					-Wdelete-incomplete -Wdelete-non-virtual-dtor
					-Wdeprecated 
					# -Wdeprecated-copy -Wdeprecated-copy-dtor (gcc 11, not in gcc 8)
					-Wdeprecated-declarations
					# -Wdeprecated-enum-enum-conversion -Wdeprecated-enum-float-conversion (gcc 11, not in gcc 10)
					-Wdisabled-optimization -Wdiv-by-zero -Wdouble-promotion
					-Wduplicated-branches -Wduplicated-cond
					# -Weffc++ (doesn't allow some advanced techniques, such as CRTP)
					-Wempty-body -Wendif-labels
					-Wenum-compare
					# -Wenum-conversion (gcc 11, not in gcc 10)
					-Wexpansion-to-defined
					# -Wexceptions  (gcc 11, not in gcc 10)
					# -Wextra (see above)
					-Wfloat-conversion # -Wfloat-equal (disallows float equality)
					-Wformat-truncation -Wformat=2
					# -Wformat-diag (gcc 11, not in gcc 9)
					-Wformat-extra-args -Wformat-nonliteral -Wformat-security -Wformat-signedness -Wformat-truncation -Wformat-y2k -Wformat-zero-length  # -Wformat-overflow=<0,2>
					-Wframe-address  # -Wframe-larger-than=<byte-size>
					-Wfree-nonheap-object -Whsa -Wif-not-aligned
					-Wignored-attributes -Wignored-qualifiers
					-Wimplicit-fallthrough#=3  # -Wimplicit-fallthrough=<0,5>
					-Winit-self
					# -Winline
					-Wint-in-bool-context -Wint-to-pointer-cast
					# -Winvalid-imported-macros (gcc 11, not in gcc 10)
					-Winvalid-memory-model -Winvalid-offsetof -Winvalid-pch
					-Wignored-qualifiers
					# -Wlarger-than=<byte-size>  # (disallow large objects types? in executable)
					-Wliteral-suffix
					-Wlogical-not-parentheses -Wlogical-op
					-Wlong-long -Wlto-type-mismatch -Wmain -Wmaybe-uninitialized
					-Wmemset-elt-size -Wmemset-transposed-args
					-Wmisleading-indentation
					# -Wmismatched-dealloc -Wmismatched-new-delete (gcc 11, not in gcc 10)
					# -Wmismatched-tags (gcc 11, not in gcc 9)
					-Wmissing-attributes -Wmissing-braces -Wmissing-declarations -Wmissing-field-initializers -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn
					# -Wmissing-profile (gcc 11, not in gcc 8)
					-Wmultichar
					# -Wmultiple-inheritance (disallows composition by inheritance)
					-Wmultistatement-macros
					# -Wnamespaces (disallows use of namespaces, seems a C-tool)
					-Wnarrowing
					-Wnoexcept -Wnoexcept-type
					-Wnormalized  #=nfc -Wnormalized=[none|id|nfc|nfkc]
					-Wnon-template-friend
					-Wnon-virtual-dtor
					-Wnonnull -Wnonnull-compare -Wnull-dereference
					-Wodr -Wold-style-cast -Wopenmp-simd -Woverflow
					-Woverlength-strings -Woverloaded-virtual
					-Wpacked -Wpacked-bitfield-compat -Wpacked-not-aligned
					# -Wpadded (disallows structs that need padding for alignment)
					-Wparentheses
					# -Wpedantic (see above)
					# -Wpessimizing-move (gcc 11, not in gcc 8)
					-Wplacement-new  #=1  -Wplacement-new=<0,2>
					-Wpmf-conversions
					-Wpointer-arith -Wpointer-compare
					-Wpragmas
					# -Wprio-ctor-dtor (gcc 11, not in gcc 8) 
					-Wpsabi
					# -Wrange-loop-construct (gcc 11, not in gcc 10)
					-Wredundant-decls
					# -Wredundant-move (gcc 11, not in gcc 8)
					# -Wredundant-tags (gcc 11, not in gcc 9)
					-Wregister
					-Wreturn-local-addr -Wreturn-type
					-Wrestrict -Wreorder
					-Wscalar-storage-order -Wsequence-point
					-Wshadow -Wshadow-compatible-local -Wshadow-local -Wshadow=compatible-local -Wshadow=local
					-Wshift-count-negative -Wshift-count-overflow -Wshift-negative-value -Wshift-overflow  #=1 -Wshift-overflow=<0,2>
					-Wsign-compare -Wsign-conversion -Wsign-promo
					-Wsized-deallocation
					-Wsizeof-array-argument 
					# -Wsizeof-array-div (gcc 11, not in gcc 10)
					-Wsizeof-pointer-div -Wsizeof-pointer-memaccess
					-Wstack-protector  # -Wstack-usage=<byte-size>
					-Wstrict-aliasing  #=3  -Wstrict-aliasing=<0,3>
					-Wstrict-null-sentinel  #=1  -Wstrict-overflow=<0,5>
					-Wstrict-overflow  #=1  -Wstrict-overflow=<0,5>
					# -Wstring-compare (gcc 11, not in gcc 9)
					-Wstringop-overflow  #=2  -Wstringop-overflow=<0,4>
					# -Wstringop-overread (gcc 11, not in gcc 10)
					-Wstringop-truncation
					-Wsubobject-linkage
					-Wsuggest-attribute=const -Wsuggest-attribute=format -Wsuggest-attribute=malloc -Wsuggest-attribute=noreturn -Wsuggest-attribute=pure -Wsuggest-final-methods -Wsuggest-final-types
					-Wswitch -Wswitch-bool -Wswitch-default -Wswitch-enum
					# -Wswitch-outside-range (gcc 11, not in gcc 9)
					-Wswitch-unreachable
					-Wsync-nand -Wsynth
					# -Wsystem-headers (expects system headers to be warning-compliant which they are not)
					-Wtautological-compare
					# -Wtemplates (disallows templates, C-tool)
					-Wtrampolines -Wtrigraphs
					# -Wtsan (gcc 11, not in gcc 10)
					-Wtype-limits -Wundef -Wuninitialized
					-Wno-unknown-pragmas  # (see above) -Wunknown-pragmas (other compilers need their own pragmas for their warnings)
					-Wunreachable-code -Wunsafe-loop-optimizations
					-Wunused -Wunused-parameter -Wunused-but-set-variable 
					# -Wunused-const-variable  #=2 TODO(correaa) add [[maybe_unused]] to niebloids
					-Wunused-function -Wunused-label -Wunused-local-typedefs -Wunused-macros -Wunused-parameter -Wunused-result -Wunused-value -Wunused-variable
					-Wuseless-cast
					-Wvarargs -Wvariadic-macros -Wvector-operation-performance
					# -Wvexing-parse (gcc 11, not in gcc 10)
					-Wvirtual-inheritance -Wvirtual-move-assign
					-Wvla
					# -Wvla-parameter (gcc 11, not in gcc 10)
					# -Wvolatile (gcc 11, not in gcc 9)
					-Wvolatile-register-var
					-Wwrite-strings
					-Wzero-as-null-pointer-constant
				>
				$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CUDA_COMPILER_ID:Clang>>:
					-Werror
					-Wall
					-Wextra
					-Wcast-align
					-Wcast-qual
					-Wconversion
					-Wdouble-promotion
					-Wformat=2
					-Wmissing-include-dirs
					-Wno-unknown-pragmas
					-Wnon-virtual-dtor
					-Wnull-dereference
					-Wmove
					-Wold-style-cast
					-Woverloaded-virtual
					-Wpointer-arith
					-Wpedantic
					-Wredundant-decls
					-Wshadow
					-Wold-style-cast
					-Woverloaded-virtual
					-Wpedantic
					-Wsign-compare
					-Wsign-conversion
					-Wstrict-aliasing
					-Wswitch-enum
					-Wtype-limits
					-Wuninitialized
					-Wunused
					-Wunused-result
					-Wzero-as-null-pointer-constant
		        >
				$<$<CXX_COMPILER_ID:Intel>:  # also IntelLLVM, XL (ibm), XLClang (ibm)
					-Werror
					-Wall
					-Wextra
					-diag-disable=remark
					-diag-error:3846
					-diag-disable=1011  # disables warning missing return at the end of non-void function
					-wd161
					-Wabi
					-Warray-bounds
					-Wcast-qual
					-Wchar-subscripts
					-Wcomment
					-Wdeprecated
					-Wenum-compare
					-Wextra-tokens
					-Wformat
					-Wformat=2
					-Wformat-security
					-Wic-pointer
					-Wnarrowing
					-Wno-return-type
					-Wnon-virtual-dtor
					-Wnonnull
					-Wmaybe-uninitialized
					-Wmain
					-Wmissing-declarations
					-Wmissing-prototypes
					-Wmultichar
					-Woverloaded-virtual
					-Woverflow
					-Wparentheses
					-Wpointer-arith
					-Wpointer-sign
					-Wreorder
					-Wreturn-type
					-Wsequence-point
					-Wshadow
					-Wsign-compare
					-Wshorten-64-to-32
					-Wno-strict-aliasing
					-Wstrict-prototypes
					-Wtrigraphs
					-Wtype-limits
					-Wuninitialized
					-Wunused
					-Wunused-but-set-variable
					-Wunused-function
					-Wunused-parameter
					-Wunused-variable
					-Wwrite-strings
				>
				$<$<OR:$<CXX_COMPILER_ID:PGI>,$<CXX_COMPILER_ID:NVHPC>>:
					-Werror
					-Wall
					-Wcast-qual
					-Wformat=2
					-Wshadow
					-Wsign-compare
					-Wuninitialized
					-Wunused
					-Wunused-parameter
				>
				$<$<CXX_COMPILER_ID:MSVC>:
					/WX
					/W4
				>
		)
	else()
		target_compile_definitions(${TEST_EXE} PRIVATE "BOOST_TEST_DYN_LINK=1")
	endif()
	add_test(NAME ${TEST_EXE} COMMAND ./${TEST_EXE})
endforeach()
