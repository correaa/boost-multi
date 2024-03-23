// Copyright 2024 Alfredo A. Correa

#include <boost/test/unit_test.hpp>

#include <multi/array.hpp>

#ifdef TBB_FOUND
#if !defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
#include <execution>
#endif
#endif

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_construct_1d) {
	multi::static_array<double, 1> const arr(multi::extensions_t<1>{multi::iextension{10}}, 1.0);
	//  multi::static_array<double, 1> arr(multi::array<double, 1>::extensions_type{10}, 1.);
	BOOST_REQUIRE( size(arr) == 10 );
	BOOST_REQUIRE( arr[1] == 1.0 );

	// #if defined(__cpp_lib_execution) && (__cpp_lib_execution >= 201603L)
	// #if defined(__INTEL_COMPILER) || defined(__NVCOMPILER)
	// multi::static_array<double, 1> arr2(arr);
	#ifdef TBB_FOUND
	#if !defined(__NVCC__) && !(defined(__clang__) && defined(__CUDA__))
	multi::static_array<double, 1> const arr2(std::execution::par, arr);

	BOOST_REQUIRE( arr2 == arr );

	#endif
	#endif
}
