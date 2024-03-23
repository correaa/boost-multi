// Copyright 2024 Alfredo A. Correa

#include <boost/test/unit_test.hpp>

#include <multi/array.hpp>

#if defined(__INTEL_COMPILER) || defined(__NVCOMPILER)
#include <execution>
#endif

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_construct_1d) {
	multi::static_array<double, 1> arr(multi::extensions_t<1>{multi::iextension{10}}, 1.0);
	//  multi::static_array<double, 1> arr(multi::array<double, 1>::extensions_type{10}, 1.);
	BOOST_REQUIRE( size(arr) == 10 );
	BOOST_REQUIRE( arr[1] == 1.0 );

	// #if defined(__cpp_lib_execution) && (__cpp_lib_execution >= 201603L)
	#if defined(__INTEL_COMPILER) || defined(__NVCOMPILER)
	// multi::static_array<double, 1> arr2(arr);
	multi::static_array<double, 1> arr2(std::execution::par, arr);
	#endif
}
