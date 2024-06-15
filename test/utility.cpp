// Copyright 2018-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wold-style-cast"
	#pragma clang diagnostic ignored "-Wundef"
	#pragma clang diagnostic ignored "-Wconversion"
	#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wundef"
	#pragma GCC diagnostic ignored "-Wconversion"
	#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#ifndef BOOST_TEST_MODULE
	#define BOOST_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>

#if defined(__clang__)
	#pragma clang diagnostic pop
#elif defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

#include <boost/multi/array.hpp>  // for array_ref, data_elements, num_el...

#include <array>        // for array
#include <iterator>     // for begin, end, iterator_traits, rend
#include <numeric>      // for iota
#include <type_traits>  // for is_same
#include <utility>      // for addressof
#include <vector>       // for vector, allocator

namespace multi = boost::multi;

// TODO(correaa) add test for reinterpret_pointer_cast

BOOST_AUTO_TEST_CASE(std_array_extensions_3d) {
	std::array<std::array<std::array<double, 5>, 4>, 3> arr = {};

	static_assert(std::is_same<typename multi::array_traits<decltype(arr)>::element, double>{});

	BOOST_REQUIRE( multi::dimensionality(arr) == 3 );

	// BOOST_REQUIRE( multi::extension(arr) == 3 );

	BOOST_REQUIRE(( multi::extensions(arr) == decltype(multi::extensions(arr)){3, 4, 5} ));

#ifndef _MSC_VER  // problem with 14.3 c++17
	using multi::data_elements;
	BOOST_REQUIRE( data_elements(arr) == &arr[0][0][0] );  // NOLINT(readability-container-data-pointer)
	BOOST_REQUIRE( data_elements(arr) ==  arr[0][0].data() );

	using multi::num_elements;
	BOOST_REQUIRE( num_elements(arr) == 60 );
#endif

	multi::array<double, 3> const marr(
#ifdef _MSC_VER  // problem with 14.3 c++17
		multi::extensions_t<3>
#endif
		{ 3, 4, 5 }
	);
	using multi::layout;
	BOOST_REQUIRE( layout(arr) == layout(marr) );

	BOOST_REQUIRE( multi::extensions(arr) == extensions(marr) );
}

BOOST_AUTO_TEST_CASE(std_array_extensions_2d) {
	std::array<std::array<double, 4>, 3> arr = {};

	static_assert(std::is_same<typename multi::array_traits<decltype(arr)>::element, double>{});

	using multi::dimensionality;
	BOOST_REQUIRE( dimensionality(arr) == 2 );

	// using multi::extension;
	// BOOST_REQUIRE( extension(arr) == 3 );

	using multi::extensions;
	BOOST_REQUIRE(( extensions(arr) == decltype(extensions(arr)){3, 4} ));

	using multi::data_elements;
	BOOST_REQUIRE( data_elements(arr) == &arr[0][0] );  // NOLINT(readability-container-data-pointer) test access
	BOOST_REQUIRE( data_elements(arr) ==  arr[0].data() );
	BOOST_REQUIRE( data_elements(arr) ==  arr.front().data() );

	using multi::num_elements;
	BOOST_REQUIRE( num_elements(arr) == 12 );

	multi::array<double, 2> const marr({ 3, 4 });
	using multi::layout;
	BOOST_REQUIRE( layout(arr) == layout(marr) );

	BOOST_REQUIRE( extensions(arr) == extensions(marr) );
}

BOOST_AUTO_TEST_CASE(std_array_extensions_1d) {
	std::array<double, 4> arr = {};

	static_assert(std::is_same<typename multi::array_traits<decltype(arr)>::element, double>{});

	using multi::dimensionality;
	BOOST_REQUIRE( dimensionality(arr) == 1 );

	// using multi::extension;
	// BOOST_REQUIRE( extension(arr) == 4 );

	using multi::extensions;
	BOOST_REQUIRE(( extensions(arr) == decltype(extensions(arr)){multi::iextension{4}} ));

	using multi::data_elements;
	BOOST_REQUIRE( data_elements(arr) == &arr[0] );  // NOLINT(readability-container-data-pointer) test access
	BOOST_REQUIRE( data_elements(arr) ==  arr.data() );

	using multi::num_elements;
	BOOST_REQUIRE( num_elements(arr) == 4 );
}

BOOST_AUTO_TEST_CASE(test_utility_1d) {
	std::array<int, 10> carr = {
		{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	};

	multi::array_ref<int, 1> marr(carr.data(), { multi::iextension{ 10 } });

	std::vector<int> varr(10);  // NOLINT(fuchsia-default-arguments-calls)
	std::iota(begin(varr), end(varr), 0);
	std::array<int, 10> aarr{};
	std::iota(begin(aarr), end(aarr), 0);

	BOOST_REQUIRE( size(marr) == 10 );

	BOOST_REQUIRE( static_cast<multi::size_t>(carr.size()) == size(marr) );
	BOOST_REQUIRE( static_cast<multi::size_t>(aarr.size()) == size(marr) );

	BOOST_REQUIRE( carr[7] == marr[7] );
	BOOST_REQUIRE( varr[7] == marr[7] );
	BOOST_REQUIRE( aarr[7] == marr[7] );

	BOOST_REQUIRE( &carr[7] == &marr[7] );

	using multi::num_elements;
	BOOST_REQUIRE( num_elements(carr) == num_elements(marr) );
	// BOOST_REQUIRE( num_elements(varr) == num_elements(marr) );
	BOOST_REQUIRE( num_elements(aarr) == num_elements(aarr) );

	using multi::data_elements;
	BOOST_REQUIRE( carr.data() == data_elements(marr) );

	BOOST_REQUIRE( *begin(varr) == *begin(marr) );

	using std::begin;
	BOOST_REQUIRE( *begin(carr) == *begin(marr) );

	using std::rend;
	BOOST_REQUIRE( *(end(varr)-1) == *(end(marr)-1) );

	using std::end;
	BOOST_REQUIRE( *(end(carr)-1) == *(end(marr)-1) );
}

BOOST_AUTO_TEST_CASE(test_utility_2d) {
	// clang-format off
	std::array<std::array<int, 10>, 3> carr{{
		{{ 00, 10, 20, 30, 40, 50, 60, 70, 80, 90 }},
		{{ 100, 110, 120, 130, 140, 150, 160, 170, 180, 190 }},
		{{ 200, 210, 220, 230, 240, 250, 260, 270, 280, 290 }},
	}};
	// clang-format on

	multi::array_ref<int, 2> marr(&carr[0][0], { 3, 10 });  // NOLINT(readability-container-data-pointer) tests access

	BOOST_REQUIRE( static_cast<multi::size_t>(carr.size()) == size(marr) );

	BOOST_REQUIRE( carr[1][7] == marr[1][7] );

	BOOST_REQUIRE( &carr[1][7] == &marr[1][7] );

	using multi::num_elements;
	BOOST_REQUIRE( num_elements(carr) == num_elements(marr) );

	using multi::data_elements;
	BOOST_REQUIRE( data_elements(carr) == data_elements(marr) );
}

BOOST_AUTO_TEST_CASE(multi_utility_test) {
	static_assert(std::is_same_v<std::iterator_traits<int const*>::value_type, int>);

	using multi::corigin;
	using multi::dimensionality;
	using multi::extension;
	using multi::extensions;
	using multi::num_elements;
	using multi::size;
	using multi::sizes;
	{
		int arr[4] = { 10, 20, 30, 40 };  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) test legacy types
		BOOST_REQUIRE( dimensionality(arr) == 1 );
		BOOST_REQUIRE( extension(arr).first() == 0 );
		BOOST_REQUIRE( extension(arr).last() == 4 );

		BOOST_REQUIRE( size(arr) == 4 );

		using boost::multi::detail::get;
		BOOST_REQUIRE( get<0>(sizes(arr)) == size(arr) );
		using multi::get_allocator;

		static_assert(std::is_same_v<decltype(get_allocator(arr)), std::allocator<int>>);

		using std::addressof;

		using multi::data_elements;
		static_assert(std::is_same_v<decltype(data_elements(arr)), int*>);
		//  BOOST_REQUIRE( data(A) == addressof(A[0]) );
		BOOST_REQUIRE(data_elements(arr) == addressof(arr[0]));
	}
	{
		int arr[2][3] = {
  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : test legacy types
			{10, 20, 30},
			{40, 50, 60},
		};
		BOOST_REQUIRE( dimensionality(arr) == 2 );
		BOOST_REQUIRE( extension(arr).first() == 0 );
		BOOST_REQUIRE( extension(arr).last() == 2 );

		arr[0][0] = 990;

		BOOST_REQUIRE( arr[0][0] == 990 );
		BOOST_REQUIRE( corigin(arr) == &arr[0][0] );
		BOOST_REQUIRE( size(arr) == 2 );

		using multi::detail::get;
		BOOST_REQUIRE( get<0>(sizes(arr)) == size(arr) );
		BOOST_REQUIRE( num_elements(arr) == 6 );

		static_assert(num_elements(arr) == 6);
	}
}
