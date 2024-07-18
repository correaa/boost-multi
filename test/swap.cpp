// Copyright 2020-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 10.
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

#include <boost/multi/array.hpp>  // for array, apply, array_types<>::ele...

// #include <algorithm>  // for copy, equal, fill_n, move
// #include <iterator>   // for size, back_insert_iterator, back...
// #include <memory>     // for unique_ptr, make_unique, allocat...
// IWYU pragma: no_include <type_traits>  // for remove_reference<>::type
// IWYU pragma: no_include <map>
// #include <utility>  // for move
// #include <vector>   // for vector, operator==, vector<>::va...

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(swap_array_1D) {
	multi::array<int, 1> arr1 = {  0,   1,   2,   3};
	multi::array<int, 1> arr2 = {100, 101, 102};

	using std::swap;
	swap(arr1, arr2);

	BOOST_TEST( arr1[1] == 101 );
	BOOST_TEST( arr2[1] ==   1 );
}

BOOST_AUTO_TEST_CASE(swap_array_2D) {
	multi::array<int, 2> arr1 = {
		{00, 01, 02, 03},
		{10, 11, 12, 13},
		{20, 21, 22, 23},
	};

	multi::array<int, 2> arr2 = {
		{100, 101, 102, 103},
		{110, 111, 112, 113},
	};

	using std::swap;
	swap(arr1, arr2);

	BOOST_TEST( arr1[1][1] == 111 );
	BOOST_TEST( arr2[1][1] ==  11 );
}

BOOST_AUTO_TEST_CASE(swap_subarray_1D) {
	multi::array<int, 1> arr1 = {  0,   1,   2,   3};
	multi::array<int, 1> arr2 = {100, 101, 102, 103};

	using std::swap;
	swap(arr1(), arr2());

	BOOST_TEST( arr1[1] == 101 );
	BOOST_TEST( arr2[1] ==   1 );
}

BOOST_AUTO_TEST_CASE(swap_subarray_2D) {
	multi::array<int, 2> arr1 = {
		{00, 01, 02, 03},
		{10, 11, 12, 13},
		{20, 21, 22, 23},
	};

	multi::array<int, 2> arr2 = {
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
	};

	using std::swap;
	swap(arr1(), arr2());

	BOOST_TEST( arr1[1][1] == 111 );
	BOOST_TEST( arr2[1][1] ==  11 );
}

BOOST_AUTO_TEST_CASE(swap_const_subarray_2D) {
	multi::array<int, 2> const arr1 = {
		{00, 01, 02, 03},
		{10, 11, 12, 13},
		{20, 21, 22, 23},
	};

	multi::array<int, 2> arr2 = {
		{100, 101, 102, 103},
		{110, 111, 112, 113},
		{120, 121, 122, 123},
	};

	// using std::swap;
	// swap(arr1(), arr2());

	BOOST_TEST( arr1[1][1] ==  11 );
	BOOST_TEST( arr2[1][1] == 111 );
}

