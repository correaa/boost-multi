// Copyright 2023-2024 Alfredo A. Correa
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

#include <boost/multi/array.hpp>  // for array, subarray, static_array  // IWYU pragma: keep

#include <algorithm>    // for std::ran  // IWYU pragma: keep
#include <complex>      // for complex, real, operator==, imag  // IWYU pragma: keep
#include <iterator>     // for size, begin, end  // IWYU pragma: keep
#include <numeric>      // for iota  // IWYU pragma: keep
#include <type_traits>  // for is_same_v  // IWYU pragma: keep
#include <utility>      // for pair  // IWYU pragma: keep

BOOST_AUTO_TEST_CASE(range_accumulate) {
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
	namespace multi = boost::multi;

	static constexpr auto accumulate = [](auto const& R) { return std::ranges::fold_left(R, 0, std::plus<>{}); };

	auto const values = multi::array<int, 2>{
		{2, 0, 2, 2},
		{2, 2, 0, 4},
		{2, 2, 0, 4},
		{2, 2, 0, 0},
		{2, 7, 0, 2},
		{2, 2, 4, 4},
	};

	boost::multi::array<int, 1, std::allocator<int>> aaa = {1, 2, 3};

	constexpr auto rowOddSum = [](auto const& arr) {
		return std::ranges::find_if(arr, [](auto const& row) { return (accumulate(row) & 1) == 1; });
	};

	auto const result = rowOddSum(values);

	BOOST_REQUIRE( result - values.begin() == 4 );
#endif
}

BOOST_AUTO_TEST_CASE(range_find) {
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
	namespace multi = boost::multi;

	using Array2D = multi::array<int, 2>;

	Array2D const a = {
		{1, 2},
		{3, 4},
	};
	{
		auto const needle = std::ranges::find_if(a, [](auto const& row) { return row[0] == 9; });
		BOOST_REQUIRE(needle == a.end());
	}
	{
		std::ranges::equal_to eto;

		auto a2 = a();

		auto a1     = a[1];
		auto a1_val = +a[1];

		bool const res = eto(a1_val, a1);
		BOOST_REQUIRE( res );
	}
	{
		auto&&     a1     = a[1];

		auto const needle = std::ranges::find(a, a1);
		BOOST_REQUIRE(needle != a.end());
		BOOST_REQUIRE( *needle == a1 );
		BOOST_REQUIRE( *needle == a[1] );
	}
	[&] {
		auto const needle = std::ranges::find(a, a[1]);
		BOOST_REQUIRE(needle != a.end());
		BOOST_REQUIRE( *needle == a[1] );
	}();
#endif
}

// #if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L)
// BOOST_AUTO_TEST_CASE(range_copy_n_1D) {
//  namespace multi = boost::multi;

//  multi::array<int, 1> const X1 = {1, 2, 3};
//  multi::array<int, 1> X2(X1.extensions());

//  std::ranges::copy_n(X1.begin(), 10, X2.begin());

//  BOOST_REQUIRE( X1 == X2 );
// }

// BOOST_AUTO_TEST_CASE(range_copy_n) {
//  namespace multi = boost::multi;

//  multi::array<int, 2> const X1({ 10, 10 }, 99);
//  multi::array<int, 2> X2(X1.extensions());

//  std::ranges::copy_n(X1.begin(), 10, X2.begin());
//  BOOST_REQUIRE( X1 == X2 );
// }
// #endif
