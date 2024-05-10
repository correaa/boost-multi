// Copyright 2023-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <algorithm>  // for std::ranges::fold_left

// Suppress warnings from boost.test
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wold-style-cast"
#  pragma clang diagnostic ignored "-Wundef"
#  pragma clang diagnostic ignored "-Wconversion"
#  pragma clang diagnostic ignored "-Wsign-conversion"
#  pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wold-style-cast"
#  pragma GCC diagnostic ignored "-Wundef"
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wsign-conversion"
#  pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#ifndef BOOST_TEST_MODULE
#  define BOOST_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(range_accumulate) {
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
	namespace multi = boost::multi;

	static constexpr auto accumulate = [](auto const& R) {return std::ranges::fold_left(R, 0, std::plus<>{});};

	auto const values = multi::array<int, 2>{
        {2, 0, 2, 2},
        {2, 2, 0, 4},
        {2, 2, 0, 4},
        {2, 2, 0, 0},
        {2, 7, 0, 2},
        {2, 2, 4, 4},
    };

    constexpr auto rowOddSum = [](auto const& arr) {
        return std::ranges::find_if(arr, [](auto const& row) {return (accumulate(row) & 1) == 1;});
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
		auto const needle = std::ranges::find_if(a, [](auto const& row) {return row[0] == 9;});
		BOOST_REQUIRE(needle == a.end());
	}
	{
		auto const needle = std::ranges::find(a, a[1]);
		BOOST_REQUIRE(needle != a.end());
		BOOST_REQUIRE( *needle == a[1] );
	}
#endif
}
