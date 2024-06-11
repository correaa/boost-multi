// Copyright 2021-2023 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <array>

#if(__cplusplus >= 202002L)
	#include <ranges>
#endif

#include <algorithm>
#include <numeric>  // for std::iota

// Suppress warnings from boost.test
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

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_rotate_3d) {
	multi::array<double, 3> arr({ 3, 4, 5 });

	BOOST_REQUIRE( std::get<0>(sizes(arr)) == 3 );
	BOOST_REQUIRE( std::get<1>(sizes(arr)) == 4 );
	BOOST_REQUIRE( std::get<2>(sizes(arr)) == 5 );

	auto&& RA = rotated(arr);
	BOOST_REQUIRE(( sizes(RA) == decltype(sizes(RA)){4, 5, 3} ));
	BOOST_REQUIRE(  &arr[0][1][2] == &RA[1][2][0] );

	auto&& UA = unrotated(arr);
	BOOST_REQUIRE(( sizes(UA) == decltype(sizes(UA)){5, 3, 4} ));
	BOOST_REQUIRE( &arr[0][1][2] == &UA[2][0][1] );

	auto&& RRA = rotated(RA);
	BOOST_REQUIRE(( sizes(RRA) == decltype(sizes(RRA)){5, 3, 4} ));
	BOOST_REQUIRE( &arr[0][1][2] == &RRA[2][0][1] );
}

BOOST_AUTO_TEST_CASE(multi_rotate_4d) {
	multi::array<double, 4> original({ 14, 14, 7, 4 });

	auto&& unrotd = original.unrotated();
	BOOST_REQUIRE(( sizes(unrotd) == decltype(sizes(unrotd)){4, 14, 14, 7} ));
	BOOST_REQUIRE( &original[0][1][2][3] == &unrotd[3][0][1][2] );

	auto&& unrotd2 = original.unrotated().unrotated();
	BOOST_REQUIRE(( sizes(unrotd2) == decltype(sizes(unrotd2)){7, 4, 14, 14} ));
	BOOST_REQUIRE( &original[0][1][2][3] == &unrotd2[2][3][0][1] );
}

BOOST_AUTO_TEST_CASE(multi_rotate_4d_op) {
	multi::array<double, 4> original({ 14, 14, 7, 4 });

	auto&& unrotd = (original.unrotated());
	BOOST_REQUIRE(( sizes(unrotd) == decltype(sizes(unrotd)){4, 14, 14, 7} ));
	BOOST_REQUIRE( &original[0][1][2][3] == &unrotd[3][0][1][2] );

	auto&& unrotd2 = (original.unrotated().unrotated());
	BOOST_REQUIRE(( sizes(unrotd2) == decltype(sizes(unrotd2)){7, 4, 14, 14} ));
	BOOST_REQUIRE( &original[0][1][2][3] == &unrotd2[2][3][0][1] );
}

BOOST_AUTO_TEST_CASE(multi_rotate_part1) {
	// clang-format off
	std::array<std::array<int, 5>, 4> stdarr = {{
		{{ 0,  1,  2,  3,  4}},
		{{ 5,  6,  7,  8,  9}},
		{{10, 11, 12, 13, 14}},
		{{15, 16, 17, 18, 19}},
	}};
	// clang-format on

	std::array<std::array<int, 5>, 4> stdarr2 = {};

	multi::array_ref<int, 2> arr(&stdarr[0][0], { 4, 5 });    // NOLINT(readability-container-data-pointer) test access
	multi::array_ref<int, 2> arr2(&stdarr2[0][0], { 4, 5 });  // NOLINT(readability-container-data-pointer) test access

	rotated(arr2) = rotated(arr);
	BOOST_REQUIRE( arr2[1][1] == 6  );
	BOOST_REQUIRE( arr2[2][1] == 11 );
	BOOST_REQUIRE( arr2[1][2] == 7  );
	BOOST_REQUIRE( (arr2.rotated() ) == (arr.rotated() ) );
	BOOST_REQUIRE( (arr2.rotated() )[2][1] == 7 );
}

BOOST_AUTO_TEST_CASE(multi_rotate) {
	{
		multi::array<int, 2> arr = {
			{ 00, 01 },
			{ 10, 11 },
		};
		BOOST_REQUIRE(       arr[1][0] == 10 );
		BOOST_REQUIRE( (arr.rotated())[0][1] == 10 );
		BOOST_REQUIRE( &     arr[1][0] == &(arr.rotated() )[0][1] );

		BOOST_REQUIRE( arr.transposed()[0][1] == 10 );
		BOOST_REQUIRE( transposed(arr)[0][1] == 10 );
		BOOST_REQUIRE( (~arr)[0][1] == 10 );
		BOOST_REQUIRE( &arr[1][0] == &arr.transposed()[0][1] );

		(arr.rotated())[0][1] = 100;
		BOOST_REQUIRE( arr[1][0] == 100 );
	}
	{
		multi::array<double, 3> arr({ 11, 13, 17 });
		BOOST_REQUIRE( & arr[3][5][7] == &   arr.transposed()[5][3][7] );
		BOOST_REQUIRE( & arr[3][5][7] == & transposed(arr)   [5][3][7] );
		BOOST_REQUIRE( & arr[3][5][7] == & (~arr)            [5][3][7] );
		BOOST_REQUIRE( & arr[3][5][7] == &   arr[3].transposed()[7][5] );
		BOOST_REQUIRE( & arr[3][5][7] == & (~arr[3])            [7][5] );

		BOOST_REQUIRE( & arr[3][5] == & (~arr)[5][3] );

		BOOST_REQUIRE( & ~~arr          == & arr      );
		BOOST_REQUIRE( &  (arr.rotated().rotated().rotated() )     == & arr       );
		BOOST_REQUIRE( &   arr()          == & (arr.rotated().rotated().rotated() ) );
		BOOST_REQUIRE( &  (arr.rotated() )     != & arr      );
		BOOST_REQUIRE( &  (arr.unrotated().rotated()) == & arr      );

		std::iota(arr.data_elements(), arr.data_elements() + arr.num_elements(), 0.1);
		BOOST_REQUIRE( ~~arr == arr );
		BOOST_REQUIRE( arr.unrotated().rotated() == arr );
	}
	{
		multi::array<int, 2> const arr = {
			{ 00, 01 },
			{ 10, 11 },
		};
		BOOST_REQUIRE(   arr.rotated() [0][1] == 10 );
		BOOST_REQUIRE( &(arr.rotated())[1][0] == &arr[0][1] );
		BOOST_REQUIRE( &(~arr)[1][0] == &arr[0][1] );
	}
}

BOOST_AUTO_TEST_CASE(multi_transposed) {
	multi::array<int, 2> const arr0 = {
		{  9, 24, 30, 9 },
		{  4, 10, 12, 7 },
		{ 14, 16, 36, 1 },
	};
	multi::array<int, 2> const arr1 = arr0.transposed();
	multi::array<int, 2> const arr2 = ~arr0;
	BOOST_REQUIRE( arr1 == arr2 );
}

BOOST_AUTO_TEST_CASE(miguel) {
	multi::array<double, 2> G2D({ 41, 35 });
	auto const&             G3D = G2D.rotated().partitioned(7).sliced(0, 3).unrotated();

	BOOST_REQUIRE( &G3D[0][0][0] == &G2D[0][0] );
}

#if(__cplusplus >= 202002L)
#if defined(__cpp_lib_ranges_repeat) && (__cpp_lib_ranges_repeat >= 202207L)
auto meshgrid(auto const& x, auto const& y) {
	return std::pair{ x.broadcasted().rotated(), y.broadcasted() };
}

template<class X1D, class Y1D>
auto meshgrid_copy(X1D const& x, Y1D const& y) {
	auto ret = std::pair{
		multi::array<typename X1D::element_type, 2>({ x.size(), y.size() }),
		multi::array<typename Y1D::element_type, 2>(std::views::repeat(y, x.size()))
	};

	std::ranges::fill(ret.first.rotated(), x);
	
	return ret;
}

BOOST_AUTO_TEST_CASE(matlab_meshgrid) {
	auto const x = multi::array{ 1, 2, 3 };
	auto const y = multi::array{ 1, 2, 3, 4, 5 };

	auto const& [X, Y] = meshgrid(x, y);

	auto const [X_copy, Y_copy] = meshgrid_copy(x, y);

	for(auto i : x.extension()) {
		for(auto j : y.extension()) {
			BOOST_TEST( X[i][j] == X_copy[i][j] );
			BOOST_TEST( Y[i][j] == Y_copy[i][j] );
		}
	}
}
#endif
#endif
