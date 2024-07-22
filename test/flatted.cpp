// Copyright 2018-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunknown-warning-option"
	#pragma clang diagnostic ignored "-Wconversion"
	#pragma clang diagnostic ignored "-Wextra-semi-stmt"
	#pragma clang diagnostic ignored "-Wold-style-cast"
	#pragma clang diagnostic ignored "-Wundef"
	#pragma clang diagnostic ignored "-Wsign-conversion"
	#pragma clang diagnostic ignored "-Wswitch-default"
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wconversion"
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wsign-conversion"
	#pragma GCC diagnostic ignored "-Wundef"
#endif

#ifndef BOOST_TEST_MODULE
	#define BOOST_TEST_MAIN
#endif

#include <boost/test/included/unit_test.hpp>

#include <boost/multi/array.hpp>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(array_flatted_2d) {
	multi::array<int, 2> arr = {
		{0, 1, 2},
		{3, 4, 5},
	};

	BOOST_REQUIRE( arr.flatted()[1] == 1 );
	BOOST_REQUIRE( arr.flatted()[4] == 4 );

	arr.flatted()[4] = 44;
}

BOOST_AUTO_TEST_CASE(array_flatted_3d) {
	multi::array<double, 3> arr({ 13, 4, 5 });

	BOOST_REQUIRE( arr.size() == 13 );
	// BOOST_REQUIRE( arr.rotated().is_flattable() );

	{
		auto&& arrRFU = arr.rotated().flatted().unrotated();  // TODO(correaa) remove flatted?
		BOOST_REQUIRE( &arrRFU[11][7] == &arr[11][1][2] );
	}
	{
		auto&& arrRFU = (arr.rotated()).flatted().unrotated();
		BOOST_REQUIRE( &arrRFU[11][7] == &arr[11][7/5][7%5] );
	}
}

BOOST_AUTO_TEST_CASE(array_flatted_3d_bis) {
	multi::array<double, 3> const arr({ 13, 4, 5 });
	BOOST_REQUIRE( arr.size() == 13 );
	// BOOST_REQUIRE( arr.is_flattable() );
	BOOST_REQUIRE( arr.flatted().size() == 52 );
}

BOOST_AUTO_TEST_CASE(empty_array_3D_flatted) {
	multi::array<double, 3> const arr;
	// BOOST_REQUIRE( arr.is_flattable() );
	BOOST_REQUIRE( arr.flatted().size() == 0 );
}

BOOST_AUTO_TEST_CASE(empty_array_2D_flatted) {
	multi::array<double, 2> const arr;
	// BOOST_REQUIRE( arr.is_flattable() );
	BOOST_REQUIRE( arr.flatted().size() == 0 );
}
