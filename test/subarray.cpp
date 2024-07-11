// Copyright 2019-2024 Alfredo A. Correa
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

#include <boost/multi/array.hpp>  // for implicit_cast, explicit_cast

#include <utility>  // for as_const

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(subarray_assignment) {
	multi::array<int, 3> A({3, 4, 5}, 99);

	auto constA2 = std::as_const(A)[2];
	BOOST_REQUIRE( constA2[1][1] == 99 );

	auto A2 = A[2];
	BOOST_REQUIRE( A2[1][1] == 99 );

	//  what(constA2, A2);
	//  A2[1][1] = 88;
}

BOOST_AUTO_TEST_CASE(subarray_base) {
	multi::array<int, 3> A({3, 4, 5}, 99);

	auto&& Asub = A();
	*Asub.base() = 88;

	BOOST_REQUIRE( A[0][0][0] == 88 );

	*A().base() = 77;

	BOOST_REQUIRE( A[0][0][0] == 77 );

	// *std::as_const(Asub).base() = 66;  // should not compile, read-only
}
