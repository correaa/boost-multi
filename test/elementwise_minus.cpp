// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/elementwise/invoke.hpp>  // for invoke
#include <boost/multi/elementwise/minus.hpp>
#include <boost/multi/restriction.hpp>  // for bind_front_t, restriction

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <functional>  // IWYU pragma: keep
#include <utility>     // IWYU pragma: keep

namespace multi = boost::multi;

auto main() -> int {
	// NOLINTNEXTLINE(readability-identifier-length)
	auto const A = multi::array<int, 2>{
		{0, 1, 2},
		{3, 4, 5},
	};
	// NOLINTNEXTLINE(readability-identifier-length)
	auto const B = multi::array<int, 2>{
		{ 0, 10, 20},
		{30, 40, 50},
	};

	BOOST_TEST( multi::elementwise::minus(A, B)[1][1] == A[1][1] - B[1][1] );

	BOOST_TEST( multi::elementwise::minus(multi::array<int, 2>{
		{0, 1, 2},
		{3, 4, 5}
	}, B)[1][1] == A[1][1] - B[1][1] );

	BOOST_TEST( multi::elementwise::minus(A[1], B[1])[1] == A[1][1] - B[1][1] );

	BOOST_TEST( multi::elementwise::invoke(std::minus<>{}, A[1], B[1])[1] == A[1][1] - B[1][1] );

	BOOST_TEST(
		multi::elementwise::invoke(
			std::minus<>{},
			A[1],
			[](auto) { return 5; } ^ A[1].extents() 
		)[1] == A[1][1] - 5 );

	return boost::report_errors();
}
