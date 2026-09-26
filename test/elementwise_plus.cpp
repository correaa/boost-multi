// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/elementwise/plus.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <utility>

namespace multi = boost::multi;

auto main() -> int {
	auto const A = multi::array<int, 2>{
		{0, 1, 2},
		{3, 4, 5}
	};
	auto const B = multi::array<int, 2>{
		{ 0, 10, 20},
		{30, 40, 50}
	};

	BOOST_TEST( multi::elementwise::plus(A, B)[1][1] == A[1][1] + B[1][1] );

	BOOST_TEST( multi::elementwise::plus(multi::array<int, 2>{
		{0, 1, 2},
		{3, 4, 5}
	}, B)[1][1] == A[1][1] + B[1][1] );

	BOOST_TEST( multi::elementwise::plus(A[1], B[1])[1] == A[1][1] + B[1][1] );

	BOOST_TEST( multi::elementwise::invoke(std::plus<>{}, A[1], B[1])[1] == A[1][1] + B[1][1] );

	BOOST_TEST(
		multi::elementwise::invoke(
			std::plus<>{},
			A[1],
			[](auto) { return 5; } ^ A[1].extents() 
		)[1]
		== A[1][1] + 5 
	);

	BOOST_TEST(
		multi::elementwise::invoke(
			std::plus<>{},
			[](auto) { return 5; } ^ A[1].extents(),
			A[1]
		)[1] == 5 + A[1][1]
	);

	BOOST_TEST(
		multi::elementwise::broadcast(
			std::plus<>{},
			[](auto) { return 5; } ^ A[1].extents(),
			A[1]
		)[1] == 5 + A[1][1]
	);

	return boost::report_errors();
}
