// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/broadcast.hpp>
#include <boost/multi/io.hpp>
// #include <boost/multi/restriction.hpp>  // for restriction, operator!=
#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <numeric>
#include <utility>

// #include <algorithm>   // IWYU pragma: keep  // for std::equal
// #include <cmath>       // for std::abs
// #include <functional>  // for std::plus  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <iostream>
// #include <iterator>  // IWYU pragma: keep
// #include <limits>    // for std::numeric_limits  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
// #include <numeric>
// IWYU pragma: no_include <tuple>    // for apply
// #include <utility>  //  for forward  //  NOLINT(misc-include-cleaner)  //  IWYU pragma: keep

namespace multi = boost::multi;

// NOLINTBEGIN(readability-identifier-length)
int main() {  // NOLINT(readability-function-cognitive-complexity)
	multi::array<int, 2> A({3, 4});
	std::iota(A.elements().begin(), A.elements().end(), 0);

	BOOST_TEST((
		A == multi::array<int, 2>{
			{0, 1, 2, 3},
			{4, 5, 6, 7},
			{8, 9, 10, 11}
		}
	));

	multi::array<int, 2> B({3, 1});
	std::iota(B.elements().begin(), B.elements().end(), 100);

	BOOST_TEST((
		B == multi::array<int, 2>{
			{100},
			{101},
			{102}
		}
	));

	multi::array<int, 1> b(3);
	std::iota(b.elements().begin(), b.elements().end(), 100);

	using multi::broadcast::operator+;  // cppcheck-suppress constStatement ;
	// auto        BT0r4 = (~B)[0].repeated(4);
	auto const& C   = ~A + b;  // (~B)[0].repeated(4);  // std::move(BT0);

	std::cout << "A + B" << C << '\n';

	// BOOST_TEST((
	// 	+C == multi::array<int, 2>{
	// 		{100, 1, 2, 3},
	// 		{105, 5, 6, 7},
	// 		{110, 9, 10, 11}
	// 	}
	// ));

	return boost::report_errors();
}
// NOLINTEND(readability-identifier-length)
