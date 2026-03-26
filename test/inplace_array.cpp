// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, dynamic_array, num_elements

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
	multi::inplace_array<int[4][4]> a2d = {
		{1, 2},
		{3, 4}
	};

	BOOST_TEST( a2d.size() == 2 );
	BOOST_TEST( a2d[1][1] == 4 );

	return boost::report_errors();
}
