// Copyright 2016 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// clang-format off
	{
		{ multi::array<int, 1> const arr({9, 9, 9}); BOOST_TEST(arr.num_elements() == 3); }
		{ multi::array<int, 1> const arr({9, 9});    BOOST_TEST(arr.num_elements() == 2); }
		{ multi::array<int, 1> const arr({9});       BOOST_TEST(arr.num_elements() == 1); }

		{ multi::array<int, 3> const arr({9, 9, 9}); BOOST_TEST(arr.num_elements() == 9L*9L*9L ); }
		{ multi::array<int, 2> const arr({9, 9});    BOOST_TEST(arr.num_elements() == 9L*9L); }
		{ multi::array<int, 1> const arr({9});       BOOST_TEST(arr.num_elements() != 9L); } // PATTERN BREAKS
	}
	// clang-format on

	return boost::report_errors();
}
