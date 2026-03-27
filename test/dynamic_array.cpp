// Copyright 2019-2026 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, array_ref, subarray, arra...

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	{
		int carr[3] = {0, 1, 2};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		multi::dynamic_array<int, 1> const darr(carr);

		BOOST_TEST( darr.size() == 3 );
		BOOST_TEST( darr[1] == 1 );
	}
	{
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
		int carr[2][3] = {
			{0, 1, 2},
			{3, 4, 5}
		};

		multi::dynamic_array<int, 2> const darr(carr);

		BOOST_TEST( darr.size() == 2 );

		BOOST_TEST( darr[1][1] == 4 );
	}

	return boost::report_errors();
}
