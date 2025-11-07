// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep  // for std::equal
#include <iterator>   // IWYU pragma: keep

#if defined(__cplusplus) && (__cplusplus >= 202002L)
#include <ranges>  // IWYU pragma: keep
#endif

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	{
		auto rst = [](auto ii, auto jj) { return (10 * ii) + jj; } ^ multi::extensions_t(5, 5);

		multi::array<int, 2> const AA = rst;

		BOOST_TEST( AA.size() == rst.size() );

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
		multi::array<int, 2> const BB = rst | std::views::reverse;

		BOOST_TEST( AA[0] == BB[4] );  // as A[0][0] == B[4][0] && A[0][1] == B[4][1] ...
		BOOST_TEST( AA[1] == BB[3] );  // as A[1][0] == B[3][0] && A[1][1] == B[3][1] ...
		BOOST_TEST( AA[2] == BB[2] );  // ...
		BOOST_TEST( AA[3] == BB[1] );
		BOOST_TEST( AA[4] == BB[0] );
#endif
	}
	return boost::report_errors();
}
