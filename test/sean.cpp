// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep  // for std::equal
#include <iterator>   // IWYU pragma: keep

#if defined(__cplusplus) && (__cplusplus >= 202002L)
#include <concepts>  // for totally_ordered
#include <ranges>    // IWYU pragma: keep
#endif

#include <tuple>        // IWYU pragma: keep
#include <type_traits>  // for std::is_same_v
// IWYU pragma: no_include <variant>        // for get, iwyu bug

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	{
		#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
		auto rst = [](auto i, auto j) { return (10 * i) + j; } ^ multi::extensions_t(5, 5);

		multi::array<int, 2> const A = rst;
		multi::array<int, 2> const B = rst | std::views::reverse;

		BOOST_TEST( A[0] == B[4] );  // as A[0][0] == B[4][0] && A[0][1] == B[4][1] ...
		BOOST_TEST( A[1] == B[3] );  // as A[1][0] == B[3][0] && A[1][1] == B[3][1] ...
		BOOST_TEST( A[2] == B[2] );  // ...
		BOOST_TEST( A[3] == B[1] );
		BOOST_TEST( A[4] == B[0] );
		#endif
	}
	return boost::report_errors();
}
