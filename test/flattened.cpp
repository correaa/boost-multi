// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for range, layout_t, get, extensions_t

#include <boost/core/lightweight_test.hpp>

// #include <algorithm>  // for copy
// #include <array>      // for array, array<>::value_type
// #include <cstddef>    // for ptrdiff_t, size_t  // IWYU pragma: keep
// #include <iterator>   // for size

// #if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<ranges>)
// #if !defined(__clang_major__) || (__clang_major__ != 16)
// #include <ranges>  // IWYU pragma: keep
// #endif
// #endif

// #include <tuple>   // for make_tuple, tuple_element<>::type
// #include <vector>  // for vector
// // IWYU pragma: no_include <version>
// #include <type_traits>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// BOOST_AUTO_TEST_CASE(layout_3)
	{
		multi::array<double, 2> arr({50, 50});
		BOOST_TEST( arr.size() == 50 );

		BOOST_TEST( arr[0].sliced(10, 20).size() == 10 );
		BOOST_TEST( size(arr[0].sliced(10, 20))  == 10 );

		static_assert(decltype(arr(0, {10, 20}))::rank_v == 1);

		BOOST_TEST( size(arr(0, {10, 20})) == 10 );

		BOOST_TEST(      arr.layout() == arr.layout()  );
		BOOST_TEST( !(arr.layout() <  arr.layout()) );

		auto const& barr = arr.flattened();
		BOOST_TEST( &barr[10] == &arr[0][10] );
	}
	{
		multi::array<double, 2> arr({6, 10});

		auto const& barr = arr.strided(2).flattened();

		BOOST_TEST( &barr [0] == &arr[0][0] );
		BOOST_TEST( &barr [1] == &arr[0][1] );
		// ...
		BOOST_TEST( &barr [9] == &arr[0][9] );

		BOOST_TEST( &barr[10] == &arr[2][0] );
		BOOST_TEST( &barr[11] == &arr[2][1] );
		BOOST_TEST( &barr[12] == &arr[2][2] );
		// ...
		BOOST_TEST( &barr[19] == &arr[2][9] );

		BOOST_TEST( &barr[20] == &arr[4][0] );
		BOOST_TEST( &barr[21] == &arr[4][1] );
		BOOST_TEST( &barr[22] == &arr[4][2] );
		// ...
		BOOST_TEST( &barr[29] == &arr[4][9] );

		BOOST_TEST( arr.num_elements() == 60 );
		BOOST_TEST( barr.size() == 30 );
	}
	{
		multi::array<double, 2> arr({6, 10});

		auto const& barr = arr.strided(2).transposed().strided(2).transposed().flattened();

		BOOST_TEST( &barr [0] == &arr[0][0] );
		BOOST_TEST( &barr [1] == &arr[0][2] );
		BOOST_TEST( &barr [2] == &arr[0][4] );
		BOOST_TEST( &barr [3] == &arr[0][6] );
		BOOST_TEST( &barr [4] == &arr[0][8] );

		BOOST_TEST( &barr [5] == &arr[2][0] );
		BOOST_TEST( &barr [6] == &arr[2][2] );
		BOOST_TEST( &barr [7] == &arr[2][4] );
		BOOST_TEST( &barr [8] == &arr[2][6] );
		BOOST_TEST( &barr [9] == &arr[2][8] );

		BOOST_TEST( &barr [10] == &arr[4][0] );
		BOOST_TEST( &barr [11] == &arr[4][2] );
		BOOST_TEST( &barr [12] == &arr[4][4] );
		BOOST_TEST( &barr [13] == &arr[4][6] );
		BOOST_TEST( &barr [14] == &arr[4][8] );

		BOOST_TEST( arr.num_elements() == 60 );
		BOOST_TEST( barr.size() == 15 );
	}
	{
		multi::array<double, 2> arr({3, 5});

		auto&& barr = arr.flattened();

		BOOST_TEST( &barr [0] == &arr[0][0] );

		BOOST_TEST( &*barr.begin() == &barr[0] );
		BOOST_TEST( &*(barr.begin() + 1) == &barr[1] );
	}
	{
		multi::array<int, 2> arr = {
			{ 0,  1,  2,  3,  4},
			{ 5,  6,  7,  8,  9},
			{10, 11, 12, 13, 14}
		};

		BOOST_TEST( arr.size() == 3 );

		auto&& barr = arr().flattened();

		BOOST_TEST( &barr [0] == &arr[0][0] );

		BOOST_TEST( &*barr.begin() == &barr[0] );
		BOOST_TEST( &*(barr.begin() + 1) == &barr[1] );

		auto it = barr.begin();
		BOOST_TEST( &*(it + 0) == &arr[0][0] );
		BOOST_TEST( &*(it + 1) == &arr[0][1] );
		BOOST_TEST( &*(it + 2) == &arr[0][2] );
		BOOST_TEST( &*(it + 3) == &arr[0][3] );
		BOOST_TEST( &*(it + 4) == &arr[0][4] );

		BOOST_TEST( &*(it + 5) == &arr[1][0] );
		BOOST_TEST( &*(it + 6) == &arr[1][1] );

		BOOST_TEST( &*it == &arr[0][0] );
		++it;
		BOOST_TEST( &*it == &arr[0][1] );
		++it;
		BOOST_TEST( &*it == &arr[0][2] );
		++it;
		BOOST_TEST( &*it == &arr[0][3] );
		++it;
		BOOST_TEST( &*it == &arr[0][4] );
		++it;
		BOOST_TEST( &*it == &arr[1][0] );
	}
	{
		multi::array<int, 2> arr_original = {
			{ 0,  1,  2,  3,  4, 99},
			{ 5,  6,  7,  8,  9, 99},
			{10, 11, 12, 13, 14, 99},
			{99, 99, 99, 99, 99, 99}
		};

		auto&& arr = arr_original({0, 3}, {0, 5});

		BOOST_TEST( arr.size() == 3 );

		auto&& barr = arr().flattened();

		BOOST_TEST( &barr [0] == &arr[0][0] );

		BOOST_TEST( &*barr.begin() == &barr[0] );
		BOOST_TEST( &*(barr.begin() + 1) == &barr[1] );

		auto it = barr.begin();
		BOOST_TEST( &*(it + 0) == &arr[0][0] );
		BOOST_TEST( &*(it + 1) == &arr[0][1] );
		BOOST_TEST( &*(it + 2) == &arr[0][2] );
		BOOST_TEST( &*(it + 3) == &arr[0][3] );
		BOOST_TEST( &*(it + 4) == &arr[0][4] );

		BOOST_TEST( &*(it + 5) == &arr[1][0] );
		BOOST_TEST( &*(it + 6) == &arr[1][1] );

		BOOST_TEST( &*it == &arr[0][0] );
		++it;
		BOOST_TEST( &*it == &arr[0][1] );
		++it;
		BOOST_TEST( &*it == &arr[0][2] );
		++it;
		BOOST_TEST( &*it == &arr[0][3] );
		++it;
		BOOST_TEST( &*it == &arr[0][4] );
		// multi::detail::what(it);
		// ++it;
		// BOOST_TEST( &*it == &arr[1][0] );
	}
	{
		multi::array<int, 1> const arr({5});

		// auto const& barr = arr.flattened();  // compilation error, good
	}
	{
		multi::array<int, 3> const arr({3, 5, 7});

		auto const& barr = arr.flattened();

		BOOST_TEST( barr.size() == 15 );

		// auto it = barr.begin();
		// BOOST_TEST( &*(it + 0)[0] == &arr[0][0][0] );
		// BOOST_TEST( &*(it + 1)[0] == &arr[0][1][0] );
		// BOOST_TEST( &*(it + 2)[0] == &arr[0][2][0] );
		// BOOST_TEST( &*(it + 3)[0] == &arr[0][3][0] );
		// BOOST_TEST( &*(it + 4)[0] == &arr[0][4][0] );

		// auto const& bbarr = barr.flattened();
	}

	return boost::report_errors();
}
