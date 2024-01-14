// Copyright 2019-2024 Alfredo A. Correa

#include <boost/test/unit_test.hpp>

#include <multi/array.hpp>

#include <algorithm>  // for std::stable_sort
#include <array>
#include <vector>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(array_1D_partial_order_syntax) {
	multi::array<int, 1> const tt = {1, 1, 1};
	multi::array<int, 1> const uu = {2, 2, 2};

	BOOST_REQUIRE(     tt <  uu   );
	BOOST_REQUIRE( !  (tt >  uu)  );
	BOOST_REQUIRE(     tt <= uu   );
	BOOST_REQUIRE( !  (tt >= uu)  );
	BOOST_REQUIRE( !  (tt == uu)  );
	BOOST_REQUIRE(    (tt != uu)  );
	BOOST_REQUIRE( not(uu <  tt)  );
	BOOST_REQUIRE(    (uu >  tt)  );
	BOOST_REQUIRE( !  (uu <= tt)  );
	BOOST_REQUIRE(    (uu >= tt)  );
}

#if defined(__cpp_lib_ranges)
BOOST_AUTO_TEST_CASE(sort_2D) {
	multi::array<int, 2> A = {
		{3, 3, 3},
		{2, 2, 2},
		{1, 1, 1},
	};
	BOOST_REQUIRE(! std::ranges::is_sorted(A));

	std::ranges::sort(A);

	BOOST_REQUIRE(  std::ranges::is_sorted(A));

	static_assert(std::permutable<boost::multi::array_iterator<int, 2, int*>>);
}

BOOST_AUTO_TEST_CASE(sort_strings) {
	auto A = multi::array<char, 2>{
		{'S', 'e', 'a', 'n', ' ', ' '},
		{'A', 'l', 'e', 'x', ' ', ' '},
		{'B', 'j', 'a', 'r', 'n', 'e'},
	};
	BOOST_REQUIRE(! std::ranges::is_sorted(A));

	std::ranges::sort(A);

	BOOST_REQUIRE(  std::ranges::is_sorted(A));

	BOOST_REQUIRE((
		A == multi::array<char, 2>{
			{'A', 'l', 'e', 'x', ' ', ' '},
			{'B', 'j', 'a', 'r', 'n', 'e' },
			{'S', 'e', 'a', 'n', ' ', ' '},
		}
	));

	std::ranges::sort(~A);
	BOOST_REQUIRE(  std::ranges::is_sorted(~A));

	static_assert(std::permutable<boost::multi::array_iterator<int, 2, int*>>);
}
#endif

BOOST_AUTO_TEST_CASE(multi_array_stable_sort) {
	std::vector<double> vec = {1.0, 2.0, 3.0};  // NOLINT(fuchsia-default-arguments-calls)
	BOOST_REQUIRE( std::is_sorted(begin(vec), end(vec)) );

	multi::array<double, 2> d2D = {
		{150.0, 16.0, 17.0, 18.0, 19.0},
		{ 30.0,  1.0,  2.0,  3.0,  4.0},
		{100.0, 11.0, 12.0, 13.0, 14.0},
		{ 50.0,  6.0,  7.0,  8.0,  9.0},
	};
	BOOST_REQUIRE( not std::is_sorted(begin(d2D), end(d2D) ) );

	std::stable_sort(begin(d2D), end(d2D));
	BOOST_REQUIRE( std::is_sorted( begin(d2D), end(d2D) ) );

	BOOST_REQUIRE((
		d2D == decltype(d2D){
			{ 30.0,  1.0,  2.0,  3.0,  4.0},
			{ 50.0,  6.0,  7.0,  8.0,  9.0},
			{100.0, 11.0, 12.0, 13.0, 14.0},
			{150.0, 16.0, 17.0, 18.0, 19.0},
		}
	));

	BOOST_REQUIRE( not std::is_sorted( begin(d2D.rotated()), end(d2D.rotated()) ) );

	std::stable_sort(begin(d2D.rotated()), end(d2D.rotated()));
	BOOST_REQUIRE( std::is_sorted( begin(d2D.rotated()), end(d2D.rotated()) ) );
	BOOST_REQUIRE( std::is_sorted( begin(d2D          ), end(d2D          ) ) );

	BOOST_REQUIRE((
		d2D == decltype(d2D){
			{ 1.0,  2.0,  3.0,  4.0,  30.0},
			{ 6.0,  7.0,  8.0,  9.0,  50.0},
			{11.0, 12.0, 13.0, 14.0, 100.0},
			{16.0, 17.0, 18.0, 19.0, 150.0},
		}
	));
}

BOOST_AUTO_TEST_CASE(multi_array_ref_stable_sort) {
	std::vector<double> vec = {1.0, 2.0, 3.0};  // NOLINT(fuchsia-default-arguments-calls)
	BOOST_REQUIRE( std::is_sorted(begin(vec), end(vec)) );

	// clang-format off
	std::array<std::array<double, 5>, 4> d2D {{
		{{150.0, 16.0, 17.0, 18.0, 19.0}},
		{{ 30.0,  1.0,  2.0,  3.0,  4.0}},
		{{100.0, 11.0, 12.0, 13.0, 14.0}},
		{{ 50.0,  6.0,  7.0,  8.0,  9.0}}
	}};
	// clang-format on

	auto&& d2D_ref = *multi::array_ptr<double, 2>(&d2D[0][0], {4, 5});  // NOLINT(readability-container-data-pointer) test access

	BOOST_REQUIRE( not std::is_sorted(begin(d2D_ref), end(d2D_ref) ) );
	std::stable_sort(begin(d2D_ref), end(d2D_ref));
	BOOST_REQUIRE( std::is_sorted( begin(d2D_ref), end(d2D_ref) ) );

	BOOST_REQUIRE( not std::is_sorted( begin(d2D_ref.rotated()), end(d2D_ref.rotated()) ) );
	std::stable_sort(begin(d2D_ref.rotated()), end(d2D_ref.rotated()));
	BOOST_REQUIRE( std::is_sorted( begin(d2D_ref.rotated()), end(d2D_ref.rotated()) ) );
}
