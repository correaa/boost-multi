// Copyright 2023-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <algorithm>  // for std::ranges::fold_left

#if(__cplusplus >= 202002L)
#include<ranges>
#endif

// Suppress warnings from boost.test
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wconversion"
#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wundef"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
// #pragma GCC diagnostic ignored "-Wstringop-overflow="
// #pragma GCC diagnostic ignored "-Warray-bounds="
#endif

#include <boost/multi/array.hpp>

#include <algorithm>  // for std::ranges::fold_left

// template<>
// inline constexpr bool std::ranges::enable_borrowed_range<
//  boost::multi::subarray<double, 2, const double*, boost::multi::layout_t<2, long int> >
// > = true;

#ifndef BOOST_TEST_MODULE
#define BOOST_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace multi = boost::multi;

#if(__cplusplus >= 202002L)
BOOST_AUTO_TEST_CASE(iota_range_experiment) {
	//  auto zero_to_six = std::ranges::views::iota(0, 6);
	//  using it = decltype(zero_to_six.begin());
	// using pp = typename std::pointer_traits<it>::element_type;

	auto two_by_three = multi::array_ref<int, 2, decltype(std::ranges::views::iota(0, 6).begin())>({2, 3}, std::ranges::views::iota(0, 6).begin());

	BOOST_REQUIRE( two_by_three.size() == 2 );

	BOOST_REQUIRE( two_by_three[0][0] == 0 );
	BOOST_REQUIRE( two_by_three[0][1] == 1 );
	BOOST_REQUIRE( two_by_three[0][2] == 2 );

	BOOST_REQUIRE( two_by_three[1][0] == 3 );
	BOOST_REQUIRE( two_by_three[1][1] == 4 );
	BOOST_REQUIRE( two_by_three[1][2] == 5 );

	// two_by_three[1][2] = 6;  // doesn't compile, good, "error: expression is not assignable"
}
#endif

BOOST_AUTO_TEST_CASE(range_accumulate) {
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
	static constexpr auto accumulate = [](auto const& R) { return std::ranges::fold_left(R, 0, std::plus<>{}); };

	auto const values = multi::array<int, 2>{
		{2, 0, 2, 2},
		{2, 2, 0, 4},
		{2, 2, 0, 4},
		{2, 2, 0, 0},
		{2, 7, 0, 2},
		{2, 2, 4, 4},
	};

	boost::multi::array<int, 1, std::allocator<int>> aaa = {1, 2, 3};

	constexpr auto rowOddSum = [](auto const& arr) {
		return std::ranges::find_if(arr, [](auto const& row) { return (accumulate(row) & 1) == 1; });
	};

	auto const result = rowOddSum(values);

	BOOST_REQUIRE( result - values.begin() == 4 );
#endif
}

BOOST_AUTO_TEST_CASE(range_find) {
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
	namespace multi = boost::multi;

	using Array2D = multi::array<int, 2>;

	Array2D const a = {
		{1, 2},
		{3, 4},
	};
	{
		auto const needle = std::ranges::find_if(a, [](auto const& row) { return row[0] == 9; });
		BOOST_REQUIRE(needle == a.end());
	}
	{
		std::ranges::equal_to eto;

		auto a2 = a();

		[[maybe_unused]] auto const& _84 = static_cast<boost::multi::subarray<int, 2, int const*, boost::multi::layout_t<2, boost::multi::size_type>> const&>(a);
		[[maybe_unused]] auto const& _85 = static_cast<boost::multi::subarray<int, 2, int const*, boost::multi::layout_t<2, boost::multi::size_type>> const&>(std::as_const(a));

		auto a1     = a[1];
		auto a1_val = +a[1];

		// [[maybe_unused]] auto const& _90 = static_cast<const boost::multi::subarray<int,1,const int *,boost::multi::layout_t<1,boost::multi::size_type>>&>(a1_val);
		// [[maybe_unused]] auto const& _91 = static_cast<const boost::multi::subarray<int,1,const int *,boost::multi::layout_t<1,boost::multi::size_type>>&>(std::as_const(a1_val));

		// static_assert( std::convertible_to<const boost::multi::array<int,1,std::allocator<int>>&, const boost::multi::subarray<int,1,const int *,boost::multi::layout_t<1,boost::multi::size_type>>&> );
		// static_assert( std::equality_comparable_with<boost::multi::array<int,1,std::allocator<int>>&,boost::multi::subarray<int,1,const int *,boost::multi::layout_t<1,boost::multi::size_type>>&> );

		bool const res = eto(a1_val, a1);
		BOOST_REQUIRE( res );
		// std::ranges::equal_to&,boost::multi::array<int,1,std::allocator<int>>&,boost::multi::subarray<int,1,const int *,boost::multi::layout_t<1,boost::multi::size_type>>&
	}

	{
		auto&&     a1     = a[1];
		auto const needle = std::ranges::find(a, a1);
		BOOST_REQUIRE(needle != a.end());
		BOOST_REQUIRE( *needle == a1 );
		BOOST_REQUIRE( *needle == a[1] );
	}

	{
		auto const needle = std::ranges::find(a, a[1]);
		BOOST_REQUIRE(needle != a.end());
		BOOST_REQUIRE( *needle == a[1] );
	}
#endif
}

// #if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L)
// BOOST_AUTO_TEST_CASE(range_copy_n_1D) {
//  namespace multi = boost::multi;

//  multi::array<int, 1> const X1 = {1, 2, 3};
//  multi::array<int, 1> X2(X1.extensions());

//  std::ranges::copy_n(X1.begin(), 10, X2.begin());

//  BOOST_REQUIRE( X1 == X2 );
// }

// BOOST_AUTO_TEST_CASE(range_copy_n) {
//  namespace multi = boost::multi;

//  multi::array<int, 2> const X1({ 10, 10 }, 99);
//  multi::array<int, 2> X2(X1.extensions());

//  std::ranges::copy_n(X1.begin(), 10, X2.begin());
//  BOOST_REQUIRE( X1 == X2 );
// }
// #endif
