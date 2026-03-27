// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, dynamic_array, num_elements

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	{
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
		multi::inplace_array<int[4][4]> a2d = {
			{1, 2},
			{3, 4}
		};

		BOOST_TEST( a2d.size() == 2 );
		BOOST_TEST( a2d[1][1] == 4 );
	}
	{
		int carr[3] = {0, 1, 2};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		multi::inplace_array<int[5]> const iparr(carr);  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		BOOST_TEST( iparr.size() == 3 );
		BOOST_TEST( iparr[1] == 1 );
	}

#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
	{
		constexpr int carr[3] = {0, 1, 2};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		static constexpr multi::inplace_array<int[5]> iparr(carr);  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		BOOST_TEST( iparr.size() == 3 );
		static_assert(iparr.size() == 3);

		BOOST_TEST( iparr[1] == 1 );
		static_assert(iparr[1] == 1);
	}
	{
		static constexpr multi::inplace_array<int[5]> iparr = {0, 1, 2};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		BOOST_TEST( iparr.size() == 3 );
		static_assert(iparr.size() == 3);

		BOOST_TEST( iparr[1] == 1 );
		static_assert(iparr[1] == 1);
	}
	{
		static constexpr multi::inplace_array<int[4]> iparr = {0, 1, 2};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

		BOOST_TEST( iparr.size() == 3 );
		static_assert(iparr.size() == 3);

		BOOST_TEST( iparr[1] == 1 );
		static_assert(iparr[1] == 1);
	}
#endif

	return boost::report_errors();
}
