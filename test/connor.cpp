// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep  // for std::equal
#include <cmath>      // for std::abs
// #include <limits>  // for std::numeric_limits
#include <iterator>  // IWYU pragma: keep

#if defined(__cplusplus) && (__cplusplus >= 202002L)
#include <concepts>  // for constructible_from  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <ranges>    // IWYU pragma: keep
#endif

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	{
#ifdef __NVCC__
		auto fun = [](auto ii) noexcept { return static_cast<float>(ii); };
		auto rst = fun ^ multi::extensions_t(6);
#else
		auto rst = [](auto ii) noexcept { return static_cast<float>(ii); } ^ multi::extensions_t(6);
#endif
		BOOST_TEST( rst.size() == 6 );

		// BOOST_TEST( std::abs( rst[0] - 0.0F ) < 1e-12F );
		// BOOST_TEST( std::abs( rst[1] - 1.0F ) < 1e-12F );
		// // ...
		// BOOST_TEST( std::abs( rst[5] - 5.0F ) < 1e-12F );

		auto rst2D = rst.partitioned(2);

		BOOST_TEST( rst2D.size() == 2 );

		using std::get;
		BOOST_TEST( get<0>(rst2D.sizes()) == 2 );
		BOOST_TEST( get<1>(rst2D.sizes()) == 3 );

		BOOST_TEST( std::abs(rst2D[0][0] - 0.0F) < 1e-12F );
		BOOST_TEST( std::abs(rst2D[0][1] - 1.0F) < 1e-12F );
		BOOST_TEST( std::abs(rst2D[0][2] - 2.0F) < 1e-12F );
		BOOST_TEST( std::abs(rst2D[1][0] - 3.0F) < 1e-12F );
		BOOST_TEST( std::abs(rst2D[1][1] - 4.0F) < 1e-12F );
		BOOST_TEST( std::abs(rst2D[1][2] - 5.0F) < 1e-12F );

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
		// static auto max_fold = []<class R>(R const& rng) { return std::ranges::fold_left(rng, std::numeric_limits<typename R::value_type>::lowest(), std::ranges::max); };
		static auto hmax = [](auto const& row) { return std::ranges::fold_left(row, std::numeric_limits<float>::lowest(), std::ranges::max); };

		auto maxs = rst2D | std::ranges::views::transform(hmax);

		BOOST_TEST(maxs.size() == 2 );

		BOOST_TEST( std::abs( maxs[0] - 2.0F) < 1e-12F );
		BOOST_TEST( std::abs( maxs[1] - 5.0F) < 1e-12F );
#if defined(__cpp_lib_ranges_zip) && (__cpp_lib_ranges_zip >= 202110L)
		// auto renorms = std::ranges::views::zip(rst2D, maxes) | std::ranges::views::transform( [](auto const& row_max) { auto const& [row, max] = row_max; return row | std::transform([&max](auto e) { return e - max;} ); } );
#endif
#endif
#endif
	}
	return boost::report_errors();
}
