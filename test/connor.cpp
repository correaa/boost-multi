// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep  // for std::equal
#include <cmath>      // for std::abs
#include <iterator>   // IWYU pragma: keep

#if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<ranges>)
#if !defined(__clang_major__) || (__clang_major__ != 16)
#include <ranges>  // IWYU pragma: keep
#endif
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

		BOOST_TEST( std::abs( rst[0] - 0.0F ) < 1e-12F );
		BOOST_TEST( std::abs( rst[1] - 1.0F ) < 1e-12F );
		// ...
		BOOST_TEST( std::abs( rst[5] - 5.0F ) < 1e-12F );

		// auto rst2D = rst.partitioned(2);

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)

#endif
	}
	return boost::report_errors();
}
