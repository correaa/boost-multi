// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#if __cplusplus >= 202302L
#include <algorithm>   // IWYU pragma: keep  // for std::equal
#include <cmath>       // for std::abs
#include <functional>  // for std::plus  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <iostream>    // for std::cout  // NOLINT(misc-include-cleaner)
#include <iterator>    // IWYU pragma: keep
#include <limits>      // for std::numeric_limits  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <tuple>       // for std::get  // NOLINT(misc-include-cleaner)

#if defined(__cplusplus) && (__cplusplus >= 202002L)
#include <concepts>  // for constructible_from  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <ranges>    // IWYU pragma: keep
#endif

#include <boost/multi/array.hpp>
// #include <boost/multi/detail/what.hpp>

namespace stdr = std::ranges;
namespace stdv = std::views;

auto printR2(auto const& lbl, auto const& arr2D) {
	// return fmt::print("{} = \n[{}]\n\n", lbl, fmt::join(arr2D, ",\n "));
	std::cout << lbl << " = \n";
	for(auto const& row : arr2D) {
		for(auto const& elem : row)
			std::cout << elem << ", ";
		std::cout << '\n';
	}
	std::cout << '\n';
}

#define FWD(var) std::forward<decltype(var)>(var)

namespace multi = boost::multi;

int main() {
	{
		multi::array a = {1, 2, 3};
		multi::array b = {4, 5, 6};

		using multi::broadcast::operator+;
		auto&& c = a + b;

		// multi::detail::what(c);
		BOOST_TEST(( c == multi::array{5, 7, 9} ));
		// BOOST_TEST( std::ranges::equal(c, multi::array{5, 7, 9}) );
	}

	// np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) + np.array([10, 20, 30])
	// array([[11, 22, 33],
	//        [14, 25, 36],
	//        [17, 28, 39]])

	return boost::report_errors();
}
#else
auto main() -> int {
	return boost::report_errors();
}
#endif
