// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// vvv this has no effect, needs to be passed directly from compilation line "-Wno-psabi"
// #ifdef __GNUC__
// #pragma GCC diagnostic ignored "-Wpsabi"  // for ranges backwards compatibility message
// #endif

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep  // for std::equal
#include <cmath>      // for std::abs
#include <iterator>   // IWYU pragma: keep
#include <tuple>      // for std::get  // NOLINT(misc-include-cleaner)

#if defined(__cplusplus) && (__cplusplus >= 202002L)
#include <concepts>    // for constructible_from  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <functional>  // for std::plus  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <iostream>    // for std::cout  // NOLINT(misc-include-cleaner)
#include <limits>      // for std::numeric_limits  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <ranges>      // IWYU pragma: keep
#endif

#include <cmath>
#include <ranges>

#if __cplusplus >= 202302L

#define FMT_HEADER_ONLY
#define FMT_USE_NONTYPE_TEMPLATE_ARGS 0
#include <fmt/ranges.h>
#include <multi/array.hpp>  // from https://gitlab.com/correaa/boost-multi

namespace stdr = std::ranges;
namespace stdv = std::views;

void printR2(auto const& label, auto const& arr2D) {
	fmt::print("{} = \n[{}]\n", label, fmt::join(arr2D, ",\n "));
}

constexpr auto maxR1 = []<class R>(R const& row) {
	return stdr::fold_left(
		row, std::numeric_limits<stdr::range_value_t<R>>::lowest(), stdr::max
	);
};

[[maybe_unused]] auto sumR1 = []<class R>(R const& rng) {
	return stdr::fold_left(rng, stdr::range_value_t<R>{}, std::plus<>{});
};

#define FWD(var) std::forward<decltype(var)>(var)

auto softmax(auto&& matrix) {
	return FWD(matrix) | stdv::transform([](auto row) {
			   return stdv::transform(
				   [max = maxR1(row)](auto elem) { return exp(elem - max); }
			   );
		   }) |
		   stdv::transform([](auto row) {
			   return stdv::transform(
				   [sum = sumR1(row)](auto elem) { return elem / sum; }
			   );
		   });
}

namespace multi = boost::multi;

int main() {
	auto const matrix =
		([](auto ii) noexcept { return static_cast<float>(ii); } ^
		 multi::extensions_t(6))
			.partitioned(2);

	static_assert(
		std::is_same_v<stdr::range_value_t<std::decay_t<decltype(matrix[0])>>, float>
	);
	// using type = stdr::range_value_t<std::decay_t<decltype(matrix[0])>>;
	// stdr::fold_left(rng, stdr::range_value_t<std::decay_t<decltype(matrix[0])>>{}, std::plus<>{});

#if 0
	sumR1(matrix[0]);

	
	fmt::print("{}", sumR1(matrix[0]));

	printR2("sm = ", softmax(matrix));

	BOOST_TEST(false);

	auto const allocated_matrix =
		multi::array<float, 2>{
			{0.0F, 1.0F, 2.0F},
			{3.0F, 4.0F, 5.0F}
    };

	fmt::print("{}", sumR1(allocated_matrix[0]));
	// printR2("sm2 = ", softmax(allocated_matrix));

	// auto sm2 = softmax(allocated_matrix);

	// auto const result_matrix = multi::array<float, 2>(sm2);
	// printR2("result = ", result_matrix);
#endif
	return boost::report_errors();
}
#else
auto main() -> int {
	return boost::report_errors();
}
#endif
