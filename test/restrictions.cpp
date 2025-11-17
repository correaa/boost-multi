// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// vvv this has no effect, needs to be passed directly from compilation line "-Wno-psabi"
// #ifdef __GNUC__
// #pragma GCC diagnostic ignored "-Wpsabi"  // for ranges backwards compatibility message
// #endif

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#if __cplusplus >= 202302L

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

#include <boost/multi/array.hpp>

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

constexpr auto maxR1 = []<class R, class V = stdr::range_value_t<R>>(R const& row, V low = std::numeric_limits<V>::lowest()) {
	return stdr::fold_left(row, low, stdr::max);
};

constexpr auto sumR1 = []<class R, class V = stdr::range_value_t<R>>(R const& rng, V zero = {}) {
	return stdr::fold_left(rng, zero, std::plus<>{});
};

#define FWD(var) std::forward<decltype(var)>(var)

auto softmax(auto&& matrix) noexcept {
	return           //
		FWD(matrix)  //
		|
		stdv::transform([](auto&& row) {
			auto max = maxR1(row);
			return        //
				FWD(row)  //
				|
				stdv::transform([=](auto ele) noexcept { return std::exp(ele - max); });
		})  //
		|
		stdv::transform([](auto&& nums) {
			auto den = sumR1(nums);
			return         //
				FWD(nums)  //
				|
				stdv::transform([=](auto num) noexcept { return num / den; });
		});
}

namespace multi = boost::multi;

namespace lazy {

template<class A>
auto operator*(typename A::element scalar, A const& a) {
	return [scalar, &a](auto... is) { return scalar * a[is...]; } ^ a.extensions();
}

namespace elementwise {

template<class A, class B>
auto operator*(A const& a, B const& b) requires(A::dimensionality == B::dimensionality) {
	assert(a.extensions() == b.extensions());
	return [&a, &b](auto... is) { return a[is...] * b[is...]; } ^ a.extensions();
}

template<class A, class B>
auto operator+(A const& a, B const& b) requires(A::dimensionality == B::dimensionality) {
	assert(a.extensions() == b.extensions());
	return [&a, &b](auto... is) { return a[is...] + b[is...]; } ^ a.extensions();
}

}  // namespace elementwise
}  // namespace lazy

int main() {
	auto const A = multi::array<int, 2>{
		{0, 1, 2},
		{3, 4, 5}
	};
	auto const B = multi::array<int, 2>{
		{ 0, 10, 20},
		{30, 40, 50}
	};

	using lazy::operator*;
	using lazy::elementwise::operator+;
	using lazy::elementwise::operator*;

	multi::array<int, 2> const C = A + (A * B) + (2.0 * B);

	std::cout << "C11 = " << C[1][1] << std::endl;
	BOOST_TEST( C[1][1] == 4 + 4*40 + 2*40 );

	return boost::report_errors();
}
#else
auto main() -> int {
	return boost::report_errors();
}
#endif
