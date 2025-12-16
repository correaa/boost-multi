// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// #include <fmt/ranges.h>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<ranges>)
#if !defined(__clang_major__) || (__clang_major__ != 16)
#include <ranges>  // IWYU pragma: keep
#endif
#endif

#if defined(__cpp_lib_ranges) && !defined(_MSC_VER)

#include <boost/multi/array.hpp>  // from https://github.com/correaa/boost-multi
#include <boost/multi/broadcast.hpp>

#include <iostream>
#include <limits>
#include <numeric>

namespace stdr = std::ranges;
namespace stdv = std::views;

void printR2(auto const& lbl, auto const& arr2D) {
	//  fmt::print("\n{} = \n[{}]\n\n", lbl, fmt::join(arr2D, ",\n "));
	std::cout << lbl << "=\n";
	for(auto const& row : arr2D) {
		for(auto const& e : row) {
			std::cout << e << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
}

auto maxR1 = []<class R, class V = stdr::range_value_t<R>>(R const& rng) noexcept {
	// fmt::print("M");
	std::cout << 'M';
#if defined(__cpp_lib_ranges_fold)
	return stdr::fold_left(rng, std::numeric_limits<V>::lowest(), stdr::max);
#else
	return std::accumulate(rng.begin(), rng.end(), std::numeric_limits<V>::lowest(), stdr::max);
#endif
};

auto sumR1 = []<class R, class V = stdr::range_value_t<R>>(R const& rng, V zero = {}) noexcept {
	// fmt::print("S");
	std::cout << 'S';
#if defined(__cpp_lib_ranges_fold)
	return stdr::fold_left(rng, zero, std::plus<>{});
#else
	return std::accumulate(rng.begin(), rng.end(), zero);
#endif
};

#define FWD(var) std::forward<decltype(var)>(var)

namespace multi = boost::multi;

auto softmax2(auto&& matrix) noexcept {
	// auto const max_per_row = [&](auto i) { return maxR1(matrix[i]); } ^ multi::extensions_t<1>{matrix.extension()};

	//  using multi::broadcast::operator-;

	// using multi::broadcast::exp;

    // auto maxR1s = [&](auto ii) { return maxR1(matrix[ii]); } ^ multi::extensions_t<1>{matrix.extension()};

	using multi::broadcast::operator-;
	using multi::broadcast::exp;

    // auto shifted_exp = [matrix = FWD(matrix), maxR1s = std::move(maxR1s)](auto i, auto j) {
	// 	return std::exp(matrix[i][j] - maxR1s[i]);
	// } ^ matrix.extensions();

	auto shifted_exp = [matrix = FWD(matrix)](auto i) {
		return exp(
            matrix - (multi::array<float, 0>(maxR1(matrix[i])).repeated(3))
        )[i];
	} ^ multi::extensions_t<1>{matrix.extension()};

	auto const sum_exp =
		[shifted_exp = std::move(shifted_exp)](auto i) {
			auto shifted_expi = shifted_exp[i];
			using multi::broadcast::operator/;
			return shifted_expi / sumR1(shifted_expi);
			// return [shifted_expi=std::move(shifted_expi), sum_expi = std::move(sum_expi)](auto j) { return shifted_expi[j] / sum_expi; } ^ multi::extensions_t<1>{3};
		} ^
		multi::extensions_t<1>{matrix.extension()};

	return sum_exp;
}

auto softmax(auto&& matrix) noexcept {
	return FWD(matrix)  //
		 |              //
		   stdv::transform([](auto&& row) {
			   auto max = maxR1(row);
			   return FWD(row) |
					  stdv::transform([=](auto ele) noexcept { return std::exp(ele - max); });
		   })  //
		 |     //
		   stdv::transform([](auto&& nums) {
			   auto d = sumR1(nums);
			   return FWD(nums) |
					  stdv::transform([=](auto n) noexcept { return n / d; });
		   });
}

namespace multi = boost::multi;

int main() {
	auto const lazy_matrix =
		([](auto i) -> float { return static_cast<float>(i); } ^ multi::extensions_t(6))
			.partitioned(2);

	printR2("lazy matrix", lazy_matrix);

	printR2("softmax of lazy array", softmax(lazy_matrix));
	printR2("softmax2 of lazy array", softmax2(lazy_matrix));

	multi::array<float, 2> const alloc_matrix = {
		{0.0F, 1.0F, 2.0F},
		{3.0F, 4.0F, 5.0F}
	};

	printR2("softmax of alloc array", softmax(alloc_matrix));

	// materialize
	multi::array<float, 2> const sofmax_copy(softmax(alloc_matrix));

	// printR2("materialized softmax", sofmax_copy);

	//    assert(std::abs(sumR1(sofmax_copy[1]) - 1.0F) < 1e-12F);

	return boost::report_errors();
}
#else
int main() { return boost::report_errors(); }
#endif