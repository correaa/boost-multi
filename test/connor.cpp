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
// #include <limits>  // for std::numeric_limits
#include <iterator>  // IWYU pragma: keep
#include <tuple>     // for std::get  // NOLINT(misc-include-cleaner)

#if defined(__cplusplus) && (__cplusplus >= 202002L)
#include <concepts>  // for constructible_from  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <iostream>  // for std::cout
#include <ranges>    // IWYU pragma: keep
#endif

namespace {

template<typename Label, class A1D>
void print_1d(Label const& label, A1D const& arr1D) {
	using std::cout;
	cout << label << ' ';
	for(auto const& elem : arr1D) {  // NOLINT(altera-unroll-loops)
		cout << elem << ", ";
	}
	cout << '\n';
}

template<typename Label, class A2D>
void print_2d(Label const& label, A2D const& arr2D) {
	using std::cout;
	cout << label << '\n';
	for(auto const& row : arr2D) {
		for(auto const& elem : row) {  // NOLINT(altera-unroll-loops)
			cout << elem << ", ";
		}
		cout << '\n';
	}
	cout << '\n';
}
}  // namespace

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	{
#ifdef __NVCC__
		auto fun = [](auto ii) noexcept { return static_cast<float>(ii); };
		auto rst = fun ^ multi::extensions_t(6);
#else
		auto rst = [](auto ii) noexcept { return static_cast<float>(ii); } ^ multi::extensions_t(6);
#endif
		print_1d("rst = ", rst);

		BOOST_TEST( rst.size() == 6 );

		// BOOST_TEST( std::abs( rst[0] - 0.0F ) < 1e-12F );
		// BOOST_TEST( std::abs( rst[1] - 1.0F ) < 1e-12F );
		// // ...
		// BOOST_TEST( std::abs( rst[5] - 5.0F ) < 1e-12F );

		auto rst2D = rst.partitioned(2);

		print_2d("rst2D = ", rst2D);

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
		namespace stdr = std::ranges;

		static auto hard_max = []<class R>(R const& row) { return stdr::fold_left(row, std::numeric_limits<std::ranges::range_value_t<R>>::lowest(), stdr::max); };

		auto maxs = rst2D | std::ranges::views::transform(hard_max);

		print_1d("maxs = ", maxs);

		BOOST_TEST(maxs.size() == 2 );

		BOOST_TEST( std::abs( maxs[0] - 2.0F) < 1e-12F );
		BOOST_TEST( std::abs( maxs[1] - 5.0F) < 1e-12F );

#if defined(__cpp_lib_ranges_zip) && (__cpp_lib_ranges_zip >= 202110L)
#define FWD(val) std::forward<decltype(val)>(val)
		auto Z = std::ranges::views::zip(rst2D, maxs);
		std::cout << get<0>(Z[0])[1] << std::endl;

		auto A = Z | stdr::views::transform(
						 [](auto row_max) noexcept { auto [row, max] = row_max; return FWD(row) | stdr::views::transform([max](auto elem) noexcept { return elem - max; }); }
					 );

		BOOST_TEST(A.size() == 2);
		BOOST_TEST(A[0].size() == 3);
		print_2d("A = ", A);

		auto Aexp = Z | stdr::views::transform([](auto row_max) noexcept {auto [row, max] = row_max; return FWD(row) | stdr::views::transform([max](auto elem) noexcept {return std::exp(elem - max);}); });

		print_2d("Aexp = ", Aexp);

		static auto sum = []<class R>(R const& rng) { return stdr::fold_left(rng, std::ranges::range_value_t<R>{}, std::plus<>{}); };

		auto den = Aexp | stdr::views::transform(sum);
		print_1d("den = ", den);
		auto Z2  = std::ranges::views::zip(Aexp, den);
		auto num = Z2 | stdr::views::transform(
							[](auto&& row_de) noexcept {
								auto&& [row, de] = FWD(row_de);
								// return row;
								return FWD(row) | stdr::views::transform([de](auto const& elem) noexcept { return elem / de; });
							}
						);
		print_2d("num = ", num);

		BOOST_TEST( static_cast<multi::size_t>(num.size()) == rst2D.size() );
		BOOST_TEST( static_cast<multi::size_t>(num.size()) == num.end() - num.begin() );

		auto softmax = [](auto const& matrix) {
			// [[maybe_unused]] auto [_, _cols] = matrix.extensions();
			auto maxs_ = matrix | std::ranges::views::transform(hard_max);
			auto Z_    = std::ranges::views::zip(matrix, maxs_);
			auto _num  = Z_ | stdr::views::transform([](auto row_max) noexcept {auto [row, max] = row_max; return FWD(row) | stdr::views::transform([max](auto elem) noexcept {return std::exp(elem - max);}); });
			auto _den  = _num | stdr::views::transform(sum);
			auto Z2_   = std::ranges::views::zip(_num, _den);
			auto ret   = Z2_  //
					 | stdr::views::transform([](auto&& row_de) noexcept { auto&& [row, de] = FWD(row_de); return FWD(row) | stdr::views::transform([de](auto const& elem) noexcept { return elem / de; }); });
			return ret;
		};

		print_2d("sm = ", softmax(rst2D));
#endif
#endif
#endif
	}
#ifndef __NVCC__
#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
#if defined(__cpp_lib_ranges_fold) && (__cpp_lib_ranges_fold >= 202207L)
	{
		namespace stdr = std::ranges;

		static auto hard_max = []<class R>(R const& row) { return stdr::fold_left(row, std::numeric_limits<stdr::range_value_t<R>>::lowest(), stdr::max); };
		static auto sum      = []<class R>(R const& rng) { return stdr::fold_left(rng, stdr::range_value_t<R>{}, std::plus<>{}); };

		auto softmax = [](auto const& matrix) {
			// [[maybe_unused]] auto [_, _cols] = matrix.extensions();
			auto maxs = matrix | stdr::views::transform(hard_max);
			auto num =
				stdr::views::zip(matrix, maxs) | stdr::views::transform(
													 [](auto row_max) noexcept {
														 auto&& [row, max] = FWD(row_max);
														 return FWD(row) | stdr::views::transform([max](auto elem) noexcept { return std::exp(elem - max); });
													 }
												 );
			auto den = num | stdr::views::transform(sum);
			return stdr::views::zip(num, den) | stdr::views::transform(
													[](auto&& row_de) noexcept {
														auto&& [row, de] = FWD(row_de);
														return FWD(row) | stdr::views::transform([de](auto const& elem) noexcept { return elem / de; });
													}
												);
		};

		auto const matrix = ([](auto ii) noexcept { return static_cast<float>(ii); } ^ multi::extensions_t(6)).partitioned(2);

		// auto matrix = parrot::range(6).as<float>().reshape({2, 3});

		print_2d("sm = ", softmax(matrix));

		auto const allocated_matrix = multi::array<float, 2>{
			{0.0F, 1.0F, 2.0F},
			{3.0F, 4.0F, 5.0F}
		};

		print_2d("sm2 = ", softmax(allocated_matrix));

		auto sm2 = softmax(allocated_matrix);
		sm2.begin();
		sm2.end();
		// auto const result_maxtrix = multi::array<float, 2>(sm2.begin(), sm2.end());
	}
#endif
#endif
#endif

	return boost::report_errors();
}
