// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <chrono>
#include <cmath>

#if defined(TBB_FOUND) && !defined(__NVCC__)
#ifndef __clang__
#include <execution>
#endif
#endif

#include <algorithm>  // for for_each
#include <iostream>   // for basic_ostream, operator<<
#include <string>     // for char_traits, operator<<

#if (__cplusplus >= 202002L) || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
#include <ranges>
#endif

namespace multi = boost::multi;

namespace {

class auto_timer : std::chrono::high_resolution_clock {
	std::string label_;
	time_point  start_ = now();

 public:
	explicit auto_timer(char const* label) : label_{label} {}

	auto_timer(auto_timer const&) = delete;
	auto_timer(auto_timer&&)      = delete;

	auto operator=(auto_timer const&) -> auto_timer& = delete;
	auto operator=(auto_timer&&) -> auto_timer&      = delete;

	~auto_timer() {
		auto const count = std::chrono::duration<double>(now() - start_).count();
		std::cerr << label_ << ": " << count << " sec\n";
	}
};
}  // namespace

auto main() -> int {  // NOLINT(bugprone-exception-escape)
	using T = double;
	{
		auto cpu_own = multi::array<T, 3>({64, 64, 64}, 0);

		auto&& cpu = cpu_own();
		{
			auto_timer const _{"triple nested for"};
			for(auto&& plane : cpu) {                                             // NOLINT(modernize-use-ranges)
				for(auto&& row : plane) {                                         // NOLINT(altera-unroll-loops)
					for(auto&& elem : row) {                                      // NOLINT(altera-unroll-loops)
						elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));  // cppcheck-suppress useStlAlgorithm;
					}
				}
			}
		}
		{
			auto_timer const _{"std::for_each"};
			std::for_each(cpu.begin(), cpu.end(), [](auto&& plane) {              // NOLINT(modernize-use-ranges,llvm-use-ranges) for C++20
				for(auto&& row : plane) {                                         // NOLINT(altera-unroll-loops)
					for(auto&& elem : row) {                                      // NOLINT(altera-unroll-loops)
						elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));  // cppcheck-suppress useStlAlgorithm;
					}
				}
			});
		}
#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
		// TODO(correaa): make it work std::ranges::for_each (elem ends being const)
		// {
		// 	auto_timer const _{"std::ranges::for_each"};
		// 	std::ranges::for_each(cpu, [](auto&& plane) {
		// 		for(auto&& row : std::forward<decltype(plane)>(plane)) {                                         // NOLINT(altera-unroll-loops)
		// 			for(auto&& elem : std::forward<decltype(row)>(row)) {         // NOLINT(altera-unroll-loops)
		// 				elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));  // cppcheck-suppress useStlAlgorithm;
		// 			}
		// 		}
		// 	});
		// }
#endif
#ifdef __cpp_lib_execution
#if defined(TBB_FOUND) && !defined(__NVCC__)
#if !defined(__clang__)
		{
			auto_timer const _{"std::for_each(std::par)"};
			std::for_each(std::execution::par, cpu.begin(), cpu.end(), [](auto&& plane) {
				for(auto&& row : plane) {  // NOLINT(altera-unroll-loops)
					for(auto&& elem : row) {
						elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));
					}
				}
			});
		}
		{
			auto_timer const _{"std::for_each(std::par, elements)"};
			std::for_each(std::execution::par, cpu.elements().begin(), cpu.elements().end(), [](auto&& elem) {
				elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));
			});
		}
#endif
#endif
#endif
		{
			auto_timer const _{"std::for_each(elements)"};
			std::for_each(cpu.elements().begin(), cpu.elements().end(), [](auto&& elem) {
				elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));
			});
		}
		{
			std::for_each(
				cpu.extents().elements().begin(),
				cpu.extents().elements().end(),
				[&cpu](auto const& coords) {
					auto [i, j, k] = coords;

					cpu[i][j][k] = static_cast<double>(i + j + k);
				}
			);
			BOOST_TEST( std::abs(cpu[1][2][3] - 6.0) < 1e-10 );
		}
		{
			std::transform(
				cpu.extents().elements().begin(),
				cpu.extents().elements().end(),
				cpu.elements().begin(),
				[](auto const& coords) {
					auto [i, j, k] = coords;

					return static_cast<double>(i + j + k);
				}
			);
			BOOST_TEST( std::abs(cpu[1][2][3] - 6.0) < 1e-10 );
		}
	}
	{
		multi::array<int, 2> arr({5, 7}, 5);
		std::for_each(arr.elements().begin(), arr.elements().end(), [](auto& element) { ++element; });

		BOOST_TEST( arr[3][2]== 6 );
	}
#if defined(__cpp_lib_ranges_zip) and (__cpp_lib_ranges_zip >= 202110L)
	{
		multi::array<int, 2> arr1({5, 7}, 5);
		multi::array<int, 2> arr2({5, 7}, 6);

		auto e1 = arr1.elements();  // bind to lvalues: zip needs viewable_range; array_ref temporaries are not views
		auto e2 = arr2.elements();

		std::ranges::for_each(
			std::views::zip(e1, e2),
			[](auto&& pair) { using std::get; thrust::swap(get<0>(pair), get<1>(pair)); }
		);

		BOOST_TEST( arr1[3][2]== 6 );
		BOOST_TEST( arr2[3][2]== 5 );
	}
	{
		multi::array<int, 2> arr1({5, 7}, 5);
		multi::array<int, 2> arr2({5, 7}, 6);

		std::ranges::for_each(
			std::views::zip(arr1.elements(), arr2.elements()),
			[](auto&& pair) { using std::get; thrust::swap(get<0>(pair), get<1>(pair)); }
		);

		BOOST_TEST( arr1[3][2]== 6 );
		BOOST_TEST( arr2[3][2]== 5 );
	}
#endif

	return boost::report_errors();
}
