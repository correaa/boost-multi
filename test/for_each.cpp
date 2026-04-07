// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <chrono>
#include <cmath>

#if defined(TBB_FOUND) && !defined(__NVCC__)
#if !defined(__clang__)
#include <execution>
#endif
#endif

#include <algorithm>  // for for_each
#include <iostream>   // for basic_ostream, operator<<
#include <string>     // for char_traits, operator<<

namespace multi = boost::multi;

namespace {

class auto_timer : std::chrono::high_resolution_clock {
	std::string label_;
	time_point  start_;

 public:
	explicit auto_timer(char const* label) : label_{label} {
		start_ = now();
	}

	auto_timer(auto_timer const&) = delete;
	auto_timer(auto_timer&&)      = delete;

	auto operator=(auto_timer const&) -> auto_timer& = delete;
	auto operator=(auto_timer&&) -> auto_timer&      = delete;

	// auto_timer() : auto_timer("") {}
	// auto elapsed() const {
	// 	cudaDeviceSynchronize() == cudaSuccess ? void() : assert(0);
	// 	struct {
	// 		long long wall;
	// 	} ret{std::chrono::duration_cast<std::chrono::nanoseconds>(now() - start_)
	// 			  .count()};
	// 	return ret;
	// }
	~auto_timer() {
		// cudaDeviceSynchronize() == cudaSuccess ? void() : assert(0);
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
			for(auto&& plane : cpu) {  // NOLINT(modernize-use-ranges)
				for(auto&& row : plane) {                             // NOLINT(altera-unroll-loops)
					for(auto&& elem : row) {                          // NOLINT(altera-unroll-loops)
						elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));
					}
				}
			}
		}

		{
			auto_timer const _{"std::for_each"};
			std::for_each(cpu.begin(), cpu.end(), [](auto&& plane) {  // NOLINT(modernize-use-ranges)
				for(auto&& row : plane) {                             // NOLINT(altera-unroll-loops)
					for(auto&& elem : row) {                          // NOLINT(altera-unroll-loops)
						elem += std::sqrt(std::pow(elem, 1.5) + std::sin(elem));
					}
				}
			});
		}
#if defined(__cpp_lib_execution)
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
	}

	return boost::report_errors();
}
