// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <chrono>
#include <cmath>

#if defined(TBB_FOUND) && !defined(__NVCC__) || __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1)
#if !defined(__clang__) && !defined(__CUDACC__)
#include <execution>
#endif
#endif

#include <numeric>

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
		auto cpu     = multi::array<T, 2>({64, 1024 * 1024}, 0);
		auto cpu_par = multi::array<T, 2>({64, 1024 * 1024});

		{
			auto_timer t{"std::for_each"};
			std::for_each(cpu.begin(), cpu.end(), [](auto&& row) {
				for(auto&& e : row) {
					e += std::sqrt(std::pow(e, 1.5) + std::sin(e));
				}
			});
		}
#if defined(__cpp_lib_execution)
#if defined(TBB_FOUND) && !defined(__NVCC__) || __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1)
#if !defined(__clang__) && !defined(__CUDACC__)
		{
			auto_timer t{"std::for_each(std::par)"};
			std::for_each(std::execution::par, cpu.begin(), cpu.end(), [](auto&& row) {
				for(auto&& e : row) {
					e += std::sqrt(std::pow(e, 1.5) + std::sin(e));
				}
			});
		}
		{
			auto_timer t{"std::for_each(std::par, elements)"};
			std::for_each(std::execution::par, cpu.elements().begin(), cpu.elements().end(), [](auto&& e) {
				e += std::sqrt(std::pow(e, 1.5) + std::sin(e));
			});
		}
#endif
#endif
#endif
	}

	return boost::report_errors();
}
