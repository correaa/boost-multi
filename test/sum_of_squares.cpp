// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/restriction.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <functional>  // for std::plus
#include <numeric>     // for std::transform_reduce
// #include <version>     // for the feature-test macros used to guard the paths below

#ifdef __cpp_lib_ranges_fold
#include <algorithm>  // for std::ranges::fold_left  (C++23)
#include <ranges>     // for std::views::transform
#endif

#if defined(__cpp_lib_parallel_algorithm) && !defined(__NVCC__) && !(defined(__clang__) && (__clang_major__ < 17) && defined(__GLIBCXX__))
#define MULTI_HAS_PARALLEL_EXECUTION 1
#include <execution>  // for std::execution::par / parallel_policy
#endif

namespace multi = boost::multi;

namespace {

auto sos(int N) {  // NOLINT(readability-identifier-length)  // N is the number of integers to sum
	using multi::range;

#ifdef __cpp_lib_ranges_fold  // C++23 ranges form (sequential)
	return std::ranges::fold_left(
		range(0, N) | std::views::transform([](auto const& e) noexcept { return e * e; }),
		0,
		std::plus<>{}
	);
#else  // C++17/20 fallback: there is no std::ranges::transform_reduce
	return std::transform_reduce(
		range(0, N).begin(), range(0, N).end(),
		0,
		std::plus<>{},
		[](auto const& e) noexcept { return e * e; }
	);
#endif
}

template<class ExecutionPolicy
#ifdef MULTI_HAS_PARALLEL_EXECUTION  // execution policies + (policy, ...) overloads
		 = std::execution::parallel_policy
#endif
		 >
auto sos(ExecutionPolicy&& ep, int N) {  // NOLINT(readability-identifier-length)  // N is the number of integers to sum
	using multi::range;

	return std::transform_reduce(
		std::forward<ExecutionPolicy>(ep),
		range(0, N).begin(), range(0, N).end(), 0,
		std::plus<>{},
		[](auto const& e) { return e * e; }
	);
}

}  // namespace

auto main() -> int {
	BOOST_TEST( sos(4) == 0 + 1 + 4 + 9 );
#ifdef MULTI_HAS_PARALLEL_EXECUTION
	BOOST_TEST( sos(std::execution::par, 4) == sos(4) );
	BOOST_TEST( sos({}, 4) == sos(4) );  // default policy via default template argument
#endif

	return boost::report_errors();
}
