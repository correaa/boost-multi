// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef _VSTD
# define _VSTD std  // NOLINT(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
#endif

#include <boost/multi/adaptors/thrust/omp.hpp>

#if !defined(__has_feature) || !__has_feature(address_sanitizer)
# include <boost/multi/array.hpp>
#endif

#include <omp.h>
#include <thrust/reduce.h>
#include <thrust/system/omp/detail/par.h>

#include <boost/core/lightweight_test.hpp>

#if !defined(__has_feature) || !__has_feature(address_sanitizer)
# include <chrono>
# include <cstdio>
# include <iostream>
#endif

namespace {

template<class Array1D>
auto serial_array_sum(Array1D const& arr) {
	auto const                   size  = arr.size();
	auto const* const            aptr  = raw_pointer_cast(arr.data_elements());
	typename Array1D::value_type total = 0.0;
	for(typename Array1D::size_type i = 0; i != size; ++i) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		total += aptr[i];                                     // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
	return total;
}

template<class Array1D>
auto parallel_array_sum(Array1D const& arr) {
	auto const                   size  = arr.size();
	auto const* const            aptr  = raw_pointer_cast(arr.data_elements());
	typename Array1D::value_type total = 0.0;
#pragma omp parallel for reduction(+ : total)                 // NOLINT(openmp-use-default-none)
	for(typename Array1D::size_type i = 0; i < size; ++i) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		total += aptr[i];                                     // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
	return total;
}

template<class Array1D>
auto parallel_idiom_array_sum(Array1D const& arr) {
	typename Array1D::value_type total = 0.0;
#pragma omp parallel for reduction(+ : total)  // NOLINT(openmp-use-default-none)
#ifndef __NVCOMPILER
	for(auto const i : arr.extension()) {      // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		// NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
		total += arr[i];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
#else
	for(auto const it = arr.extension().begin(); it < arr.extension().end(); ++it) {      // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		// NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
		total += arr[*it];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
#endif
	return total;
}

template<class Array1D>
auto thrust_array_sum(Array1D const& arr) {
	return thrust::reduce(arr.begin(), arr.end(), typename Array1D::value_type{});
}

template<class Array1D>
auto thrust_omp_array_sum(Array1D const& arr) {
	return thrust::reduce(thrust::omp::par, arr.begin(), arr.end(), typename Array1D::value_type{});
}

template<class Tp>
inline __attribute__((always_inline)) void DoNotOptimize(Tp const& value) {  // NOLINT(readability-identifier-naming)
	asm volatile("" : : "r,m"(value) : "memory");                            // NOLINT(hicpp-no-assembler)
}

template<class Tp>
inline __attribute__((always_inline)) void DoNotOptimize(Tp& value) {  // NOLINT(readability-identifier-naming)
#if defined(__clang__)
	asm volatile("" : "+r,m"(value) : : "memory");  // NOLINT(hicpp-no-assembler)
#else
	asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

}  // end namespace

// auto parallel_array_sum(int n, float const *a) {
//     float total = 0.0;
//     #pragma omp parallel for reduction(+:total)
//     for (int i = 0; i < n; i++) {
//         total += a[i];
//     }
//     return total;
// }

/**
 * @brief Solution to the Hello world exercise in OpenMP.
 **/
auto main() -> int {
// 1) Create the OpenMP parallel region
#pragma omp parallel default(none)
	{
		// 1.1) Get my thread number
		int const my_id = omp_get_thread_num();

		// 1.2) Get the number of threads inside that parallel region
		int const thread_number = omp_get_num_threads();

		// 1.3) Print everything
		printf("\"Hello world!\" from thread %d, we are %d threads.\n", my_id, thread_number);  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	}

#if !defined(__has_feature) || !__has_feature(address_sanitizer)
	namespace multi = boost::multi;

	multi::thrust::omp::array<double, 1> arr(1U << 30U);

	{
# pragma omp parallel for default(none) shared(arr)
		for(int i = 0; i < arr.size(); ++i) {  // NOLINT(altera-unroll-loops)
			arr[i] = static_cast<double>(i) * static_cast<double>(i);
		}
	}

	BOOST_TEST( arr[arr.size() - 1] == static_cast<double>(arr.size() - 1)*static_cast<double>(arr.size() - 1) );

	auto tick = std::chrono::high_resolution_clock::now();

	auto const serial = serial_array_sum(arr);
	DoNotOptimize(serial);
	std::cout << "serial " << (std::chrono::high_resolution_clock::now() - tick).count() << '\n';

	tick = std::chrono::high_resolution_clock::now();

	auto const parallel = parallel_array_sum(arr);
	DoNotOptimize(parallel);
	std::cout << "parallel " << (std::chrono::high_resolution_clock::now() - tick).count() << '\n';

	std::cout << serial << ' ' << parallel << ' ' << serial - parallel << '\n';
	BOOST_TEST( std::abs((serial / parallel) - 1.0) < 1.0e-12 );

	tick = std::chrono::high_resolution_clock::now();

	auto const parallel_idiom = parallel_idiom_array_sum(arr);
	DoNotOptimize(parallel_idiom);
	std::cout << "parallel idiom " << (std::chrono::high_resolution_clock::now() - tick).count() << '\n';

	BOOST_TEST( std::abs((parallel_idiom / parallel) - 1.0) < 1.0e-12 );

	tick = std::chrono::high_resolution_clock::now();

	auto const thrust = thrust_array_sum(arr);
	DoNotOptimize(thrust);
	std::cout << "thrust " << (std::chrono::high_resolution_clock::now() - tick).count() << '\n';

	BOOST_TEST( std::abs((thrust / parallel) - 1.0) < 1.0e-12 );

	tick = std::chrono::high_resolution_clock::now();

	auto const thrust_omp = thrust_omp_array_sum(arr);
	DoNotOptimize(thrust_omp);
	std::cout << "thrust omp " << (std::chrono::high_resolution_clock::now() - tick).count() << '\n';

	BOOST_TEST( std::abs((thrust_omp / parallel) - 1.0) < 1.0e-12 );

	multi::array<double, 1> const arr_normal{arr};
	DoNotOptimize(arr_normal);

	tick = std::chrono::high_resolution_clock::now();

	auto const thrust_omp_normal = thrust_omp_array_sum(arr_normal);
	DoNotOptimize(thrust_omp_normal);
	std::cout << "thrust omp normal " << (std::chrono::high_resolution_clock::now() - tick).count() << '\n';
	BOOST_TEST( std::abs((thrust_omp_normal / parallel) - 1.0) < 1.0e-12 );
#endif

	return boost::report_errors();
}
