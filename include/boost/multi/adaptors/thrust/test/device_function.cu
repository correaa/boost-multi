// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/thrust.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>

#include <boost/core/lightweight_test.hpp>

#include <iostream>

__device__ int square_device(int x) {
	return x * x;
}

struct square_functor {
	__device__ int operator()(int x) const {
		return square_device(x);
	}
};

namespace multi = boost::multi;

// #define _HOST

#ifndef _HOST
template<class T> using vector = multi::thrust::device_array<T, 1>;
#define DEV __device__
#else
template<class T> using vector = multi::thrust::host_array<T, 1>;
#define DEV
#endif

int main() {
	int const N = 8;

	vector<int> d_out(N);

	auto first = thrust::counting_iterator<int>(0);
	auto last  = first + N;

	thrust::transform(
		thrust::device,
		first,
		last,
		d_out.begin(),
		[] DEV(int x) { return x * x; }
	);

	// auto c2 = multi::thrust::device_restriction(multi::extensions_t<1>(N), [a = 5] __device__(int x) { return x * x + a; });
	// auto c2 = multi::thrust::device_restriction(multi::extensions_t<1>(N), multi::thrust::device_function<int>([a = 5] __device__(int x) { return x * x + a; }));

    // [[maybe_unused]] auto fff = [a = 5] __host__(int x) { return x * x + a; };
    // auto c2 
    //     = multi::thrust::device_restriction(multi::extensions_t<1>(N),
    //         multi::thrust::device_function<
    //             decltype((fff)(multi::index{}))
    //         >([a = 5] __device__(int x) { return x * x + a; })
    //     );

    // [[maybe_unused]] auto fff = [a = 5] __host__(int x) { return x * x + a; };
    // using result_type = decltype((fff)(multi::index{}));
    // auto c2 
    //     = multi::thrust::device_restriction(multi::extensions_t<1>(N),
    //         multi::thrust::device_function<
    //             result_type
    //         >([a = 5] __device__(int x) { return x * x + a; })
    //     );


    // auto c2
    //     = multi::thrust::device_restriction(
    //         multi::extensions_t<1>(N),
    //         [=]() {
    //             [[maybe_unused]] auto fff = [a = 5] __host__(int x) { return x * x + a; };
    //             using result_type = decltype((fff)(multi::index{}));
    //             return multi::thrust::device_function<result_type>([a = 5] __device__(int x) { return x * x + a; });
    //         }()
    // );

    // auto c2
    // = multi::thrust::device_restriction(
    //         multi::extensions_t<1>(N),
    //         [a = 5]() {
    //             // [[maybe_unused]] auto fff = [=] __host__(int x) { return x * x + a; };
    //             // using result_type = typename multi::thrust::result_helper<decltype(fff)>::type;
    //             return [=] __device__(int x) { return x * x + a; };
    //         }()
    // );

#if defined(__CUDACC__) && \
   (__CUDACC_VER_MAJOR__ < 12 || \
   (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 5))

    auto c2
    = multi::thrust::device_restriction(
            multi::extensions_t<1>(N),
            BOOST_MULTI_DEVICE_LAMBDA_LEGACY(BOOST_MULTI_CAPTURE(a = 5, b = 0), (auto x) { int c = a, d = b; return  x * x + c + d; })
    );

#else

    auto c2
    = multi::thrust::device_restriction(
            multi::extensions_t<1>(N),
            [a = 5, b = 0] BOOST_MULTI_DEVICE_LAMBDA((auto x) { int c = a, d = b; return  x * x + c + d; })
    );

#endif

	// auto it = c2.begin();
	// static_assert(std::is_default_constructible_v<multi::extensions_t<1>::iterator>);
	// static_assert(std::is_trivially_copyable_v<multi::extensions_t<1>::iterator>);
    // static_assert(std::is_default_constructible_v<decltype(it)>);

	// // decltype(it) dc;

	// thrust::copy(c2.begin(), c2.end(), d_out.begin());

	// multi::thrust::host_array<int, 1> h_out = d_out;

	// BOOST_TEST( h_out[1] == 6 );
	// BOOST_TEST( h_out[2] == 9 );

	return boost::report_errors();
}
