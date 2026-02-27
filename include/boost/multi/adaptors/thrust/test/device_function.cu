// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/thrust.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>

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
		first,
		last,
		d_out.begin(),
		[] DEV(int x) { return x * x; }
	);

	// auto c2 = multi::restricted<1>( [] DEV (int x) { return x * x; } ,  {N} );

	// thrust::copy(

	// );

	multi::thrust::host_array<int, 1> h_out = d_out;

	for(int v : h_out)
		std::cout << v << " ";
}
