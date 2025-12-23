// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <thrust/complex.h>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

    multi::thrust::universal_array<thrust::complex<double>, 2> arr_gold({50, 70});

	auto [is, js] = arr_gold.extensions();
	for(auto i : is) {
		for(auto j : js) {
			arr_gold[i][j] = thrust::complex<double>{static_cast<double>(i), static_cast<double>(j)};
		}
	}

	multi::thrust::universal_array<thrust::complex<double>, 2> arr =
		[] __host__ __device__ (multi::index i, multi::index j) {
			return thrust::complex<double>{static_cast<double>(i), static_cast<double>(j)};
		} ^
		multi::extensions_t<2>({50, 70});

	BOOST_TEST( arr == arr_gold );

	return boost::report_errors();
}
