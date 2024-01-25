// Copyright 2024 Alfredo A. Correa

// this header contains a generic fft algorithm
// when compiled using -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -DNDEBUG -mfpmath=sse -march=native -funroll-loops -fargument-noalias"

#ifndef MULTI_ALGORITHM_FFT_HPP
#define MULTI_ALGORITHM_FFT_HPP

#include <execution>  // for par  // needs linking to TBB library
#include <numeric>  // for inner_product and transform_reduce

namespace boost {
namespace multi {

template<class It, class Size, class It2>
auto dft_naive(It first, Size N, It2 d_first, Size M) {
	for(Size k = 0; k != M; ++k) {
		d_first[k] = 0.0;
		for(Size n = 0; n!= N; ++n) {
			d_first[k] += std::exp(std::complex<double>{0.0, 1.0}*2.0*M_PI*k*n/N)*first[n];
		}
	}
	return d_first + N;
}

template<class It, class Size, class It2>
auto dft_naive(It first, Size N, It2 d_first) {
	return dft_naive(first, N, d_first, N);
}

}  // end namespace multi
}  // end namespace boost
#endif
