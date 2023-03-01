// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUBLAS gemv"
#include<boost/test/unit_test.hpp>

#include <multi/adaptors/cuda/cublas.hpp>

#include <multi/adaptors/blas/gemm.hpp>
#include <multi/adaptors/blas/axpy.hpp>
#include <multi/adaptors/blas/nrm2.hpp>
#include <multi/adaptors/thrust.hpp>

#include<thrust/complex.h>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_blas_gemv_real_complex) {
	namespace blas = multi::blas;
	using complex = thrust::complex<double>;

	multi::thrust::cuda::array<complex, 2> const M_gpu =  {  // NOLINT(readability-identifier-length) BLAS naming
		{ { 9.0, 0.0}, {24.0, 0.0}, {30.0, 0.0}, {9.0, 0.0} },
		{ { 4.0, 0.0}, {10.0, 0.0}, {12.0, 0.0}, {7.0, 0.0} },
		{ {14.0, 0.0}, {16.0, 0.0}, {36.0, 0.0}, {1.0, 0.0} },
	};

	multi::thrust::cuda::array<complex, 1> const X_gpu = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0}, {4.1, 0.0} };

	multi::thrust::cuda::array<complex, 1> Y_gpu = { {4.0, 0.0}, {5.0, 0.0}, {6.0, 0.0} };

	blas::gemv(/*alpha*/ 1.1, M_gpu, X_gpu, /*beta*/ 1.2, Y_gpu);  // y = a*M*x + b*y

	multi::array<complex, 1> const Y_copy = Y_gpu;

	using blas::operators::operator-;
	BOOST_REQUIRE_SMALL( +blas::nrm2(Y_copy - multi::array<complex, 1>{ {214.02, 0.0}, {106.43, 0.0}, {188.37, 0.0} }) , 1e-13);
}
