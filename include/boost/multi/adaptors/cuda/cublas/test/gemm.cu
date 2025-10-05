// Copyright 2023-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/cuda/cublas.hpp>

#include <boost/multi/adaptors/blas/axpy.hpp>
#include <boost/multi/adaptors/blas/gemm.hpp>
#include <boost/multi/adaptors/blas/gemv.hpp>
#include <boost/multi/adaptors/blas/nrm2.hpp>

#include <boost/multi/adaptors/thrust.hpp>

#include <thrust/complex.h>

#include <numeric>  // for std::iota

namespace multi = boost::multi;

#include <boost/core/lightweight_test.hpp>

int main() {
	// BOOST_AUTO_TEST_CASE(multi_blas_gemv_complex)
	{
		namespace blas = multi::blas;
		using complex  = thrust::complex<double>;
		// NOLINTNEXTLINE(readability-identifier-length) BLAS naming
		multi::thrust::cuda::array<complex, 2> const M_gpu = {
			{ {9.0, 0.0}, {24.0, 0.0}, {30.0, 0.0}, {9.0, 0.0}},
			{ {4.0, 0.0}, {10.0, 0.0}, {12.0, 0.0}, {7.0, 0.0}},
			{{14.0, 0.0}, {16.0, 0.0}, {36.0, 0.0}, {1.0, 0.0}},
		};

		multi::thrust::cuda::array<complex, 1> const X_gpu = {
			{1.1, 0.0},
			{2.1, 0.0},
			{3.1, 0.0},
			{4.1, 0.0}
		};

		multi::thrust::cuda::array<complex, 1> Y_gpu = {
			{4.0, 0.0},
			{5.0, 0.0},
			{6.0, 0.0}
		};

		blas::gemv(/*alpha*/ 1.1, M_gpu, X_gpu, /*beta*/ 1.2, Y_gpu);  // y = a*M*x + b*y

		multi::array<complex, 1> const Y_copy = Y_gpu;

		using blas::operators::operator-;
		BOOST_TEST(+blas::nrm2(Y_copy - 
			multi::array<complex, 1>{
				{214.02, 0.0},
				{106.43, 0.0},
				{188.37, 0.0}
        	}
		) < 1e-13);
	}

	// BOOST_AUTO_TEST_CASE(cublas_gemv_real)
	{
		namespace blas = multi::blas;
		using T        = double;
		// NOLINTNEXTLINE(readability-identifier-length) BLAS naming
		multi::thrust::cuda::array<T, 2> const M_gpu = {
			{ 9.0, 24.0, 30.0, 9.0},
			{ 4.0, 10.0, 12.0, 7.0},
			{14.0, 16.0, 36.0, 1.0},
		};

		multi::thrust::cuda::array<T, 1> const X_gpu = {1.1, 2.1, 3.1, 4.1};

		multi::thrust::cuda::array<T, 1> Y_gpu = {4.0, 5.0, 6.0};

		blas::gemv(/*alpha*/ 1.1, M_gpu, X_gpu, /*beta*/ 1.2, Y_gpu);  // y = a*M*x + b*y

		multi::array<T, 1> const Y_copy = Y_gpu;

		using blas::operators::operator-;
		BOOST_TEST(+blas::nrm2(Y_copy - multi::array{214.02, 106.43, 188.37}) < 1e-13);
	}

	// BOOST_AUTO_TEST_CASE(cublas_gemm_nh)
	{
		namespace blas = multi::blas;

		using complex = thrust::complex<double>;
		complex const I{0.0, 1.0};  // NOLINT(readability-identifier-length) imaginary unit

		{
						// NOLINTNEXTLINE(readability-identifier-length) BLAS naming
			multi::thrust::cuda::array<complex, 2> const a = {
				{1.0 - 2.0 * I, 9.0 - 1.0 * I},
				{2.0 + 3.0 * I, 1.0 - 2.0 * I},
			};

			multi::thrust::cuda::array<complex, 2> c({2, 2}, {9999.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
			blas::gemm({1.0, 0.0}, a, a, {0.0, 0.0}, c);                      // c=aa†, c†=aa†

			multi::array<complex, 2> const c_copy = c;
			BOOST_TEST( c_copy[1][0] == 16.0 -  2.0*I );
			BOOST_TEST( c_copy[0][1] == 14.0 - 38.0*I );
		}
		{
									// NOLINTNEXTLINE(readability-identifier-length) BLAS naming
			multi::thrust::cuda::array<complex, 2> const a = {
				{1.0 - 2.0 * I, 9.0 - 1.0 * I},
				{2.0 + 3.0 * I, 1.0 - 2.0 * I},
			};

			auto const c = +blas::gemm(complex{1.0, 0.0}, a, a);  // c=aa†, c†=aa†

			multi::array<complex, 2> const c_copy = c;
			BOOST_TEST( c_copy[1][0] == 16.0 -  2.0*I );
			BOOST_TEST( c_copy[0][1] == 14.0 - 38.0*I );
		}
		{
						// NOLINTNEXTLINE(readability-identifier-length) BLAS naming
			multi::thrust::cuda::array<complex, 2> const a = {
				{1.0 - 2.0 * I, 9.0 - 1.0 * I},
				{2.0 + 3.0 * I, 1.0 - 2.0 * I},
			};

			multi::thrust::cuda::array<complex, 2> c({2, 2}, {0.0, 0.0});
			c += blas::gemm(complex{1.0, 0.0}, a, a);  // c=aa†, c†=aa†

			multi::array<complex, 2> const c_copy = c;
			BOOST_TEST( c_copy[1][0] == 16.0 -  2.0*I );
			BOOST_TEST( c_copy[0][1] == 14.0 - 38.0*I );
		}

		// Chris
		{
			int m   = 12;
			int n   = 10;
			int ell = 8;

			multi::array<double, 3> a({m, n, ell});
			multi::array<double, 2> b({n, ell});

			std::iota(a.elements().begin(), a.elements().end(), 20.0);
			std::iota(b.elements().begin(), b.elements().end(), 30.0);

			multi::array<double, 1> c_gold(m, 0.0);

			for(int k = 0; k != m; ++k) {
				for(int j = 0; j != n; ++j) {
					for(int i = 0; i != ell; ++i) {
						c_gold[k] += a[k][j][i] * b[j][i];
					}
				}
			}

			multi::array<double, 1> c_flat(m, 0.0);

			for(int k = 0; k != m; ++k) {
				for(int ji = 0; ji != a[k].elements().size(); ++ji) {
					c_flat[k] += a[k].elements()[ji] * b.elements()[ji];
				}
			}

			BOOST_TEST( c_gold == c_flat );

			// gpu::run(m, n, [...] (auto k, auto j) {
			//      for (int i =0; i <l ; ++i) {
			//        c[k] +=a[k][j][i]) * b[j][i];
			//        }
			//  });
		}
	}

	return boost::report_errors();
}
