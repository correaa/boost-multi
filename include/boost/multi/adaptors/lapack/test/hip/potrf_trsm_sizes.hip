// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Device potrf (hipSOLVER) and trsm (hipBLAS) over a sweep of sizes, checked on the host.
// rocSOLVER and rocBLAS switch to blocked algorithms above some size, so small-matrix tests
// do not exercise the kernels that large problems use; inq's orthogonalize (Cholesky + trsm
// of a 100x100 overlap matrix) failed on the MI300X simulator while 3x3 to 37x37 passed.

#include <boost/multi/adaptors/lapack/hipsolver.hpp>  // first: defines MULTI_USE_HIP for the other adaptors (the CI also passes it globally)

#include <boost/multi/adaptors/cuda/cublas.hpp>  // before blas/trsm.hpp, so the device contexts are visible to it

#include <boost/multi/adaptors/blas/trsm.hpp>
#include <boost/multi/adaptors/lapack/filling.hpp>
#include <boost/multi/adaptors/lapack/potrf.hpp>

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <thrust/complex.h>
#include <thrust/system/hip/memory.h>

#include <boost/core/lightweight_test.hpp>

#include <algorithm>  // for max
#include <cstdio>     // for printf
#include <random>     // for mt19937, uniform_real_distribution

namespace multi = boost::multi;

namespace {
using complex = ::thrust::complex<double>;  // not std::complex: the latter currently breaks the host <-> device array copy

template<class T, multi::dimensionality_type D>
using device_array = multi::array<T, D, ::thrust::hip::allocator<T>>;

auto random_matrix(multi::ssize_t n, std::mt19937& gen) {  // NOLINT(readability-identifier-length) conventional name
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	multi::array<complex, 2> ret({n, n});
	for(auto&& row : ret) {
		for(auto&& elem : row) {
			elem = complex{dist(gen), dist(gen)};
		}
	}
	return ret;
}

// well conditioned Hermitian positive definite matrix, A = M^H M / n + I
auto hpd_matrix(multi::ssize_t n, std::mt19937& gen) {  // NOLINT(readability-identifier-length) conventional name
	auto const M = random_matrix(n, gen);  // NOLINT(readability-identifier-length) conventional name

	multi::array<complex, 2> A({n, n}, complex{0.0, 0.0});  // NOLINT(readability-identifier-length) conventional name
	for(multi::index i = 0; i != n; ++i) {
		for(multi::index j = 0; j != n; ++j) {
			for(multi::index k = 0; k != n; ++k) {
				A[i][j] += ::thrust::conj(M[k][i]) * M[k][j];
			}
			A[i][j] /= static_cast<double>(n);
		}
		A[i][i] += 1.0;
	}
	return A;
}

auto max_abs(multi::array<complex, 2> const& A) {  // NOLINT(readability-identifier-length) conventional name
	double ret = 0.0;
	for(auto const& row : A) {
		for(auto const& elem : row) {
			ret = std::max(ret, abs(elem));
		}
	}
	return ret;
}

// max |L.R - C| / max |C|, with plain loops so the reference does not depend on any BLAS
auto product_error(multi::array<complex, 2> const& L, multi::array<complex, 2> const& R, multi::array<complex, 2> const& C) {  // NOLINT(readability-identifier-length) conventional name
	double ret = 0.0;
	for(multi::index i = 0; i != C.size(); ++i) {
		for(multi::index j = 0; j != (~C).size(); ++j) {
			complex sum{0.0, 0.0};
			for(multi::index k = 0; k != R.size(); ++k) {
				sum += L[i][k] * R[k][j];
			}
			ret = std::max(ret, abs(sum - C[i][j]));
		}
	}
	return ret / max_abs(C);
}

auto hermitian_transpose(multi::array<complex, 2> const& A) {  // NOLINT(readability-identifier-length) conventional name
	multi::array<complex, 2> ret({(~A).size(), A.size()});
	for(multi::index i = 0; i != A.size(); ++i) {
		for(multi::index j = 0; j != (~A).size(); ++j) {
			ret[j][i] = ::thrust::conj(A[i][j]);
		}
	}
	return ret;
}

auto identity(multi::ssize_t n) {  // NOLINT(readability-identifier-length) conventional name
	multi::array<complex, 2> ret({n, n}, complex{0.0, 0.0});
	for(multi::index i = 0; i != n; ++i) {
		ret[i][i] = 1.0;
	}
	return ret;
}
}  // end unnamed namespace

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	namespace blas = multi::blas;

	double const tolerance = 1.0e-12;

	std::mt19937 gen(42);  // NOLINT(cert-msc32-c,cert-msc51-cpp) fixed seed, reproducible

	// sizes around the usual block-size thresholds of the GPU libraries; 37 and 100 are inq's passing and failing cases
	for(multi::ssize_t const n : {16, 32, 37, 48, 63, 64, 65, 96, 100, 128}) {  // NOLINT(readability-identifier-length) conventional name
		auto const A_host = hpd_matrix(n, gen);  // NOLINT(readability-identifier-length) conventional name

		// potrf: A = U^H U, with U upper triangular
		device_array<complex, 2> U_dev = A_host;  // NOLINT(readability-identifier-length) conventional name

		auto&& res = multi::lapack::potrf(multi::lapack::filling::upper, U_dev);
		BOOST_TEST( res.size() == n );  // fully positive definite

		multi::array<complex, 2> U = U_dev;  // NOLINT(readability-identifier-length) conventional name
		for(multi::index i = 0; i != n; ++i) {
			for(multi::index j = 0; j != i; ++j) {
				U[i][j] = 0.0;  // the lower triangle is not referenced, it can hold anything
			}
		}
		auto const UH = hermitian_transpose(U);  // NOLINT(readability-identifier-length) conventional name

		double const potrf_error = product_error(UH, U, A_host);

		// trsm on the left: X = U^-1 B
		auto const B_host = random_matrix(n, gen);  // NOLINT(readability-identifier-length) conventional name

		device_array<complex, 2> const U_tri = U;  // NOLINT(readability-identifier-length) conventional name
		device_array<complex, 2>       X_dev = B_host;  // NOLINT(readability-identifier-length) conventional name

		blas::trsm(blas::side::left, blas::filling::upper, complex{1.0, 0.0}, U_tri, X_dev);

		double const trsm_left_error = product_error(U, multi::array<complex, 2>(X_dev), B_host);

		// trsm the way inq's matrix::invert_triangular calls it: on the right, with a hermitized lower factor L = U^H,
		// which computes Y = (L^H)^-1 = U^-1
		device_array<complex, 2> const L_dev = UH;  // NOLINT(readability-identifier-length) conventional name
		device_array<complex, 2>       Y_dev = identity(n);  // NOLINT(readability-identifier-length) conventional name

		blas::trsm(blas::side::right, blas::filling::upper, complex{1.0, 0.0}, blas::H(L_dev), Y_dev);

		double const trsm_inq_error = product_error(U, multi::array<complex, 2>(Y_dev), identity(n));

		std::printf("n = %3ld   potrf %.3e   trsm left %.3e   trsm right (inq) %.3e\n", static_cast<long>(n), potrf_error, trsm_left_error, trsm_inq_error);  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)

		BOOST_TEST_LT( potrf_error    , tolerance );
		BOOST_TEST_LT( trsm_left_error, tolerance );
		BOOST_TEST_LT( trsm_inq_error , tolerance );
	}

	return boost::report_errors();
}
