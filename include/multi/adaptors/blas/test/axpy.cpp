// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2019-2023 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS axpy"
#include <boost/test/unit_test.hpp>

#include <multi/adaptors/blas/axpy.hpp>
#include <multi/adaptors/blas/operations.hpp>
#include <multi/adaptors/complex.hpp>

#include <multi/array.hpp>

#include <complex>

namespace multi = boost::multi;
namespace blas  = multi::blas;

using complex = std::complex<double>;

BOOST_AUTO_TEST_CASE(multi_blas_axpy_real) {
	multi::array<double, 2> arr = {
		{1.0,  2.0,  3.0,  4.0},
		{5.0,  6.0,  7.0,  8.0},
		{9.0, 10.0, 11.0, 12.0},
	};
	auto const                    AC = arr;
	multi::array<double, 1> const b  = arr[2];  // NOLINT(readability-identifier-length) BLAS naming

	blas::axpy(2.0, b, arr[1]);  // daxpy
	BOOST_REQUIRE( arr[1][2] == 2.0*b[2] + AC[1][2] );
}

BOOST_AUTO_TEST_CASE(multi_blas_axpy_double) {
	multi::array<double, 2> const const_arr = {
		{1.0,  2.0,  3.0,  4.0},
		{5.0,  6.0,  7.0,  8.0},
		{9.0, 10.0, 11.0, 12.0},
	};
	multi::array<double, 2>       arr = const_arr;
	multi::array<double, 1> const b   = const_arr[2];  // NOLINT(readability-identifier-length) conventional name in BLAS

	blas::axpy(2.0, b, arr[1]);  // A[1] = 2*b + A[1], A[1]+= a*A[1]
	BOOST_REQUIRE( arr[1][2] == 2.0*b[2] + const_arr[1][2] );

	auto const I = complex{0, 1};  // NOLINT(readability-identifier-length) imaginary unit

	multi::array<complex, 1> AC = {1.0 + 2.0 * I, 3.0 + 4.0 * I, 4.0 - 8.0 * I};
	multi::array<complex, 1> BC(extensions(AC), complex{0.0, 0.0});

	blas::axpy(+1.0, blas::real(AC), blas::real(BC));
	blas::axpy(-1.0, blas::imag(AC), blas::imag(BC));

	//  BOOST_REQUIRE( BC[2] == std::conj(AC[2]) );
	BOOST_REQUIRE( BC[2] == conj(AC[2]) );
}

BOOST_AUTO_TEST_CASE(multi_blas_axpy_complex) {
	multi::array<complex, 2> arr = {
		{{1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {4.0, 0.0}},
		{{5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},  {8.0, 0.0}},
		{{9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}},
	};
	auto const const_arr = arr;

	multi::array<complex, 1> const x = arr[2];  // NOLINT(readability-identifier-length) BLAS naming
	blas::axpy(complex{2.0, 0.0}, x, arr[1]);  // zaxpy (2. is promoted to 2+I*0 internally and automatically)
	BOOST_REQUIRE( arr[1][2] == 2.0*x[2] + const_arr[1][2] );
}

BOOST_AUTO_TEST_CASE(multi_blas_axpy_complex_as_operator_plus_equal) {
	using complex = std::complex<double>;

	multi::array<complex, 2> arr = {
		{{1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {4.0, 0.0}},
		{{5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},  {8.0, 0.0}},
		{{9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}},
	};
	auto const                     carr = arr;
	multi::array<complex, 1> const y    = arr[2];  // NOLINT(readability-identifier-length) BLAS naming
	arr[1] += blas::axpy(2.0, y);  // zaxpy (2. is promoted to 2+I*0 internally and automatically)
	BOOST_REQUIRE( arr[1][2] == 2.0*y[2] + carr[1][2] );
}

BOOST_AUTO_TEST_CASE(multi_blas_axpy_complex_as_operator_minus_equal) {
	multi::array<complex, 2> arr = {
		{{1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {4.0, 0.0}},
		{{5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},  {8.0, 0.0}},
		{{9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}},
	};
	auto const                     AC = arr;
	multi::array<complex, 1> const x  = arr[2];  // NOLINT(readability-identifier-length) BLAS naming
	arr[1] -= blas::axpy(2.0, x);  // zaxpy (2. is promoted to 2+I*0 internally and automatically)
	BOOST_REQUIRE( arr[1][2] == -2.0*x[2] + AC[1][2] );
}

BOOST_AUTO_TEST_CASE(multi_blas_axpy_complex_context) {
	multi::array<complex, 2> arr = {
		{{1.0, 0.0},  {2.0, 0.0},  {3.0, 0.0},  {4.0, 0.0}},
		{{5.0, 0.0},  {6.0, 0.0},  {7.0, 0.0},  {8.0, 0.0}},
		{{9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}},
	};
	auto const                     arr_copy = arr;
	multi::array<complex, 1> const arr2     = arr[2];
	blas::axpy(blas::context{}, complex{2.0, 0.0}, arr2, arr[1]);  // zaxpy (2. is promoted to 2+I*0 internally and automatically)
	BOOST_REQUIRE( arr[1][2] == 2.0*arr2[2] + arr_copy[1][2] );
}

BOOST_AUTO_TEST_CASE(multi_blas_axpy_operator_minus) {
	// NOLINTNEXTLINE(readability-identifier-length) BLAS naming
	multi::array<complex, 1> x = {
		{10.0, 0.0},
		{11.0, 0.0},
		{12.0, 0.0},
		{13.0, 0.0},
	};
	multi::array<complex, 1> const y = x;  // NOLINT(readability-identifier-length) BLAS naming

	using blas::operators::operator-;

	BOOST_REQUIRE( (x - y)[0] == 0.0 );
	BOOST_REQUIRE( (y - x)[0] == 0.0 );

	using blas::operators::operator+;

	BOOST_REQUIRE( (x - (y+y))[0] == -x[0] );
	BOOST_REQUIRE( ((x+x) - y)[0] == +x[0] );

	multi::array<complex, 2> arr = {
		{{1.0, 0.0}, {2.0, 0.0}},
		{{3.0, 0.0}, {4.0, 0.0}},
	};
	multi::array<complex, 1> const arr2 = {
		{1.0, 0.0},
		{2.0, 0.0},
	};
	BOOST_REQUIRE( (arr[0] - arr2)[0] == 0.0 );
	BOOST_REQUIRE( (arr[0] - arr2)[1] == 0.0 );

	multi::array<complex, 1> X = {  /* NOLINT(readability-identifier-length) BLAS naming */
		{10.0, 0.0},
		{11.0, 0.0},
		{12.0, 0.0},
		{13.0, 0.0},
	};
	multi::array<complex, 1> const Y = {  /* NOLINT(readability-identifier-length) BLAS naming*/
		{10.0, 0.0},
		{11.0, 0.0},
		{12.0, 0.0},
		{13.0, 0.0},
	};

	using blas::operators::operator-=;
	X -= Y;
	BOOST_REQUIRE( X[0] == 0.0 );
}
