#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX -DADD_ $0 -o $0x `pkg-config --libs blas` -lboost_unit_test_framework \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_GEMV_HPP
#define MULTI_ADAPTORS_BLAS_GEMV_HPP

#include "../blas/core.hpp"
#include "./../../detail/../utility.hpp"
#include "./../../detail/../array_ref.hpp"

namespace boost{
namespace multi{namespace blas{

template<class Trans, class T, class ItA, class Size, class Itx, class Ity>
Ity gemv_n(Trans IN, T a, ItA A_first, Size n, Itx x_first, T beta, Ity y_first){
	assert( IN == 'N' or IN == 'T' or IN == 'C' );
//	assert( A_first->stride() == 1 );
	gemv(IN, size(*A_first), n, a, base(A_first), stride(A_first), base(x_first), stride(x_first), beta, base(y_first), stride(y_first));
	switch(IN){
		case 'N' : return y_first + size(*A_first);
		case 'T' : return y_first + n;
		case 'C' : return y_first + n;
	}
	return y_first;
}

template<class Trans, class T, class ItA, class Itx, class Ity>
Ity gemv(Trans IN, T a, ItA A_first, ItA A_last, Itx x_first, T beta, Ity y_first){
//	assert( stride(A_first) == stride(A_last) and A_first->size()==A_last->size() );
	return gemv_n(IN, a, A_first, std::distance(A_first, A_last), x_first, beta, y_first);
}

template<class Trans, class T, class A2D, class X1D, class Y1D>
Y1D&& gemv(Trans IN, T a, A2D const& A, X1D const& x, T beta, Y1D&& y){
	assert( not(IN == 'T') or size(A) == size(y) );
	assert( not(IN == 'N') or size(A) == size(x) );
	assert( not(IN == 'C') or size(A) == size(x) );
	using std::begin; using std::end;
	auto e = gemv(IN, a, begin(A), end(A), begin(x), beta, begin(y)); (void)e;
	assert( e == end(y) );
	return std::forward<Y1D>(y);
} //y := alpha*A*x + beta*y,

template<class T, class A2D, class X1D, class Y1D>
Y1D&& gemv(T a, A2D const& A, X1D const& x, T beta, Y1D&& y){
	return stride(A)==1?gemv('N', a, rotated(A), x, beta, y):gemv('T', a, A, x, beta, y);
}

template<class A2D, class X1D, class Y1D>
Y1D&& gemv(A2D const& A, X1D const& x, Y1D&& y){
	return gemv(1., A, x, 0., std::forward<Y1D>(y));
}

}}}

#if not __INCLUDE_LEVEL__ // _TEST_MULTI_ADAPTORS_BLAS_GEMV

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi blas gemv"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../array.hpp"
#include "../../utility.hpp"

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

using std::cout;
namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_blas_gemv){
	namespace blas = multi::blas;

	using std::abs;
	{
		multi::array<double, 2> const M = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		multi::array<double, 1> const X = {1.1,2.1,3.1, 4.1};
		multi::array<double, 1> Y = {4.,5.,6.};
		double a = 1.1, b = 1.2;
		blas::gemv('T', a, M, X, b, Y); // y = a*M*x + b*y

		multi::array<double, 1> const Y3 = {214.02, 106.43, 188.37}; // = 1.1 {{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}.{1.1, 2.1, 3.1, 4.1} + 1.2 {4., 5., 6.}
//		cout << abs(Y[1] - Y3[1]) << std::endl;
		assert( abs(Y[1] - Y3[1]) < 2e-14 );
	}
#if 0
	{
		double const M[3][4] = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		double const X[4] = {1.1,2.1,3.1, 4.1};
		double Y[3] = {4.,5.,6.};
		double const a = 1.1;
		double const b = 1.2;
		gemv('T', a, M, X, b, Y); // y = a*M*x + b*y
		double Y3[3] = {214.02, 106.43, 188.37};
		assert( abs(Y[1] - Y3[1]) < 2e-14 );
	}

	{
		multi::array<double, 2> const M = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		multi::array<double, 1> const X = {1.1,2.1,3.1};
		multi::array<double, 1> Y = {4.,5.,6., 7.};
		double a = 1.8, b = 1.6;
		gemv('N', a, M, X, b, Y); // y = a*(M^T)*x + b*y, y^T = a*(x^T)*M + b*y^T
		multi::array<double, 1> const Y3 = {117.46, 182.6, 315.24, 61.06}; // =1.8 Transpose[{{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}].{1.1, 2.1, 3.1} + 1.6 {4., 5., 6., 7.}
		assert( abs(Y[2] - Y3[2]) < 1e-13 );
	}
	{
		multi::array<double, 2> const M = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		multi::array<double, 1> const X = {1.1,2.1,3.1, 4.1};
		multi::array<double, 1> Y = {4.,5.,6.};
		double a = 1.1, b = 1.2;
		gemv(a, M, X, b, Y); // y = a*M*x + b*y
		multi::array<double, 1> const Y3 = {214.02, 106.43, 188.37}; // = 1.1 {{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}.{1.1, 2.1, 3.1, 4.1} + 1.2 {4., 5., 6.}
		assert( std::abs(Y[1] - Y3[1]) < 2e-14 );
	}
	{
		double const M[3][4] = {
			{ 9., 24., 30., 9.},
			{ 4., 10., 12., 7.},
			{14., 16., 36., 1.}
		};
		assert( M[2][0] == 14. );
		double const X[4] = {1.1,2.1,3.1, 4.1};
		double Y[3] = {4.,5.,6.};
		double a = 1.1, b = 1.2;
		gemv(a, M, X, b, Y); // y = a*M*x + b*y
		double const Y3[3] = {214.02, 106.43, 188.37};
		assert( std::abs(Y[1] - Y3[1]) < 2e-14 );
	}
	{
		multi::array<double, 2> const M = {
			{ 9., 4., 14.},
			{24., 10., 16.},
			{30., 12., 36.},
			{9., 7., 1.}
		}; assert( M[0][2] == 14. );
		multi::array<double, 1> const X = {1.1,2.1,3.1, 4.1};
		multi::array<double, 1> Y = {4.,5.,6.};
		double a = 1.1, b = 1.2;
		gemv(a, rotated(M), X, b, Y); // y = a*M*x + b*y

		multi::array<double, 1> const Y3 = {214.02, 106.43, 188.37}; // = 1.1 {{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}.{1.1, 2.1, 3.1, 4.1} + 1.2 {4., 5., 6.}
		assert( abs(Y[1] - Y3[1]) < 2e-14 );
	}
#endif
}

#endif
#endif

