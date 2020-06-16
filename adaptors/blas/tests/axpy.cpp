#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 -o $0x `pkg-config --libs blas` -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS axpy"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include       "../../blas.hpp"
#include "../../../array.hpp"

#include<complex>

template<class T> constexpr std::add_const_t<T>& as_const(T&& t) noexcept{return t;}

namespace multi = boost::multi;
namespace blas = multi::blas;

BOOST_AUTO_TEST_CASE(multi_blas_axpy){
	{
		multi::array<double, 2> A = {
			{1.,  2.,  3.,  4.},
			{5.,  6.,  7.,  8.},
			{9., 10., 11., 12.}
		};
		auto const AC = A;
		multi::array<double, 1> const B = A[2];
		auto A1cpy = blas::axpy(2., B, as_const(A[1])); // daxpy
		blas::axpy(2., B, A[1]); // daxpy
		BOOST_REQUIRE( A[1][2] == 2.*B[2] + AC[1][2] );
		assert( A1cpy == A[1] );
	}
	{
		using complex = std::complex<double>;
		multi::array<complex, 2> A = {
			{1.,  2.,  3.,  4.},
			{5.,  6.,  7.,  8.},
			{9., 10., 11., 12.}
		};
		auto const AC = A;
		multi::array<complex, 1> const B = A[2];
		blas::axpy(2., B, A[1]); // zaxpy (2. is promoted to 2+I*0 internally and automatically)
		BOOST_REQUIRE( A[1][2] == 2.*B[2] + AC[1][2] );
	}
}

