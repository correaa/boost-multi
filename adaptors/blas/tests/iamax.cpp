#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x `pkg-config --libs blas` -lboost_unit_test_framework&&$0x&&rm $0x; exit
#endif
// © Alfredo A. Correa 2019

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../blas/iamax.hpp"

#include "../../../array.hpp"

#include<complex>
#include<cassert>

using std::cout;
namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_iamax){
	{
		using complex = std::complex<double>;
		using Z = complex; 
		Z const I{0.,1};
		multi::array<Z, 2> const A = {
			{1. + 2.*I,  2.,  3.,  4.},
			{5.,  6. + 3.*I,  7.,  8.},
			{9., 10., 11.+ 4.*I, 12.}
		};
		using multi::blas::iamax;
		BOOST_REQUIRE(iamax(A[1]) == std::distance(begin(A[1]), std::max_element(begin(A[1]), end(A[1]), [](auto&& a, auto&& b){using std::abs; return abs(real(a))+abs(imag(a)) < abs(real(b))+abs(imag(b));})));
	}
}

