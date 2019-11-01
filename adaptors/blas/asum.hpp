#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_SCAL .DCATCH_CONFIG_MAIN $0.cpp -o$0x `pkg-config --cflags --libs blas` &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_DOT_HPP
#define MULTI_ADAPTORS_BLAS_DOT_HPP

#include "../blas/core.hpp"

namespace boost{
namespace multi{namespace blas{

template<class It, typename Size>
auto asum_n(It first, Size n)
->decltype(asum(n, base(first), stride(first))){
	return asum(n, base(first), stride(first));}

using std::distance;

template<class It>
auto asum(It f, It last)
->decltype(asum_n(f, distance(f, last))){assert(stride(f) == stride(last));
	return asum_n(f, distance(f, last));}

using std::begin; using std::end;

template<class X1D> 
auto asum(X1D const& x)
->decltype(asum(begin(x), end(x))){assert( not offset(x) );
	return asum(begin(x), end(x));}

}}
}

#if _TEST_MULTI_ADAPTORS_BLAS_SCAL

#include<catch.hpp>

#include "../../array.hpp"
//#include "../../utility.hpp"

#include<numeric> // accumulate

namespace multi = boost::multi;

using complex = std::complex<double>;
using multi::blas::asum;

TEST_CASE("multi::blas::asum double", "double"){
	multi::array<double, 2> const A = {
		{1.,  2.,  3.,  4.},
		{-5.,  6.,  -7.,  8.},
		{9., 10., 11., 12.}
	};
	REQUIRE(asum(A[1]) == std::accumulate(begin(A[1]), end(A[1]), 0., [](auto&& a, auto&& b){return a+abs(b);}));
}

TEST_CASE("multi::blas::asum complex", "complex"){
	constexpr complex I{0, 1};
	multi::array<complex, 2> const A = {
		{ 1. + 1.*I,  2.,  3.,  4.},
		{-5. + 3.*I,  6.,  -7.,  8.},
		{ 9. - 2.*I, 10., 11., 12.}
	};
	REQUIRE(asum(rotated(A)[0]) == 1.+1. + 5.+3. + 9.+2.);
}

TEST_CASE("multi::blas::asum double c-array", "double"){
	double A[3][4] = {
		{1.,  2.,  3.,  4.},
		{-5.,  6.,  -7.,  8.},
		{9., 10., 11., 12.}
	};
	using std::begin; using std::end;
	REQUIRE(asum(A[1]) == std::accumulate(begin(A[1]), end(A[1]), 0., [](auto&& a, auto&& b){return a+abs(b);}));
}


#endif
#endif

