#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_NRM2 .DCATCH_CONFIG_MAIN $0.cpp -o$0x `pkg-config --cflags --libs blas`&&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_NRM2_HPP
#define MULTI_ADAPTORS_BLAS_NRM2_HPP

#include "../blas/core.hpp"

namespace boost{
namespace multi{
namespace blas{

template<class It, class Size>
auto nrm2_n(It first, Size n)
->decltype(nrm2(n, base(first), stride(first))){
	return nrm2(n, base(first), stride(first));}

template<class It>
auto nrm2(It f, It l)
->decltype(nrm2_n(f, std::distance(f, l))){assert(stride(f)==stride(l));
	return nrm2_n(f, std::distance(f, l));}

template<class X1D> 
auto nrm2(X1D const& x)
->decltype(nrm2(begin(x), end(x))){assert( not offset(x) );
	return nrm2(begin(x), end(x));}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_NRM2

#include<catch.hpp>

#include "../../array.hpp"

#include "../blas/dot.hpp"

namespace multi = boost::multi;

TEST_CASE("multi_adaptor_multi", "[nrm2]"){
	multi::array<double, 2> const cA = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	using multi::blas::nrm2;
	using multi::blas::dot;
	using std::sqrt;
	REQUIRE( nrm2(cA[1]) == sqrt(dot(cA[1], cA[1])) );
}

#endif
#endif

