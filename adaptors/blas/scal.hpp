#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -D_TEST_MULTI_ADAPTORS_BLAS_SCAL $0.cpp -o $0x `pkg-config --libs blas` -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_SCAL_HPP
#define MULTI_ADAPTORS_BLAS_SCAL_HPP

#include "../blas/core.hpp"
#include "../../config/nodiscard_.hpp"

#include<cassert>

namespace boost{
namespace multi{
namespace blas{

using blas::core::scal;

template<typename T, class It, typename Size>
auto scal_n(T a, It first, Size count)
->decltype(scal(count, a, base(first), stride(first)), first + count){
	return scal(count, a, base(first), stride(first)), first + count;}

template<typename T, class It>
auto scal(T a, It f, It l)
->decltype(scal_n(a, f, std::distance(f, l))){assert(stride(f) == stride(l));
	return scal_n(a, f, std::distance(f, l));}

template<typename T, class X1D>
auto scal(T a, X1D&& m)
->decltype(scal(a, begin(m), end(m)), std::forward<X1D>(m)){
	return scal(a, begin(m), end(m)), std::forward<X1D>(m);}

template<typename T, class X1D> 
NODISCARD("when second argument is const")
auto scal(T a, X1D const& m)->std::decay_t<decltype(scal(a, m.decay()))>{
	return scal(a, m.decay());
}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_SCAL

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS scal"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../array.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_blas_scal_real){
	{
		multi::array<double, 2> A = {
			{1.,  2.,  3.,  4.},
			{5.,  6.,  7.,  8.},
			{9., 10., 11., 12.}
		};

		using multi::blas::scal;
		auto S = scal(2., rotated(A)[1]);

		BOOST_REQUIRE( A[2][1] == 20 );
		BOOST_REQUIRE( S[0] == 4 );
	}
	{
		multi::array<double, 2> const A = {
			{1.,  2.,  3.,  4.},
			{5.,  6.,  7.,  8.},
			{9., 10., 11., 12.}
		};
		using multi::blas::scal;
		auto rA1_scaled = scal(2., A[1]);
		BOOST_REQUIRE( size(rA1_scaled) == 4 );
		BOOST_REQUIRE( rA1_scaled[1] == 12 );
	}
}

#endif
#endif

