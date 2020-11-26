#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXXX $CXXFLAGS $0 -o $0x `pkg-config --libs blas` -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_AXPY_HPP
#define MULTI_ADAPTORS_BLAS_AXPY_HPP

#include "../../adaptors/blas/core.hpp"
#include "../../config/NODISCARD.hpp"
#include "../../array_ref.hpp"

namespace boost{
namespace multi{namespace blas{

using core::axpy;

template<class T, class It1, class Size, class OutIt>
auto axpy_n(T alpha, It1 first, Size n, OutIt d_first)
->decltype(axpy(n, alpha, base(first), stride(first), base(d_first), stride(d_first)), d_first + n){
	return axpy(n, alpha, base(first), stride(first), base(d_first), stride(d_first)), d_first + n;}

template<class T, class It1, class OutIt>
auto axpy(T alpha, It1 first, It1 last, OutIt d_first)
->decltype(axpy_n(alpha, first, last - first, d_first)){
	return axpy_n(alpha, first, last - first, d_first);}

using std::begin;
using std::end;

template<class X1D, class Y1D, typename = decltype( std::declval<Y1D&&>()[0] = 0. )>
auto axpy(typename X1D::element alpha, X1D const& x, Y1D&& y)
->decltype(axpy_n(alpha, x.begin(), x.size(), y.begin()), std::forward<Y1D>(y)){assert(size(x)==size(y)); // intel doesn't like ADL in deduced/sfinaed return types
	return axpy_n(alpha, begin(x), size(x), begin(y)), std::forward<Y1D>(y);
}

template<class X1D, class Y1D>
NODISCARD("because input is read-only")
auto axpy(typename X1D::element alpha, X1D const& x, Y1D const& y)
->std::decay_t<decltype(axpy(alpha, x, typename array_traits<Y1D>::decay_type{y}))>{
	return axpy(alpha, x, typename array_traits<Y1D>::decay_type{y});}

template<class X1D, class Y1D>
Y1D&& axpy(X1D const& x, Y1D&& y){return axpy(+1., x, std::forward<Y1D>(y));}

template<class T, class X1D, class Y1D>
NODISCARD("because input is read-only")
auto axpy(X1D const& x, Y1D const& y){
	return axpy(x, typename array_traits<Y1D>::decay_type{y});
}

namespace operators{

	template<class X1D, class Y1D> auto operator+=(X1D&& x, Y1D const& other) DECLRETURN(axpy(+1., other, std::forward<X1D>(x)))
	template<class X1D, class Y1D> auto operator-=(X1D&& x, Y1D const& other) DECLRETURN(axpy(-1., other, std::forward<X1D>(x)))

	template<class X1D, class Y1D> auto operator+(X1D const& x, Y1D const& y)->std::decay_t<decltype(x.decay())>{auto X=x.decay(); X+=y; return X;}
	template<class X1D, class Y1D> auto operator-(X1D const& x, Y1D const& y)->std::decay_t<decltype(x.decay())>{auto X=x.decay(); X-=y; return X;}

}


}}

}
#endif

