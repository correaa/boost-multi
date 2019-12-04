#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_DOT $0.cpp -o $0x `pkg-config --cflags --libs blas` &&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_DOT_HPP
#define MULTI_ADAPTORS_BLAS_DOT_HPP

#include "../blas/core.hpp"
#include "../blas/numeric.hpp"
#include "../blas/operations.hpp"

namespace boost{
namespace multi{

//template<typename Size, class It2>
//auto dot_n(multi::array_iterator<std::complex<double>, 1, const std::complex<double>*> first1, Size n, It2 first2){
//	return multi::blas::dotu(n, base(first1), stride(first1), base(first2), stride(first2));
//}

//template<typename Size, class It2>
//auto dot_n(multi::array_iterator<std::complex<double>, 1, multi::blas::conjugater<const std::complex<double>*>, std::complex<double> > first1, Size n, It2 first2){
//	return multi::blas::dotc(n, base(first1).underlying(), stride(first1), base(first2), stride(first2));
//}

namespace blas{

//template<class R, class It1, class It2>
//auto dot(It1 first1, It1 last1, It2 first2){
//	assert( stride(first1) == stride(first2) );
//	auto d = std::distance(first1, last1);
//	return dot<R>(d, base(first1), stride(first1), base(first2), stride(first2));
//}

//template<class It1, typename Size, class It2>
//auto dot_n(It1 first1, Size n, It2 first2){
//	return dot(n, base(first1), stride(first1), base(first2), stride(first2));
//}

//template<class It1, class It2>
//auto dot(It1 first1, It1 last1, It2 first2){
//	assert( stride(first1)==stride(first2) );
//	using std::distance;
//	return dot_n(first1, distance(first1, last1), first2);
//}

template<class R, class X1D, class Y1D>
auto dot(X1D const& x, Y1D const& y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	return dot<R>(begin(x), end(x), begin(y));
}

template<class X1D, class Y1D, class R>
R&& dot(X1D const& x, Y1D const& y, R&& r){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	multi::blas::core::dot(size(x), base(x), stride(x), base(y), stride(y), &r);
	return std::forward<R>(r);
//	return dot(begin(x), end(x), begin(y));
}

template<class It1, class Size, class It2>
auto dotu(It1 first1, Size n, It2 first2){
	return dotu(n, base(first1), stride(first1), base(first2), stride(first2));
}

template<class It1, class It2>
auto dotu(It1 first1, It1 last1, It2 first2){
	assert( stride(first1) == stride(last1) );
	return dot(first1, std::distance(first1, last1), first2);
}

template<class X1D, class Y1D>
auto dotu(X1D const& x, Y1D const& y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	return dotu(begin(x), end(x), begin(y));
}

template<class It1, typename Size, class It2>
auto dotc_n(It1 first1, Size n, It2 first2){
	dotc(n, base(first1), stride(first1), base(first2), stride(first2));
	return first2 + n;
}

template<class It1, class It2>
auto dotc(It1 first1, It1 last1, It2 first2){
	assert( stride(first1) == stride(last1) );
	return dotc_n(first1, std::distance(first1, last1), first2);
}

template<class X1D, class Y1D>
auto dotc(X1D const& x, Y1D const& y){
	assert( size(x) == size(y) );
	assert( not offset(x) and not offset(y) );
	return dotc(begin(x), end(x), begin(y));
}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_DOT

#include "../../array.hpp"
#include "../../utility.hpp"

#include "../blas/nrm2.hpp"

#include<cassert>
#include<numeric> // inner_product

namespace multi = boost::multi;

int main(){
{
	multi::array<double, 2> const cA = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	
	using multi::blas::dot;
	{
		double d;
		dot(cA[1], cA[2], d);
		assert( d==std::inner_product(begin(cA[1]), begin(cA[2]), end(cA[1]), 0.) );
	}
	using multi::blas::nrm2;
	using std::sqrt;
	{
		double s;
		dot(cA[1], cA[1], s);
		assert( sqrt(s)==nrm2(cA[1]) );
	}
}
{
	using complex = std::complex<double>;
	constexpr complex I{0, 1};
	multi::array<complex, 2> const A = {
		{1. +    I,  2. + 3.*I,  3.+2.*I,  4.-9.*I},
		{5. + 2.*I,  6. + 6.*I,  7.+2.*I,  8.-3.*I},
		{9. + 1.*I, 10. + 9.*I, 11.+1.*I, 12.+2.*I}
	};
	using multi::blas::dot;
	using multi::blas::C;
	complex d;
	dot(C(A[1]), A[1], d);
	assert( d==std::inner_product(begin(C(A[1])), end(C(A[1])), begin(A[1]), complex{0}) );
	using std::conj;
	assert( d==std::inner_product(begin(A[1]), end(A[1]), begin(A[1]), complex{0}, std::plus<complex>{}, [](auto&& a, auto&& b){return conj(a)*b;}) );
}
}

#endif
#endif

