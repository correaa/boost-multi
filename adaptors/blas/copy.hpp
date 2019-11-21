#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_COPY $0.cpp -o $0x `pkg-config --cflags --libs blas`&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_COPY_HPP
#define MULTI_ADAPTORS_BLAS_COPY_HPP

#include "../blas/core.hpp"

namespace boost{
namespace multi{
namespace blas{

using multi::blas::core::copy;

template<class It, typename Size, class OutIt>
auto copy_n(It first, Size n, OutIt d_first)
->decltype(copy(n, base(first), stride(first), base(d_first), stride(d_first)), d_first + n){
	return copy(n, base(first), stride(first), base(d_first), stride(d_first)), d_first + n;}

//template<class It1, class OutIt>
//auto copy(It1 f, It1 l, OutIt d)
//->decltype(blas::copy_n(f, std::distance(f, l), d)){assert(stride(f)==stride(l));
//	return blas::copy_n(f, std::distance(f, l), d);
//}

template<class X1D, class Y1D>
Y1D&& copy(X1D const& x, Y1D&& y){assert(size(x)==size(y)); assert(offset(x)==0 and offset(y)==0);
	copy(size(x), base(x), stride(x), base(y), stride(y));
//	auto e = blas::copy(begin(x), end(x), begin(y)); (void)e; assert(e==end(y));
	return std::forward<Y1D>(y);
}

template<class X1D, class Ret = typename X1D::decay_type> // TODO multi::array_traits<X1D>::decay_type
Ret copy(X1D const& x){
	assert( not offset(x) );
	return copy(x, Ret(size(x), get_allocator(x)));
}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_COPY

#include "../../array.hpp"
#include "../../utility.hpp"

#include<complex>
#include<cassert>

using std::cout;
namespace multi = boost::multi;

int main(){
	{
		multi::array<double, 1> const A = {1., 2., 3., 4.};
		multi::array<double, 1> B = {5., 6., 7., 8.};
		using multi::blas::copy;
		copy(A, B);
		assert( B == A );
	}
	{
		using complex = std::complex<double>;
		multi::array<complex, 1> const A = {1., 2., 3., 4.};
		multi::array<complex, 1> B = {5., 6., 7., 8.};
		using multi::blas::copy;
		copy(A, B);
		assert( B == A );		
	}
	{
		multi::array<double, 2> const A = {
			{1., 2., 3.},
			{4., 5., 6.},
			{7., 8., 9.}
		};
		multi::array<double, 1> B(3);
		using multi::blas::copy;
		copy(rotated(A)[0], B);
		assert( B == rotated(A)[0] );
	}
}

#endif
#endif

