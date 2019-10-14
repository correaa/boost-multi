#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_OPERATIONS $0.cpp -o $0x `pkg-config --cflags --libs blas` &&$0x&&rm $0x $0.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_BLAS_OPERATIONS_HPP
#define MULTI_ADAPTORS_BLAS_OPERATIONS_HPP

#include "../blas/core.hpp"
#include "../../array_ref.hpp"

namespace boost{
namespace multi{
namespace blas{

template<class M> decltype(auto) transposed(M const& m){return rotated(m);}

template<class ElementPtr = void*>
struct conj_proxy_impl : 
	std::iterator_traits<ElementPtr>
{
	ElementPtr p_;
	using value_type = decltype(conj(std::declval<typename std::iterator_traits<ElementPtr>::value_type>()));
	using reference = value_type;
	explicit conj_proxy_impl(ElementPtr p) : p_{p}{}
	explicit operator ElementPtr() const{return p_;}
	decltype(auto) operator*() const{return conj(*p_);}
	auto operator+(difference_type d) const{return conj_proxy_impl{p_ + d};}
	auto operator++() const{return conj_proxy_impl{++p_};}
	decltype(auto) operator+=(difference_type d){p_+=d; return *this;}
	bool operator==(conj_proxy_impl const& other) const{return p_ == other.p_;}
	bool operator!=(conj_proxy_impl const& other) const{return p_ != other.p_;}
	difference_type operator-(conj_proxy_impl const& other) const{return p_ - other.p_;}
	ElementPtr const& underlying() const{return p_;}
};

template<class T, typename = decltype(conj(*std::declval<T>()))>
std::true_type conj_detect(T t);
std::false_type conj_detect(...);

template<class T = void*> struct conj_proxy_aux{using type = std::conditional_t<decltype(conj_detect(std::declval<T const>())){}, conj_proxy_impl<T>, T>;};
template<class T> struct conj_proxy_aux<conj_proxy_impl<T>>{using type = T;};
template<> struct conj_proxy_aux<void*>{using type = conj_proxy_impl<void*>;};

template<class... Ts> using conj_proxy_t = typename conj_proxy_aux<Ts...>::type;

template<class M> decltype(auto) conjugated(M const& m){
//	using multi::static_array_cast;
	return multi::static_array_cast<typename M::element, conj_proxy_t<typename M::element_ptr> >(m);
}

template<class M> 
auto conjugated_transposed(M const& m){return conjugated(transposed(m));}

template<class M> decltype(auto) N(M const& m){return m;}
template<class M> auto T(M const& m){return transposed(m);}
template<class M> auto C(M const& m){return conjugated_transposed(m);}
template<class M> auto H(M const& m){return C(m);}

template<class M> auto TH(M const& m){return conjugated(m);}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_OPERATIONS

#include "../../array.hpp"
#include "../../utility.hpp"
#include "../blas/nrm2.hpp"

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

using std::cout;
namespace multi = boost::multi;

int main(){
	auto const I = std::complex<double>(0., 1.);
	multi::array<std::complex<double>, 2> A = {
		{1. + 8.*I,  2.-I,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	
	multi::array<std::complex<double>, 2> Aconj0 = multi::blas::conjugated(A);
//	assert( Aconj0[0][1] == std::conj(A[0][1]) );

	
	auto const& Aconj = multi::blas::conjugated(A);
	assert( Aconj[0][1] == std::conj(A[0][1]) );
	auto const& AC1 = multi::blas::conjugated(multi::blas::transposed(A));
	assert( AC1[1][0] == std::conj(A[0][1]) );
	
	auto const& AC2 = multi::blas::C(A);
	assert( AC2[2][1] == AC1[2][1] );
	{
		using multi::blas::C;
		assert( C(C(A))[1][3] == A[1][3] );
	}
	
	multi::array<double, 2> a = {{1.,2.},{3.,4.}};
//	assert( rotated(multi::blas::C(a)) == a );
}

#endif
#endif

