#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_OPERATIONS $0.cpp -o $0x `pkg-config --cflags --libs blas` &&$0x&&rm $0x $0.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_BLAS_OPERATIONS_HPP
#define MULTI_ADAPTORS_BLAS_OPERATIONS_HPP

#include "../blas/core.hpp"
#include "../../array_ref.hpp"
#include "../blas/numeric.hpp"

namespace boost{
namespace multi{namespace blas{

enum class uplo : char{L='L', U='U'};

template<class M> decltype(auto) transposed(M const& m){return rotated(m);}
template<class M> decltype(auto) transposed(M&       m){return rotated(m);}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element, class C=detail::conjugater<typename D::element_ptr>>
decltype(auto) conjugated(A&& a){
	return multi::static_array_cast<E, C>(std::forward<A>(a));
}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element>
decltype(auto) conjugated_transposed(A&& a){
	return transposed(conjugated(a));
}

template<class A>
decltype(auto) identity(A&& a){return std::forward<A>(a);}

template<class A>
decltype(auto) hermitized(A&& a){return conjugated_transposed(std::forward<A>(a));}

template<class A>
decltype(auto) transposed(A&& a){return rotated(std::forward<A>(a));}

#if 0
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
	std::allocator<value_type> default_allocator() const{return {};}
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
#endif

template<class M> decltype(auto) N(M&& m){return m;}
template<class M> decltype(auto) T(M&& m){return transposed(m);}
template<class M> decltype(auto) C(M&& m){return conjugated_transposed(m);}
//template<class M> auto H(M const& m){return C(m);}
//template<class M> auto TH(M const& m){return conjugated(m);}

}}

namespace multi{
	using blas::N;
	using blas::T;
	using blas::C;
}

}

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

template<class M> 
decltype(auto) print(M const& C){
	using boost::multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j) cout<< C[i][j] <<' ';
		cout<<std::endl;
	}
	return cout<<"---"<<std::endl;
}

namespace multi = boost::multi;
using complex = std::complex<double>;
auto const I = complex(0., 1.);

//template<class... T> void what(T&&...);

//boost::multi::array<std::complex<double>, 2l, std::allocator<std::complex<double> > >&, 
//boost::multi::array<std::complex<double>, 2l, std::allocator<std::complex<double>*> >&)

int main(){

	multi::array<complex, 2> const A = {
		{1. - 3.*I, 6.  + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};

	print(A);
	print(multi::blas::conjugated(A));

	auto&& Aconjd = multi::blas::conjugated(A);
	assert( Aconjd[1][2] == conj(A[1][2]) );
	multi::array<complex, 2> Aconj = multi::blas::conjugated(A);
	assert( Aconj[1][2] == conj(A[1][2]) );
	assert( Aconjd == Aconj );

	auto&& Atranspd = multi::blas::transposed(A);
	assert( Atranspd[1][2] == A[2][1] );
	multi::array<complex, 2> Atransp = multi::blas::transposed(A);
	assert( Atransp[1][2] == A[2][1] );
	assert( Atransp == Atranspd );

	auto&& Aconjdtranspd = multi::blas::conjugated_transposed(A); (void)Aconjdtranspd;
	assert( Aconjdtranspd[1][2] == conj(A[2][1]) );
	multi::array<complex, 2> Aconjtransp = multi::blas::conjugated_transposed(A);
	assert( Aconjtransp[1][2] == conj(A[2][1]) );
	assert( Aconjdtranspd == Aconjtransp );

{
	multi::array<complex, 2> const A = {
		{1. - 3.*I, 6.  + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};
//	auto&& Aconjd = multi::blas::conjugated(A);
//	assert( Aconjd[1][2] == conj(A[1][2]) );
//	multi::array<complex, 2> Aconj = multi::blas::conjugated(A);
//	assert( Aconj[1][2] == conj(A[1][2]) );
//	assert( Aconjd == Aconj );

	auto&& Atranspd = multi::blas::T(A);
	assert( Atranspd[1][2] == A[2][1] );
	multi::array<complex, 2> Atransp = multi::blas::transposed(A);
	assert( Atransp[1][2] == A[2][1] );
	assert( Atransp == Atranspd );

	auto&& Aconjdtranspd = multi::blas::C(A); (void)Aconjdtranspd;
	assert( Aconjdtranspd[1][2] == conj(A[2][1]) );
	multi::array<complex, 2> Aconjtransp = multi::blas::conjugated_transposed(A);
	assert( Aconjtransp[1][2] == conj(A[2][1]) );
	assert( Aconjdtranspd == Aconjtransp );

}
	

//	assert( A[1]
#if 0
	{
		multi::array<complex, 2> Aconj_copy = multi::blas::conjugated(A);
		assert( Aconj_copy[0][1] == std::conj(A[0][1]) );
	//	what(multi::blas::conjugated(A));

		auto Aconj_copy2 = multi::blas::conjugated(A).decay();
	//	what(Aconj_copy, Aconj_copy2);
//		static_assert(std::is_same<decltype(Aconj_copy), decltype(Aconj_copy2)>{}, "!");

	//	what(multi::pointer_traits<std::complex<double>>::default_allocator_type{});
	//	what(multi::pointer_traits<boost::multi::blas::conj_proxy_impl<std::complex<double>*>>::default_allocator_type{});
	}
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
#endif
}

#endif
#endif

