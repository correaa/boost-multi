#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc -x cu --expt-relaxed-constexpr`#c++ -Wall -Wextra -Wpedantic` -D_TEST_MULTI_ADAPTORS_BLAS_OPERATIONS $0.cpp -o $0x `pkg-config --libs blas` -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_OPERATIONS_HPP
#define MULTI_ADAPTORS_BLAS_OPERATIONS_HPP

#include    "../blas/core.hpp"
#include    "../blas/asum.hpp"
#include    "../blas/numeric.hpp"

#include "../../array_ref.hpp"

//#include<experimental/functional> // std::identity

namespace boost{
namespace multi{namespace blas{

enum class trans : char{N='N', T='T', C='C'};

enum class real_operation : char{
	transposition = static_cast<char>(trans::N),
	identity      = static_cast<char>(trans::T),
};

real_operation transpose(real_operation op){
	switch(op){
		case real_operation::transposition: return real_operation::identity;
		case real_operation::identity: return real_operation::transposition;
	} __builtin_unreachable();
}

enum class complex_operation : char{
	hermitian = static_cast<char>(trans::N),
	identity  = static_cast<char>(trans::C),
};
complex_operation hermitize(complex_operation op){
	switch(op){
		case complex_operation::hermitian: return complex_operation::identity;
		case complex_operation::identity: return complex_operation::hermitian;
	} __builtin_unreachable();
}

class operation{
	enum class impl_t : char{
		identity,// = static_cast<char>(trans::N), 
		transposition,// = static_cast<char>(real_operation::transposition), 
		hermitian// = static_cast<char>(complex_operation::hermitian)
	};
	impl_t impl_;
public:
#if __cplusplus > 201900L
	operation(std::identity<>) : impl_{impl_t::identity}{}
#endif
	operation(complex_operation cop) : impl_{[=]{switch(cop){
		case complex_operation::identity  : return impl_t::identity;
		case complex_operation::hermitian : return impl_t::hermitian;
	} __builtin_unreachable();}()}{}
	operation(real_operation rop) : impl_{[=]{switch(rop){
		case real_operation::identity      : return impl_t::identity;
		case real_operation::transposition : return impl_t::transposition;
	} __builtin_unreachable();}()}{}
	constexpr operation(impl_t impl) : impl_{impl}{}
	constexpr operator complex_operation() const{switch(impl_){
		case impl_t::identity      : return complex_operation::identity; 
		case impl_t::transposition : assert(0);
		case impl_t::hermitian     : return complex_operation::hermitian;
	} __builtin_unreachable();}
	constexpr operator real_operation() const{switch(impl_){
		case impl_t::identity      : return real_operation::identity;
		case impl_t::transposition : return real_operation::transposition;
		case impl_t::hermitian     : assert(0); // default:return{};
	} __builtin_unreachable();}
	constexpr operator char() const{return static_cast<char>(impl_);}
	friend bool operator==(operation const& o1, operation const& o2){return o1.impl_==o2.impl_;}
	friend bool operator==(complex_operation const& o1, operation const& o2){return operation(o1)==o2;}
	friend bool operator==(operation const& o1, complex_operation const& o2){return o1==operation(o2);}
	friend bool operator==(real_operation const& o1, operation const& o2){return operation(o1)==o2;}
	friend bool operator==(operation const& o1, real_operation const& o2){return o1==operation(o2);}
	static operation const identity; //= impl_t::identity;
	static operation const hermitian; //= impl_t::hermitian;
	static operation const transposition; //= impl_t::transposition;
};

/*inline*/ operation const operation::identity{operation::impl_t::identity};
/*inline*/ operation const operation::hermitian{operation::impl_t::hermitian};
/*inline*/ operation const operation::transposition{operation::impl_t::transposition};

//operation const& identity = operation::identity;
//operation const& hermitian = operation::hermitian;
//operation const& transposition = operation::transposition;

template<class M> decltype(auto) transposed(M const& m){return rotated(m);}
//template<class M> decltype(auto) transposed(M&       m){return rotated(m);}

template<class T, typename = decltype(std::declval<typename T::element>().imag())>
std::true_type is_complex_array_aux(T const&);
std::false_type is_complex_array_aux(...);

template <typename T> struct is_complex_array: decltype(is_complex_array_aux(std::declval<T const&>())){};

template<class ComplexPtr> std::true_type is_conjugated_aux(multi::blas::detail::conjugater<ComplexPtr> const&);
template<class T> std::false_type is_conjugated_aux(T const&);

template<class A>
struct is_conjugated_t : decltype(is_conjugated_aux(typename std::decay_t<A>::element_ptr{})){};

template<class A> constexpr bool is_conjugated(A const&){return is_conjugated_t<A>{};}

template<class A> constexpr bool is_not_conjugated(A const& a){return is_complex_array<A>{} and not is_conjugated(a);}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element, class C=detail::conjugater<typename D::element_ptr>, typename = std::enable_if_t<not is_conjugated_t<std::decay_t<A>>()> >
decltype(auto) conjugated(A&& a, void* = 0){
	return multi::static_array_cast<E, C>(std::forward<A>(a));
}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element, class C=typename D::element_ptr::underlying_type, typename = std::enable_if_t<is_conjugated_t<std::decay_t<A>>{}> >
decltype(auto) conjugated(A&& a){
	return multi::static_array_cast<E, C>(std::forward<A>(a));
}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element>
decltype(auto) conjugated_transposed(A&& a){
	return transposed(conjugated(a));
}

template<class ComplexPtr> std::true_type is_hermitized_aux(multi::blas::detail::conjugater<ComplexPtr> const&);
template<class T> std::false_type is_hermitized_aux(T const&);

template<class A>
struct is_hermitized : std::decay_t<decltype(is_hermitized_aux(typename std::decay_t<A>::element_ptr{}))>{};

template<class A> decltype(auto) identity(A&& a){return std::forward<A>(a);}


template<class A>
decltype(auto) hermitized(A&& a, std::true_type){
	return conjugated_transposed(std::forward<A>(a));
}

template<class A>
decltype(auto) hermitized(A&& a, std::false_type){
	return transposed(std::forward<A>(a));
}

template<class A>
decltype(auto) hermitized(A&& a){
#if __cpp_if_constexpr>=201606
	if constexpr(is_complex_array<std::decay_t<A>>{}){
		return conjugated_transposed(std::forward<A>(a));
	}else{
		return transposed(std::forward<A>(a));
	}
#else
	return hermitized(std::forward<A>(a), is_complex_array<std::decay_t<A>>{});
#endif
}

template<class A>
decltype(auto) transposed(A&& a){return rotated(std::forward<A>(a));}

template<class M> decltype(auto) N(M&& m){return m;}
template<class M> decltype(auto) T(M&& m){return transposed(m);}
template<class M> decltype(auto) C(M&& m){return conjugated_transposed(m);}

}}

namespace multi{
	using blas::N;
	using blas::T;
	using blas::C;
}

}

#if _TEST_MULTI_ADAPTORS_BLAS_OPERATIONS

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

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

template<class T> void what();

BOOST_AUTO_TEST_CASE(m){
	multi::array<complex, 2> const A = {
		{1. - 3.*I, 6.  + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};
	using multi::blas::hermitized;
	assert( hermitized(A)[0][1] == conj(A[1][0]) );
	static_assert( multi::blas::is_conjugated_t<decltype(hermitized(A))>{} , "!" );


	static_assert( not multi::blas::is_conjugated_t<std::decay_t<decltype( conjugated(hermitized(A)) )>>{}, "!");
	static_assert( not multi::blas::is_hermitized<std::decay_t<decltype( conjugated(hermitized(A)) )>>{}, "!");


}

BOOST_AUTO_TEST_CASE(is_complex_array_test){
	static_assert(multi::blas::is_complex_array<multi::array<std::complex<double>, 2>>{}, "!");
}

#if 0
BOOST_AUTO_TEST_CASE(multi_adaptors_blas_operations_enums){
	BOOST_REQUIRE( multi::blas::operation::identity == multi::blas::real_operation::identity );
	BOOST_REQUIRE( multi::blas::operation::transposition == multi::blas::real_operation::transposition );
	BOOST_REQUIRE( multi::blas::operation::hermitian == multi::blas::complex_operation::hermitian );
	BOOST_REQUIRE( multi::blas::operation::identity == multi::blas::complex_operation::identity );

	BOOST_REQUIRE( multi::blas::operation{multi::blas::real_operation::identity} == multi::blas::real_operation::identity );
	BOOST_REQUIRE( multi::blas::operation{multi::blas::real_operation::transposition} == multi::blas::real_operation::transposition );
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_operations){

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

	auto&& Aconjdconjd = multi::blas::conjugated(Aconjd);
	assert( Aconjdconjd[1][2] == A[1][2] );
	assert( &Aconjdconjd[1][2] == &A[1][2] );

	auto&& Atranspd = multi::blas::transposed(A);
	assert( Atranspd[1][2] == A[2][1] );
	multi::array<complex, 2> Atransp = multi::blas::transposed(A);
	assert( Atransp[1][2] == A[2][1] );
	assert( Atransp == Atranspd );

	auto&& Aconjdtranspd = multi::blas::conjugated_transposed(A); (void)Aconjdtranspd;
	assert( Aconjdtranspd[1][2] == conj(A[2][1]) );
	auto Aconjtransp = multi::blas::conjugated_transposed(A).decay();
	
	assert( Aconjtransp[1][2] == conj(A[2][1]) );
	assert( Aconjdtranspd == Aconjtransp );

	
{
	multi::array<complex, 2> const A = {
		{1. - 3.*I, 6.  + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};
	using multi::blas::hermitized;
	assert( hermitized(A)[0][1] == conj(A[1][0]) );
//	[]{}(hermitized(A));
	static_assert( multi::blas::is_conjugated<decltype(hermitized(A))>{} , "!");

	using multi::blas::conjugated;
//	[]{}(conjugated(conjugated(A)));

	using multi::blas::hermitized;
	[]{}(hermitized(hermitized(A)));

//	static_assert( not multi::blas::is_conjugated<decltype(hermitized(hermitized(A)))>{} , "!");

//	[]{}(hermitized(hermitized(A)));
//	[]{}(conjugated(conjugated(A)));

	static_assert( multi::blas::is_complex_array<std::decay_t<decltype(A)>>{} , "!");
//	auto&& AH = multi::blas::hermitized(A);
//	auto c = AH[0][0].imag();
//	static_assert( multi::blas::is_complex_array<std::decay_t<decltype(AH)>>{} , "!");

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
	
}
#endif
#endif
#endif

