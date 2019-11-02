#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_OPERATIONS $0.cpp -o $0x `pkg-config --cflags --libs blas` &&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_OPERATIONS_HPP
#define MULTI_ADAPTORS_BLAS_OPERATIONS_HPP

#include    "../blas/core.hpp"
#include    "../blas/asum.hpp"
#include    "../blas/numeric.hpp"

#include "../../array_ref.hpp"

namespace boost{
namespace multi{namespace blas{

enum class trans : char{N='N', T='T', C='C'};
enum class uplo  : char{L='L', U='U'};

enum class triangular : char{
	lower = static_cast<char>(uplo::U),
	upper = static_cast<char>(uplo::L)
};

triangular flip(triangular side){
	switch(side){
		case triangular::lower: return triangular::upper;
		case triangular::upper: return triangular::lower;
	} __builtin_unreachable();
}

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
	static operation const identity; //= impl_t::identity;
	static operation const hermitian; //= impl_t::hermitian;
	static operation const transposition; //= impl_t::transposition;
};

/*inline*/ operation const operation::identity{operation::impl_t::identity};
/*inline*/ operation const operation::hermitian{operation::impl_t::hermitian};
/*inline*/ operation const operation::transposition{operation::impl_t::transposition};

template<class M> decltype(auto) transposed(M const& m){return rotated(m);}
template<class M> decltype(auto) transposed(M&       m){return rotated(m);}

template<class A>
constexpr bool is_conjugated(){//A&& a){
	using ptr = typename std::decay_t<A>::element_ptr;//decltype(base(a));
	return
		   std::is_same<std::decay_t<ptr>, boost::multi::blas::detail::conjugater<const std::complex<double>*>>{}
		or std::is_same<std::decay_t<ptr>, boost::multi::blas::detail::conjugater<      std::complex<double>*>>{}
	;
}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element, class C=detail::conjugater<typename D::element_ptr>, typename = std::enable_if_t<not is_conjugated<A>()> >
decltype(auto) conjugated(A&& a){
	return multi::static_array_cast<E, C>(std::forward<A>(a));
}

template<class A, typename D=std::decay_t<A>, typename E=typename D::element, class C=typename D::element_ptr::underlying_type, typename = std::enable_if_t<is_conjugated<A>()> >
decltype(auto) conjugated(A&& a, void* = 0){
	return multi::static_array_cast<E, C>(std::forward<A>(a));
}


template<class A, typename D=std::decay_t<A>, typename E=typename D::element>
decltype(auto) conjugated_transposed(A&& a){
	return transposed(conjugated(a));
}

template<class A>
constexpr bool is_hermitized(){//A&& a){
	using ptr = typename std::decay_t<A>::element_ptr;//decltype(base(a));
	return
		   std::is_same<std::decay_t<ptr>, boost::multi::blas::detail::conjugater<const std::complex<double>*>>{}
		or std::is_same<std::decay_t<ptr>, boost::multi::blas::detail::conjugater<      std::complex<double>*>>{}
	;
}


template<class A>
decltype(auto) identity(A&& a){return std::forward<A>(a);}

template<class T, typename = decltype(std::declval<T const&>()[0][0].imag())>
std::true_type is_complex_array_aux(T const&);
std::false_type is_complex_array_aux(...);

template <typename T> struct is_complex_array: decltype(is_complex_array_aux(std::declval<T const&>())){};

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

template<class A2D>
triangular detect_triangular(A2D const& A, std::true_type){
	{
		for(auto i = size(A); i != 0; --i){
			auto const asum_up = blas::asum(begin(A[i-1])+i, end(A[i-1]));
			if(asum_up!=asum_up) return triangular::lower;
			else if(asum_up!=0.) return triangular::upper;

			auto const asum_lo = blas::asum(begin(rotated(A)[i-1])+i, end(rotated(A)[i-1]));
			if(asum_lo!=asum_lo) return triangular::upper;
			else if(asum_lo!=0.) return triangular::lower;
		}
	}
	return triangular::lower;
}

template<class A2D>
triangular detect_triangular(A2D const& A, std::false_type){
	return flip(detect_triangular(hermitized(A)));
}

template<class A2D>
triangular detect_triangular(A2D const& A){
#if __cpp_if_constexpr>=201606
	if constexpr(not is_hermitized<A2D>()){
		for(auto i = size(A); i != 0; --i){
			auto const asum_up = blas::asum(begin(A[i-1])+i, end(A[i-1]));
			if(asum_up!=asum_up) return triangular::lower;
			else if(asum_up!=0.) return triangular::upper;

			auto const asum_lo = blas::asum(begin(rotated(A)[i-1])+i, end(rotated(A)[i-1]));
			if(asum_lo!=asum_lo) return triangular::upper;
			else if(asum_lo!=0.) return triangular::lower;
		}
	}else{
		return flip(detect_triangular(hermitized(A)));
	}
#else
	detect_triangular(A, std::integral_constant<bool, not is_hermitized<A2D>()>{});
#endif
	return triangular::lower;
}


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

template<class T> void what();

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
	static_assert( multi::blas::is_complex_array<std::decay_t<decltype(A)>>{} , "!");
	auto&& AH = multi::blas::hermitized(A);
	auto c = AH[0][0].imag();
	static_assert( multi::blas::is_complex_array<std::decay_t<decltype(AH)>>{} , "!");

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

