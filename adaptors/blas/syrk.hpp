#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc -std=c++14 `#-Wall -Wextra -Wpedantic` `#-Wfatal-errors` -D_TEST_MULTI_ADAPTORS_BLAS_SYRK .DCATCH_CONFIG_MAIN.o $0.cpp -o $0x \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_SYRK_HPP
#define MULTI_ADAPTORS_BLAS_SYRK_HPP

#include "../blas/core.hpp"
#include "../blas/copy.hpp"

#include "../blas/numeric.hpp"
#include "../blas/operations.hpp"

namespace boost{
namespace multi{namespace blas{

enum class real_operation : char{
	transposition = static_cast<char>(trans::N),
	identity      = static_cast<char>(trans::T),
};

enum class triangular : char{
	lower = static_cast<char>(uplo::U),
	upper = static_cast<char>(uplo::L),
};

real_operation transpose(real_operation op){
	switch(op){
		case real_operation::transposition: return real_operation::identity;
		case real_operation::identity : return real_operation::transposition;
		default: return {}; // for gcc and nvcc
	}
}

triangular flip(triangular side){
	switch(side){
		case triangular::lower: return triangular::upper;
		case triangular::upper: return triangular::lower;
		default: return {}; // for gcc and nvcc
	}
}

enum class complex_operation : char{
	hermitian = static_cast<char>(trans::N),
	identity  = static_cast<char>(trans::C),
};

class operation{
	enum class impl_t : char{
		identity,// = static_cast<char>(trans::N), 
		transposition,// = static_cast<char>(real_operation::transposition), 
		hermitian// = static_cast<char>(complex_operation::hermitian)
	};
	impl_t impl_;
public:
	operation(complex_operation cop) : impl_{[&]{switch(cop){
		case complex_operation::identity: return impl_t::identity;
		case complex_operation::hermitian: return impl_t::hermitian;
		default: return impl_t{}; // for gcc and nvcc
	}}()}{}
	operation(real_operation rop) : impl_{[&]{switch(rop){
		case real_operation::identity: return impl_t::identity;
		case real_operation::transposition: return impl_t::transposition;
	}}()}{}
	constexpr operation(impl_t impl) : impl_{impl}{}
	constexpr operator complex_operation() const{switch(impl_){
		case impl_t::identity: return complex_operation::identity; 
		case impl_t::transposition: assert(0);
		case impl_t::hermitian: return complex_operation::hermitian;
		default: return {}; // for gcc and nvcc
	}}
	constexpr operator real_operation() const{switch(impl_){
			case impl_t::identity: return real_operation::identity;
			case impl_t::transposition: return real_operation::transposition;
			case impl_t::hermitian: assert(0);
	}}
	static operation const identity; //= impl_t::identity;
	static operation const hermitian; //= impl_t::hermitian;
	static operation const transposition; //= impl_t::transposition; 
};

/*inline*/ operation const operation::identity{operation::impl_t::identity};
/*inline*/ operation const operation::hermitian{operation::impl_t::hermitian};
/*inline*/ operation const operation::transposition{operation::impl_t::transposition};

template<class T, typename = decltype(imag(std::declval<T>()[0])[0])>
std::true_type is_complex_array_aux(T&&);
std::false_type is_complex_array_aux(...);

template <typename T> struct is_complex_array: decltype(is_complex_array_aux(std::declval<T>())){};

template<typename AA, typename BB, class A2D, class C2D>
C2D&& syrk(triangular c_side, real_operation a_op, AA alpha, A2D const& a, BB beta, C2D&& c){
	if(stride(c)!=1){
		assert( stride(a)!=1 ); // sources and destination are incompatible layout
		assert( size(c)==(a_op==real_operation::transposition?size(*begin(a)):size(a)) );
		syrk(
			static_cast<char>(c_side), static_cast<char>(a_op), size(c), 
			a_op==real_operation::transposition?size(a):size(*begin(a)), 
			alpha, base(a), stride(a), beta, base(c), stride(c)
		);
	}else syrk(flip(c_side), transpose(a_op), alpha, rotated(a), beta, rotated(c));
	return std::forward<C2D>(c);
}

template<typename AA, typename BB, class A2D, class C2D>
C2D&& syrk(triangular c_side, AA alpha, A2D const& a, BB beta, C2D&& c){
	if(stride(a)==1){
		if(stride(c)==1) syrk(flip(c_side), real_operation::transposition, alpha, rotated(a), beta, rotated(std::forward<C2D>(c)));
		else             syrk(c_side      , real_operation::transposition, alpha, rotated(a), beta,        (std::forward<C2D>(c)));
	}else{
		if(stride(c)==1) syrk(flip(c_side), real_operation::identity     , alpha,        (a), beta, rotated(std::forward<C2D>(c)) );
		else             syrk(c_side      , real_operation::identity     , alpha,        (a), beta,        (std::forward<C2D>(c)));
	}
	return std::forward<C2D>(c);
}

template<typename AA, class A2D, class C2D>
C2D&& syrk(triangular c_side, AA alpha, A2D const& a, C2D&& c){
	return syrk(c_side, alpha, a, 0., std::forward<C2D>(c));
}

template<typename AA, class A2D, class C2D>
C2D&& syrk(AA alpha, A2D const& a, C2D&& c){
	if(stride(c)==1) syrk(triangular::upper, alpha, a, rotated(std::forward<C2D>(c)));
	else             syrk(triangular::lower, alpha, a,         std::forward<C2D>(c) );
	for(typename std::decay_t<C2D>::difference_type i = 0; i != size(c); ++i)
		blas::copy(begin(rotated(c)[i])+i+1, end(rotated(c)[i]), begin(c[i])+i+1);
	return std::forward<C2D>(c);
}

template<typename AA, class A2D, class Ret = typename A2D::decay_type>
Ret syrk(AA alpha, A2D const& A){
	Ret ret({size(A), size(A)});
	return syrk(alpha, A, ret);
	return ret;
}

template<class A2D, class R = typename A2D::decay_type>
auto syrk(A2D const& A){return syrk(1., A);}

#if 0
template<class UL, class Op, typename ScalarA, typename ScalarB, class A2D, class C2D>
C2D&& syrk(UL uplo, Op op, ScalarA alpha, A2D const& A, ScalarB beta, C2D&& C){
	assert(stride(*begin(A))==1); assert(stride(*begin(C))==1);
	switch(op){
		case 'T': assert(size(A) == size(C));
			syrk(uplo, op, size(C), size(*begin(A)), alpha, base(A), stride(A), beta, base(C), stride(C)); break;
		case 'N': assert(size(*begin(A))==size(C));
			syrk(uplo, op, size(C), size(A), alpha, base(A), stride(A), beta, base(C), stride(C)); break;
		default: assert(0);
	}
	return std::forward<C2D>(C);
}
#endif

#if 0
template<class UpLo, class A2D, class C2D, typename ScalarA, typename ScalarB>
C2D&& syrk(UpLo ul, ScalarA alpha, A2D const& A, ScalarB beta, C2D&& C){
	if(stride(A)==1){
		if(stride(C)==1) syrk(ul==U?L:U, T, alpha, rotated(A), beta, rotated(std::forward<C2D>(C)));
		else             syrk(ul       , T, alpha, rotated(A), beta,        (std::forward<C2D>(C)));
	}else{
		if(stride(C)==1) syrk(ul==U?L:U, N, alpha,        (A), beta, rotated(std::forward<C2D>(C)) );
		else             syrk(ul       , N, alpha,        (A), beta,        (std::forward<C2D>(C)));
	}
	return std::forward<C2D>(C);
}

template<class UpLo, typename ScalarA, class A2D, class C2D>
C2D&& syrk(UpLo ul, ScalarA alpha, A2D const& A, C2D&& C){
	return syrk(ul, alpha, A, 0., std::forward<C2D>(C));
}

template<typename AA, class A2D, class C2D>
C2D&& syrk(AA a, A2D const& A, C2D&& C){
	if(stride(C)==1) syrk(L, a, A, rotated(std::forward<C2D>(C)));
	else             syrk(U, a, A,         std::forward<C2D>(C) );
	for(typename std::decay_t<C2D>::difference_type i = 0; i != size(C); ++i)
		blas::copy(begin(rotated(C)[i])+i+1, end(rotated(C)[i]), begin(C[i])+i+1);
	return std::forward<C2D>(C);
}

template<class A2D, class C2D>
C2D&& syrk(A2D const& A, C2D&& C){return syrk(1., A, std::forward<C2D>(C));}

#endif

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_SYRK

#include "../blas/gemm.hpp"

#include "../../array.hpp"
#include "../../utility.hpp"

#include <boost/timer/timer.hpp>

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

#include<catch.hpp>

using std::cout;
using std::cerr;

namespace multi = boost::multi;

template<class M> decltype(auto) print(M const& C){
	using boost::multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j)
			std::cout << C[i][j] << ' ';
		std::cout << std::endl;
	}
	return std::cout << std::endl;
}

TEST_CASE("multi::blas::syrk enums", "[report]"){
	REQUIRE( multi::blas::complex_operation::hermitian == multi::blas::operation::hermitian );
	REQUIRE( multi::blas::complex_operation::identity == multi::blas::operation::identity );
	multi::blas::operation op = multi::blas::complex_operation::identity;
}

TEST_CASE("multi::blas::syrk real", "[report]"){

	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::transposition, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[2][1] == 19. ); 
		REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::transposition, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[1][2] == 19. );
		REQUIRE( c[2][1] == 9999. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[1][0] == 34. ); 
		REQUIRE( c[0][1] == 9999. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::identity, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, a⊤a, `c` in lower triangular
		REQUIRE( c[0][1] == 34. ); 
		REQUIRE( c[1][0] == 9999. );
	}
	{
		multi::array<double, 2> const at = rotated(a);
		multi::array<double, 2> ct({2, 2}, 9999.);
		auto&& a_ = rotated(at);
		auto&& c_ = rotated(ct);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::identity, 1., a_, 0., c_); // c⊤=c=a⊤a=(a⊤a)⊤, a⊤a, `c` in lower triangular
		REQUIRE( c_[0][1] == 34. ); 
		REQUIRE( c_[1][0] == 9999. );
	}
}

TEST_CASE("multi::blas::syrk complex (real case)", "[report]"){
	using complex = std::complex<double>;
	multi::array<complex, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::transposition, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[2][1] == 19. );
		REQUIRE( c[1][2] == 9999. );
	}
}

TEST_CASE("multi::blas::syrk complex", "[report]"){
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::transposition, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[2][1] == complex(-3., -34.) );
		REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in lower triangular
		REQUIRE( c[1][0] == complex(18., -21.) );
		REQUIRE( c[0][1] == 9999. );
	}
	{
		multi::array<complex, 2> const at = rotated(a);
		multi::array<complex, 2> ct({2, 2}, 9999.);
		auto&& a_ = rotated(at);
		auto&& c_ = rotated(ct);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::identity, 1., a_, 0., c_); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in upper triangular
		REQUIRE( c_[0][1] == complex(18., -21.) ); 
		REQUIRE( c_[1][0] == 9999. );
	}
}

TEST_CASE("multi::blas::syrk automatic operation complex", "[report]"){
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., a, 0., c); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in lower triangular
		REQUIRE( c[1][0]==complex(18., -21.) );
		REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::transposed;
		syrk(triangular::lower, 1., transposed(a), 0., c); // c⊤=c=a⊤a=(aa⊤)⊤, `c` in lower triangular
		REQUIRE( c[2][1]==complex(-3.,-34.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::transposed;
		syrk(triangular::lower, 1., rotated(a), 0., c); // c⊤=c=a⊤a=(aa⊤)⊤, `c` in lower triangular
		REQUIRE( c[2][1]==complex(-3.,-34.) );
		REQUIRE( c[1][2]==9999. );
	}
}

TEST_CASE("multi::blas::syrk automatic operation real", "[report]"){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., a, 0., c); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in lower triangular
		REQUIRE( c[1][0] == 34. );
		REQUIRE( c[0][1] == 9999. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::upper, 1., a, 0., c); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in upper triangular
		REQUIRE( c[0][1] == 34. );
		REQUIRE( c[1][0] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., rotated(a), 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[2][1] == 19. );
		REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::transposed;
		using multi::blas::triangular;
		syrk(triangular::lower, 1., transposed(a), 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[2][1] == 19. );
		REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::transposed;
		using multi::blas::triangular;
		syrk(triangular::upper, 1., transposed(a), 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in upper triangular
		REQUIRE( c[1][2] == 19. );
		REQUIRE( c[2][1] == 9999. );
	}
	{
		multi::array<double, 2> const at = rotated(a);
		multi::array<double, 2> ct({2, 2}, 9999.);
		auto&& a_ = rotated(at);
		auto&& c_ = rotated(ct);
		using multi::blas::triangular;
		syrk(triangular::upper, 1., a_, 0., c_); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in upper triangular
		print(c_);
		REQUIRE( c_[0][1] == 34. ); 
		REQUIRE( c_[1][0] == 9999. );
	}
}

TEST_CASE("multi::blas::syrk automatic implicit zero", "[report]"){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., a, c); // c⊤=c=aa⊤=(aa⊤)⊤, `c` in lower triangular
		REQUIRE( c[1][0] == 34. );
		REQUIRE( c[0][1] == 9999. );
	}
}



TEST_CASE("multi::blas::syrk automatic symmetrization", "[report]"){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::syrk;
		syrk(1., a, c); // c⊤=c=aa⊤=(aa⊤)⊤
		REQUIRE( c[1][0] == 34. );
		REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::syrk;
		multi::array<double, 2> c = syrk(1., a); // c⊤=c=aa⊤=(aa⊤)⊤
		REQUIRE( c[1][0] == 34. );
		REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::syrk;
		multi::array<double, 2> c = syrk(a); // c⊤=c=aa⊤=(aa⊤)⊤
		REQUIRE( c[1][0] == 34. );
		REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::transposed;
		using multi::blas::syrk;
		multi::array<double, 2> c = syrk(transposed(a)); // c⊤=c=a⊤a=(a⊤a)⊤
		REQUIRE( c[2][1] == 19. );
		REQUIRE( c[1][2] == 19. );
	}
#if 0
	{
		{
			multi::array<complex, 2> C({2, 2}, 9999.);
			syrk(1., rotated(A), C); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C lower triangular
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			multi::array<complex, 2> C({2, 2}, 9999.);
			syrk(1., rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			multi::array<complex, 2> C({2, 2}, 9999.);
			syrk(rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			complex C[2][2];
			using multi::rotated;
			syrk(rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			auto C = syrk(1., A); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is everywhere)
			assert( C[1][2]==complex(-3.,-34.) );
		}
		{
//			what(rotated(syrk(A)));
			multi::array C = rotated(syrk(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
			print(C) <<"---\n";
		}
		
	}
#if 0
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		auto C = rotated(syrk(A)).decay(); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
		print(C) <<"---\n";
//		print(C) <<"---\n";
	}
	return 0;
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		auto C = syrk(rotated(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering)
		print(C) <<"---\n";
	}
#endif
#endif
}

TEST_CASE("multi::blas::syrk herk fallback", "[report]"){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	using multi::blas::triangular;
	using multi::blas::real_operation;
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⊤=c=a⊤a=(a⊤a)⊤, `c` in lower triangular
		REQUIRE( c[1][0] == 34. ); 
		REQUIRE( c[0][1] == 9999. );
	}
}

#endif
#endif

