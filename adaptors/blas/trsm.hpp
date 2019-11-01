#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc --compiler-options -std=c++17,-Wall,-Wextra,-Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_TRSM .DCATCH_CONFIG_MAIN.o $0.cpp -o $0x \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_TRSM_HPP
#define MULTI_ADAPTORS_BLAS_TRSM_HPP

#include "../blas/core.hpp"

#include "../blas/asum.hpp" 
//#include "../blas/copy.hpp" 
//#include "../blas/numeric.hpp"
//#include "../blas/scal.hpp" 
//#include "../blas/syrk.hpp" // fallback to real case

//#include<type_traits> // void_t

#include "../blas/operations.hpp" // uplo

namespace boost{
namespace multi{namespace blas{

enum class SIDE : char{L='L', R='R'};
enum class DIAG : char{U='U', N='N'};

enum class side : char{left = static_cast<char>(SIDE::R), right = static_cast<char>(SIDE::L)};

side swap(side s){
	switch(s){
		case side::left: return side::right;
		case side::right: return side::left;
	} __builtin_unreachable();
}

enum class diagonal : char{
	unit = static_cast<char>(DIAG::U), 
	non_unit = static_cast<char>(DIAG::N), general = non_unit
};

template<typename AA, class A2D, class B2D>
B2D&& trsm(side a_side, triangular a_nonz, operation a_op, diagonal a_diag, AA alpha, A2D const& a, B2D&& b){
//	if(s=='R' and op=='N') assert(size(A)==size(B));
//	if(s=='L' and op=='N') assert(size(B[0])==size(A[0]));
//	if(s=='R' and (op=='T' or op=='C')) assert(size(A[0])==size(B));
//	if(s=='L' and (op=='T' or op=='C')) assert(size(B[0])==size(A));
	if(stride(a)==1){trsm(a_side, flip(a_nonz), transpose(a_op), a_diag, alpha, rotated(a), b); return std::forward<B2D>(b);}
	if(stride(b)==1){trsm(swap(a_side), a_nonz, transpose(a_op), a_diag, alpha, a, rotated(b)); return std::forward<B2D>(b);}
	assert(stride(*begin(a))==1);
	assert(stride(*begin(b))==1);
	char OP = [&]{
		if(a_op==operation::identity) return 'N';
		if(a_op==operation::transposition) return 'T';
		if(a_op==operation::hermitian) return 'C'; 
		__builtin_unreachable();
	}();
	multi::blas::trsm(
		static_cast<char>(a_side), static_cast<char>(a_nonz), OP, static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base(a), 
		stride(a), base(b), stride(b)
	);
	return std::forward<B2D>(b);
}

template<typename AA, class A2D, class B2D>
B2D&& trsm(side a_side, triangular a_nonz, diagonal a_diag, AA alpha, A2D const& a, B2D&& b){
	if constexpr(not multi::blas::is_hermitized<A2D>()){
		if(stride(a)==1) trsm(a_side, flip(a_nonz), operation::transposition, a_diag, alpha, rotated(a), std::forward<B2D>(b));
		else             trsm(a_side, a_nonz      , operation::identity     , a_diag, alpha,         a , std::forward<B2D>(b));
	}else{
		if(stride(a)==1) trsm(a_side, flip(a_nonz), operation::hermitian, a_diag, alpha, hermitized(a), std::forward<B2D>(b));
		else assert(0);
	}
	return std::forward<B2D>(b);
}

template<typename AA, class A2D, class B2D>
B2D&& trsm(triangular a_nonz, diagonal a_diag, AA a, A2D const& A, B2D&& B){
	if(stride(B)==1) trsm(side::right, a_nonz==triangular::lower?triangular::upper:triangular::lower, a_diag, a, rotated(A), rotated(B));
	else             trsm(side::left , a_nonz             , a_diag, a, A, std::forward<B2D>(B));
	return std::forward<B2D>(B);
}

template<typename AA, class A2D, class B2D>
decltype(auto) trsm(triangular a_nonz, AA alpha, A2D const& a, B2D&& b){
	return trsm(a_nonz, diagonal::general, alpha, a, std::forward<B2D>(b));
}

template<typename AA, class A2D, class B2D>
decltype(auto) trsm(AA a, A2D const& A, B2D&& B){
	return trsm(detect_triangular(A), diagonal::general, a, A, std::forward<B2D>(B));
}

template<typename AA, class A2D, class B2D, class Ret = typename B2D::decay_type>
#if __cplusplus>=201703L
#if __has_cpp_attribute(nodiscard)>=201603
[[nodiscard
#if __has_cpp_attribute(nodiscard)>=201907
("result is returned because third argument is const")
#endif
]]
#endif
#endif 
auto trsm(AA alpha, A2D const& a, B2D const& b){
	return trsm(alpha, a, decay(b));
}

template<class A2D, class B2D>
decltype(auto) trsm(A2D const& a, B2D&& b){return trsm(1., a, std::forward<B2D>(b));}

template<class A2D, class B2D>
#if __cplusplus>=201703L
#if __has_cpp_attribute(nodiscard)>=201603
[[nodiscard
#if __has_cpp_attribute(nodiscard)>=201907
("result is returned because second argument is const")
#endif
]]
#endif
#endif
auto trsm(A2D const& a, B2D const& b){return trsm(1., a, b);}

#if 0
enum triangular_storage{lower='L', upper='U'};

template<class UL, class Op, class AA, class BB, class A2D, class C2D>
C2D&& herk(UL uplo, Op op, AA a, A2D const& A, BB b, C2D&& C){
	if(op == 'C'){
		assert(size(A) == size(C));
		assert(stride(*begin(A))==1); assert(stride(*begin(C))==1);
		herk(uplo, op, size(C), size(*begin(A)), a, base(A), stride(A), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	if(op == 'N'){
		assert(size(*begin(A))==size(C));
		assert(stride(*begin(A))==1); assert(stride(*begin(C))==1);
		herk(uplo, op, size(C), size(A), a, base(A), stride(A), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	return std::forward<C2D>(C);
//	assert(0);
}

template<class UL, typename AA, typename BB, class A2D, class C2D>
C2D&& herk(UL uplo, AA a, A2D const& A, BB b, C2D&& C){
	if(stride(A)==1) return herk(uplo, 'C', a, rotated(A), b, std::forward<C2D>(C));
	else             return herk(uplo, 'N', a,         A , b, std::forward<C2D>(C));
}

template<class UL, typename AA, class A2D, class C2D>
C2D&& herk(UL uplo, AA a, A2D const& A, C2D&& C){
	if(stride(A)==1) return herk(uplo, 'C', a, rotated(A), 0., std::forward<C2D>(C));
	else             return herk(uplo, 'N', a,         A , 0., std::forward<C2D>(C));
}

template<class T, typename = decltype(imag(std::declval<T>()[0])[0])>
std::true_type is_complex_array_aux(T&&);
std::false_type is_complex_array_aux(...);

template <typename T> struct is_complex_array: decltype(is_complex_array_aux(std::declval<T>())){};

template<typename AA, class A2D, class C2D, typename = std::enable_if_t<is_complex_array<std::decay_t<C2D>>{}>>
C2D&& herk(AA a, A2D const& A, C2D&& C){
	if(stride(C)==1) herk('L', a, A, rotated(std::forward<C2D>(C)));
	else             herk('U', a, A,         std::forward<C2D>(C) );
	using multi::rotated;
	using multi::size;
	for(typename std::decay_t<C2D>::difference_type i = 0; i != size(C); ++i){
		blas::copy(begin(rotated(C)[i])+i+1, end(rotated(C)[i]), begin(C[i])+i+1);
		blas::scal(-1., begin(imag(C[i]))+i+1, end(imag(C[i])));
	}
	return std::forward<C2D>(C);
}

template<typename AA, class A2D, class C2D, typename = std::enable_if_t<not is_complex_array<std::decay_t<C2D>>{}>>
C2D&& herk(AA a, A2D const& A, C2D&& C, void* = 0){
	return syrk(a, A, std::forward<C2D>(C));
}

template<class AA, class A2D, class R = typename A2D::decay_type>
R herk(AA a, A2D const& A){return herk(a, A, R({size(rotated(A)), size(rotated(A))}));}

template<class A2D, class R = typename A2D::decay_type>
auto herk(A2D const& A){return herk(1., A);}
#endif

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_TRSM

namespace multi = boost::multi;

#if 0
#include "../blas/gemm.hpp"

#include "../../array.hpp"
#include "../../utility.hpp"

#include <boost/timer/timer.hpp>

#include<complex>
#include<cassert>

#include<numeric>
#include<algorithm>

using std::cout;
using std::cerr;


void f(double*){}
#endif
#include<catch.hpp>

#include<iostream>
#include<vector>

template<class M> decltype(auto) print(M const& C){
	using boost::multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j)
			std::cout << C[i][j] << ' ';
		std::cout << std::endl;
	}
	return std::cout << std::endl;
}

multi::array<double, 2> const A = {
	{ 1.,  3.,  4.},
	{NAN,  7.,  1.},
	{NAN, NAN,  8.}
};

TEST_CASE("multi::blas::trsm double RLNN", "[report]"){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, triangular::upper, operation::identity, diagonal::general, 1., A, B); // B=solve(A.x=alpha*B, x) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
	REQUIRE( B[1][2] == Approx(0.107143) );
}
TEST_CASE("multi::blas::trsm double LLNN", "[report]"){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, operation::identity, diagonal::general, 1., A, B); // B=B.A⁻¹, B⊤=(A⊤)⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE(B[1][2] == Approx(-0.892857));
}
TEST_CASE("multi::blas::trsm double RLTN", "[report]"){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, triangular::upper, operation::transposition, diagonal::general, 1., A, B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] ==  Approx(-1.57143) );
}
TEST_CASE("multi::blas::trsm double LLTN", "[report]"){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, operation::transposition, diagonal::general, 1., A, B); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] ==  Approx(0.125) );
}

TEST_CASE("multi::blas::trsm double LLTN Att", "[report]"){
	multi::array<double, 2> At = rotated(A);
	auto&& Att = rotated(At);
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, operation::transposition, diagonal::general, 1., Att, B); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] ==  Approx(0.125) );
}

TEST_CASE("multi::blas::trsm double LLTN Btt", "[report]"){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	multi::array<double, 2> Bt = rotated(B);
	auto&& Btt = rotated(Bt);
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, operation::transposition, diagonal::general, 1., A, Btt); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( Btt[1][2] ==  Approx(0.125) );
}

TEST_CASE("multi::blas::trsm double LLTN Att Btt", "[report]"){
	multi::array<double, 2> At = rotated(A);
	auto&& Att = rotated(At);
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	multi::array<double, 2> Bt = rotated(B);
	auto&& Btt = rotated(Bt);
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, operation::transposition, diagonal::general, 1., Att, Btt); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( Btt[1][2] ==  Approx(0.125) );
}
TEST_CASE("multi::blas::trsm double RLxN", "[report]"){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(side::left, triangular::upper, diagonal::general, 1., A, B); // B=A⁻¹.B, B⊤=B⊤.A⊤⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] == Approx(0.107143) );
}

TEST_CASE("multi::blas::trsm double RLxN trans", "[report]"){
	multi::array<double, 2> const A = {
		{ 1., NAN,  NAN},
		{ 3.,  7.,  NAN},
		{ 4.,  1.,   8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::diagonal;
	using multi::blas::transposed;
	trsm(side::left, triangular::upper, diagonal::general, 1., transposed(A), B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] == Approx(0.107143) );
}

TEST_CASE("multi::blas::trsm double RLxN rot", "[report]"){
	multi::array<double, 2> const A = {
		{ 1., NAN,  NAN},
		{ 3.,  7.,  NAN},
		{ 4.,  1.,   8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(side::left, triangular::upper, diagonal::general, 1., rotated(A), B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] == Approx(0.107143) );
}

TEST_CASE("multi::blas::trsm double LLxN ", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, diagonal::general, 1., A, B); // B=B.A⁻¹, B⊤=A⊤⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] == Approx(-0.892857) );
}

TEST_CASE("multi::blas::trsm double LLxN Att", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> const At = rotated(A);
	auto&& Att = rotated(At);
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, diagonal::general, 1., Att, B); // B=B.A⁻¹, B⊤=A⊤⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( B[1][2] == Approx(-0.892857) );
}

TEST_CASE("multi::blas::trsm double LLxN Att Btt", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> const At = rotated(A);
	auto&& Att = rotated(At);
	multi::array<double, 2> const B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	multi::array<double, 2> Bt = rotated(B);
	auto&& Btt = rotated(Bt);
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(side::right, triangular::upper, diagonal::general, 1., Att, Btt); // B=B.A⁻¹, B⊤=A⊤⁻¹.B⊤, A is upper triangular (implicit zeros below)
	REQUIRE( Btt[1][2] == Approx(-0.892857) );
}

TEST_CASE("multi::blas::trsm double xLxN", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(triangular::upper, diagonal::general, 1., A, B); // B=A⁻¹.B, B⊤=B⊤.A⊤⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE(B[1][2] == Approx(0.107143));
}

TEST_CASE("multi::blas::trsm double xLxN      trans", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::triangular;
	using multi::blas::diagonal;
	using multi::blas::transposed;
	trsm(triangular::upper, diagonal::general, 1., A, transposed(B)); // B⊤=A⁻¹.B⊤, B=B.A⊤⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE(B[1][2] == Approx(0.125));
}

TEST_CASE("multi::blas::trsm double xLxN      rot", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::triangular;
	using multi::blas::diagonal;
	trsm(triangular::upper, diagonal::general, 1., A, rotated(B)); // B⊤=A⁻¹.B⊤, B=B.A⊤⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE(B[1][2] == Approx(0.125));
}

TEST_CASE("multi::blas::trsm double xLxx      rot", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::triangular;
	trsm(triangular::upper, 1., A, rotated(B)); // B⊤=A⁻¹.B⊤, B=B.A⊤⁻¹, A is upper triangular (implicit zeros below)
	REQUIRE( rotated(B)[1][2] == Approx(0.5357142857) );
	REQUIRE( B[1][2] == Approx(0.125));
}

TEST_CASE("multi::blas::trsm double xxxx", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	trsm(1., A, B); // B=A⁻¹.B, B⊤=B⊤.A⊤⁻¹, Solve(A.X=B, X), A upper or lower triangular (explicit zeros or NAN on the other triangle)
	REQUIRE(B[1][2] == Approx(0.107143));
}

TEST_CASE("multi::blas::trsm double xxxx rot", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	trsm(1., rotated(A), B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, B=Solve(A⊤.X=B, X), Solve(T[X].A=T[B], X), A is upper or lower triangular (explicit zeros or NAN on the other triangular)
	REQUIRE(B[1][2] == Approx(-1.57143));
}

TEST_CASE("multi::blas::trsm double xxxx normal rot", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	trsm(1., A, rotated(B)); // B⊤=Inv[A].B⊤, B=B.Inv[A⊤], Solve(T[A].T[X]=T[B], X), Solve(T[X].A=T[B], X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE(B[1][2] == Approx(0.125));
}

TEST_CASE("multi::blas::trsm double xxxx trans trans", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	trsm(1., rotated(A), rotated(B)); // B⊤=Inv[A⊤].B⊤, B=B.Inv[A], Solve(A⊤.X=B⊤, X), Solve(T[X].A=T[B], X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE(B[1][2] == Approx(-0.892857));
}

TEST_CASE("multi::blas::trsm double xxxx rot rot", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	trsm(1., rotated(A), rotated(B)); // B⊤=Inv[A⊤].B⊤, B=B.Inv[A], Solve(A⊤.T[X]=B⊤, X), Solve(T[X].A=T[B], X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE(B[1][2] == Approx(-0.892857));
}

TEST_CASE("multi::blas::trsm double xxxx ret", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> const B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	auto C = trsm(1., A, B); // C=Inv[A].B, C⊤=B⊤.Inv[A⊤], C<-Solve(A.X=B, X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE( B[1][2] == 1. );
	REQUIRE( C[1][2] == Approx(0.107143) );
}

TEST_CASE("multi::blas::trsm double xxxx ret as_const", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	auto C = trsm(1., A, std::as_const(B)); // C=Inv[A].B, T[C]=T[B].Inv[T[A]], C<-Solve(A.X=B, X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE( B[1][2] == 1. );
	REQUIRE( C[1][2] == Approx(0.107143) );
}

TEST_CASE("multi::blas::trsm double xxxx ret rotated", "[report]"){
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<double, 2> const B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
// TODO: make this work
//	auto C = trsm(1., A, rotated(B)); // C=Inv[A].B⊤, C⊤=B.Inv[A⊤], C<-Solve(A.X=B⊤, X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
//	print(C)<<"---\n";
//	what(B.rotated());
//	REQUIRE(B[1][2] == 1.);
//	REQUIRE(C[1][2] == Approx(0.107143));
}

TEST_CASE("multi::blas::trsm complex xxxx trans trans", "[report]"){
	using complex = std::complex<double>;
	multi::array<complex, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<complex, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	using multi::blas::transposed;
	trsm(1., transposed(A), transposed(B)); // B⊤=Inv[A⊤].B⊤, B=B.Inv[A], Solve(A⊤.X⊤=B⊤, X), Solve(T[X].A=T[B], X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE( real(B[1][2]) == Approx(-0.892857));
	REQUIRE( imag(B[1][2]) == 0 );
}

TEST_CASE("multi::blas::trsm complex xxxx trans trans real factor", "[report]"){
	using complex = std::complex<double>;
	multi::array<complex, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<complex, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	using multi::blas::transposed;
	trsm(2., transposed(A), transposed(B)); // B⊤=Inv[A⊤].B⊤, B=B.Inv[A], Solve(A⊤.X⊤=B⊤, X), Solve(T[X].A=T[B], X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE( real(B[1][2]) == Approx(2.*-0.892857));
	REQUIRE( imag(B[1][2]) == 0 );
}

TEST_CASE("multi::blas::trsm complex xxxx trans trans complex factor", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0, 1};
	multi::array<complex, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::array<complex, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::trsm;
	using multi::blas::transposed;
	trsm(2.+ 2.*I, transposed(A), transposed(B)); // B⊤=alpha*Inv[A⊤].B⊤, B=alpha*B.Inv[A], Solve(A⊤.X⊤=B⊤, X), Solve(T[X].A=T[B], X), if A is upper or lower triangular (w/explicit zeros or NAN on the other triangular)
	REQUIRE( real(B[1][2]) == Approx(real((2.+2.*I)*-0.892857)));
	REQUIRE( imag(B[1][2]) == Approx(imag((2.+2.*I)*-0.892857)) );
}

TEST_CASE("multi::blas::trsm complex xxxx identity trans", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::trsm;
	using multi::blas::transposed;
	trsm(1., A, transposed(B)); // B⊤=Inv[A].B⊤, B=B.Inv[A⊤], Solve(A.X⊤=B⊤, X), Solve(T[X].A⊤=T[B], X), A is upper OR lower triangular (explicit zeros or NAN on the other triangular)
	REQUIRE( real(B[1][2]) == Approx(0.0882353) );
	REQUIRE( imag(B[1][2]) == Approx(-0.147059) );
}

TEST_CASE("multi::blas::trsm complex xxxx identity identity", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::trsm;
	trsm(1., A, B); // B=Inv[A].B, B⊤=B⊤.Inv[A⊤], Solve(A.X=B, X), Solve(X⊤.A⊤=⊤B, X), A is upper OR lower triangular (explicit zeros or NAN on the other triangular)
	REQUIRE( real(B[1][2]) == Approx(0.393213) );
	REQUIRE( imag(B[1][2]) == Approx(-0.980995) );
}

TEST_CASE("multi::blas::trsm complex xxxx identity identity complex factor", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::trsm;
	trsm(3.+1.*I, A, B); // B=alpha*Inv[A].B, B⊤=alpha*B⊤.Inv[A⊤], Solve(A.X=B, X), Solve(X⊤.A⊤=⊤B, X), A is upper OR lower triangular (explicit zeros or NAN on the other triangular)
	REQUIRE( real(B[1][2]) == Approx(2.1606) );
	REQUIRE( imag(B[1][2]) == Approx(-2.54977) );
}

TEST_CASE("multi::blas::trsm complex RLCN ", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, triangular::upper, operation::hermitian, diagonal::general, 1., A, B); // B=Inv[A⊹].B, B⊹=B⊹.Inv[A], Solve(A⊹.X=B, X), Solve(X⊹.A=B⊹, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(B[1][2]) == Approx(0.916923) );
	REQUIRE( imag(B[1][2]) == Approx(-0.504615) );
}

TEST_CASE("multi::blas::trsm complex RLCN complex factor", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, triangular::upper, operation::hermitian, diagonal::general, 2.+5.*I, A, B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(B[1][2]) == Approx(4.35692) );
	REQUIRE( imag(B[1][2]) == Approx(3.57538) );
}

TEST_CASE("multi::blas::trsm complex RLNN hermitized", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	using multi::blas::hermitized;

	trsm(side::left, triangular::lower, diagonal::general, 1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(B[1][2]) == Approx(0.916923) );
	REQUIRE( imag(B[1][2]) == Approx(-0.504615) );
}

TEST_CASE("multi::blas::trsm complex RxNN hermitized", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::side;
	using multi::blas::triangular;
	using multi::blas::operation;
	using multi::blas::diagonal;
	using multi::blas::hermitized;

	trsm(triangular::lower, diagonal::general, 1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(B[1][2]) == Approx(0.916923) );
	REQUIRE( imag(B[1][2]) == Approx(-0.504615) );
}

TEST_CASE("multi::blas::trsm complex RxNx hermitized", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	using multi::blas::triangular;
	using multi::blas::hermitized;

	trsm(triangular::lower, 1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(B[1][2]) == Approx(0.916923) );
	REQUIRE( imag(B[1][2]) == Approx(-0.504615) );
}

TEST_CASE("multi::blas::trsm complex xxxx hermitized", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};

	using multi::blas::triangular;

	REQUIRE( multi::blas::detect_triangular(A)==triangular::upper );
	REQUIRE( multi::blas::detect_triangular(rotated(A))==triangular::lower );
	REQUIRE( multi::blas::detect_triangular(multi::blas::hermitized(A))==triangular::lower );
	
	using multi::blas::hermitized;
	trsm(1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)

	REQUIRE( real(B[1][2]) == Approx(0.916923) );
	REQUIRE( imag(B[1][2]) == Approx(-0.504615) );
}

TEST_CASE("multi::blas::trsm complex xxxx hermitized complex factor", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};

	using multi::blas::triangular;	
	using multi::blas::hermitized;
	trsm(2.+1.*I, hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(B[1][2]) == Approx(2.33846) );
	REQUIRE( imag(B[1][2]) == Approx(-0.0923077) );
}

TEST_CASE("multi::blas::trsm complex xxxx hermitized complex factor return value", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> const B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};

	using multi::blas::triangular;	
	using multi::blas::hermitized;
	auto C = trsm(2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(C[1][2]) == Approx(2.33846) );
	REQUIRE( imag(C[1][2]) == Approx(-0.0923077) );
}

TEST_CASE("multi::blas::trsm complex xxxx hermitized return value", "[report]"){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> const B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};

	using multi::blas::hermitized;
	auto C = trsm(hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	REQUIRE( real(C[1][2]) == Approx(0.916923) );
	REQUIRE( imag(C[1][2]) == Approx(-0.504615) );
}



#endif
#endif

