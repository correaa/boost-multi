#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&`#nvcc`clang++ -std=c++17 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_TRSM $0.cpp -o $0x -lboost_unit_test_framework \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_TRSM_HPP
#define MULTI_ADAPTORS_BLAS_TRSM_HPP

#include "../blas/core.hpp"

#include "../blas/operations.hpp" // uplo
#include "../blas/side.hpp"

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

template<class A> auto trsm_base_aux(A&& a, std::false_type){return base(a);}
template<class A> auto trsm_base_aux(A&& a, std::true_type){return underlying(base(a));}

template<typename AA, class A2D, class B2D>
B2D&& trsm(side a_side, fill a_nonz, operation a_op, diagonal a_diag, AA alpha, A2D const& a, B2D&& b){
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
	using core::trsm;
	auto base_a = trsm_base_aux(a, is_hermitized<A2D>{}); (void)base_a;
	auto base_b = trsm_base_aux(b, is_hermitized<B2D>{}); (void)base_b;
	trsm(
		static_cast<char>(a_side), static_cast<char>(a_nonz), OP, static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, 
		stride(a), base_b, stride(b)
	);
	return std::forward<B2D>(b);
}

template<typename AA, class A2D, class B2D>
B2D&& trsm(side a_side, fill a_nonz, diagonal a_diag, AA alpha, A2D const& a, B2D&& b){
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
B2D&& trsm(fill a_nonz, diagonal a_diag, AA a, A2D const& A, B2D&& B){
	if(stride(B)==1) trsm(side::right, a_nonz==fill::lower?fill::upper:fill::lower, a_diag, a, rotated(A), rotated(B));
	else             trsm(side::left , a_nonz             , a_diag, a, A, std::forward<B2D>(B));
	return std::forward<B2D>(B);
}

template<typename AA, class A2D, class B2D>
decltype(auto) trsm(fill a_nonz, AA alpha, A2D const& a, B2D&& b){
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
	return trsm(alpha, a, b.decay());
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

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../array.hpp"

namespace multi = boost::multi;

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

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_RLNN, *utf::tolerance(0.00001)){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;

	trsm(side::left, fill::upper, operation::identity, diagonal::general, 1., A, B); // B=solve(A.x=alpha*B, x) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == 0.107143 );
}
BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLNN, *utf::tolerance(0.00001)){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, operation::identity, diagonal::general, 1., A, B); // B=B.A⁻¹, B⊤=(A⊤)⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == -0.892857 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_RLTN, *utf::tolerance(0.00001)){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, fill::upper, operation::transposition, diagonal::general, 1., A, B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] ==  -1.57143 );
}
BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLTN){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, operation::transposition, diagonal::general, 1., A, B); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == 0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLTN_Att){
	multi::array<double, 2> At = rotated(A);
	auto&& Att = rotated(At);
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, operation::transposition, diagonal::general, 1., Att, B); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_REQUIRE( B[1][2] == 0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLTN_Btt){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	multi::array<double, 2> Bt = rotated(B);
	auto&& Btt = rotated(Bt);
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, operation::transposition, diagonal::general, 1., A, Btt); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_REQUIRE( Btt[1][2] ==  0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLTN_Att_Btt){
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
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, operation::transposition, diagonal::general, 1., Att, Btt); // B=B.A⊤⁻¹, B⊤=A⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_REQUIRE( Btt[1][2] ==  0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_RLxN, * utf::tolerance(0.00001)){
	multi::array<double, 2> B = {
		{1., 3., 4.},
		{2., 7., 1.},
		{3., 4., 2.}
	};
	using multi::blas::side;
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(side::left, fill::upper, diagonal::general, 1., A, B); // B=A⁻¹.B, B⊤=B⊤.A⊤⁻¹, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_RLxN_trans, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	using multi::blas::transposed;
	trsm(side::left, fill::upper, diagonal::general, 1., transposed(A), B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_RLxN_rot, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(side::left, fill::upper, diagonal::general, 1., rotated(A), B); // B=A⊤⁻¹.B, B⊤=B⊤.A⁻¹, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLxN, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, diagonal::general, 1., A, B); // B=B.A⁻¹, B⊤=A⊤⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == -0.892857 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLxN_Att, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, diagonal::general, 1., Att, B); // B=B.A⁻¹, B⊤=A⊤⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_TEST( B[1][2] == -0.892857 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_LLxN_Att_Btt, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(side::right, fill::upper, diagonal::general, 1., Att, Btt); // B=B.A⁻¹, B⊤=A⊤⁻¹.B⊤, A is upper triangular (implicit zeros below)
	BOOST_TEST( Btt[1][2] == -0.892857 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xLxN, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(fill::upper, diagonal::general, 1., A, B); // B=A⁻¹.B, B⊤=B⊤.A⊤⁻¹, A is upper triangular (implicit zeros below)
	BOOST_TEST(B[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xLxN_trans){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	using multi::blas::transposed;
	trsm(fill::upper, diagonal::general, 1., A, transposed(B)); // B⊤=A⁻¹.B⊤, B=B.A⊤⁻¹, A is upper triangular (implicit zeros below)
	BOOST_REQUIRE( B[1][2] == 0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xLxN_rot){
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
	using multi::blas::fill;
	using multi::blas::diagonal;
	trsm(fill::upper, diagonal::general, 1., A, rotated(B)); // B⊤=A⁻¹.B⊤, B=B.A⊤⁻¹, A is upper triangular (implicit zeros below)
	BOOST_REQUIRE( B[1][2] == 0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xLxx_rot, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	trsm(fill::upper, 1., A, rotated(B)); // B⊤=A⁻¹.B⊤, B=B.A⊤⁻¹, A is upper triangular (implicit zeros below)
	BOOST_TEST( rotated(B)[1][2] == 0.5357142857  );
	BOOST_TEST( B[1][2] == 0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_rot, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == -1.57143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_normal_rot, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == 0.125 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_trans_trans, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == -0.892857 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_rot_rot, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == -0.892857 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_ret, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == 1. );
	BOOST_TEST( C[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_ret_as_const, *utf::tolerance(0.00001)){
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
	BOOST_TEST( B[1][2] == 1. );
	BOOST_TEST( C[1][2] == 0.107143 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_xxxx_ret_rotated){
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

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_trans_trans, *utf::tolerance(0.00001)){
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
	BOOST_TEST( real(B[1][2]) == -0.892857 );
	BOOST_TEST( imag(B[1][2]) == 0 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_trans_trans_real_factor, *utf::tolerance(0.00001)){
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
	BOOST_TEST( real(B[1][2]) == 2.*-0.892857 );
	BOOST_TEST( imag(B[1][2]) == 0 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_trans_trans_complex_factor, *utf::tolerance(0.00001)){
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
	BOOST_TEST( real(B[1][2]) == real((2.+2.*I)*-0.892857) );
	BOOST_TEST( imag(B[1][2]) == imag((2.+2.*I)*-0.892857) );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_identity_trans, *utf::tolerance(0.00001)){
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
	BOOST_TEST( real(B[1][2]) == 0.0882353 );
	BOOST_TEST( imag(B[1][2]) == -0.147059 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_identity_identity, *utf::tolerance(0.00001)){
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
	BOOST_TEST( real(B[1][2]) == 0.393213 );
	BOOST_TEST( imag(B[1][2]) == -0.980995 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_identity_identity_complex_factor, *utf::tolerance(0.0001)){
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
	BOOST_TEST( real(B[1][2]) == 2.1606 );
	BOOST_TEST( imag(B[1][2]) == -2.54977 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_RLCN, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, fill::upper, operation::hermitian, diagonal::general, 1., A, B); // B=Inv[A⊹].B, B⊹=B⊹.Inv[A], Solve(A⊹.X=B, X), Solve(X⊹.A=B⊹, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(B[1][2]) == 0.916923 );
	BOOST_TEST( imag(B[1][2]) == -0.504615 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_RLCN_complex_factor, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	trsm(side::left, fill::upper, operation::hermitian, diagonal::general, 2.+5.*I, A, B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(B[1][2]) == 4.35692 );
	BOOST_TEST( imag(B[1][2]) == 3.57538 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_RLNN_hermitized, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	using multi::blas::hermitized;

	trsm(side::left, fill::lower, diagonal::general, 1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(B[1][2]) == 0.916923 );
	BOOST_TEST( imag(B[1][2]) == -0.504615 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_RxNN_hermitized, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::operation;
	using multi::blas::diagonal;
	using multi::blas::hermitized;

	trsm(fill::lower, diagonal::general, 1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(B[1][2]) == 0.916923 );
	BOOST_TEST( imag(B[1][2]) == -0.504615 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_RxNx_hermitized, *utf::tolerance(0.00001)){
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
	using multi::blas::fill;
	using multi::blas::hermitized;

	trsm(fill::lower, 1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(B[1][2]) == 0.916923 );
	BOOST_TEST( imag(B[1][2]) == -0.504615 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_hermitized, *utf::tolerance(0.00001)){
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

	using multi::blas::fill;

	BOOST_REQUIRE( multi::blas::detect_triangular(A)==fill::upper );
	BOOST_REQUIRE( multi::blas::detect_triangular(rotated(A))==fill::lower );
	BOOST_REQUIRE( multi::blas::detect_triangular(multi::blas::hermitized(A))==fill::lower );
	
	using multi::blas::hermitized;
	trsm(1., hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)

	BOOST_TEST( real(B[1][2]) == 0.916923 );
	BOOST_TEST( imag(B[1][2]) == -0.504615 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_hermitized_complex_factor, *utf::tolerance(0.00001)){
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

	using multi::blas::fill;	
	using multi::blas::hermitized;
	trsm(2.+1.*I, hermitized(A), B); // B=Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(B[1][2]) == 2.33846 );
	BOOST_TEST( imag(B[1][2]) == -0.0923077 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_hermitized_complex_factor_return_value, *utf::tolerance(0.00001)){
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

	using multi::blas::fill;	
	using multi::blas::hermitized;
	auto C = trsm(2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(C[1][2]) == 2.33846 );
	BOOST_TEST( imag(C[1][2]) == -0.0923077 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_xxxx_hermitized_return_value, *utf::tolerance(0.00001)){
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
	BOOST_TEST( real(C[1][2]) == 0.916923 );
	BOOST_TEST( imag(C[1][2]) == -0.504615 );
}



#endif
#endif

