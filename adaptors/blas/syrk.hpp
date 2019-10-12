#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -Ofast -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_SYRK $0.cpp -o $0x \
`pkg-config --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
&&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_SYRK_HPP
#define MULTI_ADAPTORS_BLAS_SYRK_HPP

#include "../blas/core.hpp"
#include "../blas/copy.hpp"

namespace boost{
namespace multi{namespace blas{

template<class UL, class Op, typename AA, typename BB, class A2D, class C2D>
C2D&& syrk(UL uplo, Op op, AA a, A2D const& A, BB b, C2D&& C){
	if(op == 'C' or op == 'T'){
		assert(size(A) == size(C)); assert(stride(*begin(A))==1); assert(stride(*begin(C))==1);
		syrk(uplo, op, size(C), size(*begin(A)), a, base(A), stride(A), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	if(op == 'N'){
		assert(size(*begin(A))==size(C)); assert(stride(*begin(A))==1); assert(stride(*begin(C))==1);
		syrk(uplo, op, size(C), size(A), a, base(A), stride(A), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	assert(0);
	return std::forward<C2D>(C);
}

template<class UL, typename AA, typename BB, class A2D, class C2D>
C2D&& syrk(UL uplo, AA a, A2D const& A, BB b, C2D&& C){
	if(stride(A)==1) return syrk(uplo, 'T', a, rotated(A), b, std::forward<C2D>(C));
	else             return syrk(uplo, 'N', a,         A , b, std::forward<C2D>(C));
}

template<class UL, typename AA, class A2D, class C2D>
C2D&& syrk(UL uplo, AA a, A2D const& A, C2D&& C){
	return syrk(uplo, a, A, 0., std::forward<C2D>(C));
}

template<typename AA, class A2D, class C2D>
C2D&& syrk(AA a, A2D const& A, C2D&& C){
	if(stride(C)==1) syrk('L', a, A, rotated(std::forward<C2D>(C)));
	else             syrk('U', a, A,         std::forward<C2D>(C) );
	for(typename std::decay_t<C2D>::difference_type i = 0; i != size(C); ++i)
		blas::copy(begin(rotated(C)[i])+i+1, end(rotated(C)[i]), begin(C[i])+i+1);
	return std::forward<C2D>(C);
}

template<class A2D, class C2D>
C2D&& syrk(A2D const& A, C2D&& C){return syrk(1., A, std::forward<C2D>(C));}

template<typename AA, class A2D, class R = typename A2D::decay_type>
R syrk(AA a, A2D const& A){return syrk(a, A, R({size(rotated(A)), size(rotated(A))}));}

template<class A2D, class R = typename A2D::decay_type>
auto syrk(A2D const& A){return syrk(1., A);}

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

void f(double*){}

int main(){
	using multi::blas::gemm;
	using multi::blas::syrk;
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> C({3, 3}, 9999.);
		syrk('U', 'N', 1., A, 0., C); // C^T = C =  A^T*A = (A^T*A)^T, A^T*A, A and C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> C({3, 3}, 9999.);
		syrk('L', 'N', 1., A, 0., C); // C^T = C =  A^T*A = (A^T*A)^T, A^T*A, A and C are C-ordering, information in C upper triangular
	//	syrk('L', 'N', 1., A, 0., rotated(C)); // error, rotated(C) is not C-ordering
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		syrk('U', 'N', 1., A, 0., C); // C^T = C =  A^T*A = (A^T*A)^T, A^T*A, A and C are C-ordering, information in C upper triangular
	//	syrk('L', 'N', 1., A, 0., rotated(C)); // error, rotated(C) is not C-ordering
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		syrk('U', 'N', 1., A, 0., C); // C^T = C =  A^T*A = (A^T*A)^T, A^T*A, A and C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		syrk('U', 'T', 1., A, 0., C); // C^T = C =  A*A^T = (A*A^T)^T, A^T*A, A and C are C-ordering, information in C lower triangular
	//	syrk('U', 'C', 1., A, 0., C); // error 'C' is not valid for syrk??
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		syrk('U', 1., A, 0., C); // C^T = C =  A^T*A = (A*A^T)^T, A*A^T, A and C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		syrk('U', 1., rotated(A), 0., C); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		syrk(1., rotated(A), C); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		syrk(1., rotated(A), rotated(C)); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		syrk(rotated(A), rotated(C)); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		complex const A[2][3] = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		complex C[2][2];
		using multi::rotated;
		syrk(rotated(A), rotated(C)); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		auto C = syrk(A); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is everywhere)
		print(C) <<"---\n";
	}
	return 0; 
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C = rotated(syrk(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
		print(C) <<"---\n";
	}
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
}

#endif
#endif

