#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"">$0.cpp)&&clang++ -Ofast -std=c++14 -Wall -Wextra -Wpedantic `#-Wfatal-errors` -D_TEST_MULTI_ADAPTORS_BLAS_HERK $0.cpp -o $0x \
`#-lblas` \
-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_BLAS_HERK_HPP
#define MULTI_ADAPTORS_BLAS_HERK_HPP

#include "../blas/core.hpp"

namespace boost{
namespace multi{namespace blas{

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
	assert(0);
}

template<class UL, class AA, class BB, class A2D, class C2D>
C2D&& herk(UL uplo, AA a, A2D const& A, BB b, C2D&& C){
	if(stride(A)==1) return herk(uplo, 'C', a, rotated(A), b, std::forward<C2D>(C));
	else             return herk(uplo, 'N', a,         A , b, std::forward<C2D>(C));
}

template<class AA, class BB, class A2D, class C2D>
C2D&& herk(AA a, A2D const& A, BB b, C2D&& C){
	if(stride(C)==1) herk('L', a, A, b, rotated(std::forward<C2D>(C)));
	else             herk('U', a, A, b,         std::forward<C2D>(C) );
	return std::forward<C2D>(C);
}

template<class A2D, class C2D>
C2D&& herk(A2D const& A, C2D&& C){return herk(1., A, 0., std::forward<C2D>(C));}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_HERK

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
	using multi::blas::herk;
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk('U', 'N', 1., A, 0., C); // CC^H = CC =  A^H*A = (A^H*A)^H, A^H*A, A and C are C-ordering, information in C lower triangular
	//	herk('U', 'N', 1., A, 0., C); // error: C must be C-ordering
	//	herk('U', 'N', 1., rotated(A), 0., C); // error: A must be C-ordering
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk('L', 'N', 1., A, 0., C); // CC^H = CC =  A^H*A = (A^H*A)^H, A^H*A, A and C are C-ordering, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk('U', 'C', 1., A, 0., C); // CC^H = CC =  A*A^H, A and C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk('L', 'C', 1., A, 0., C); // CC^H = CC =  A*A^H, A and C are C-ordering, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk('U', 1., A, 0., C); // CC^H = CC =  A^H*A, C is C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk('U', 1., rotated(A), 0., C); // CC^H = CC =  A*A^H, C is C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk(1., A, 0., C); // CC^H = CC =  A^H*A, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk(1., A, 0., rotated(C)); // CC^H = CC =  A^H*A, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(1., rotated(A), 0., C); // CC^H = CC =  A*A^H, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(1., rotated(A), 0., rotated(C)); // CC^H = CC =  A*A^H, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk(A, C); // CC^H = CC =  A^H*A, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(rotated(A), C); // C^H = C =  A*A^H, also C = (A^T)^H*(A^T) information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(rotated(A), rotated(C)); // CC^H = CC =  A*A^H, information in C upper triangular
		print(C) <<"---\n";
	}
	return 0;
}

#endif
#endif

