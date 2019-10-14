#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -Ofast -std=c++14 -Wall -Wextra -Wpedantic `#-Wfatal-errors` -D_TEST_MULTI_ADAPTORS_BLAS_GEMM $0.cpp -o $0x \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_BLAS_GEMM_HPP
#define MULTI_ADAPTORS_BLAS_GEMM_HPP
#include "../blas/core.hpp"

namespace boost{
namespace multi{
namespace blas{

struct op{enum : char{N='N', T='T', C='C'};};

//struct conj{template<class T> auto operator()(T const& t) const{using std::conj; return conj(t);}};

template<class Op, class AA, class BB, class It1, class Size1, class It2, class Size2, class Out>
auto gemm_n(
	Op opA, Op opB, AA const& a, 
	It1 Af, Size1 An, It2 Bf, Size2 Bn, BB const& b, 
	Out Cf
){
	assert( Af->stride() == 1 );
	assert( Bf->stride() == 1 );
	assert( Cf->stride() == 1 );
	switch(opA){
	case op::N: assert( Cf->size() == Af->size() );
		switch(opB){
		case op::N:
			gemm(opA, opB, Cf->size(), Bn, An, a, base(Af), stride(Af), base(Bf), stride(Bf), b, base(Cf), stride(Cf));
			return Cf + Bn;
		case op::T: case op::C:
			gemm(opA, opB, Cf->size(), Bn, An, a, base(Af), stride(Af), base(Bf), stride(Bf), b, base(Cf), stride(Cf));
			return Cf + Bf->size();
		}
	case op::T: case op::C: // assert( Cf->size() == An );
		switch(opB){
		case op::N:
			gemm(op::N, op::T, Cf->size(), Cf->size(), Bn, a, base(Af), stride(Af), base(Bf), stride(Bf), b, base(Cf), stride(Cf));
			return Cf + Bn;
		case op::T: case op::C:
			gemm(opA, opB, Cf->size(), An, Bn, a, base(Af), stride(Af), base(Bf), stride(Bf), b, base(Cf), stride(Cf));
			return Cf + Bf->size();
		}
	}
	assert(0);
	return Cf;
}

template<class Op, class AA, class BB, class A2D, class B2D, class C2D>
decltype(auto) gemm(Op opA, Op opB, AA a, A2D const& A, B2D const& B, BB b, C2D&& C){
//	gemm_n(opA, opB, a, begin(A), size(A), begin(B), size(B), b, begin(C));
//	return std::forward<C2D>(C);
#if 1
	using std::begin;
	assert(stride(*begin(A)) == 1 and stride(*begin(B))==1 and stride(*begin(C))==1);
	if((opA == 'T' or opA == 'C') and (opB == 'T' or opB == 'C')){ // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		assert(size(*begin(A)) == size(B) and size(A)==size(*begin(C)) and size(*begin(B)) == size(C));
		gemm(opA, opB, size(A), size(C), size(B), a, base(A), stride(A), base(B), stride(B), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	if(opA == 'N' and (opB == 'T' or opB == 'C')){
		assert(size(A) == size(B) and size(*begin(A))==size(*begin(C)) and size(*begin(B)) == size(C));
		gemm(opA, opB, size(*begin(A)), size(C), size(B), a, base(A), stride(A), base(B), stride(B), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	if((opA == 'T' or opA == 'C') and (opB == 'N')){
		assert(size(*begin(A)) == size(*begin(B)) and size(A)==size(*begin(C)) and size(B) == size(C));
		gemm(opA, opB, size(A), size(C), size(*begin(B)), a, base(A), stride(A), base(B), stride(B), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	if((opA == 'N') and (opB == 'N')){
		assert(size(A) == size(*begin(B)) and size(*begin(A))==size(*begin(C)) and size(B) == size(C));
		gemm(opA, opB, size(*begin(A)), size(C), size(*begin(B)), a, base(A), stride(A), base(B), stride(B), b, base(C), stride(C));
		return std::forward<C2D>(C);
	}
	assert(0);
	return std::forward<C2D>(C);
#endif
}

template<class AA, class BB, class A2D, class B2D, class C2D>
decltype(auto) gemm(AA a, A2D const& A, B2D const& B, BB b, C2D&& C){
	if(stride(*begin(A)) == 1 and stride(*begin(B))==1 and stride(*begin(C))==1){
		assert( size(A) == size(rotated(C)) );
		assert( size(rotated(B)) == size(C) );
		return gemm('T', 'T', a, A, B, b, C);
	}
	assert(0);
	return std::forward<C2D>(C);
}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_GEMM

#include "../blas/herk.hpp"

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
	using complex = std::complex<double>;
	using multi::blas::gemm;
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<complex, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<complex, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.} 
		};
		multi::array<complex, 2> C({4, 2}); 
		gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		double A[2][3] = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		double B[3][4] = {
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		double C[4][2];
		gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm(1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<double, 2> const A = {
			{1., 9.}, 
			{3., 7.}, 
			{4., 1.}
		};
		multi::array<double, 2> const B = {
			{ 11., 12., 4., 0.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<double, 2> C({4, 2}, 0.);
		gemm('N', 'T', 1., A, B, 0., C); // C^T = A^T*B , C = (A^T*B)^T, C = B^T*A , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n";
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {
			{11., 7., 11.}, 
			{12., 19., 12.}, 
			{4., 1., 4.}, 
			{3., 2., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm('T', 'N', 1., A, B, 0., C); // C^T = A*B^T , C = (A*B^T)^T, C = B*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<double, 2> const A = {
			{1., 9.}, 
			{3., 7.}, 
			{4., 1.}
		};
		multi::array<double, 2> const B = {
			{11., 7., 11.}, 
			{12., 19., 12.}, 
			{4., 1., 4.}, 
			{3., 2., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm('N', 'N', 1., A, B, 0., C); // C^T = A^T*B^T , C = (A^T*B^T)^T, C = B*A, if A, B, and C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> C({2, 2});
		gemm('T', 'N', 1., A, A, 0., C); // C^T = C = A*A^T , C = (A*A^T)^T, C = A*A^T = C^T, if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	constexpr auto const I = complex{0., 1.};
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2});
		gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
		print(C) << "---\n";
		multi::array<complex, 2> CC({2, 2});
		using multi::blas::herk;
		herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	{
		complex const A[2][3] = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		complex C[2][2];
		gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
		print(C) << "---\n";
		complex CC[2][2];
		using multi::blas::herk;
		herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3});
		gemm('N', 'C', 1., A, A, 0., C); // C^H = C = A^H*A , C = (A^H*A)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
	//	print(C) << "---\n";
		multi::array<complex, 2> CC({3, 3});
		using multi::blas::herk;
		herk('U', 'N', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	return 0;
	{
	{
		using std::complex;
		{
			multi::array<complex<double>, 2> const A({1000, 1000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.1);
			multi::array<std::complex<double>, 2> C({size(A), size(A)});
			{
			boost::timer::auto_cpu_timer t;
			using multi::blas::herk;
			herk('U', 'C', 1., A, 0., C); // C^H = C = A*A^H, C = (A*A^H)^H, C = A*A^H, C lower triangular. if A, B, C are c-ordering (e.g. array or array_ref)
			}
			cerr << C[10][1] << std::endl;
		}
		{
			multi::array<complex<double>, 2> const A({1000, 1000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.1);
			multi::array<std::complex<double>, 2> C({size(A), size(A)});
			{
			boost::timer::auto_cpu_timer t;
			gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
			}
			cerr << C[10][1] << std::endl;
		}
		{
			multi::array<complex<double>, 2> const A({2000, 2000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.1);
			multi::array<complex<double>, 2> const B({2000, 2000}); std::iota(data_elements(B), data_elements(B) + num_elements(B), 1.2);
			multi::array<std::complex<double>, 2> C({size(A), size(A)});
			{
			boost::timer::auto_cpu_timer t;
			gemm('C', 'N', 1., A, B, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
			}
			cerr << C[1222][134] << std::endl;
		}
	//	print(C) << "---\n";
	//	print(CC) << "---\n";
	}

	}
	return 0; 
#if 0
	{
		multi::array<double, 2> const A = {
			{ 1., 1., 1.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 1., 0., 0.},
			{ 1., 0., 0.},
			{ 1., 0., 0.},
		};
		using multi::blas::gemm;

		multi::array<double, 2> C1({3, 3});
		gemm('N', 'N', 1., B, A, 0., C1); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C1);

		multi::array<double, 2> C2({3, 3});
		gemm('N', 'N', 1., A, B, 0., C2); // C = B*A , C^T = (A^T).(B^T) , if A, B, C are c-ordering
		print(C2);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 1., 1.},
			{ 0., 0., 0.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 1., 0., 0., 0., 0.},
			{ 1., 0., 0., 0., 0.},
			{ 1., 0., 0., 0., 0,}
		};
		multi::array<double, 2> C({4, 5});

		using multi::blas::gemm;
		gemm('N', 'N', 1., B, A, 0., C); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2., 3.},
			{ 1., 4., 5.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm('T', 'N', 1., A, A, 0., C); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0.},
			{ 1., 1., 1.}
		};
		multi::array<double, 2> const B = {
			{ 0., 1., 0.},
			{ 0., 1., 0.},
			{ 0., 1., 0.}
		};
		multi::array<double, 2> C({2, 3});

		using multi::blas::gemm;
		gemm('N', 'N', 1., B, A, 0., C); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0.},
			{ 0., 0., 1.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 0., 1., 0.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> C({3, 3});

		using multi::blas::gemm; 
		gemm('T', 'T', 1., B, A, 0., C); // C = (B*A)^T , C^T = A*B, C=A^T*B^T , if A, B, C are c-ordering
		print(C);

	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0., 0.},
			{ 0., 1., 0., 0.},
			{ 0., 0., 0., 0.},
			{ 0., 0., 0., 0.},
			{ 0., 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 0., 0., 0., 0., 0.},
			{ 0., 1., 0., 0., 0.},
			{ 0., 0., 0., 0., 0.}
		};
		multi::array<double, 2> C({4, 3});

		using multi::blas::gemm;
		gemm('T', 'T', 1., B, A, 0., C); //C = (B*A)^T, C^T = A*B, C=A^T*B^T, if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0.},
			{ 0., 1., 0.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 0., 0., 0., 0.},
			{ 0., 1., 0., 0.},
		};
		multi::array<double, 2> C({3, 2});

		using multi::blas::gemm;
		gemm('T', 'T', 1., B, A, 0., C); //C = (B*A)^T, C^T = A*B, C=A^T*B^T, if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1.},
			{ 0., 1.}
		};
		multi::array<double, 2> const B = {
			{ 1., 1.},
			{ 0., 0.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm; 
		gemm('T', 'N', 1., B, A, 0., C); // C = A*(B^T) , C^T = B*(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1., 0., 0.},
			{ 0., 1., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 1., 1., 1., 1.},
			{ 0., 0., 0., 0.},
			{ 0., 0., 0., 0.}
		};
		multi::array<double, 2> C({2, 3});
		using multi::blas::gemm;
		gemm('T', 'N', 1., B, A, 0., C); // C = A*(B^T) , C^T = B*(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1.},
			{ 0., 1.}
		};
		multi::array<double, 2> const B = {
			{ 1., 1.},
			{ 0., 0.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm('N', 'T', 1., B, A, 0., C); // C = ((B^T)*A)^T , C^T = B^T*A, C = (A^T)*B , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1., 0., 0.},
			{ 0., 1., 0., 0.},
		};
		multi::array<double, 2> const B = {
			{ 1., 1., 1.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> C({4, 3});
		using multi::blas::gemm;
		gemm('N', 'T', 1., B, A, 0., C); // C = ((B^T)*A)^T , C^T = B^T*A, C = (A^T)*B , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, B, 0., C); // C = A.B
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), B, 0., C); // C = A^T.B
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, rotated(B), 0., C); // C = A.B^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), rotated(B), 0., C); // C = A^T.B^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, B, 0., rotated(C)); // C^T = A.B, C = B^T.A^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), B, 0., rotated(C)); // C^T = A^T.B, C = B^T.A
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, rotated(B), 0., rotated(C)); // C^T = A.B^T, C = B.A^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), rotated(B), 0., rotated(C)); // C^T = A^T.B^T, C = B.A
		print(C);
	}
#endif
}

#endif
#endif

