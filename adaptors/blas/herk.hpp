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
	if(stride(*begin(C))!=1){
		if(uplo == 'U'){
			herk('L', op, a, A, b, rotated(C));
			return std::forward<C2D>(C);
		}
		if(uplo == 'L'){
			herk('U', op, a, A, b, rotated(C));
			return std::forward<C2D>(C);
		}
		assert(0);
	}
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
	if(stride(A)==1) herk('L', 'C', a, rotated(A), b, std::forward<C2D>(C));
	else             herk('L', 'N', a,         A , b, std::forward<C2D>(C));
	for(std::ptrdiff_t i = 0; i != size(C); ++i)
		for(std::ptrdiff_t j = 0; j != i; ++j)
			C[j][i] = conj(C[i][j]);
	return std::forward<C2D>(C);
}


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
	using std::complex;
	constexpr auto const I = complex<double>{0., 1.};
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> C({2, 2});
		gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
		print(C) << "---\n";
		multi::array<std::complex<double>, 2> CC({2, 2});
		herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	{
		complex<double> const A[2][3] = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		complex<double> C[2][2];
		gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
		print(C) << "---\n";
		complex<double> CC[2][2];
		herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> C({3, 3});
		gemm('N', 'C', 1., A, A, 0., C); // C^H = C = A^H*A , C = (A^H*A)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
	//	print(C) << "---\n";
		multi::array<complex<double>, 2> CC({3, 3});
		using multi::blas::herk;
		herk('U', 'N', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> CC({3, 3});
		using multi::blas::herk;
		herk('L', 'N', 1., A, 0., CC); // CC^H = CC = A^H*A = (A^H*A)^H, A^H*A, A C-ordering, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> CC({2, 2});
		using multi::blas::herk;
		herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A^H*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> CC({3, 3});
		using multi::blas::herk;
		herk('U', 1., A, 0., CC); // CC^H = CC =  A^H*A = (A^H*A)^H, A^H*A, A C-ordering, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A = {
			{1. + 3.*I, 9. + 1.*I}, 
			{3. - 2.*I, 7. - 8.*I}, 
			{4. + 1.*I, 1. - 3.*I}
		};
		multi::array<complex<double>, 2> CC({3, 3});
		using multi::blas::herk;
		herk('U', 1., rotated(A), 0., CC); // CC^H = CC =  A*A^H = (A*A^H)^H, A*A^H, A C-ordering, C upper triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> CC({3, 3});
		using multi::blas::herk;
		herk(1., A, 0., CC); // CC^H = CC =  A^H*A = (A^H*A)^H, A^H*A, A C-ordering, C lower triangular
		print(CC) << "---\n";
	}
	return 0;
	{
		multi::array<complex<double>, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex<double>, 2> CC({2, 2});
		using multi::blas::herk;
		herk('U', 1., rotated(A), 0., CC); // CC^H = CC =  A*A^H = (A*A^H)^H, A*A^H, A C-ordering, C lower triangular
		print(CC) << "---\n";
	}
	{
		multi::array<complex<double>, 2> const A_aux = {
			{1. + 3.*I, 9. + 1.*I}, 
			{3. - 2.*I, 7. - 8.*I}, 
			{4. + 1.*I, 1. - 3.*I}
		};
		auto const& A = rotated(A_aux);
		multi::array<complex<double>, 2> CC({2, 2});
		using multi::blas::herk;
		herk('L', 1., rotated(A), 0., rotated(CC)); // CC^H = CC =  A*A^H = (A*A^H)^H, A*A^H, A C-ordering, C upper triangular
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

