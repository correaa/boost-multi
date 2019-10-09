#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -Ofast -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_HERK $0.cpp -o $0x \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_HERK_HPP
#define MULTI_ADAPTORS_BLAS_HERK_HPP

#include "../blas/core.hpp"
#include "../blas/copy.hpp" 
#include "../blas/numeric.hpp"
#include "../blas/scal.hpp" 
#include "../blas/syrk.hpp" // fallback to real case

#include<type_traits> // void_t

namespace boost{
namespace multi{namespace blas{

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
	using multi::blas::syrk;
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	{
		multi::array<complex, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk('U', 'N', 1., A, 0., C); // CC^H = CC =  A^H*A = (A^H*A)^H, A^H*A, A and C are C-ordering, information in C lower triangular
		print(C) <<"---\n";
	}
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
		herk(1., A, C); // CC^H = CC =  A^H*A, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk(1., A, rotated(C)); // CC^H = CC =  A^H*A, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(1., rotated(A), C); // CC^H = CC =  A*A^H, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(1., rotated(A), rotated(C)); // CC^H = CC =  A*A^H, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk(1., A, C); // C^H = C =  A^H*A = (A^H*A)^H, information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(1., rotated(A), C); // C^H = C =  A*A^H, also C = (A^T)^H*(A^T) information in C lower triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2}, 9999.);
		herk(1., rotated(A), rotated(C)); // C^H = C =  A*A^H, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3}, 9999.);
		herk(1., A, rotated(C)); // C^H = C =  A^H*A, information in C upper triangular
		print(C) <<"---\n";
	}
	{
		multi::array<complex, 2> const A({2000, 2000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.2);
		multi::array<complex, 2> C({2000, 2000}, 9999.);
		boost::timer::auto_cpu_timer t;
		herk(1., rotated(A), rotated(C)); // C^H = C =  A*A^H, information in C upper triangular
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
	//	multi::array<double, 2> C({2, 2}, 9999.);
		auto C = herk(1., rotated(A)); // CC^H = CC =  A*A^H, information in C lower triangular
		print(C) <<"---\n";
	}
	return 0;
}
	
#endif
#endif

