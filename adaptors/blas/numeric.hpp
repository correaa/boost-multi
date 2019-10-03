#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -Ofast -std=c++17 -Wall -Wextra -Wpedantic `#-Wfatal-errors` -D_TEST_MULTI_ADAPTORS_BLAS_NUMERIC $0.cpp -o $0x \
`pkg-config --libs blas64` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_BLAS_NUMERIC_HPP
#define MULTI_ADAPTORS_BLAS_NUMERIC_HPP

//#include "../blas/core.hpp"
#include "../../array_ref.hpp"

namespace boost{
namespace multi{namespace blas{

//#include "../blas/core.hpp"

template<class A, typename C = typename std::decay_t<A>::element_type, typename T = typename C::value_type>
decltype(auto) real(A&& a){
	struct Complex_{T real_; T imag_;};
	auto&& Acast = multi::reinterpret_array_cast<Complex_>(a);
	return multi::member_array_cast<T>(Acast, &Complex_::real_);
}

template<class A, typename C = typename std::decay_t<A>::element_type, typename T = typename C::value_type>
decltype(auto) imag(A&& a){
	struct Complex_{T real_; T imag_;};
	auto&& Acast = multi::reinterpret_array_cast<Complex_>(a);
	return multi::member_array_cast<T>(Acast, &Complex_::imag_);
}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_NUMERIC

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

int main(){
	using multi::blas::gemm;
	using multi::blas::herk;
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};

	multi::array A = {
		{1., 3., 4.}, 
		{9., 7., 1.}
	};
	multi::array<complex, 2> Acomplex = A;
	multi::array B = {
		{1. - 3.*I, 6.  + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};
	multi::array Breal = {
		{1., 6.},
		{8., 2.},
		{2., 1.}
	};
	multi::array Bimag = {
		{-3., +2.},
		{+2., +4.},
		{-1., +1.}
	};
	using multi::blas::real;
	using multi::blas::imag;
	assert( real(B) == Breal );
	assert( imag(B) == Bimag );

	multi::array_ref rB(reinterpret_cast<double*>(data_elements(B)), {size(B), 2*size(*begin(B))});


//	using multi::blas::imag;

//	assert( real(A)[1][2] == 1. );
//	assert( imag(A)[1][2] == -3. );

//	print(A) <<"--\n";
//	print(real(A)) <<"--\n";
//	print(imag(A)) <<"--\n";

	multi::array<complex, 2> C({2, 2});
	multi::array_ref rC(reinterpret_cast<double*>(data_elements(C)), {size(C), 2*size(*begin(C))});
//	gemm('T', 'T', 1., A, B, 0., C);
//	gemm('T', 'T', 1., A, B, 0., C);
//	gemm('T', 'T', 1., real(A), B, 0., C);
}

#endif
#endif

