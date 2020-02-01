#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && clang++ `#-DNDEBUG` -O3 -std=c++14 -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_LAPACK_CORE -DADD_ $0x.cpp -o $0x.x -lblas -llapack && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
// Alfredo A. Correa 2019 ©

#ifndef MULTI_ADAPTORS_LAPACK_CORE_HPP
#define MULTI_ADAPTORS_LAPACK_CORE_HPP

//#include<iostream>
#include<cassert>
#include<complex>

//#include <cblas/cblas.h>

#define s float
#define d double
#define c std::complex<s>
#define z std::complex<d>
#define v void 

#define INT int
#define INTEGER INT const&

#define LDA const int& lda
#define N INTEGER n
#define CHARACTER char const&
#define INFO int& info
#define UPLO CHARACTER
#define JOBZ CHARACTER
#define LAPACK(NamE) NamE##_

#define xPOTRF(T)     v LAPACK(T##potrf)(UPLO, N, T*, LDA, INFO)
//#define xHEEV(T)      v LAPACK(T##heev)(JOBZ, UPLO, N, LDA, T*, T*, T*, T*, INFO)

extern "C"{
xPOTRF(s)   ; xPOTRF(d)    ;
xPOTRF(c)   ; xPOTRF(z)    ;
}

#undef JOBZ
#undef UPLO
#undef INFO
#undef CHARACTER
#undef N
#undef LDA

#undef INTEGER
#undef INT


#define xpotrf(T) template<class S> v potrf(char uplo, S n, T *x, S incx, int& info){LAPACK(T##potrf)(uplo, n, x, incx, info);}

namespace core{
xpotrf(s) xpotrf(d)
xpotrf(c) xpotrf(z)
}

#undef s
#undef d
#undef c
#undef z
#undef v



#define TRANS const char& trans

///////////////////////////////////////////////////////////////////////////////

#if _TEST_MULTI_ADAPTORS_LAPACK_CORE

#include "../../array.hpp"
#include "../../utility.hpp"

#include<iostream>
#include<numeric>
#include<vector>

namespace multi = boost::multi;
using std::cout; 

int main(){
	using core::potrf;

	std::vector<double> v = {
		2., 1.,
		1., 2.
	};
	cout 
		<< v[0] <<'\t'<< v[1] <<'\n'
		<< v[2] <<'\t'<< v[3] <<'\n' << std::endl
	;
	int info;
	potrf('U', 2, v.data(), 2, info);
	cout << "error " << info << std::endl;
	cout 
		<< v[0] <<'\t'<< v[1] <<'\n'
		<< v[2] <<'\t'<< v[3] <<'\n'
	;
	cout << std::endl;
}

#endif
#endif

