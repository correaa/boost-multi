#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -std=c++14 -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_CORE -DADD_ $0.cpp -o $0x -lblas -lcudart&&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_CUDA_HPP
#define MULTI_ADAPTORS_BLAS_CUDA_HPP

//#include "../../utility.hpp"
//#include "../../array.hpp" // allocating multi::arrays for output
#include "../../memory/adaptors/cuda/ptr.hpp"
#include "../../memory/adaptors/cuda/managed/ptr.hpp"

#include<iostream> // debug
#include<cublas_v2.h>
//#include <cblas/cblas.h>
#include<complex>

namespace boost{
namespace multi{
namespace blas{

template<class T> class cublas;

template<>
class cublas<double>{
public:
	template<class... Args>
	static auto gemm(Args... args){return cublasDgemm(args...);}
};

template<>
class cublas<float>{
public:
	template<class... Args>
	static auto gemm(Args... args){return cublasSgemm(args...);}
};

template<>
class cublas<std::complex<double>>{
	template<class T> static decltype(auto) to_cu(T&& t){return std::forward<T>(t);}
	static decltype(auto) to_cu(std::complex<double> const* t){return reinterpret_cast<cuDoubleComplex const*>(t);}	
	static decltype(auto) to_cu(std::complex<double>* t){return reinterpret_cast<cuDoubleComplex*>(t);}	
public:
	template<class... Args>
	static auto gemm(Args&&... args){return cublasZgemm(to_cu(std::forward<Args>(args))...);}
};

template<>
class cublas<std::complex<float>>{
	template<class T> static decltype(auto) to_cu(T&& t){return std::forward<T>(t);}
	static decltype(auto) to_cu(std::complex<float> const* t){return reinterpret_cast<cuComplex const*>(t);}	
	static decltype(auto) to_cu(std::complex<float>* t){return reinterpret_cast<cuComplex*>(t);}	
public:
	template<class... Args>
	static auto gemm(Args&&... args){return cublasCgemm(to_cu(std::forward<Args>(args))...);}
};

template<typename T, typename AA, typename BB, class C, typename S>
void gemm(C transA, C transB, S m, S n, S k, AA a, multi::memory::cuda::ptr<T const> A, S lda, multi::memory::cuda::ptr<T const> B, S ldb, BB beta, multi::memory::cuda::ptr<T> CC, S ldc){
	cublasHandle_t handle;
	{cublasStatus_t s = cublasCreate(&handle); assert(s==CUBLAS_STATUS_SUCCESS);}
	cublasOperation_t cutransA = [transA](){
		switch(transA){
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
		} assert(0); return CUBLAS_OP_N;
	}();
	cublasOperation_t cutransB = [transB](){
		switch(transB){
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
		} assert(0); return CUBLAS_OP_N;
	}();
	T Talpha{a};
	T Tbeta{beta};
	cublasStatus_t s = cublas<T>::gemm(handle, cutransA, cutransB, m, n, k, &Talpha, static_cast<T const*>(A), lda, static_cast<T const*>(B), ldb, &Tbeta, static_cast<T*>(CC), ldc);
	if(s!=CUBLAS_STATUS_SUCCESS){
		std::cerr << [&](){switch(s){
			case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
			case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
		} return "<unknown>";}() << std::endl;
	}
	assert( s==CUBLAS_STATUS_SUCCESS ); (void)s;
	cublasDestroy(handle);
}

template<typename T, typename AA, typename BB, class C, class S>
void gemm(C transA, C transB, S m, S n, S k, AA const& a, multi::memory::cuda::managed::ptr<T const> A, S lda, multi::memory::cuda::managed::ptr<T const> B, S ldb, BB const& beta, multi::memory::cuda::managed::ptr<T> CC, S ldc){
	gemm(transA, transB, m, n, k, a, boost::multi::memory::cuda::ptr<T const>(A), lda, boost::multi::memory::cuda::ptr<T const>(B), ldb, beta, boost::multi::memory::cuda::ptr<T>(CC), ldc);
}

}}}

#if 0
#define xgemm(T) template<class C, class S> v gemm(C transA, C transB, S m, S n, S k, T const& a, T const* A, S lda, T const* B, S ldb, T const& beta, T* CC, S ldc){BLAS(T##gemm)(transA, transB, BC(m), BC(n), BC(k), a, A, BC(lda), B, BC(ldb), beta, CC, BC(ldc));}
#define xsyrk(T) template<class UL, class C, class S> v syrk(UL ul, C transA, S n, S k, T alpha, T const* A, S lda, T beta, T* CC, S ldc){BLAS(T##syrk)(ul, transA, BC(n), BC(k), alpha, A, BC(lda), beta, CC, BC(ldc));}
#define xherk(T) template<class UL, class C, class S, class Real> v herk(UL ul, C transA, S n, S k, Real alpha, T const* A, S lda, Real beta, T* CC, S ldc){BLAS(T##herk)(ul, transA, BC(n), BC(k), alpha, A, BC(lda), beta, CC, BC(ldc));}
#define xtrsm(T) template<class C, class UL, class Di, class S> v trsm(C side, UL ul, C transA, Di di, S m, S n, T alpha, T const* A, S lda, T* B, S ldb){BLAS(T##trsm)(side, ul, transA, di, BC(m), BC(n), alpha, A, lda, B, ldb);}

xgemm(s) xgemm(d) xgemm(c) xgemm(z)
xsyrk(s) xsyrk(d) xsyrk(c) xsyrk(z)
                  xherk(c) xherk(z)
xtrsm(s) xtrsm(d) xtrsm(c) xtrsm(z)

#undef xgemm
#undef xsyrk
#undef xherk
#undef xtrsm

}}}
#endif

///////////////////////////////////////////////////////////////////////////////

#if _TEST_MULTI_ADAPTORS_BLAS_CORE

#include "../../array.hpp"
#include "../../utility.hpp"

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

using std::cout;
namespace multi = boost::multi;

int main(){}

#endif
#endif

