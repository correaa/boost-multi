#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_CUDA $0.cpp -o $0x `pkg-config --libs blas` -lcudart&&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_CUDA_HPP
#define MULTI_ADAPTORS_BLAS_CUDA_HPP

//#ifdef MULTI_ADAPTORS_BLAS_GEMM_HPP
//#error "blas/cuda.hpp must be included before blas/gemm.hpp"
//#endif
//#include "../../utility.hpp"
//#include "../../array.hpp" // allocating multi::arrays for output
#include "../../memory/adaptors/cuda/ptr.hpp"
#include "../../memory/adaptors/cuda/managed/ptr.hpp"

#include<cublas_v2.h>

#include<iostream> // debug

#include<complex>
#include<memory>

namespace boost{
namespace multi{
//namespace blas{

struct cublas_context{
	cublasHandle_t h_;
	cublas_context(){
		{cublasStatus_t s = cublasCreate(&h_); assert(s==CUBLAS_STATUS_SUCCESS);}
	}
	int version() const{
		int ret;
		{cublasStatus_t s = cublasGetVersion(h_, &ret); assert(s==CUBLAS_STATUS_SUCCESS);}
		return ret;
	}
	//set_stream https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream
	//get_stream https://docs.nvidia.com/cuda/cublas/index.html#cublasgetstream
	//get_pointer_mode https://docs.nvidia.com/cuda/cublas/index.html#cublasgetpointermode
	//set_pointer_mode https://docs.nvidia.com/cuda/cublas/index.html#cublasgetpointermode
	~cublas_context() noexcept{cublasDestroy(h_);}
};

template<class T> class cublas;

template<>
class cublas<double> : cublas_context{
public:
	template<class... Args>
	static auto gemm(Args... args){return cublasDgemm(args...);}
	template<class... Args>
	static auto scal(Args... args){return cublasDscal(args...);}
	template<class... Args>
	static auto syrk(Args&&... args){return cublasDsyrk(args...);}
	template<class... Args> void copy(Args... args){
		cublasStatus_t s = cublasDcopy(h_, args...); assert(s==CUBLAS_STATUS_SUCCESS);
	}
	template<class... Args> 
	void iamax(Args... args){auto s = cublasIdamax(h_, args...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

template<>
class cublas<float> : cublas_context{
public:
	template<class... Args>
	static auto gemm(Args... args){return cublasSgemm(args...);}
	template<class... Args>
	static auto scal(Args... args){return cublasSscal(args...);}
	template<class... Args>
	static auto syrk(Args&&... args){return cublasSsyrk(args...);}
	template<class... Args> void copy(Args... args){
		cublasStatus_t s = cublasScopy(h_, args...); assert(s==CUBLAS_STATUS_SUCCESS);
	}
	template<class... Args> 
	void iamax(Args... as){auto s=cublasIsamax(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

template<>
class cublas<std::complex<double>> : cublas_context{
	static_assert(sizeof(std::complex<double>)==sizeof(cuDoubleComplex), "!");
	template<class T> static decltype(auto) to_cu(T&& t){return std::forward<T>(t);}
	static decltype(auto) to_cu(std::complex<double> const* t){return reinterpret_cast<cuDoubleComplex const*>(t);}	
	static decltype(auto) to_cu(std::complex<double>* t){return reinterpret_cast<cuDoubleComplex*>(t);}	
public:
	template<class... Args>
	static auto gemm(Args&&... args){return cublasZgemm(to_cu(std::forward<Args>(args))...);}
	template<class... Args>
	static auto herk(Args&&... args){return cublasZherk(to_cu(std::forward<Args>(args))...);}
	template<class... Args>
	static auto scal(Args&&... args)
	->decltype(cublasZscal(to_cu(std::forward<Args>(args))...)){
		return cublasZscal(to_cu(std::forward<Args>(args))...);}
	template<class Handle, class Size, class... Args2>
	static auto scal(Handle h, Size s, double* alpha, Args2&&... args2){
		return cublasZdscal(h, s, alpha, to_cu(std::forward<Args2>(args2))...);
	}
	template<class... Args> void copy(Args... args){
		auto s = cublasZcopy(h_, to_cu(args)...); assert(s==CUBLAS_STATUS_SUCCESS);
	}
	template<class... Args> 
	void iamax(Args... as){auto s=cublasIzamax(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

template<>
class cublas<std::complex<float>> : cublas_context{
	static_assert(sizeof(std::complex<float>)==sizeof(cuComplex), "!");
	template<class T> static decltype(auto) to_cu(T&& t){return std::forward<T>(t);}
	static decltype(auto) to_cu(std::complex<float> const* t){return reinterpret_cast<cuComplex const*>(t);}	
	static decltype(auto) to_cu(std::complex<float>* t){return reinterpret_cast<cuComplex*>(t);}	
public:
	template<class... Args>
	static auto gemm(Args&&... args){return cublasCgemm(to_cu(std::forward<Args>(args))...);}
	template<class... Args>
	static auto herk(Args&&... args){return cublasCherk(to_cu(std::forward<Args>(args))...);}
	template<class... Args>
	static auto scal(Args&&... args)
	->decltype(cublasZscal(to_cu(std::forward<Args>(args))...)){
		return cublasZscal(to_cu(std::forward<Args>(args))...);}
	template<class Handle, class Size, class... Args2>
	static auto scal(Handle h, Size s, float* alpha, Args2&&... args2){
		return cublasZdscal(h, s, alpha, to_cu(std::forward<Args2>(args2))...);
	}
	template<class... Args> void copy(Args... args){
		cublasStatus_t s = cublasCcopy(h_, to_cu(args)...); assert(s==CUBLAS_STATUS_SUCCESS);
	}
	template<class... Args> 
	void iamax(Args... as){auto s=cublasIcamax(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

namespace memory{
namespace cuda{

template<class T, typename S>
S iamax(S n, cuda::ptr<T const> x, S incx){
	int ret = n;
	cublas<T>{}.iamax(n, static_cast<T const*>(x), incx, &ret);
	return ret - 1;
}

template<class T, class TA, class S> 
void scal(S n, TA a, multi::memory::cuda::ptr<T> x, S incx){
	cublasHandle_t handle;
	{cublasStatus_t s = cublasCreate(&handle); assert(s==CUBLAS_STATUS_SUCCESS);}
	cublasStatus_t s = cublas<T>::scal(handle, n, &a, static_cast<T*>(x), incx);
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

template<class Tconst, class T, class S> 
void copy(S n, cuda::ptr<Tconst> x, S incx, cuda::ptr<T> y, S incy){
	cublas<T>{}.copy(n, static_cast<T const*>(x), incx, static_cast<T*>(y), incy);
}

//template<class T, class S> 
//void copy(S n, multi::memory::cuda::ptr<T const> x, S incx, multi::memory::cuda::ptr<T> y, S incy){
//}

template<class Tconst, class T, class UL, class C, class S, class Real>
void syrk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::ptr<T> CC, S ldc){
	cublasHandle_t handle;
	{cublasStatus_t s = cublasCreate(&handle); assert(s==CUBLAS_STATUS_SUCCESS);}
	cublasFillMode_t uplo = [ul](){
		switch(ul){
			case 'U': return CUBLAS_FILL_MODE_UPPER;
			case 'L': return CUBLAS_FILL_MODE_LOWER;
		} assert(0); return CUBLAS_FILL_MODE_UPPER;
	}();
	cublasOperation_t cutransA = [transA](){
		switch(transA){
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
		} assert(0); return CUBLAS_OP_N;
	}();
	cublasStatus_t s = cublas<T>::syrk(handle, uplo, cutransA, n, k, &alpha, static_cast<T const*>(A), lda, &beta, static_cast<T*>(CC), ldc);
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

template<class Tconst, class T, class UL, class C, class S, class Real>
void herk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::ptr<T> CC, S ldc){
	cublasHandle_t handle;
	{cublasStatus_t s = cublasCreate(&handle); assert(s==CUBLAS_STATUS_SUCCESS);}
	cublasFillMode_t uplo = [ul](){
		switch(ul){
			case 'U': return CUBLAS_FILL_MODE_UPPER;
			case 'L': return CUBLAS_FILL_MODE_LOWER;
		} assert(0); return CUBLAS_FILL_MODE_UPPER;
	}();
	cublasOperation_t cutransA = [transA](){
		switch(transA){
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
		} assert(0); return CUBLAS_OP_N;
	}();
	cublasStatus_t s = cublas<T>::herk(handle, uplo, cutransA, n, k, &alpha, static_cast<T const*>(A), lda, &beta, static_cast<T*>(CC), ldc);
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


}}}}

namespace boost{namespace multi{namespace memory{namespace cuda{namespace managed{//namespace boost::multi::memory::cuda::managed{

template<class T, typename S>
S iamax(S n, cuda::managed::ptr<T const> x, S incx){
	return cuda::iamax(n, cuda::ptr<T const>(x), incx);
}

template<class T, class TA, class S> 
void scal(S n, TA a, multi::memory::cuda::managed::ptr<T> x, S incx){
	scal(n, a, multi::memory::cuda::ptr<T>(x), incx);
}

template<typename AA, typename BB, class S>
void gemm(char transA, char transB, S m, S n, S k, AA const& a, multi::memory::cuda::managed::ptr<double const> A, S lda, multi::memory::cuda::managed::ptr<double const> B, S ldb, BB const& beta, multi::memory::cuda::managed::ptr<double> CC, S ldc){
	gemm(transA, transB, m, n, k, a, boost::multi::memory::cuda::ptr<double const>(A), lda, boost::multi::memory::cuda::ptr<double const>(B), ldb, beta, boost::multi::memory::cuda::ptr<double>(CC), ldc);
}

template<class Tconst, class T, class UL, class C, class S, class Real>
void herk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::managed::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::managed::ptr<T> CC, S ldc){
	herk(ul, transA, n, k, alpha, boost::multi::memory::cuda::ptr<Tconst>(A), lda, beta, boost::multi::memory::cuda::ptr<T>(CC), ldc);
}

template<class Tconst, class T, class UL, class C, class S, class Real>
void syrk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::managed::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::managed::ptr<T> CC, S ldc){
	syrk(ul, transA, n, k, alpha, boost::multi::memory::cuda::ptr<Tconst>(A), lda, beta, boost::multi::memory::cuda::ptr<T>(CC), ldc);
}

}}}}}//}

///////////////////////////////////////////////////////////////////////////////

#if _TEST_MULTI_ADAPTORS_BLAS_CUDA

#include "../../array.hpp"
#include "../../utility.hpp"

namespace multi = boost::multi;

int main(){}

#endif
#endif

