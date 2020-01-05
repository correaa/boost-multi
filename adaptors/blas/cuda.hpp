#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_CUDA $0.cpp -o $0x `pkg-config --libs blas` -lcudart -lcublas&&$0x&&rm $0x $0.cpp; exit
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

class cublas_context{
protected:
	cublasHandle_t h_;
public:
	cublas_context(){
		cublasStatus_t s = cublasCreate(&h_); assert(s==CUBLAS_STATUS_SUCCESS); (void)s;
	}
	int version() const{
		int ret;
		cublasStatus_t s = cublasGetVersion(h_, &ret); assert(s==CUBLAS_STATUS_SUCCESS); (void)s;
		return ret;
	}
	~cublas_context() noexcept{cublasDestroy(h_);}
	//set_stream https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream
	//get_stream https://docs.nvidia.com/cuda/cublas/index.html#cublasgetstream
	//get_pointer_mode https://docs.nvidia.com/cuda/cublas/index.html#cublasgetpointermode
	//set_pointer_mode https://docs.nvidia.com/cuda/cublas/index.html#cublasgetpointermode
};

template<class T> class cublas;

template<>
class cublas<float> : cublas_context{
public:
	template<class... Args>
	static auto gemm(Args... args){return cublasSgemm(args...);}
	template<class... Args>
	static auto scal(Args... args){return cublasSscal(args...);}
	template<class... Args>
	static auto syrk(Args&&... args){return cublasSsyrk(args...);}
	template<class... As> void copy (As... as){auto s=cublasScopy (h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void iamax(As... as){auto s=cublasIsamax(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void asum(As... as){auto s=cublasSasum(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void trsm(As... as){auto s=cublasStrsm(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void dot(As... as){auto s=cublasSdot(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

template<>
class cublas<double> : cublas_context{
public:
	template<class... As> void dot(As... as){auto s=cublasDdot(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
public:
	template<class... Args>
	static auto gemm(Args... args){return cublasDgemm(args...);}
	template<class... Args>
	static auto scal(Args... args){return cublasDscal(args...);}
	template<class... Args>
	static auto syrk(Args&&... args){return cublasDsyrk(args...);}
	template<class... As> void copy (As... as){auto s=cublasDcopy (h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void iamax(As... as){auto s=cublasIdamax(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void asum(As... as){auto s=cublasDasum(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void trsm(As... as){auto s=cublasDtrsm(h_, as...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

template<>
class cublas<std::complex<float>> : cublas_context{
	static_assert(sizeof(std::complex<float>)==sizeof(cuComplex), "!");
	template<class T> static decltype(auto) to_cu(T&& t){return std::forward<T>(t);}
	static decltype(auto) to_cu(std::complex<float> const* t){return reinterpret_cast<cuComplex const*>(t);}	
	static decltype(auto) to_cu(std::complex<float>* t){return reinterpret_cast<cuComplex*>(t);}	
public:
	template<class... As> void dotu(As... as){auto s=cublasCdotu(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void dotc(As... as){auto s=cublasCdotc(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
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
	template<class... As> void copy (As... as){auto s=cublasCcopy (h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void iamax(As... as){auto s=cublasIcamax(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void asum(As... as){auto s=cublasScasum(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void trsm(As... as){auto s=cublasCtrsm(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

template<>
class cublas<std::complex<double>> : cublas_context{
	static_assert(sizeof(std::complex<double>)==sizeof(cuDoubleComplex), "!");
	template<class T> static decltype(auto) to_cu(T&& t){return std::forward<T>(t);}
	static decltype(auto) to_cu(std::complex<double> const* t){return reinterpret_cast<cuDoubleComplex const*>(t);}	
	static decltype(auto) to_cu(std::complex<double>* t){return reinterpret_cast<cuDoubleComplex*>(t);}	
public:
	template<class... As> void asum(As... as){auto s=cublasDzasum(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS); (void)s;}
	template<class... As> void dotu(As... as){auto s=cublasZdotu(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS); (void)s;}
	template<class... As> void dotc(As... as){auto s=cublasZdotc(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS); (void)s;}
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
	template<class... As> void copy (As... as){auto s=cublasZcopy (h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void iamax(As... as){auto s=cublasIzamax(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
	template<class... As> void trsm(As... as){auto s=cublasZtrsm(h_, to_cu(as)...); assert(s==CUBLAS_STATUS_SUCCESS);}
};

namespace memory{
namespace cuda{

template<class ComplexTconst, typename S>//, typename T = typename std::decay_t<ComplexTconst>::value_type>
auto asum(S n, cuda::ptr<ComplexTconst> x, S incx){
	decltype(std::abs(ComplexTconst{})) r; cublas<std::decay_t<ComplexTconst>>{}.asum(n, static_cast<ComplexTconst*>(x), incx, &r); return r;
}

template<class T, typename S>
S iamax(S n, cuda::ptr<T const> x, S incx){
	int r; cublas<T>{}.iamax(n, static_cast<T const*>(x), incx, &r); return r-1;
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

template<class X, class Y, class R, class S>
auto dot(S n, cuda::ptr<X> x, S incx, cuda::ptr<Y> y, S incy, cuda::ptr<R> result)
->decltype(cublas<R>{}.dot(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result))){
	return cublas<R>{}.dot(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result));}

template<class X, class Y, class R, class S>
auto dotc(S n, cuda::ptr<X> x, S incx, cuda::ptr<Y> y, S incy, cuda::ptr<R> result)
->decltype(cublas<R>{}.dotc(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result))){
	return cublas<R>{}.dotc(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result));}

template<class X, class Y, class R, class S>
auto dotu(S n, cuda::ptr<X> x, S incx, cuda::ptr<Y> y, S incy, cuda::ptr<R> result)
->decltype(cublas<R>{}.dotu(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result))){
	return cublas<R>{}.dotu(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result));}

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

template<typename TconstA, typename TconstB, typename T, typename AA, typename BB, class C, typename S>
void gemm(C transA, C transB, S m, S n, S k, AA a, multi::memory::cuda::ptr<TconstA> A, S lda, multi::memory::cuda::ptr<TconstB> B, S ldb, BB beta, multi::memory::cuda::ptr<T> CC, S ldc){
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

template<class Side, class Fill, class Trans, class Diag, typename Size, class Tconst, class T>
void trsm(Side /*cublasSideMode_t*/ side, /*cublasFillMode_t*/ Fill uplo, /*cublasOperation_t*/ Trans trans, /*cublasDiagType_t*/ Diag diag,
                           Size m, Size n, T alpha, cuda::ptr<Tconst> A, Size lda, cuda::ptr<T> B, Size ldb){
	cublasOperation_t trans_cu = [&]{
		switch(trans){
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
		} __builtin_unreachable();
	}();
	cublas<T>{}.trsm(
		side=='L'?CUBLAS_SIDE_LEFT:CUBLAS_SIDE_RIGHT, uplo=='L'?CUBLAS_FILL_MODE_LOWER:CUBLAS_FILL_MODE_UPPER, trans_cu, diag=='N'?CUBLAS_DIAG_NON_UNIT:CUBLAS_DIAG_UNIT, m, n, &alpha, static_cast<Tconst*>(A), lda, static_cast<T*>(B), ldb);
}

}}}}

namespace boost{namespace multi{namespace memory{namespace cuda{namespace managed{//namespace boost::multi::memory::cuda::managed{

template<class Tconst, typename S>
auto asum(S n, cuda::managed::ptr<Tconst> x, S incx){
	return asum(n, cuda::ptr<Tconst>(x), incx);
}

template<class T, typename S>
S iamax(S n, cuda::managed::ptr<T const> x, S incx){
	return cuda::iamax(n, cuda::ptr<T const>(x), incx);
}

template<class T, class TA, class S> 
void scal(S n, TA a, multi::memory::cuda::managed::ptr<T> x, S incx){
	scal(n, a, multi::memory::cuda::ptr<T>(x), incx);
}

template<class X, class Y, class R, class S>
auto dot(S n, cuda::managed::ptr<X> x, S incx, cuda::managed::ptr<Y> y, S incy, cuda::managed::ptr<R> result)
->decltype(cuda::dot(n, cuda_pointer_cast(x), incx, cuda_pointer_cast(y), incy, result)){
	return cuda::dot(n, cuda_pointer_cast(x), incx, cuda_pointer_cast(y), incy, result);}

template<class X, class Y, class R, class S>
auto dotu(S n, cuda::managed::ptr<X> x, S incx, cuda::managed::ptr<Y> y, S incy, cuda::managed::ptr<R> result)
->decltype(cuda::dotu(n, cuda_pointer_cast(x), incx, cuda_pointer_cast(y), incy, cuda_pointer_cast(result))){
	return cuda::dotu(n, cuda_pointer_cast(x), incx, cuda_pointer_cast(y), incy, cuda_pointer_cast(result));}

template<class X, class Y, class R, class S>
auto dotc(S n, cuda::managed::ptr<X> x, S incx, cuda::managed::ptr<Y> y, S incy, cuda::managed::ptr<R> result)
->decltype(cuda::dotc(n, cuda_pointer_cast(x), incx, cuda_pointer_cast(y), incy, cuda_pointer_cast(result))){
	return cuda::dotc(n, cuda_pointer_cast(x), incx, cuda_pointer_cast(y), incy, cuda_pointer_cast(result));}

template<typename AA, typename BB, class S, class TconstA, class TconstB, class T>
void gemm(char transA, char transB, S m, S n, S k, AA const& a, multi::memory::cuda::managed::ptr<TconstA> A, S lda, multi::memory::cuda::managed::ptr<TconstB> B, S ldb, BB const& beta, multi::memory::cuda::managed::ptr<T> CC, S ldc){
	gemm(transA, transB, m, n, k, a, boost::multi::memory::cuda::ptr<TconstA>(A), lda, boost::multi::memory::cuda::ptr<TconstB>(B), ldb, beta, boost::multi::memory::cuda::ptr<T>(CC), ldc);
}

template<class Tconst, class T, class UL, class C, class S, class Real>
void herk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::managed::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::managed::ptr<T> CC, S ldc){
	herk(ul, transA, n, k, alpha, boost::multi::memory::cuda::ptr<Tconst>(A), lda, beta, boost::multi::memory::cuda::ptr<T>(CC), ldc);
}

template<class Tconst, class T, class UL, class C, class S, class Real>
void syrk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::managed::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::managed::ptr<T> CC, S ldc){
	syrk(ul, transA, n, k, alpha, boost::multi::memory::cuda::ptr<Tconst>(A), lda, beta, boost::multi::memory::cuda::ptr<T>(CC), ldc);
}

template<class Side, class Fill, class Trans, class Diag, typename Size, class Tconst, class T>
void trsm(Side /*cublasSideMode_t*/ side, /*cublasFillMode_t*/ Fill uplo, /*cublasOperation_t*/ Trans trans, /*cublasDiagType_t*/ Diag diag,
                           Size m, Size n, T alpha, cuda::managed::ptr<Tconst> A, Size lda, cuda::managed::ptr<T> B, Size ldb){
	return trsm(side, uplo, trans, diag, m, n, alpha, cuda::ptr<Tconst>(A), lda, cuda::ptr<T>(B), ldb);
}

}}}}}

///////////////////////////////////////////////////////////////////////////////

#if _TEST_MULTI_ADAPTORS_BLAS_CUDA

#include "../../array.hpp"
#include "../../utility.hpp"
#include<cassert>

namespace multi = boost::multi;

int main(){
	multi::cublas_context c;
	assert( c.version() >= 10100 );
}

#endif
#endif

