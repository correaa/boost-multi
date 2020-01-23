#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_ADAPTORS_BLAS_CUDA $0.cpp -o $0x `pkg-config --libs blas` -lcudart -lcublas&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_CUDA_HPP
#define MULTI_ADAPTORS_BLAS_CUDA_HPP

//#ifdef MULTI_ADAPTORS_BLAS_GEMM_HPP
//#error "blas/cuda.hpp must be included before blas/gemm.hpp"
//#endif
//#include "../../utility.hpp"
//#include "../../array.hpp" // allocating multi::arrays for output
#include "../../memory/adaptors/cuda/ptr.hpp"
#include "../../memory/adaptors/cuda/managed/ptr.hpp"
#include "../../memory/adaptors/cuda/managed/allocator.hpp"

#include<cublas_v2.h>

//#include<iostream> // debug

//#include <boost/log/trivial.hpp>

#include<complex>
//#include<memory>

///////////////////
#include<system_error>
namespace boost{
namespace multi{

enum class cublas_error : typename std::underlying_type<cublasStatus_t>::type{
	success               = CUBLAS_STATUS_SUCCESS,
	not_initialized       = CUBLAS_STATUS_NOT_INITIALIZED,
	allocation_failed     = CUBLAS_STATUS_ALLOC_FAILED,
	invalid_value         = CUBLAS_STATUS_INVALID_VALUE,
	architecture_mismatch = CUBLAS_STATUS_ARCH_MISMATCH,
	mapping_error         = CUBLAS_STATUS_MAPPING_ERROR,
	execution_failed      = CUBLAS_STATUS_EXECUTION_FAILED,
	internal_error        = CUBLAS_STATUS_INTERNAL_ERROR,
	not_supported         = CUBLAS_STATUS_NOT_SUPPORTED,
	license_error         = CUBLAS_STATUS_LICENSE_ERROR
};
std::string inline cublas_string(enum cublas_error err){ //https://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas
	switch(err){
		case cublas_error::success              : return "CUBLAS_STATUS_SUCCESS";
		case cublas_error::not_initialized      : return "CUBLAS_STATUS_NOT_INITIALIZED";
		case cublas_error::allocation_failed    : return "CUBLAS_STATUS_ALLOC_FAILED";
		case cublas_error::invalid_value        : return "CUBLAS_STATUS_INVALID_VALUE";
		case cublas_error::architecture_mismatch: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case cublas_error::mapping_error        : return "CUBLAS_STATUS_MAPPING_ERROR";
		case cublas_error::execution_failed     : return "CUBLAS_STATUS_EXECUTION_FAILED";
		case cublas_error::internal_error       : return "CUBLAS_STATUS_INTERNAL_ERROR";
		case cublas_error::not_supported        : return "CUBLAS_STATUS_NOT_SUPPORTED";
		case cublas_error::license_error        : return "CUBLAS_STATUS_LICENSE_ERROR";
	}
	return "cublas status <unknown>";
}
struct cublas_error_category : std::error_category{
	char const* name() const noexcept override{return "cublas wrapper";}
	std::string message(int err) const override{return cublas_string(static_cast<enum cublas_error>(err));}
	static error_category& instance(){static cublas_error_category instance; return instance;}
};
inline std::error_code make_error_code(cublas_error err) noexcept{
	return std::error_code(int(err), cublas_error_category::instance());
}

template<class CublasFunction>
auto cublas_call(CublasFunction f){
	return [=](auto... args){
		auto s = (enum cublas_error)(f(args...));
		if( s != cublas_error::success ) throw std::system_error{make_error_code(s), "cannot call cublas function "};
	};
}

#define CUBLAS_(FunctionPostfix) boost::multi::cublas_call(cublas##FunctionPostfix)

}}

namespace std{template<> struct is_error_code_enum<::boost::multi::cublas_error> : true_type{};}
////////////


namespace boost{
namespace multi{


using v = void;
using s = float;
using d = double;
using c = cuComplex;
using z = cuDoubleComplex;

template<class T = void> struct cublas1{};
template<class T = void> struct cublas3{};

template<> struct cublas1<s>{
	template<class...A> static auto iamax(A...a){return cublasIsamax(a...);}
	template<class...A> static auto asum (A...a){return cublasSasum(a...);}
	template<class...A> static auto copy (A...a){return cublasScopy(a...);}
	template<class...A> static auto dot  (A...a){return cublasSdot(a...);}	
	template<class...A> static auto dotu (A...a){return cublasSdot(a...);}	
	template<class...A> static auto dotc (A...a){return cublasSdot(a...);}	
	template<class...A> static auto scal (A...a){return cublasSscal(a...);}
};
template<> struct cublas1<d>{
	template<class...A> static auto iamax(A...a){return cublasIdamax(a...);}
	template<class...A> static auto asum (A...a){return cublasDasum(a...);}
	template<class...A> static auto copy (A...a){return cublasDcopy(a...);}
	template<class...A> static auto dot  (A...a){return cublasDdot(a...);}
	template<class...A> static auto dotu (A...a){return cublasDdotu(a...);}	
	template<class...A> static auto dotc (A...a){return cublasDdotc(a...);}	
	template<class...A> static auto scal (A...a){return cublasDscal(a...);}
};
template<> struct cublas1<c>{
	template<class...A> static auto iamax(A...a){return cublasIcamax(a...);}
	template<class...A> static auto asum (A...a){return cublasCasum(a...);}
	template<class...A> static auto copy (A...a){return cublasCcopy(a...);}
	template<class...A> static auto dot  (A...a){return cublasCdot(a...);}
	template<class...A> static auto dotu (A...a){return cublasCdotu(a...);}	
	template<class...A> static auto dotc (A...a){return cublasCdotc(a...);}	
	template<class...A> static auto scal (A...a){return cublasCscal(a...);}
};
template<> struct cublas1<z>{
	template<class...A> static auto iamax(A...a){return cublasIzamax(a...);}
	template<class...A> static auto asum (A...a){return cublasZasum(a...);}
	template<class...A> static auto copy (A...a){return cublasZcopy(a...);}
	template<class...A> static auto dot  (A...a){return cublasZdot(a...);}
	template<class...A> static auto dotu (A...a){return cublasZdotu(a...);}	
	template<class...A> static auto dotc (A...a){return cublasZdotc(a...);}	
	template<class...A> static auto scal (A...a){return cublasZscal(a...);}
};

template<> struct cublas1<void>{
// 2.5.1. cublasI<t>amax() https://docs.nvidia.com/cuda/cublas/index.html#cublasi-lt-t-gt-amax
	template<class T> static cublasStatus_t iamax(cublasHandle_t handle, int n, const T* x, int incx, int *result){return cublas1<T>::iamax(handle, n, x, incx, result);}
// 2.5.3. cublas<t>asum() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-asum
	template<class T> static cublasStatus_t asum(cublasHandle_t handle, int n, const T* x, int incx, T* result){return cublas1<T>::asum(handle, n, x, incx, result);}
// 2.5.5. cublas<t>copy() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-copy
	template<class T> static cublasStatus_t copy(cublasHandle_t handle, int n, const T* x, int incx, T* y, int incy){return cublas1<T>::copy(handle, n, x, incx, y, incy);}
// 2.5.6. cublas<t>dot() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot
	template<class T> static cublasStatus_t dot(cublasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result){return cublas1<T>::dot(handle, n, x, incx, y, incy, result);}
	template<class T> static auto dotu(cublasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result)
	->decltype(cublas1<T>::dotu(handle, n, x, incx, y, incy, result)){
		return cublas1<T>::dotu(handle, n, x, incx, y, incy, result);}
	template<class T> static auto dotc(cublasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result)
	->decltype(cublas1<T>::dotc(handle, n, x, incx, y, incy, result)){
		return cublas1<T>::dotc(handle, n, x, incx, y, incy, result);}
// 2.5.12. cublas<t>scal()	https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-scale
	template<class T> static cublasStatus_t scal(cublasHandle_t handle, int n, const T* alpha, T* x, int incx){return cublas1<T>::scal(handle, n, alpha, x, incx);}
};

template<> struct cublas3<void>{
// 2.7.1. cublas<t>gemm() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
	template<class T> static cublasStatus_t gemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const T           *alpha,
                           const T           *A, int lda,
                           const T           *B, int ldb,
                           const T           *beta,
                           T           *C, int ldc){return cublas3<T>::gemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta);}
// 2.7.6. cublas<t>syrk() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-syrk
	template<class T> static cublasStatus_t syrk(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const T           *alpha,
                           const T           *A, int lda,
                           const T           *beta,
                           T           *C, int ldc){return cublas3<T>::syrk(handle, uplo, trans, n, k, alpha, A, lda, beta);}
// 2.7.13. cublas<t>herk() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-herk
	template<class T> static cublasStatus_t herk(cublasHandle_t handle,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           int n, int k,
                           const float  *alpha,
                           const T       *A, int lda,
                           const float  *beta,
                           cuComplex       *C, int ldc){return cublas3<T>::herk(handle, uplo, trans, n, k, alpha, A, lda, beta);}
// 2.7.10. cublas<t>trsm() https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-trsm
	template<class T> static cublasStatus_t trsm(cublasHandle_t handle,
                           cublasSideMode_t side, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           float           *B, int ldb){return cublas3<T>::trsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);}
};

template<> struct cublas3<s>{
	template<class...A> static auto syrk (A...a){return cublasSsyrk(a...);}
};
template<> struct cublas3<d>{
	template<class...A> static auto syrk (A...a){return cublasDsyrk(a...);}
};
template<> struct cublas3<c>{
	template<class...A> static auto syrk (A...a){return cublasCsyrk(a...);}
	template<class...A> static auto herk (A...a){return cublasCherk(a...);}
};
template<> struct cublas3<z>{
	template<class...A> static auto syrk (A...a){return cublasZsyrk(a...);}
	template<class...A> static auto herk (A...a){return cublasZherk(a...);}
};


namespace cublas{
struct context : std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)>{
	context()  : std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)>(
		[]{cublasHandle_t h; CUBLAS_(Create)(&h); return h;}(), &cublasDestroy
	){}
	int version() const{
		int ret; CUBLAS_(GetVersion)(get(), &ret); return ret;
	}
	~context() noexcept = default;
	//set_stream https://docs.nvidia.com/cuda/cublas/index.html#cublassetstream
	//get_stream https://docs.nvidia.com/cuda/cublas/index.html#cublasgetstream
	//get_pointer_mode https://docs.nvidia.com/cuda/cublas/index.html#cublasgetpointermode
	//set_pointer_mode https://docs.nvidia.com/cuda/cublas/index.html#cublasgetpointermode
	template<class...A> auto iamax(A...a) const{return cublas1<>::iamax(get(), a...);}
	template<class...A> auto asum (A...a) const{return cublas1<>::asum(get(), a...);}
	template<class...A> auto scal (A...a) const{return cublas1<>::scal(get(), a...);}
	template<class...A> auto dot  (A...a) const{return cublas1<>::dot(get(), a...);}
	template<class...A> auto dotu (A...a) const->decltype(cublas1<>::dotu(get(), a...)){return cublas1<>::dotu(get(), a...);}
	template<class...A> auto dotc (A...a) const->decltype(cublas1<>::dotc(get(), a...)){return cublas1<>::dotc(get(), a...);}
	template<class...A> auto copy (A...a) const{return cublas1<>::copy(get(), a...);}

	template<class...A> auto gemm (A...a) const{return cublas3<>::gemm(get(), a...);}
	template<class...A> auto syrk (A...a) const{return cublas3<>::syrk(get(), a...);}
	template<class...A> auto herk (A...a) const{return cublas3<>::herk(get(), a...);}
	template<class...A> auto trsm (A...a) const{return cublas3<>::trsm(get(), a...);}
};
}

}}

namespace boost{
namespace multi{

namespace memory{
namespace cuda{

template<class T> static decltype(auto) translate(T&& t){return std::forward<T>(t);}
static decltype(auto) translate(std::complex<double> const* t){return reinterpret_cast<cuDoubleComplex const*>(t);}	
static decltype(auto) translate(std::complex<double>* t){return reinterpret_cast<cuDoubleComplex*>(t);}	

template<class Tconst, typename S>
S iamax(S n, cuda::ptr<Tconst> x, S incx){
	int r; cublas::context{}.iamax(n, translate(raw_pointer_cast(x)), incx, &r); return r-1;
}

template<class ComplexTconst, typename S>//, typename T = typename std::decay_t<ComplexTconst>::value_type>
auto asum(S n, cuda::ptr<ComplexTconst> x, S incx){
	decltype(std::abs(ComplexTconst{})) r;
	cublas::context{}.asum(n, raw_pointer_cast(x), incx, &r);
	return r;
}

template<class Tconst, class T, class S> 
void copy(S n, cuda::ptr<Tconst> x, S incx, cuda::ptr<T> y, S incy){
	cublas::context{}.copy(n, translate(raw_pointer_cast(x)), incx, translate(raw_pointer_cast(y)), incy);
}

template<class T, class TA, class S> 
void scal(S n, TA a, multi::memory::cuda::ptr<T> x, S incx){
	T aa{a};
	cublas::context{}.scal(n, translate(&aa), translate(raw_pointer_cast(x)), incx);
}

template<class X, class Y, class R, class S>
auto dot(S n, cuda::ptr<X> x, S incx, cuda::ptr<Y> y, S incy, cuda::ptr<R> result)
->decltype(cublas::context{}.dot(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result))){
	return cublas::context{}.dot(n, static_cast<X*>(x), incx, static_cast<Y*>(y), incy, static_cast<R*>(result));}

template<class X, class Y, class R, class S>
auto dotu(S n, cuda::ptr<X> x, S incx, cuda::ptr<Y> y, S incy, cuda::ptr<R> result)
->decltype(cublas::context{}.dotu(n, translate(raw_pointer_cast(x)), incx, translate(raw_pointer_cast(y)), incy, translate(raw_pointer_cast(result)))){
	return cublas::context{}.dotu(n, translate(raw_pointer_cast(x)), incx, translate(raw_pointer_cast(y)), incy, translate(raw_pointer_cast(result)));}

template<class X, class Y, class R, class S>
auto dotc(S n, cuda::ptr<X> x, S incx, cuda::ptr<Y> y, S incy, cuda::ptr<R> result)
->decltype(cublas::context{}.dotc(n, translate(raw_pointer_cast(x)), incx, translate(raw_pointer_cast(y)), incy, translate(raw_pointer_cast(result)))){
	return cublas::context{}.dotc(n, translate(raw_pointer_cast(x)), incx, translate(raw_pointer_cast(y)), incy, translate(raw_pointer_cast(result)));}

template<class Tconst, class T, class UL, class C, class S, class Real>
void syrk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::ptr<T> CC, S ldc){
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
	cublasStatus_t s = cublas::context{}.syrk(uplo, cutransA, n, k, &alpha, static_cast<T const*>(A), lda, &beta, static_cast<T*>(CC), ldc);
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
}

template<class Tconst, class T, class UL, class C, class S, class Real>
void herk(UL ul, C transA, S n, S k, Real alpha, multi::memory::cuda::ptr<Tconst> A, S lda, Real beta, multi::memory::cuda::ptr<T> CC, S ldc){
//	BOOST_LOG_TRIVIAL(trace) <<"cublas::herk called on size/stride " <<n <<" "<< lda;
//	cublasHandle_t handle;
//	{cublasStatus_t s = cublasCreate(&handle); assert(s==CUBLAS_STATUS_SUCCESS);}
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
	cublasStatus_t s = cublas::context{}.herk(uplo, cutransA, n, k, &alpha, static_cast<T const*>(A), lda, &beta, static_cast<T*>(CC), ldc);
//https://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas
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
//	cublasDestroy(handle);
}

template<typename TconstA, typename TconstB, typename T, typename AA, typename BB, class C, typename S>
void gemm(C transA, C transB, S m, S n, S k, AA a, multi::memory::cuda::ptr<TconstA> A, S lda, multi::memory::cuda::ptr<TconstB> B, S ldb, BB beta, multi::memory::cuda::ptr<T> CC, S ldc){
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
	cublasStatus_t s = cublas::context{}.gemm(cutransA, cutransB, m, n, k, &Talpha, static_cast<T const*>(A), lda, static_cast<T const*>(B), ldb, &Tbeta, static_cast<T*>(CC), ldc);
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
}

template<class Side, class Fill, class Trans, class Diag, typename Size, class Tconst, class T, class Alpha>
void trsm(Side /*cublasSideMode_t*/ side, /*cublasFillMode_t*/ Fill uplo, /*cublasOperation_t*/ Trans trans, /*cublasDiagType_t*/ Diag diag,
                           Size m, Size n, Alpha alpha, cuda::ptr<Tconst> A, Size lda, cuda::ptr<T> B, Size ldb){
	cublasOperation_t trans_cu = [&]{
		switch(trans){
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
		} __builtin_unreachable();
	}();
	T alpha_{alpha};
	cublas::context{}.trsm(
		side=='L'?CUBLAS_SIDE_LEFT:CUBLAS_SIDE_RIGHT, uplo=='L'?CUBLAS_FILL_MODE_LOWER:CUBLAS_FILL_MODE_UPPER, trans_cu, diag=='N'?CUBLAS_DIAG_NON_UNIT:CUBLAS_DIAG_UNIT, m, n, &alpha_, static_cast<Tconst*>(A), lda, static_cast<T*>(B), ldb);
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
	multi::cublas::context c;
	assert( c.version() >= 10100 );
}

#endif
#endif

