// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2020

#ifndef MULTI_ADAPTORS_CUDA_CUBLAS_CONTEXT_HPP
#define MULTI_ADAPTORS_CUDA_CUBLAS_CONTEXT_HPP

#include "../../../adaptors/cuda/cublas/call.hpp"

#include "../../../adaptors/blas/traits.hpp"
#include "../../../adaptors/blas/core.hpp"

#include "../../../memory/adaptors/cuda/ptr.hpp"
#include "../../../memory/adaptors/cuda/managed/ptr.hpp"

#include<mutex>

namespace boost{
namespace multi::cuda::cublas{

class operation{
	cublasOperation_t impl_;
public:
	operation(char trans) : impl_{[=]{
		switch(trans){
		case 'N': return CUBLAS_OP_N;
		case 'T': return CUBLAS_OP_T;
		case 'C': return CUBLAS_OP_C;
		default : assert(0);
		}
		return cublasOperation_t{};
	}()}{}
	operator cublasOperation_t() const{return impl_;}
};

class side{
	cublasSideMode_t impl_;
public:
	side(char trans) : impl_{[=]{
		switch(trans){
		case 'L': return CUBLAS_SIDE_LEFT;
		case 'R': return CUBLAS_SIDE_RIGHT;
		}
		assert(0); return cublasSideMode_t{};
	}()}{}
	operator cublasSideMode_t() const{return impl_;}
};

class filling{
	cublasFillMode_t impl_;
public:
	filling(char trans) : impl_{[=]{
		switch(trans){
		case 'L': return CUBLAS_FILL_MODE_LOWER;
		case 'U': return CUBLAS_FILL_MODE_UPPER;
		}
		assert(0); return cublasFillMode_t{};
	}()}{}
	operator cublasFillMode_t() const{return impl_;}
};

class diagonal{
	cublasDiagType_t impl_;
public:
	diagonal(char trans) : impl_{[=]{
		switch(trans){
		case 'N': return CUBLAS_DIAG_NON_UNIT;
		case 'U': return CUBLAS_DIAG_UNIT;
		}
		assert(0); return cublasDiagType_t{};
	}()}{}
	operator cublasDiagType_t() const{return impl_;}
};

using blas::is_z;
using blas::is_d;
using std::is_assignable;
using std::is_convertible_v;

template<class Derived>
struct basic_context{
	using ssize_t = int;
	static int version(){int ret; cublas::call<cublasGetVersion>(nullptr, &ret); return ret;}
	template<class ALPHA, class AAP, class AA = typename std::pointer_traits<AAP>::element_type, class BBP, class BB = typename std::pointer_traits<BBP>::element_type, class BETA, class CCP, class CC = typename std::pointer_traits<CCP>::element_type,
		std::enable_if_t<
			is_z<AA>{} and is_z<BB>{} and is_z<CC>{} and is_assignable<CC&, decltype(ALPHA{}*AA{}*BB{})>{} and
			std::is_convertible_v<AAP, memory::cuda::ptr<AA>> and std::is_convertible_v<BBP, memory::cuda::ptr<BB>> and std::is_convertible_v<CCP, memory::cuda::ptr<CC>>
		,int> =0
	>
	void gemm(char transA, char transB, ssize_t m, ssize_t n, ssize_t k, ALPHA const* alpha, AAP aa, ssize_t lda, BBP bb, ssize_t ldb, BETA const* beta, CCP cc, ssize_t ldc){
		cublas::call<cublasZgemm>(static_cast<Derived&>(*this).get(), cublas::operation{transA}, cublas::operation{transB}, m, n, k, (cuDoubleComplex const*)alpha, (cuDoubleComplex const*)raw_pointer_cast(aa), lda, (cuDoubleComplex const*)raw_pointer_cast(bb), ldb, (cuDoubleComplex const*)beta, (cuDoubleComplex*)raw_pointer_cast(cc), ldc);
	}
	template<class ALPHA, class AAP, class AA = typename pointer_traits<AAP>::element_type, class BBP, class BB = typename pointer_traits<BBP>::element_type,
		std::enable_if_t<
			is_z<AA>{} and is_z<BB>{} and is_assignable<BB&, decltype(AA{}*BB{}/ALPHA{})>{} and is_assignable<BB&, decltype(ALPHA{}*BB{}/AA{})>{} and 
			is_convertible_v<AAP, memory::cuda::ptr<AA>> and is_convertible_v<BBP, memory::cuda::ptr<BB>>
		,int> =0
	>
	void trsm(char side, char ul, char transA, char diag, ssize_t m, ssize_t n, ALPHA alpha, AAP aa, ssize_t lda, BBP bb, ssize_t ldb){
		cublas::call<cublasZtrsm>(static_cast<Derived&>(*this).get(), cublas::side{side}, cublas::filling{ul}, cublas::operation{transA}, cublas::diagonal{diag}, m, n, (cuDoubleComplex const*)&alpha, (cuDoubleComplex const*)raw_pointer_cast(aa), lda, (cuDoubleComplex*)raw_pointer_cast(bb), ldb);
	}
};

class unsynchronized_context : std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)>, public basic_context<unsynchronized_context>{
	using pimpl_ = std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)>;
public:
	using std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)>::get;
	unsynchronized_context() : pimpl_{[]{cublasHandle_t h; cublasCreate(&h); return h;}(), &cublasDestroy}{}
	static int version(){int ret; cublas::call<cublasGetVersion>(nullptr, &ret); return ret;}
};

struct context{
	static unsynchronized_context& get_instance(){
		thread_local unsynchronized_context ctxt;
		return ctxt;
	};
	static int version(){int ret; cublas::call<cublasGetVersion>(nullptr, &ret); return ret;}
	template<class... Args> auto gemm(Args... args) const
//	->decltype(get_instance().gemm(args...))
	{	return get_instance().gemm(args...);}
};

}
}

namespace boost::multi::blas{

	template<> struct is_context<boost::multi::cuda::cublas::unsynchronized_context > : std::true_type{};
	template<> struct is_context<boost::multi::cuda::cublas::unsynchronized_context&> : std::true_type{};

	template<> struct is_context<boost::multi::cuda::cublas::context > : std::true_type{};
	template<> struct is_context<boost::multi::cuda::cublas::context&> : std::true_type{};

	template<class Ptr, class T = typename std::pointer_traits<Ptr>::element_type, std::enable_if_t<std::is_convertible<Ptr, multi::memory::cuda::ptr<T>>{}, int> =0>
	boost::multi::cuda::cublas::unsynchronized_context* default_context_of(Ptr const&){
		namespace multi = boost::multi;
		return &multi::cuda::cublas::context::get_instance();
	}

	template<class T>
	boost::multi::cuda::cublas::unsynchronized_context* default_context_of(boost::multi::memory::cuda::managed::ptr<T> const&){
		namespace multi = boost::multi;
		return &multi::cuda::cublas::context::get_instance();
	}

}

#endif

