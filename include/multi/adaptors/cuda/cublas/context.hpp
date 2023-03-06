// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2020-2022 Alfredo A. Correa
#pragma once

#include <multi/config/MARK.hpp>
#include <multi/adaptors/cuda/cublas/call.hpp>

#include <multi/adaptors/blas/traits.hpp>
#include <multi/adaptors/blas/core.hpp>

#include <thrust/system/cuda/memory.h>  // for thrust::cuda::pointer

#include<mutex>

namespace boost {
namespace multi::cuda::cublas {

class operation {
	cublasOperation_t impl_;

 public:
	explicit operation(char trans) : impl_{[=]{
		switch(trans) {
			case 'N': return CUBLAS_OP_N;
			case 'T': return CUBLAS_OP_T;
			case 'C': return CUBLAS_OP_C;
			default : assert(0);
		}
		return cublasOperation_t{};
	}()} {}
	operator cublasOperation_t() const{return impl_;}
};

class side {
	cublasSideMode_t impl_;

 public:
	explicit side(char trans) : impl_{[=] {
		switch(trans) {
			case 'L': return CUBLAS_SIDE_LEFT;
			case 'R': return CUBLAS_SIDE_RIGHT;
		}
		assert(0); return cublasSideMode_t{};
	}()} {}
	operator cublasSideMode_t() const {return impl_;}
};

class filling {
	cublasFillMode_t impl_;

 public:
	explicit filling(char trans) : impl_{[=] {
		switch(trans) {
			case 'L': return CUBLAS_FILL_MODE_LOWER;
			case 'U': return CUBLAS_FILL_MODE_UPPER;
		}
		assert(0); return cublasFillMode_t{};
	}()} {}
	operator cublasFillMode_t() const {return impl_;}
};

class diagonal {
	cublasDiagType_t impl_;

 public:
	explicit diagonal(char trans) : impl_{[=] {
		switch(trans) {
			case 'N': return CUBLAS_DIAG_NON_UNIT;
			case 'U': return CUBLAS_DIAG_UNIT;
		}
		assert(0); return cublasDiagType_t{};
	}()} {}
	operator cublasDiagType_t() const {return impl_;}
};

using blas::is_s;
using blas::is_d;
using blas::is_c;
using blas::is_z;

using std::is_assignable;
using std::is_assignable_v;
using std::is_convertible_v;

enum type {S, D, C, Z};

template<class T>
constexpr auto type_of(T const& = {}) -> cublas::type {
	static_assert(is_s<T>{} or is_d<T>{} or is_c<T>{} or is_z<T>{});
	     if(is_s<T>{}) {return S;}
	else if(is_d<T>{}) {return D;}
	else if(is_c<T>{}) {return C;}
	else if(is_z<T>{}) {return Z;}
}

template<class T>
constexpr auto data_cast(T* p) {
	     if constexpr(is_s<T>{}) {return reinterpret_cast<float          *>(p);}
	else if constexpr(is_d<T>{}) {return reinterpret_cast<double         *>(p);}
	else if constexpr(is_c<T>{}) {return reinterpret_cast<cuComplex      *>(p);}
	else if constexpr(is_z<T>{}) {return reinterpret_cast<cuDoubleComplex*>(p);}
}

template<class T>
constexpr auto data_cast(T const* p) {
	     if constexpr(is_s<T>{}) {return reinterpret_cast<float           const*>(p);}
	else if constexpr(is_d<T>{}) {return reinterpret_cast<double          const*>(p);}
	else if constexpr(is_c<T>{}) {return reinterpret_cast<cuComplex       const*>(p);}
	else if constexpr(is_z<T>{}) {return reinterpret_cast<cuDoubleComplex const*>(p);}
}

template<cublas::type T> constexpr auto cublas_gemv = std::enable_if_t<T!=T>{};

template<> constexpr auto cublas_gemv<S> = cublasSgemv;
template<> constexpr auto cublas_gemv<D> = cublasDgemv;
template<> constexpr auto cublas_gemv<C> = cublasCgemv;
template<> constexpr auto cublas_gemv<Z> = cublasZgemv;

#define DECLRETURN(ExpR) -> decltype(ExpR) {return ExpR;}  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing
#define JUSTRETURN(ExpR)                   {return ExpR;}  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing

template<cublas::type T> struct cublas {
	static constexpr auto gemv = cublas_gemv<T>;
};

class context : private std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)> {
	using pimpl_t = std::unique_ptr<std::decay_t<decltype(*cublasHandle_t{})>, decltype(&cublasDestroy)>;
	cudaStream_t stream() const {cudaStream_t streamId; cuda::cublas::call<cublasGetStream>(this->get(), &streamId); return streamId;}
	template<auto Function, class... Args>
	void sync_call(Args... args) const {
		call<Function>(const_cast<context*>(this)->get(), args...);
		this->synchronize();
	}
	template<auto Function, class... Args>
	void sync_call(Args... args) {
		call<Function>(this->get(), args...);
		this->synchronize();
	}

 public:
	using pimpl_t::get;
	static context& get_instance() {
		thread_local context ctxt;
		return ctxt;
	};
	context() : pimpl_t{[] {cublasHandle_t h; cublasCreate(&h); return h;}(), &cublasDestroy} {}
	using ssize_t = int;
	static int version() {int ret; cuda::cublas::call<cublasGetVersion>(nullptr, &ret); return ret;}
	void synchronize() const {
		cudaError_t e = cudaDeviceSynchronize();
		//cudaError_t e = cudaStreamSynchronize(stream());
		if(e != cudaSuccess) {throw std::runtime_error{"cannot synchronize stream in cublas context"};}
	}

	template<class ALPHA, class XP, class X = typename std::pointer_traits<XP>::element_type,
		class = decltype(std::declval<X&>() *= ALPHA{}),
		std::enable_if_t<std::is_convertible_v<XP, ::thrust::cuda::pointer<X>>, int> = 0
	>
	void scal(ssize_t n, ALPHA const& alpha, XP x, ssize_t incx) const {
		if(is_d<X>{}) {sync_call<cublasDscal>(n, (double          const*)alpha, (double         *)raw_pointer_cast(x), incx);}
		if(is_z<X>{}) {sync_call<cublasZscal>(n, (cuDoubleComplex const*)alpha, (cuDoubleComplex*)raw_pointer_cast(x), incx);}
	}

	template<class ALPHA, class XP, class X = typename std::pointer_traits<XP>::element_type, class YP, class Y = typename std::pointer_traits<YP>::element_type,
		typename = decltype(std::declval<Y&>() = ALPHA{}*X{} + Y{}),
		std::enable_if_t<std::is_convertible_v<XP, ::thrust::cuda::pointer<X>> and std::is_convertible_v<YP, ::thrust::cuda::pointer<Y>>, int> = 0
	>
	void axpy(ssize_t n, ALPHA const* alpha, XP x, ssize_t incx, YP y, ssize_t incy) {
		if(is_d<X>{}) {sync_call<cublasDaxpy>(n, (double          const*)alpha, (double          const*)raw_pointer_cast(x), incx, (double         *)raw_pointer_cast(y), incy);}
		if(is_z<X>{}) {sync_call<cublasZaxpy>(n, (cuDoubleComplex const*)alpha, (cuDoubleComplex const*)raw_pointer_cast(x), incx, (cuDoubleComplex*)raw_pointer_cast(y), incy);}
	}

	template<class ALPHA, class AAP, class AA = typename std::pointer_traits<AAP>::element_type, class XXP, class XX = typename std::pointer_traits<XXP>::element_type, class BETA, class YYP, class YY = typename std::pointer_traits<YYP>::element_type,
		typename = decltype(std::declval<YY&>() = ALPHA{}*(AA{}*XX{} + AA{}*XX{})),
		std::enable_if_t<std::is_convertible_v<AAP, ::thrust::cuda::pointer<AA>> and std::is_convertible_v<XXP, ::thrust::cuda::pointer<XX>> and std::is_convertible_v<YYP, ::thrust::cuda::pointer<YY>>, int> = 0
	>
	auto gemv(char transA, ssize_t m, ssize_t n, ALPHA const* alpha, AAP aa, ssize_t lda, XXP xx, ssize_t incx, BETA const* beta, YYP yy, ssize_t incy) {
		if(is_d<AA>{}) {sync_call<cublasDgemv>(operation{transA}, m, n, (double const         *)alpha, (double          const*)::thrust::raw_pointer_cast(aa), lda, (double          const*)::thrust::raw_pointer_cast(xx), incx, (double          const*)beta, (double         *)::thrust::raw_pointer_cast(yy), incy);}
		if(is_z<AA>{}) {sync_call<cublasZgemv>(operation{transA}, m, n, (cuDoubleComplex const*)alpha, (cuDoubleComplex const*)::thrust::raw_pointer_cast(aa), lda, (cuDoubleComplex const*)::thrust::raw_pointer_cast(xx), incx, (cuDoubleComplex const*)beta, (cuDoubleComplex*)::thrust::raw_pointer_cast(yy), incy);}
	}

	template<class ALPHA, class AAP, class AA = typename std::pointer_traits<AAP>::element_type, class BBP, class BB = typename std::pointer_traits<BBP>::element_type, class BETA, class CCP, class CC = typename std::pointer_traits<CCP>::element_type,
		typename = decltype(std::declval<CC&>() = ALPHA{}*(AA{}*BB{} + AA{}*BB{})),
		class = std::enable_if_t<std::is_convertible_v<AAP, ::thrust::cuda::pointer<AA>> and std::is_convertible_v<BBP, ::thrust::cuda::pointer<BB>> and std::is_convertible_v<CCP, ::thrust::cuda::pointer<CC>>>
	>
	void gemm(char transA, char transB, ssize_t m, ssize_t n, ssize_t k, ALPHA const* alpha, AAP aa, ssize_t lda, BBP bb, ssize_t ldb, BETA const* beta, CCP cc, ssize_t ldc) {
		MULTI_MARK_SCOPE("cublasXgemm");
		if(is_d<AA>{}) {sync_call<cublasDgemm>(cuda::cublas::operation{transA}, cuda::cublas::operation{transB}, m, n, k, (double          const*)alpha, (double          const*)::thrust::raw_pointer_cast(aa), lda, (double          const*)::thrust::raw_pointer_cast(bb), ldb, (double          const*)beta, (double         *)::thrust::raw_pointer_cast(cc), ldc);}
		if(is_z<AA>{}) {sync_call<cublasZgemm>(cuda::cublas::operation{transA}, cuda::cublas::operation{transB}, m, n, k, (cuDoubleComplex const*)alpha, (cuDoubleComplex const*)::thrust::raw_pointer_cast(aa), lda, (cuDoubleComplex const*)::thrust::raw_pointer_cast(bb), ldb, (cuDoubleComplex const*)beta, (cuDoubleComplex*)::thrust::raw_pointer_cast(cc), ldc);}
	}

	template<class ALPHA, class AAP, class AA = typename std::pointer_traits<AAP>::element_type, class BBP, class BB = typename std::pointer_traits<BBP>::element_type,
		std::enable_if_t<
			is_z<AA>{} and is_z<BB>{} and is_assignable<BB&, decltype(AA{}*BB{}/ALPHA{})>{} and is_assignable<BB&, decltype(ALPHA{}*BB{}/AA{})>{} and 
			is_convertible_v<AAP, ::thrust::cuda::pointer<AA>> and is_convertible_v<BBP, ::thrust::cuda::pointer<BB>>
		,int> =0
	>
	void trsm(char side, char ul, char transA, char diag, ssize_t m, ssize_t n, ALPHA alpha, AAP aa, ssize_t lda, BBP bb, ssize_t ldb) {
		sync_call<cublasZtrsm>(cuda::cublas::side{side}, cuda::cublas::filling{ul}, cuda::cublas::operation{transA}, cuda::cublas::diagonal{diag}, m, n, (cuDoubleComplex const*)&alpha, (cuDoubleComplex const*)raw_pointer_cast(aa), lda, (cuDoubleComplex*)raw_pointer_cast(bb), ldb);
	}

	template<class ALPHA, class AAP, class AA = typename std::pointer_traits<AAP>::element_type, class BBP, class BB = typename std::pointer_traits<BBP>::element_type,
		std::enable_if_t<
			is_d<AA>{} and is_d<BB>{} and is_assignable<BB&, decltype(AA{}*BB{}/ALPHA{})>{} and is_assignable<BB&, decltype(ALPHA{}*BB{}/AA{})>{} and 
			is_convertible_v<AAP, ::thrust::cuda::pointer<AA>> and is_convertible_v<BBP, ::thrust::cuda::pointer<BB>>
		,int> =0
	>
	void trsm(char side, char ul, char transA, char diag, ssize_t m, ssize_t n, ALPHA alpha, AAP aa, ssize_t lda, BBP bb, ssize_t ldb) {
		sync_call<cublasDtrsm>(
			cuda::cublas::side{side},
			cuda::cublas::filling{ul},
			cuda::cublas::operation{transA},
			cuda::cublas::diagonal{diag}, 
			m, n, (double const*)&alpha, (double const*)raw_pointer_cast(aa), lda, (double*)raw_pointer_cast(bb), ldb
		);
	}

	template<
		class XXP, class XX = typename std::pointer_traits<XXP>::element_type,
		class YYP, class YY = typename std::pointer_traits<YYP>::element_type,
		class RRP, class RR = typename std::pointer_traits<RRP>::element_type,
		std::enable_if_t<
			is_d<XX>{} and is_d<YY>{} and is_d<RR>{} and is_assignable<RR&, decltype(XX{}*YY{})>{} and
			is_convertible_v<XXP, ::thrust::cuda::pointer<XX>> and is_convertible_v<YYP, ::thrust::cuda::pointer<YY>> and is_convertible_v<RRP, RR*>
		, int> =0
	>
	void dot(int n, XXP xx, int incx, YYP yy, int incy, RRP rr) {
		assert(0);
		cublasPointerMode_t mode;
		auto s = cublasGetPointerMode(get(), &mode); assert( s == CUBLAS_STATUS_SUCCESS );
		assert( mode == CUBLAS_POINTER_MODE_HOST );
		sync_call<cublasDdot>(n, raw_pointer_cast(xx), incx, raw_pointer_cast(yy), incy, rr);
	}

	template<
		class XXP, class XX = typename std::pointer_traits<XXP>::element_type,
		class YYP, class YY = typename std::pointer_traits<YYP>::element_type,
		class RRP, class RR = typename std::pointer_traits<RRP>::element_type,
		std::enable_if_t<
			is_z<XX>{} and is_z<YY>{} and is_z<RR>{} and is_assignable<RR&, decltype(XX{}*YY{})>{} and
			is_convertible_v<XXP, ::thrust::cuda::pointer<XX>> and is_convertible_v<YYP, ::thrust::cuda::pointer<YY>>
			and (is_convertible_v<RRP, ::thrust::cuda::pointer<RR>> or is_convertible_v<RRP, RR*>)
		, int> =0
	>
	void dotc(int n, XXP xx, int incx, YYP yy, int incy, RRP rr) {
		cublasPointerMode_t mode;
		auto s = cublasGetPointerMode(get(), &mode); assert( s == CUBLAS_STATUS_SUCCESS );
		assert( mode == CUBLAS_POINTER_MODE_HOST );
	//  cublasSetPointerMode(get(), CUBLAS_POINTER_MODE_DEVICE);
		if constexpr(is_convertible_v<RRP, ::thrust::cuda::pointer<RR>>) {
			sync_call<cublasZdotc>(n, (cuDoubleComplex const*)::thrust::raw_pointer_cast(xx), incx, (cuDoubleComplex const*)::thrust::raw_pointer_cast(yy), incy, (cuDoubleComplex*)::thrust::raw_pointer_cast(rr) );
		} else {
			sync_call<cublasZdotc>(n, (cuDoubleComplex const*)::thrust::raw_pointer_cast(xx), incx, (cuDoubleComplex const*)::thrust::raw_pointer_cast(yy), incy, (cuDoubleComplex*)rr);
		}
	//  cublasSetPointerMode(get(), CUBLAS_POINTER_MODE_HOST);
	}
};

}  // end namespace multi::cuda::cublas
}  // end namespace boost

namespace boost::multi::blas {

	template<> struct is_context<boost::multi::cuda::cublas::context > : std::true_type {};
	template<> struct is_context<boost::multi::cuda::cublas::context&> : std::true_type {};

	template<class Ptr, class T = typename std::pointer_traits<Ptr>::element_type, std::enable_if_t<std::is_convertible<Ptr, ::thrust::cuda::pointer<T>>{}, int> =0>
	boost::multi::cuda::cublas::context* default_context_of(Ptr const&) {
		namespace multi = boost::multi;
		return &multi::cuda::cublas::context::get_instance();
	}

	template<class T, class R>
	boost::multi::cuda::cublas::context* default_context_of(::thrust::pointer<T, ::thrust::cuda_cub::tag, R> const&) {
		namespace multi = boost::multi;
		return &multi::cuda::cublas::context::get_instance();
	}
}
