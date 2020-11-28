#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
$CXXX $CXXFLAGS -O3 $0 -o $0x -lboost_unit_test_framework -lboost_timer \
`pkg-config --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_GEMM_HPP
#define MULTI_ADAPTORS_BLAS_GEMM_HPP

#include "../blas/core.hpp"

#include "../blas/numeric.hpp"
#include "../blas/operations.hpp"

#include "../blas/gemv.hpp"

#include "../../config/NODISCARD.hpp"
#include "../../config/MARK.hpp"

namespace boost{
namespace multi{
namespace blas{

using multi::blas::core::gemm;
using multi::blas::core::gemv;

template<class M> bool is_c_ordering(M const& m){
	return stride(rotated(m))==1 and size(m)!=1;
}

double const& conj(double const& x){return x;}
float  const& conj(float  const& x){return x;}

template<class A, std::enable_if_t<not is_conjugated<A>{}, int> =0>
auto gemm_base_aux(A&& a){return base(a);}

template<class A, std::enable_if_t<    is_conjugated<A>{}, int> =0>
auto gemm_base_aux(A&& a){return underlying(base(a));}

template<class Context, class A, class B, class C>
C&& gemm(Context& ctx, typename A::element alpha, A const& a, B const& b, typename A::element beta, C&& c)
//->decltype(ctx.gemm('N', 'T', size(~c), size(a), size(b), &alpha, gemm_base_aux(b), stride( b), gemm_base_aux(a), stride(~a), &beta, gemm_base_aux(c), size(b)) , std::forward<C>(c))
{

	MULTI_MARK_SCOPE("multi::blas::gemm with context");

	if(c.is_empty()){
		assert(a.is_empty() and b.is_empty());
		return std::forward<C>(c);
	}

	assert( size(~a) == size( b) );
	assert( size( a) == size( c) );
	assert( size(~b) == size(~c) );
	
	auto base_a = gemm_base_aux(a);
	auto base_b = gemm_base_aux(b);
	auto base_c = gemm_base_aux(c);

	assert( stride(a)==1 or stride(~a)==1 );
	assert( stride(b)==1 or stride(~b)==1 );
	assert( stride(c)==1 or stride(~c)==1 );
	
	     if(stride(c)==1 and stride(~c)!=1) blas::gemm(ctx, alpha, ~b, ~a, beta, ~c);
	else if(is_conjugated<C>{}) blas::gemm(ctx, conj(alpha), conj(a), conj(b), conj(beta), conj(c));
	else{
		;;;;; if(stride(~a)==1 and stride(~b)==1 and not is_conjugated<A>{} and not is_conjugated<B>{}){
			if(size(a)==1) ctx.gemm('N', 'N', size(~c), size(a), size(b), &alpha, base_b, stride( b), base_a, size(b)   , &beta, base_c, size(b)  );
			else           ctx.gemm('N', 'N', size(~c), size(a), size(b), &alpha, base_b, stride( b), base_a, stride( a), &beta, base_c, stride(c));
		}else if(stride( a)==1 and stride(~b)==1 and     is_conjugated<A>{} and not is_conjugated<B>{}) ctx.gemm('N', 'C', size(~c), size(a), size(b), &alpha, base_b, stride( b), base_a, stride(~a), &beta, base_c, stride(c));
		else if(stride( a)==1 and stride(~b)==1 and not is_conjugated<A>{} and not is_conjugated<B>{}){
			if(size(a)==1) ctx.gemm('N', 'T', size(~c), size(a), size(b), &alpha, base_b, stride( b), base_a, stride(~a), &beta, base_c, size(b));
			else           ctx.gemm('N', 'T', size(~c), size(a), size(b), &alpha, base_b, stride( b), base_a, stride(~a), &beta, base_c, stride(c));
		}
		else if(stride(~a)==1 and stride( b)==1 and not is_conjugated<A>{} and     is_conjugated<B>{}) ctx.gemm('C', 'N', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, stride( a), &beta, base_c, stride(c));
		else if(stride( a)==1 and stride( b)==1 and     is_conjugated<A>{} and     is_conjugated<B>{}) ctx.gemm('C', 'C', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, stride(~a), &beta, base_c, stride(c));
		else if(stride( a)==1 and stride( b)==1 and not is_conjugated<A>{} and     is_conjugated<B>{}) ctx.gemm('C', 'T', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, stride(~a), &beta, base_c, stride(c));
		else if(stride(~a)==1 and stride( b)==1 and not is_conjugated<A>{} and not is_conjugated<B>{}){
			if(size(a)==1) ctx.gemm('T', 'N', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, size(b)   , &beta, base_c, stride(c));
			else           ctx.gemm('T', 'N', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, stride( a), &beta, base_c, stride(c));
		}
		else if(stride( a)==1 and stride( b)==1 and     is_conjugated<A>{} and not is_conjugated<B>{}) ctx.gemm('T', 'C', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, stride(~a), &beta, base_c, stride(c));
		else if(stride( a)==1 and stride( b)==1 and not is_conjugated<A>{} and not is_conjugated<B>{}) ctx.gemm('T', 'T', size(~c), size(a), size(b), &alpha, base_b, stride(~b), base_a, stride(~a), &beta, base_c, stride(c));
		else                                                                                           assert(0&&" case not implemented in blas");
	}
	return std::forward<C>(c);
}

template<class A, class B, class C>
C&& gemm(typename A::element alpha, A const& a, B const& b, typename A::element beta, C&& c){
//->decltype(gemm('N', 'T', size(~c), size(a), size(b), &alpha, gemm_base_aux(b), stride( b), gemm_base_aux(a), stride(~a), &beta, gemm_base_aux(c), size(b)) , std::forward<C>(c)){
	using multi::blas::default_allocator_of;
	auto ctx = default_context_of(gemm_base_aux(a)); // ADL
	return gemm(ctx, alpha, a, b, beta, std::forward<C>(c));
}

template<class A2D, class B2D, class C2D = typename A2D::decay_type>
NODISCARD("because input arguments are const")
auto gemm(typename A2D::element a, A2D const& A, B2D const& B){
	assert(get_allocator(A) == get_allocator(B));
	return gemm(a, A, B, 0., C2D({size(A), size(rotated(B))}, get_allocator(A)));
}

template<class Context, class A2D, class B2D, class C2D = typename A2D::decay_type>
NODISCARD("because input arguments are const")
auto gemm(Context&& ctx, typename A2D::element a, A2D const& A, B2D const& B)
->std::decay_t<decltype(gemm(std::forward<Context>(ctx), a, A, B, 0., C2D({size(A), size(rotated(B))}, get_allocator(A))))>{
	assert(get_allocator(A) == get_allocator(B));
	return gemm(std::forward<Context>(ctx), a, A, B, 0., C2D({size(A), size(rotated(B))}, get_allocator(A)));
}

template<class A2D, class B2D> 
auto gemm(A2D const& A, B2D const& B)
->decltype(gemm(1., A, B)){
	return gemm(1., A, B);}

template<class Context, class A2D, class B2D> 
auto gemm(Context&& ctx, A2D const& A, B2D const& B)
->decltype(gemm(std::forward<Context>(ctx), 1., A, B)){
	return gemm(std::forward<Context>(ctx), 1., A, B);}

namespace operators{
	template<class A2D, class B2D> 
	auto operator*(A2D const& A, B2D const& B)
	->decltype(gemm(1., A, B)){
		return gemm(1., A, B);}
}

}}}

#if not __INCLUDE_LEVEL__ // _TEST_MULTI_ADAPTORS_BLAS_GEMM

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../array.hpp"
#include "../../utility.hpp"

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>
#include<random>

#include <boost/timer/timer.hpp>

#include "../blas/axpy.hpp"
#include "../blas/dot.hpp"
#include "../blas/nrm2.hpp"

namespace multi = boost::multi;

#endif
#endif

