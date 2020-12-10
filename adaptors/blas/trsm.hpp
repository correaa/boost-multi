#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
$CXXX $CXXFLAGS $0 -o $0.$X -lboost_unit_test_framework `pkg-config --cflags --libs blas` -lboost_timer&&$0.$X&&rm $0.$X;exit
#endif
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_TRSM_HPP
#define MULTI_ADAPTORS_BLAS_TRSM_HPP

#include "../blas/core.hpp"

#include "../blas/operations.hpp" // uplo
#include "../blas/filling.hpp"
#include "../blas/side.hpp"

namespace boost{
namespace multi::blas{

enum class diagonal : char{
	    unit = 'U', 
	non_unit = 'N', general = non_unit
};

using core::trsm;

template<class A2D, class B2D>
decltype(auto) trsm(blas::side a_side, blas::filling a_fill, blas::diagonal a_diag, typename A2D::element_type alpha, A2D const& a, B2D&& b){
	;;;; if(a_side == blas::side::left ) assert(size(~a) >= size( b));
	else if(a_side == blas::side::right) assert(size( a) >= size(~b));

	assert( stride( a) == 1 or stride(~a) == 1 );
	assert( stride( b) == 1 or stride(~b) == 1 );

	if(size(b)!=0){
		;;;; if constexpr(not is_conjugated<A2D>{} and not is_conjugated<B2D>{}){
			;;;; if(stride(~a)==1 and stride(~b)==1) trsm((char)     a_side , (char)+a_fill, 'N', (char)a_diag, size(~b), size( b),      alpha ,            base(a) , stride( a),            base(b) , stride( b));
			else if(stride( a)==1 and stride(~b)==1) trsm((char)     a_side , (char)-a_fill, 'T', (char)a_diag, size(~b), size( b),      alpha ,            base(a) , stride(~a),            base(b) , stride( b));
			else if(stride( a)==1 and stride( b)==1) trsm((char)swap(a_side), (char)-a_fill, 'N', (char)a_diag, size( b), size(~b),      alpha ,            base(a) , stride(~a),            base(b) , stride(~b));
			else if(stride(~a)==1 and stride( b)==1) trsm((char)swap(a_side), (char)+a_fill, 'T', (char)a_diag, size( b), size(~b),      alpha ,            base(a) , stride( a),            base(b) , stride(~b));
			else assert(0 && "not implemented in blas");
		}else if constexpr(   is_conjugated<A2D>{} and not is_conjugated<B2D>{}){
			;;;; if(stride(~a)==1 and stride(~b)==1) assert(0 && "not implemented in blas");
			else if(stride( a)==1 and stride(~b)==1) trsm((char)     a_side , (char)-a_fill, 'C', (char)a_diag, size(~b), size( b),      alpha , underlying(base(a)), stride(~a),            base(b) , stride( b));
			else if(stride( a)==1 and stride( b)==1) assert(0 && "not implemented in blas");
			else if(stride(~a)==1 and stride( b)==1) trsm((char)swap(a_side), (char)+a_fill, 'C', (char)a_diag, size( b), size(~b),      alpha , underlying(base(a)), stride( a),            base(b) , stride(~b));
			else assert(0 && "not implemented in blas");
		}else if constexpr(not is_conjugated<A2D>{} and    is_conjugated<B2D>{}){
			;;;; if(stride(~a)==1 and stride(~b)==1) assert(0);
			else if(stride( a)==1 and stride(~b)==1) trsm((char)     a_side , (char)-a_fill, 'C', (char)a_diag, size(~b), size( b), conj(alpha),            base(a) , stride(~a), underlying(base(b)), stride( b));
			else if(stride( a)==1 and stride( b)==1) assert(0);
			else if(stride(~a)==1 and stride( b)==1) trsm((char)swap(a_side), (char)+a_fill, 'C', (char)a_diag, size( b), size(~b), conj(alpha),            base(a) , stride( a), underlying(base(b)), stride(~b));
			else assert(0 && "not implemented in blas");
		}else if constexpr(   is_conjugated<A2D>{} and     is_conjugated<B2D>{}){
			assert(0);
			;;;; if(stride(~a)==1 and stride(~b)==1) assert(0 && "not implemented in blas");
			else if(stride( a)==1 and stride(~b)==1) trsm((char)     a_side , (char)-a_fill, 'T', (char)a_diag, size(~b), size( b), conj(alpha), underlying(base(a)), stride(~a), underlying(base(b)), stride( b));
			else if(stride( a)==1 and stride( b)==1) assert(0 && "not implemented in blas");
			else if(stride(~a)==1 and stride( b)==1) trsm((char)swap(a_side), (char)+a_fill, 'T', (char)a_diag, size( b), size(~b), conj(alpha), underlying(base(a)), stride( a), underlying(base(b)), stride(~b));
			else assert(0 && "not implemented in blas");
		}
	}
	return std::forward<B2D>(b);
}

template<class A2D, class B2D>
auto trsm(blas::side a_side, blas::filling a_fill, typename A2D::element_type alpha, A2D const& a, B2D&& b)
->decltype(trsm(a_side, a_fill, blas::diagonal::general, alpha, a, std::forward<B2D>(b))){
	return trsm(a_side, a_fill, blas::diagonal::general, alpha, a, std::forward<B2D>(b));}

}}

#endif

