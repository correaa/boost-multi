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

//#include "../../utility.hpp" // TODO include extensions only

namespace boost{
namespace multi{
namespace blas{

using core::gemm;

template<class It>
auto xbase_aux(It const& it, std::true_type const&)
->decltype(underlying(base(it))){
	return underlying(base(it));}

template<class It>
auto xbase_aux(It const& it, std::false_type const&)
->decltype(base(it)){
	return base(it);}

template<class It>
auto xbase(It const& it)
->decltype(xbase_aux(it, std::integral_constant<bool, is_conjugated<It>{}>{})){
	return xbase_aux(it, std::integral_constant<bool, is_conjugated<It>{}>{});}

template<class Context, class It2DA, class Size, class It2DB, class It2DC>
auto gemm_n(Context&& ctxt, typename It2DA::element alpha, It2DA a_first, Size a_count, It2DB b_first, typename It2DA::element beta, It2DC c_first)
->decltype(std::forward<Context>(ctxt).gemm('N', 'N', b_first->size(), a_count, a_first->size(), &alpha, xbase(b_first), b_first->size()  , xbase(a_first), a_first->size(), &beta, base(c_first), c_first->size()  ), It2DC{})
try{
	if(a_count != 0){
	assert( b_first->size() == c_first->size() );
	assert( a_first.stride()==1 or a_first->stride()==1 );
	assert( b_first.stride()==1 or b_first->stride()==1 );
	assert( c_first.stride()==1 or c_first->stride()==1 );

	;;;;; if constexpr(!is_conjugated<It2DA>{} and !is_conjugated<It2DB>{}){
		;;;;; if(a_first->stride()==1 and b_first->stride()==1 and c_first->stride()==1){
			;;;; if  (a_count==1 and b_first->size()==1){std::forward<Context>(ctxt).gemm('N', 'N', b_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first->size()  , base(a_first), a_first->size() , &beta, base(c_first), c_first->size()  );}
			else if  (a_count==1)                       {std::forward<Context>(ctxt).gemm('N', 'N', b_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first. stride(), base(a_first), a_first->size()  , &beta, base(c_first), c_first->size()  );}
			else                                        {std::forward<Context>(ctxt).gemm('N', 'N', b_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first. stride(), base(a_first), a_first. stride(), &beta, base(c_first), c_first. stride());}
		}else if(a_first->stride()==1 and b_first->stride()==1 and c_first. stride()==1){
			if  (a_count==1)        {std::forward<Context>(ctxt).gemm('T', 'T', a_count, b_first->size(), a_first->size(), &alpha, base(a_first), a_first. stride(), base(b_first), b_first->size() , &beta, base(c_first), a_first->size()  );}
			else                    {std::forward<Context>(ctxt).gemm('T', 'T', a_count, b_first->size(), a_first->size(), &alpha, base(a_first), a_first. stride(), base(b_first), b_first.stride(), &beta, base(c_first), c_first->stride());}
		}else if(a_first. stride()==1 and b_first->stride()==1 and c_first->stride()==1){
			if  (a_count==1)        {std::forward<Context>(ctxt).gemm('N', 'T', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first. stride(), base(a_first), a_first->stride(), &beta, base(c_first), a_first->size()  );}
			else                    {std::forward<Context>(ctxt).gemm('N', 'T', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first. stride(), base(a_first), a_first->stride(), &beta, base(c_first), c_first.stride());}
		}else if(a_first. stride()==1 and b_first->stride()==1 and c_first. stride()==1){
			if  (a_count==1)        {std::forward<Context>(ctxt).gemm('N', 'T', a_count, b_first->size(), a_first->size(), &alpha, base(a_first), a_first->stride(), base(b_first), a_first->size()  , &beta, base(c_first), b_first->size()  );}
			else                    {std::forward<Context>(ctxt).gemm('N', 'T', a_count, b_first->size(), a_first->size(), &alpha, base(a_first), a_first->stride(), base(b_first), b_first. stride(), &beta, base(c_first), c_first->stride());}
		}else if(a_first->stride()==1 and b_first.stride()==1 and c_first. stride()==1){
			;;;; if(a_count==1 and b_first->size()){std::forward<Context>(ctxt).gemm('N', 'N', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first->size()  , base(a_first), a_first->size()  , &beta, base(c_first), c_first->stride());}
			else if(a_count==1)                    {std::forward<Context>(ctxt).gemm('N', 'T', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first->stride(), base(a_first), a_first->size()  , &beta, base(c_first), c_first->stride());}
			else                                   {std::forward<Context>(ctxt).gemm('N', 'T', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first->stride(), base(a_first), a_first.stride() , &beta, base(c_first), c_first->stride());}
		}else if(a_first->stride()==1 and b_first. stride()==1 and c_first->stride()==1){
			if  (a_count==1)        {std::forward<Context>(ctxt).gemm('T', 'N', a_count, c_first->size(), a_first->size(), &alpha, base(b_first), b_first->stride(), base(a_first), a_first->size()  , &beta, base(c_first), c_first.stride());}
			else                    {std::forward<Context>(ctxt).gemm('T', 'N', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first->stride(), base(a_first), a_first.stride(), &beta, base(c_first), c_first.stride());}
		}else if(a_first. stride()==1 and b_first.stride( )==1 and c_first. stride()==1){
		                            {std::forward<Context>(ctxt).gemm('N', 'N', c_first->size(), a_count, a_first->size(), &alpha, base(a_first), a_first->stride(), base(b_first), b_first->stride(), &beta, base(c_first), c_first->stride());}
		}else if(a_first. stride()==1 and b_first.stride( )==1 and c_first->stride()==1){
		                            {std::forward<Context>(ctxt).gemm('T', 'T', a_count, c_first->size(), a_first->size(), &alpha, base(b_first), b_first->stride(), base(a_first), a_first->stride(), &beta, base(c_first), c_first. stride());}
		}else assert(0);
	}else if constexpr(!is_conjugated<It2DA>{} and  is_conjugated<It2DB>{}){
		;;;;; if(a_first->stride()==1 and b_first->stride()==1 and c_first->stride()==1){
		                            {std::forward<Context>(ctxt).gemm('N', 'C', c_first->size(), a_count, a_first->size(), &alpha, underlying(base(b_first)), b_first. stride(), base(a_first), a_first. stride(), &beta, base(c_first), c_first.stride());}
		}else if(a_first->stride()==1 and b_first. stride()==1 and c_first->stride()==1){
			if  (a_count==1)        {std::forward<Context>(ctxt).gemm('C', 'N', a_count, c_first->size(), a_first->size(), &alpha, underlying(base(b_first)), b_first->stride(), base(a_first), a_first->size()  , &beta, base(c_first), c_first.stride());}
			else                    {std::forward<Context>(ctxt).gemm('C', 'N', c_first->size(), a_count, a_first->size(), &alpha, underlying(base(b_first)), b_first->stride(), base(a_first), a_first.stride(), &beta, base(c_first), c_first.stride());}
		}else if(a_first->stride()==1 and b_first. stride()==1 and c_first. stride()==1){
		                            {std::forward<Context>(ctxt).gemm('C', 'N', c_first->size(), a_count, a_first->size(), &alpha, underlying(base(b_first)), b_first->stride(), base(a_first), a_first. stride(), &beta, base(c_first), c_first->stride());}
		}else if(a_first. stride()==1 and b_first. stride()==1 and c_first. stride()==1){
		                            {std::forward<Context>(ctxt).gemm('C', 'T', c_first->size(), a_count, a_first->size(), &alpha, underlying(base(b_first)), b_first->stride(), base(a_first), a_first->stride(), &beta, base(c_first), c_first->stride());}
		}else if(a_first. stride()==1 and b_first. stride()==1 and c_first->stride()==1){
		                            {std::forward<Context>(ctxt).gemm('C', 'T', a_count, c_first->size(), a_first->size(), &alpha, underlying(base(b_first)), b_first->stride(), base(a_first), a_first->stride(), &beta, base(c_first), c_first. stride());}
		}else assert(0);
	}else if constexpr( is_conjugated<It2DA>{} and !is_conjugated<It2DB>{}){
		;;;;; if(a_first. stride()==1 and b_first->stride()==1 and c_first->stride()==1){
			if  (a_count==1)        {std::forward<Context>(ctxt).gemm('N', 'C', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first. stride(), underlying(base(a_first)), a_first->stride(), &beta, base(c_first), a_first->size()  );}
			else                    {std::forward<Context>(ctxt).gemm('N', 'C', c_first->size(), a_count, a_first->size(), &alpha, base(b_first), b_first. stride(), underlying(base(a_first)), a_first->stride(), &beta, base(c_first), c_first.stride());}
		}else assert(0);
	}else if constexpr( is_conjugated<It2DA>{} and  is_conjugated<It2DB>{}){
		;;;;; if(a_first. stride()==1 and b_first. stride()==1 and c_first->stride()==1){
		                            {std::forward<Context>(ctxt).gemm('C', 'C', a_count, c_first->size(), a_first->size(), &alpha, underlying(base(b_first)), b_first->stride(), underlying(base(a_first)), a_first->stride(), &beta, base(c_first), c_first. stride());}
		}else assert(0);
	}
	}
	return c_first + a_count;
}catch(std::logic_error& e){
	throw std::logic_error(
		"couldn't do "+std::string(__PRETTY_FUNCTION__)+" of layout a_count="+std::to_string(a_count)
		+" a_strides="+std::to_string(a_first.stride())+","+std::to_string(a_first->stride())+" a->size="+std::to_string(a_first->size())
		+" b_strides="+std::to_string(b_first.stride())+","+std::to_string(b_first->stride())+" b->size="+std::to_string(a_first->size())
		+" c_strides="+std::to_string(c_first.stride())+","+std::to_string(c_first->stride())+" c->size="+std::to_string(a_first->size())
		+" because " + e.what()
	);
}

template<class It2DA, class Size, class It2DB, class It2DC, class Context = blas::context> // TODO automatic deduction of context
auto gemm_n(typename It2DA::element alpha, It2DA a_first, Size a_count, It2DB b_first, typename It2DA::element beta, It2DC c_first)
->decltype(gemm_n(Context{}, alpha, a_first, a_count, b_first, beta, c_first)){
	return gemm_n(Context{}, alpha, a_first, a_count, b_first, beta, c_first);}

template<class Context, class A, class B, class C>
C&& gemm(Context&& ctx, typename A::element alpha, A const& a, B const& b, typename A::element beta, C&& c)
//->decltype(ctx.gemm('N', 'T', size(~c), size(a), size(b), &alpha, gemm_base_aux(b), stride( b), gemm_base_aux(a), stride(~a), &beta, gemm_base_aux(c), size(b)) , std::forward<C>(c))
{
//	MULTI_MARK_SCOPE("multi::blas::gemm with context");

//	if(c.is_empty()){
//		assert(a.is_empty() and b.is_empty());
//		return std::forward<C>(c);
//	}

	assert( size( a) == size( c) );
	if(not a.is_empty()) assert( size(~a) == size( b) );
//	assert( size(~b) == size(~c) );

//	assert( stride(a)==1 or stride(~a)==1 );
//	assert( stride(b)==1 or stride(~b)==1 );
//	assert( stride(c)==1 or stride(~c)==1 );

//	if(stride(c)==1 and stride(~c)!=1){
//		blas::gemm(std::forward<Context>(ctx), alpha, ~b, ~a, beta, ~c);
//		return std::forward<C>(c);
//	}

	if constexpr(is_conjugated<C>{}) blas::gemm(std::forward<Context>(ctx), conj(alpha), conj(a), conj(b), conj(beta), conj(c));
	else{                             blas::gemm_n(std::forward<Context>(ctx), alpha, begin(a), size(a), begin(b), beta, begin(c));
//		;;;;; if constexpr(!is_conjugated<A>{} and !is_conjugated<B>{}){
//			;;;;; if(stride(~a)==1 and stride(~b)==1){
//				if(size(a)==1){std::forward<Context>(ctx).gemm('N', 'N', size(~c), size(a), size(b), &alpha, base(b), stride( b), base(a), size(b)   , &beta, base(c), size(b)  );}
//				else          {std::forward<Context>(ctx).gemm('N', 'N', size(~c), size(a), size(b), &alpha, base(b), stride( b), base(a), stride( a), &beta, base(c), stride(c));}
//			}else if(stride( a)==1 and stride(~b)==1){
//				if(size(a)==1){std::forward<Context>(ctx).gemm('N', 'T', size(~c), size(a), size(b), &alpha, base(b), stride( b), base(a), stride(~a), &beta, base(c), size(b)  );}
//				else          {std::forward<Context>(ctx).gemm('N', 'T', size(~c), size(a), size(b), &alpha, base(b), stride( b), base(a), stride(~a), &beta, base(c), stride(c));}
//			}else if(stride(~a)==1 and stride( b)==1){
//				if(size(a)==1){std::forward<Context>(ctx).gemm('T', 'N', size(~c), size(a), size(b), &alpha, base(b), stride(~b), base(a), size(b)   , &beta, base(c), stride(c));}
//				else          {std::forward<Context>(ctx).gemm('T', 'N', size(~c), size(a), size(b), &alpha, base(b), stride(~b), base(a), stride( a), &beta, base(c), stride(c));}
//			}else if(stride( a)==1 and stride( b)==1){
//				               std::forward<Context>(ctx).gemm('T', 'T', size(~c), size(a), size(b), &alpha, base(b), stride(~b), base(a), stride(~a), &beta, base(c), stride(c));
//			}else assert(0);
//		}else if constexpr(!is_conjugated<A>{} and  is_conjugated<B>{}){
//			;;;; if(stride(~a)==1 and stride( b)==1){std::forward<Context>(ctx).gemm('C', 'N', size(~c), size(a), size(b), &alpha, underlying(base(b)), stride(~b), base(a), stride( a), &beta, base(c), stride(c));}
//			else if(stride( a)==1 and stride( b)==1){std::forward<Context>(ctx).gemm('C', 'T', size(~c), size(a), size(b), &alpha, underlying(base(b)), stride(~b), base(a), stride(~a), &beta, base(c), stride(c));}
//			else assert(0);
//		}else if constexpr( is_conjugated<A>{} and !is_conjugated<B>{}){
//			;;;; if(stride( a)==1 and stride(~b)==1){std::forward<Context>(ctx).gemm('N', 'C', size(~c), size(a), size(b), &alpha, base(b), stride( b), underlying(base(a)), stride(~a), &beta, base(c), stride(c));}
//			else if(stride( a)==1 and stride( b)==1){std::forward<Context>(ctx).gemm('T', 'C', size(~c), size(a), size(b), &alpha, base(b), stride(~b), underlying(base(a)), stride(~a), &beta, base(c), stride(c));}
//			else assert(0);
//		}else if constexpr( is_conjugated<A>{} and  is_conjugated<B>{}){
//			;;;; if(stride( a)==1 and stride( b)==1){std::forward<Context>(ctx).gemm('C', 'C', size(~c), size(a), size(b), &alpha, underlying(base(b)), stride(~b), underlying(base(a)), stride(~a), &beta, base(c), stride(c));}
//			else assert(0&&"case not implemented in blas");
//		}
	}
	return std::forward<C>(c);
}

template<class A, class B, class C>
C&& gemm(typename A::element alpha, A const& a, B const& b, typename A::element beta, C&& c){
//->decltype(gemm('N', 'T', size(~c), size(a), size(b), &alpha, gemm_base_aux(b), stride( b), gemm_base_aux(a), stride(~a), &beta, gemm_base_aux(c), size(b)) , std::forward<C>(c)){
//	using multi::blas::default_allocator_of;
//	auto ctx = default_context_of(gemm_base_aux(a)); // ADL
	return gemm(blas::context{}, alpha, a, b, beta, std::forward<C>(c));
}

//template<class A2D, class B2D, class C2D = typename A2D::decay_type>
//NODISCARD("because input arguments are const")
//auto gemm(typename A2D::element a, A2D const& A, B2D const& B){
//	assert(get_allocator(A) == get_allocator(B));
//	return gemm(a, A, B, 0., C2D({size(A), size(rotated(B))}, get_allocator(A)));
//}

//template<class Context, class A2D, class B2D, class C2D = typename A2D::decay_type>
//NODISCARD("because input arguments are const")
//auto gemm(Context&& ctx, typename A2D::element a, A2D const& A, B2D const& B)
//->std::decay_t<decltype(gemm(std::forward<Context>(ctx), a, A, B, 0., C2D({size(A), size(rotated(B))}, get_allocator(A))))>{
//	assert(get_allocator(A) == get_allocator(B));
//	return gemm(std::forward<Context>(ctx), a, A, B, 0., C2D({size(A), size(rotated(B))}, get_allocator(A)));
//}

//template<class A2D, class B2D> 
//auto gemm(A2D const& A, B2D const& B)
//->decltype(gemm(1., A, B)){
//	return gemm(1., A, B);}

//template<class Context, class A2D, class B2D, class = std::enable_if_t<blas::is_context<Context>{}> > 
//auto gemm(Context&& ctx, A2D const& A, B2D const& B)
//->decltype(gemm(std::forward<Context>(ctx), 1., A, B)){
//	return gemm(std::forward<Context>(ctx), 1., A, B);}

template<class ContextPtr, class Scalar, class ItA, class ItB, class DecayType>
class gemm_range;

template<class Ext>
struct gemm_reference{
	Ext x;
	Ext const& extensions() const{return x;}
	friend Ext const& extensions(gemm_reference const& self){return self.extensions();}
};

template<class ContextPtr, class Scalar, class ItA, class ItB>
class gemm_iterator{
	ContextPtr ctxtp_;
	Scalar s_;
	ItA a_it_;
	ItB b_begin_;
	gemm_iterator(ContextPtr ctxtp, Scalar s, ItA a_it, ItB b_begin) : ctxtp_{ctxtp}, s_{s}, a_it_{a_it}, b_begin_{b_begin}{}
	template<class ContextPtr2, class Scalar2, class ItA2, class ItB2, class DecayType2>
	friend class gemm_range;
public:
	gemm_iterator(gemm_iterator const&) = default;
	using difference_type = typename std::iterator_traits<ItA>::difference_type;
	using value_type = typename std::iterator_traits<ItA>::value_type;
	using pointer = void;
	using reference = void;
	using iterator_category = std::random_access_iterator_tag;
//	using iterator_category = std::output_iterator_tag;
//	friend difference_type distance(gemv_iterator const& a, gemv_iterator const& b){assert(a.v_first_ == b.v_first_);
//		return b.m_it_ - a.m_it_;
//	}
	friend difference_type operator-(gemm_iterator const& a, gemm_iterator const& b){assert(a.b_begin_ == b.b_begin_);
		return a.a_it_ - b.a_it_;
	}
	friend bool operator==(gemm_iterator const& a, gemm_iterator const& b){return a.a_it_ == b.a_it_;}
	friend bool operator!=(gemm_iterator const& a, gemm_iterator const& b){return a.a_it_ != b.a_it_;}

	template<class ItOut> 
	friend auto copy(gemm_iterator const& first, gemm_iterator const& last, ItOut d_first)
	->decltype(blas::gemm_n(*std::declval<ContextPtr>(), std::declval<Scalar>(), std::declval<ItA>(), std::declval<ItA>() - std::declval<ItA>(), std::declval<ItB>(), 0., d_first)){assert( first.s_ == last.s_ );
		return blas::gemm_n(*first.ctxtp_              , first.s_              , first.a_it_        , last.a_it_ - first.a_it_                 , first.b_begin_     , 0., d_first);}

	template<class ItOut>
	friend auto uninitialized_copy(gemm_iterator const& first, gemm_iterator const& last, ItOut const& d_first)
	->decltype(blas::gemm_n(*std::declval<ContextPtr>(), std::declval<Scalar>(), std::declval<ItA>(), std::declval<ItA>() - std::declval<ItA>(), std::declval<ItB>(), 0., d_first)){assert( first.s_ == last.s_ );
		return blas::gemm_n(*first.ctxtp_              , first.s_              , first.a_it_        , last.a_it_ - first.a_it_                 , first.b_begin_     , 0., d_first);}

	gemm_reference<decltype(b_begin_->extensions())> operator*() const{return {b_begin_->extensions()};}
};

template<class ContextPtr, class Scalar, class ItA, class ItB, class DecayType>
class gemm_range{
	ContextPtr ctxtp_;
	Scalar s_;
	ItA a_begin_;
	ItA a_end_;
	ItB b_begin_;
public:
	gemm_range(gemm_range const&) = delete;
	gemm_range(ContextPtr ctxtp, Scalar s, ItA a_first, ItA a_last, ItB b_first) : ctxtp_{ctxtp}, s_{s}, a_begin_{a_first}, a_end_{a_last}, b_begin_{b_first}{}
	using iterator = gemm_iterator<ContextPtr, Scalar, ItA, ItB>;
	using decay_type = DecayType;
	using size_type = typename decay_type::size_type;
	iterator begin() const{return {ctxtp_, s_, a_begin_, b_begin_};}
	iterator end()   const{return {ctxtp_, s_, a_end_  , b_begin_};}
	friend auto begin(gemm_range const& self){return self.begin();}
	friend auto end  (gemm_range const& self){return self.end  ();}
	size_type size() const{return a_end_ - a_begin_;}
	typename decay_type::extensions_type extensions() const{return size()*b_begin_->extensions();}
	friend auto extensions(gemm_range const& self){return self.extensions();}
//	operator decay_type() const{return decay_type(*this);} // do not use curly { }
	decay_type operator+() const try{return *this;}
	catch(std::exception const& e){
		throw std::logic_error(
			"in " + std::string(__PRETTY_FUNCTION__) + "\nCouldn't decay product of arrays of size " + std::to_string(size()) +"x"+ std::to_string(a_begin_->size()) + " and " + 
			std::to_string(a_begin_->size())+ "x" +std::to_string(b_begin_->size()) + " into " + std::to_string(size()) +"x" + std::to_string(b_begin_->size()) +
			"\nbecause\n"+e.what()
		);
	}
};

template<class ContextPtr, class Scalar, class A2D, class B2D, class=std::enable_if_t<is_context<decltype(*ContextPtr{})>{}> >
gemm_range<ContextPtr, Scalar, typename A2D::const_iterator, typename B2D::const_iterator, typename A2D::decay_type/*B2D*/> 
gemm(ContextPtr ctxtp, Scalar s, A2D const& a, B2D const& b){
	return {ctxtp, s, begin(a), end(a), begin(b)};
}

template<               class Scalar, class A2D, class B2D> 
auto gemm(                Scalar s, A2D const& a, B2D const& b){
	return blas::gemm<blas::context*>(nullptr, s, a, b);
}

namespace operators{
	template<class A2D, class B2D> 
	auto operator*(A2D const& A, B2D const& B)
	->decltype(+blas::gemm(1., A, B)){
		return +blas::gemm(1., A, B);}
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

