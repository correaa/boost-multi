// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_GEMV_HPP
#define MULTI_ADAPTORS_BLAS_GEMV_HPP

#include "../blas/core.hpp"

#include "../blas/dot.hpp"

#include "./../../detail/../utility.hpp"
#include "./../../detail/../array_ref.hpp"

//#include<cblas64.h>

#include "../blas/operations.hpp"

namespace boost{
namespace multi::blas{

using core::gemv;

template<class A, class MIt, class Size, class XIt, class B, class YIt>
auto gemv_n(A a, MIt m_first, Size count, XIt x_first, B b, YIt y_first){
	assert(m_first->stride()==1 or m_first.stride()==1); // blas doesn't implement this case
	if constexpr(not is_conjugated<MIt>{}){
		;;;; if(m_first .stride()==1) gemv('N', count, m_first->size(), a, m_first.base()            , m_first->stride(), x_first.base(), x_first.stride(), b, y_first.base(), y_first.stride());
		else if(m_first->stride()==1) gemv('T', m_first->size(), count, a, m_first.base()            , m_first. stride(), x_first.base(), x_first.stride(), b, y_first.base(), y_first.stride());
		else                          assert(0);
	}else{
		;;;; if(m_first->stride()==1) gemv('C', m_first->size(), count, a, underlying(m_first.base()), m_first. stride(), x_first.base(), x_first.stride(), b, y_first.base(), y_first.stride());
		else if(m_first. stride()==1) assert(0); // not implemented in blas (use cblas?)
		else                          assert(0); // not implemented in blas
	}
	struct{
		MIt m_last;
		YIt y_last;
	} ret{m_first + count, y_first + count};
	return ret;
}

//template<class A, std::enable_if_t<not is_conjugated<A>{}, int> =0>
//auto gemv_base_aux(A&& a){return base(a);}

//template<class A, std::enable_if_t<    is_conjugated<A>{}, int> =0>
//auto gemv_base_aux(A&& a){return underlying(base(a));}
//template<class A, class X, class Y>
//auto gemv(typename A::element alpha, A const& a, X const& x, typename A::element beta, Y&& y)
//->decltype(gemv('N', x.size(), y.size(), alpha, gemv_base_aux(a), a.rotated().stride(), x.base(), x.stride(), beta, y.base(), y.stride()), std::forward<Y>(y)){
//	assert(a.rotated().size() == x.size());
//	assert(a.size() == y.size());
//	
//	auto base_A = gemv_base_aux(a);
//	
//	assert(stride(a)==1 or stride(~a)==1);
//	
//	assert(y.size() == a.size());
//	     if(stride( a)==1 and not is_conjugated<A>{})        gemv('N'                          , size(y), size(x),  alpha, base_A, stride(~a), base(x), stride(x),  beta, base(y), stride(y));
//	else if(stride(~a)==1 and not is_conjugated<A>{})        gemv('T'                          , size(x), size(y),  alpha, base_A, stride( a), base(x), stride(x),  beta, base(y), stride(y));
//	else if(stride(~a)==1 and     is_conjugated<A>{})        gemv('C'                          , size(x), size(y),  alpha, base_A, stride( a), base(x), stride(x),  beta, base(y), stride(y));
//	else if(stride( a)==1 and     is_conjugated<A>{}){
//		assert(0&&"case not supported by blas");
//	//	cblas_zgemv(CblasRowMajor, CblasConjTrans, size(x), size(y), &alpha, base_A, stride(~a), base(x), stride(x), &beta, base(y), stride(y));
//	}
//	
//	return std::forward<Y>(y);
//}

//template<class A2D, class X1D, class Y1D>
//auto gemv(A2D const& A, X1D const& x, Y1D&& y)
//->decltype(gemv_n(1., A, A.size(), x.begin(), 0., std::forward<Y1D>(y).begin())){
//	return gemv_n(1., A, A.size(), x.begin(), 0., std::forward<Y1D>(y).begin());}

//template<class Alloc, class A2D, class X1D, typename T = typename A2D::element>
//NODISCARD("")
//auto gemv(A2D const& A, X1D const& x, Alloc const& alloc = {})->std::decay_t<
//decltype(gemv(1., A, x, 0., multi::array<T, 1, Alloc>(A.size(), alloc)))>{
//	return gemv(1., A, x, 0., multi::array<T, 1, Alloc>(A.size(), alloc));}

//template<class A2D, class X1D> NODISCARD()
//auto gemv(A2D const& A, X1D const& x){return gemv(A, x, get_allocator(x));}

//namespace operators{
//	template<class A2D, class X1D> auto operator%(A2D const& A, X1D const& x) DECLRETURN(gemv(A, x))
//}

template<class Scalar, class It2D, class It1D>
class gemv_iterator{
	Scalar alpha_ = 1.;
	It2D m_it_; 
	It1D v_first_;
public:
	using difference_type = typename std::iterator_traits<It2D>::difference_type;
	using value_type = typename std::iterator_traits<It1D>::value_type;
	using pointer = void;
	using reference = void;
	using iterator_category = std::random_access_iterator_tag;
//	using iterator_category = std::output_iterator_tag;
//	friend difference_type distance(gemv_iterator const& a, gemv_iterator const& b){assert(a.v_first_ == b.v_first_);
//		return b.m_it_ - a.m_it_;
//	}
	friend difference_type operator-(gemv_iterator const& a, gemv_iterator const& b){assert(a.v_first_ == b.v_first_);
		return a.m_it_ - b.m_it_;
	}
	template<class It1DOut>
	friend auto copy_n(gemv_iterator first, difference_type count, It1DOut result){
		return blas::gemv_n(first.alpha_, first.m_it_, count, first.v_first_, 0., result).y_last;
	}
	template<class It1DOut>
	friend auto copy(gemv_iterator first, gemv_iterator last, It1DOut result){return copy_n(first, last - first, result);}
	template<class It1DOut>
	friend auto uninitialized_copy(gemv_iterator first, gemv_iterator last, It1DOut result){
		static_assert(std::is_trivially_default_constructible<typename It1DOut::value_type>{});
		return copy(first, last, result);
	}
	gemv_iterator(Scalar alpha, It2D m_it, It1D v_first) 
		: alpha_{alpha}, m_it_{m_it}, v_first_{v_first}{}
	value_type operator*() const{return 0.;}
};

template<class Scalar, class It2D, class It1D, class DecayType = void>
class gemv_range{
	Scalar alpha_ = 1.;
	It2D m_begin_;
	It2D m_end_;
	It1D v_first_;
public:
	gemv_range(Scalar alpha, It2D m_first, It2D m_last, It1D v_first) 
		: alpha_{alpha}, m_begin_{m_first}, m_end_{m_last}, v_first_{v_first}{
		assert(m_begin_.stride() == m_end_.stride());
	}
//	gemv_range(It2D m_first, It2D m_last, It1D v_first) : gemv_range{1., m_first, m_last, v_first}{}
	using iterator = gemv_iterator<Scalar, It2D, It1D>;
	using decay_type = DecayType;
	iterator begin() const{return {alpha_, m_begin_, v_first_};}
	iterator end()   const{return {alpha_, m_end_  , v_first_};}
	size_type size() const{return end() - begin();}
	decay_type decay() const{
		decay_type ret; 
		ret = *this;
		return ret;
	}
	friend auto operator+(gemv_range const& self){return self.decay();}
};

template<class Scalar, class M, class V>
auto gemv(Scalar s, M const& m, V const& v)
{//->decltype(gemv_range{s, m, v}){
	assert(size(~m) == size(v));
	return gemv_range<Scalar, typename M::const_iterator, typename V::const_iterator, typename V::decay_type>(s, m.begin(), m.end(), v.begin());}

template<class M, class V>
auto gemv(M const& m, V const& v)
{//->decltype(gemv(1., m, v)){
	return gemv(1., m, v);}
	
namespace operators{
	template<class M, class V>
	auto operator%(M const& m, V const& v)
	->decltype(+blas::gemv(m, v)){
		return +blas::gemv(m, v);}
}

}
}

#endif

