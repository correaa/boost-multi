// Copyright 2019-2024 Alfredo A. Correa

#ifndef BOOST_MULTI_ADAPTORS_BLAS_HERK_HPP
#define BOOST_MULTI_ADAPTORS_BLAS_HERK_HPP
#pragma once

#include <boost/multi/adaptors/blas/copy.hpp>
#include <boost/multi/adaptors/blas/core.hpp>
#include <boost/multi/adaptors/blas/filling.hpp>
#include <boost/multi/adaptors/blas/operations.hpp>
#include <boost/multi/adaptors/blas/side.hpp>
#include <boost/multi/adaptors/blas/syrk.hpp>  // fallback to real case

// IWYU pragma: no_include "boost/multi/adaptors/blas/traits.hpp"      // for blas

namespace boost::multi::blas {

template<class A,
	std::enable_if_t<! is_conjugated<A>{}, int> =0>   // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto base_aux(A&& array)
->decltype((std::forward<A>(array)).base()) {
	return (std::forward<A>(array)).base(); }

template<class A,
	std::enable_if_t<    is_conjugated<A>{}, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto base_aux(A&& array)
->decltype(underlying((std::forward<A>(array)).base())) {
	return underlying((std::forward<A>(array)).base()); }

using core::herk;

template<class ContextPtr, class Scalar, class ItA, class DecayType>
class herk_range {
	ContextPtr ctxtp_;
	Scalar s_;
	ItA a_begin_;
	ItA a_end_;

 public:
	herk_range(herk_range const&) = delete;
	herk_range(herk_range&&) = delete;
	auto operator=(herk_range const&) -> herk_range& = delete;
	auto operator=(herk_range&&) -> herk_range& = delete;
	~herk_range() = default;

	herk_range(ContextPtr ctxtp, Scalar s, ItA a_first, ItA a_last)  // NOLINT(bugprone-easily-swappable-parameters,readability-identifier-length) BLAS naming
	: ctxtp_{ctxtp}
	, s_{s}, a_begin_{std::move(a_first)}, a_end_{std::move(a_last)}
	{}

	// using iterator = herk_iterator<ContextPtr, Scalar, ItA>;
	using decay_type = DecayType;
	using size_type = typename decay_type::size_type;

	//        auto begin()          const& -> iterator {return {ctxtp_, s_, a_begin_, b_begin_};}
	//        auto end()            const& -> iterator {return {ctxtp_, s_, a_end_  , b_begin_};}
	// friend auto begin(gemm_range const& self) {return self.begin();}
	// friend auto end  (gemm_range const& self) {return self.end  ();}

	// auto size() const -> size_type {return a_end_ - a_begin_;}

	// auto extensions() const -> typename decay_type::extensions_type {return size()*(*b_begin_).extensions();}
	// friend auto extensions(gemm_range const& self) {return self.extensions();}

	// auto operator+() const -> decay_type {return *this;} // TODO(correaa) : investigate why return decay_type{*this} doesn't work
	// template<class Arr>
	// friend auto operator+=(Arr&& a, gemm_range const& self) -> Arr&& {  // NOLINT(readability-identifier-length) BLAS naming
	//  blas::gemm_n(self.ctxtp_, self.s_, self.a_begin_, self.a_end_ - self.a_begin_, self.b_begin_, 1., a.begin());
	//  return std::forward<Arr>(a);
	// }
	// friend auto operator*(Scalar factor, gemm_range const& self) {
	//  return gemm_range{self.ctxtp_, factor*self.s_, self.a_begin_, self.a_end_, self.b_begin_};
	// }
};

template<class AA, class BB, class A2D, class C2D, class = typename A2D::element_ptr,
	std::enable_if_t<is_complex_array<C2D>{}, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto herk(filling c_side, AA alpha, A2D const& a, BB beta, C2D&& c) -> C2D&& {  // NOLINT(readability-function-cognitive-complexity,readability-identifier-length) 74, BLAS naming
	assert( a.size() == c.size() ); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
	assert( c.size() == c.rotated().size() ); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
	if(c.is_empty()) {return std::forward<C2D>(c);}
	if constexpr(is_conjugated<C2D>{}) {
		herk(flip(c_side), alpha, a, beta, hermitized(c));
		return std::forward<C2D>(c);
	}

	auto base_a = base_aux(a);  // NOLINT(llvm-qualified-auto,readability-qualified-auto) TODO(correaa)
	auto base_c = base_aux(c);  // NOLINT(llvm-qualified-auto,readability-qualified-auto) TODO(correaa)
	if constexpr(is_conjugated<A2D>{}) {
	//  auto& ctxt = *blas::default_context_of(underlying(a.base()));
		// if you get an error here might be due to lack of inclusion of a header file with the backend appropriate for your type of iterator
		if     (stride(a)==1 && stride(c)!=1) {herk(c_side==filling::upper?'L':'U', 'N', size(c), a.rotated().size(), &alpha, base_a, a.rotated().stride(), &beta, base_c, stride(c));}
		else if(stride(a)==1 && stride(c)==1) {
			if(size(a)==1)                     {herk(c_side==filling::upper?'L':'U', 'N', size(c), a.rotated().size(), &alpha, base_a, a.rotated().stride(), &beta, base_c, stride(c));}
			else                               {assert(0);} // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
		}
		else if(stride(a)!=1 && stride(c)==1) { herk(c_side==filling::upper?'U':'L', 'C', size(c), a.rotated().size(), &alpha, base_a, stride(        a ), &beta, base_c, c.rotated().stride());}
		else if(stride(a)!=1 && stride(c)!=1) { herk(c_side==filling::upper?'L':'U', 'C', size(c), a.rotated().size(), &alpha, base_a, stride(        a ), &beta, base_c, stride(        c ));}
		else                                  { assert(0);} // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
	} else {
	//  auto& ctxt = *blas::default_context_of(           a.base() );
		if     (stride(a)!=1 && stride(c)!=1) { herk(c_side==filling::upper?'L':'U', 'C', size(c), a.rotated().size(), &alpha, base_a, stride(        a ), &beta, base_c, stride(c));}
		else if(stride(a)!=1 && stride(c)==1) {
			if(size(a)==1)                    { herk(c_side==filling::upper?'L':'U', 'N', size(c), a.rotated().size(), &alpha, base_a, a.rotated().stride(), &beta, base_c, c.rotated().stride());}
			else                              { assert(0);} // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
		}
		else if(stride(a)==1 && stride(c)!=1) {assert(0);} // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
		else if(stride(a)==1 && stride(c)==1) {herk(c_side==filling::upper?'U':'L', 'N', size(c), a.rotated().size(), &alpha, base_a, a.rotated().stride(), &beta, base_c, c.rotated().stride());}
	//  else                                   {assert(0);} // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
	}

	return std::forward<C2D>(c);
}

template<class AA, class BB, class A2D, class C2D, class = typename A2D::element_ptr,
	std::enable_if_t<! is_complex_array<C2D>{}, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto herk(filling c_side, AA alpha, A2D const& a, BB beta, C2D&& c)  // NOLINT(readability-identifier-length) BLAS naming
->decltype(syrk(c_side, alpha, a, beta, std::forward<C2D>(c))) {
	return syrk(c_side, alpha, a, beta, std::forward<C2D>(c)); }

template<class AA, class A2D, class C2D, class = typename A2D::element_ptr>
auto herk(filling c_side, AA alpha, A2D const& a, C2D&& c)  // NOLINT(readability-identifier-length) BLAS naming
->decltype(herk(c_side, alpha, a, 0., std::forward<C2D>(c))) {
	return herk(c_side, alpha, a, 0., std::forward<C2D>(c)); }

template<typename AA, class A2D, class C2D>
auto herk(AA alpha, A2D const& a, C2D&& c)  // NOLINT(readability-identifier-length) BLAS naming
->decltype(herk(filling::lower, alpha, a, herk(filling::upper, alpha, a, std::forward<C2D>(c)))) {
	return herk(filling::lower, alpha, a, herk(filling::upper, alpha, a, std::forward<C2D>(c))); }

template<class A2D, class C2D>
auto herk(A2D const& a, C2D&& c)  // NOLINT(readability-identifier-length) BLAS naming
->decltype(herk(1., a, std::forward<C2D>(c))) {
	return herk(1., a, std::forward<C2D>(c)); }

template<class AA, class A2D, class Ret = typename A2D::decay_type>
[[nodiscard]]  // ("when argument is read-only")
auto herk(AA alpha, A2D const& a) {  // NOLINT(readability-identifier-length) BLAS naming
	return herk(alpha, a, Ret({size(a), size(a)}));//Ret({size(a), size(a)}));//, get_allocator(a)));
}

template<class T> struct numeric_limits : std::numeric_limits<T> {};
template<class T> struct numeric_limits<std::complex<T>> : std::numeric_limits<std::complex<T>> {
	static auto quiet_NaN() -> std::complex<T> {auto nana = numeric_limits<T>::quiet_NaN(); return {nana, nana};}  // NOLINT(readability-identifier-naming) conventional std name
};

template<class AA, class A2D, class Ret = typename A2D::decay_type>
[[nodiscard]]  // ("because argument is read-only")]]
auto herk(filling cs, AA alpha, A2D const& a)  // NOLINT(readability-identifier-length) BLAS naming
->std::decay_t<
decltype(  herk(cs, alpha, a, Ret({size(a), size(a)}, 0., get_allocator(a))))> {
	return herk(cs, alpha, a, Ret({size(a), size(a)},
#ifdef NDEBUG
		numeric_limits<typename Ret::element_type>::quiet_NaN(),
#endif
		get_allocator(a)
	));
}

template<class A2D> auto herk(filling s, A2D const& a)  // NOLINT(readability-identifier-length) BLAS naming
->decltype(herk(s, 1., a)) {
	return herk(s, 1., a); }

template<class A2D> auto herk(A2D const& array) {
		return herk(1., array);
}

}  // end namespace boost::multi::blas
#endif
