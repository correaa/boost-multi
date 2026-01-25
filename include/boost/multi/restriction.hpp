// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_RESTRICTION_HPP
#define BOOST_MULTI_RESTRICTION_HPP

#include <boost/multi/detail/layout.hpp>
#include <boost/multi/utility.hpp>

namespace boost::multi::detail {

template<class T>
constexpr auto make_restriction(std::initializer_list<T> const& il) {
	return [il](multi::index i0) { return il.begin()[i0]; } ^ multi::extensions(il);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

template<class T>
constexpr auto make_restriction(std::initializer_list<std::initializer_list<T>> const& il) {
	return [il](multi::index i0, multi::index i1) { return il.begin()[i0].begin()[i1]; } ^ multi::extensions(il);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template<class T>
constexpr auto make_restriction(std::initializer_list<std::initializer_list<std::initializer_list<T>>> const& il) {
	return [il](multi::index i0, multi::index i1, multi::index i2) { return il.begin()[i0].begin()[i1].begin()[i2]; } ^ multi::extensions(il);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

}  // namespace boost::multi::detail

namespace boost::multi {

template<dimensionality_type D, typename Fun>
auto restrict(Fun&& fun, extensions_t<D> const& ext) {
	return restriction<D, Fun>(ext, std::forward<Fun>(fun));
}

template<typename Fun>
auto restrict(Fun&& fun, extensions_t<0> const& ext) {
	return restriction<0, Fun>(ext, std::forward<Fun>(fun));
}
template<typename Fun>
auto restrict(Fun&& fun, extensions_t<1> const& ext) {
	return restriction<1, Fun>(ext, std::forward<Fun>(fun));
}
template<typename Fun>
auto restrict(Fun&& fun, extensions_t<2> const& ext) {
	return restriction<2, Fun>(ext, std::forward<Fun>(fun));
}

}  // namespace boost::multi

#endif  // BOOST_MULTI_RESTRICTION_HPP
