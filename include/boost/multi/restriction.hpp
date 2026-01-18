// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_RESTRICTION_HPP
#define BOOST_MULTI_RESTRICTION_HPP

#include <boost/multi/detail/layout.hpp>

namespace boost::multi {

template<class T>
constexpr auto make_restriction(std::initializer_list<T> const& il) {
	return [il](multi::index i) { return il.begin()[i]; } ^ multi::extensions(il);
}

template<class T>
constexpr auto make_restriction(std::initializer_list<std::initializer_list<T>> const& il) {
	return [il](multi::index i, multi::index j) { return il.begin()[i].begin()[j]; } ^ multi::extensions(il);
}

template<class T>
constexpr auto make_restriction(std::initializer_list<std::initializer_list<std::initializer_list<T>>> const& il) {
	return [il](multi::index i, multi::index j, multi::index k) { return il.begin()[i].begin()[j].begin()[k]; } ^ multi::extensions(il);
}

}  // namespace boost::multi

#endif  // BOOST_MULTI_RESTRICTION_HPP
