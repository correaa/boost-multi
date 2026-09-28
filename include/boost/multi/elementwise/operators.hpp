// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ELEMENTWISE_OPERATORS_HPP
#define BOOST_MULTI_ELEMENTWISE_OPERATORS_HPP

#include "boost/multi/elementwise/broadcast.hpp"
#include "boost/multi/elementwise/invoke.hpp"

#include <cassert>
#include <functional>

namespace boost::multi::elementwise {

/// yields a array with the `+` operation (plus) applied lazily elementwise to two arrays
template<class A, class B, std::enable_if_t<has_dimensionality<std::decay_t<A>>::value || has_dimensionality<std::decay_t<B>>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
auto operator+(A&& alpha, B&& omega) {
	return elementwise::broadcast(::std::plus<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

/// yields a array with the `-` operation (minus) applied lazily elementwise to two arrays
template<class A, class B, std::enable_if_t<has_dimensionality<std::decay_t<A>>::value || has_dimensionality<std::decay_t<B>>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
auto operator-(A&& alpha, B&& omega) {
	return elementwise::broadcast(::std::minus<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

/// yields an array expression with the `*` operation (multiplies) applied lazily elementwise to two arrays.
template<class A, class B>
auto operator*(A&& alpha, B&& omega) {
	return elementwise::broadcast(::std::multiplies<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

/// yields an array expression with the `/` operation (divides) applied lazily elementwise to two arrays
template<class A, class B>
auto operator/(A&& alpha, B&& omega) {
	return elementwise::broadcast(::std::divides<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

template<class A, class B>
auto operator%(A&& alpha, B&& omega) {
	return elementwise::broadcast(::std::modulus<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

/// yields an array expression with the `-` operation (unary prefix negate) applied lazily elementwise to one arrays
template<class A>
auto operator-(A&& alpha) {
	return elementwise::broadcast(::std::negate<>{}, std::forward<A>(alpha));
}

}  // namespace boost::multi::elementwise

#endif
