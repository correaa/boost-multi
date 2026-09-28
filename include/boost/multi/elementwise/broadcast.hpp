// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ELEMENTWISE_BROADCAST_HPP
#define BOOST_MULTI_ELEMENTWISE_BROADCAST_HPP

#include "boost/multi/restriction.hpp"
#include "boost/multi/utility.hpp"
#include "boost/multi/elementwise/invoke.hpp"

namespace boost::multi::elementwise {

/// yields an array expression that would result from invoking a function of `n` arguments to corresponding elements of arrays (extents and dimensionality are adapted when possible).
template<class Fun, class A>
auto broadcast(Fun&& fun, A&& alpha) {
	return elementwise::invoke(std::forward<Fun>(fun), std::forward<A>(alpha));  // qualified: unqualified `invoke` is ADL-hijacked by `std::invoke` when Fun is a std:: functor (e.g. std::divides<>)
}

template<class Fun, class A, class B>
auto broadcast(Fun&& fun, A&& alpha, B&& beta) {
	if constexpr(!multi::has_dimensionality<std::decay_t<A>>::value) {
		return elementwise::broadcast(
			std::forward<Fun>(fun),
			[alpha_ = std::forward<A>(alpha)]() { return alpha_; } ^ multi::extents_t<0>{},
			std::forward<B>(beta)
		);
	} else if constexpr(!multi::has_dimensionality<std::decay_t<B>>::value) {
		return elementwise::broadcast(
			std::forward<Fun>(fun),
			std::forward<A>(alpha),
			[beta_ = std::forward<B>(beta)]() { return beta_; } ^ multi::extents_t<0>{}
		);
	} else {
		if constexpr(std::decay_t<A>::dimensionality < std::decay_t<B>::dimensionality) {
			using std::get;
			return elementwise::broadcast(
				std::forward<Fun>(fun),
				std::forward<A>(alpha).repeated(get<std::decay_t<B>::dimensionality - std::decay_t<A>::dimensionality - 1>(beta.sizes())),
				std::forward<B>(beta)
			);
		} else if constexpr(std::decay_t<A>::dimensionality > std::decay_t<B>::dimensionality) {
			using std::get;
			return elementwise::broadcast(
				std::forward<Fun>(fun),
				std::forward<A>(alpha),
				std::forward<B>(beta).repeated(get<std::decay_t<A>::dimensionality - std::decay_t<B>::dimensionality - 1>(alpha.sizes()))
			);
		} else {
			return elementwise::invoke(  // qualified: unqualified `invoke` is ADL-hijacked by `std::invoke` when Fun is a std:: functor (e.g. std::divides<>)
				std::forward<Fun>(fun),
				std::forward<A>(alpha),
				std::forward<B>(beta)
			);
		}
	}
}

}  // namespace boost::multi::elementwise

#endif
