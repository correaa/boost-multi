// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ELEMENTWISE_INVOKE_HPP
#define BOOST_MULTI_ELEMENTWISE_INVOKE_HPP

#include <boost/multi/restriction.hpp>
#include <boost/multi/utility.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <tuple>
#include <utility>

namespace boost::multi::elementwise {

template<class Fun, class A, class... Bs>
auto invoke(Fun&& fun, A const& alpha, Bs const&... bs) {
	assert(((alpha.extents() == bs.extents()) && ...));
	return
		[fun_   = std::forward<Fun>(fun),
		 homes_ = std::make_tuple(alpha.home(), bs.home()...)](auto... idxs) {
			using ::boost::multi::detail::invoke_square;
			return std::apply(
				[&](auto const&... homes) { return fun_(invoke_square(homes, idxs...)...); },
				homes_
			);
		}
	^ alpha.extents();
}

template<class Fun, class A, class B>
auto broadcast(Fun&& fun, A&& alpha, B&& beta) {
	if constexpr(!multi::has_dimensionality<std::decay_t<A>>::value) {
		return broadcast(
			std::forward<Fun>(fun),
			[alpha_ = std::forward<A>(alpha)]() { return alpha_; } ^ multi::extents_t<0>{},
			std::forward<B>(beta)
		);
	} else if constexpr(!multi::has_dimensionality<std::decay_t<B>>::value) {
		return broadcast(
			std::forward<Fun>(fun),
			std::forward<A>(alpha),
			[beta_ = std::forward<B>(beta)]() { return beta_; } ^ multi::extents_t<0>{}
		);
	} else {
		if constexpr(std::decay_t<A>::dimensionality < std::decay_t<B>::dimensionality) {
			return broadcast(
				std::forward<Fun>(fun),
				std::forward<A>(alpha).repeated(beta.size()),
				std::forward<B>(beta)
			);
		} else if constexpr(std::decay_t<A>::dimensionality > std::decay_t<B>::dimensionality) {
			return broadcast(
				std::forward<Fun>(fun),
				std::forward<A>(alpha),
				std::forward<B>(beta).repeated(alpha.size())
			);
		} else {
			return invoke(
				std::forward<Fun>(fun),
				std::forward<A>(alpha),
				std::forward<B>(beta)
			);
		}
	}
}

}  // namespace boost::multi::elementwise

#endif
