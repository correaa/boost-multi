// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ELEMENTWISE_INVOKE_HPP
#define BOOST_MULTI_ELEMENTWISE_INVOKE_HPP

#include "boost/multi/restriction.hpp"
#include "boost/multi/utility.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <tuple>
#include <utility>

namespace boost::multi::elementwise {

/// yields an array expression that would result from invoking a function of `n` arguments to corresponding elements of `n` arrays (all extents must match).
template<class Fun, class A, class... Bs>
auto invoke(Fun&& fun, A&& alpha, Bs&&... bs) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	auto const exts = alpha.extents();
	assert(((exts == bs.extents()) && ...));
	return
		[fun_   = std::forward<Fun>(fun),
		 homes_ = std::make_tuple(std::forward<A>(alpha).home(), std::forward<Bs>(bs).home()...)](auto... idxs) {
			using ::boost::multi::detail::invoke_square;
			return std::apply(
				[&](auto const&... homes) { return fun_(invoke_square(homes, idxs...)...); },
				homes_
			);
		}
	^ exts;
}

}  // namespace boost::multi::elementwise

#endif  // BOOST_MULTI_ELEMENTWISE_INVOKE_HPP
