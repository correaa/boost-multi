// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ELEMENTWISE_PLUS_HPP
#define BOOST_MULTI_ELEMENTWISE_PLUS_HPP

#include <boost/multi/elementwise/invoke.hpp>

#include <cassert>
#include <functional>

namespace boost::multi::elementwise {

template<class A, class B>
auto plus(A const& alpha, B const& omega) {
	return invoke(::std::plus<>{}, alpha, omega);
}

}  // namespace boost::multi::elementwise

#endif
