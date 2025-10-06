// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_WHAT_HPP
#define BOOST_MULTI_DETAIL_WHAT_HPP

namespace boost::multi::detail {
	template<class... Ts> auto what() -> std::tuple<Ts&&...>        = delete;
	template<class... Ts> auto what(Ts&&...) -> std::tuple<Ts&&...> = delete;  // NOLINT(cppcoreguidelines-missing-std-forward)

	template<int V> auto what_value() -> std::integral_constant<int, V> = delete;
	template<int V> struct what_value_t;
}  // namespace boost::multi::detail

#endif  // BOOST_MULTI_DETAIL_WHAT_HPP
