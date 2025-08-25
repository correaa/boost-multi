// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_EXTENSIONS_HPP
#define BOOST_MULTI_DETAIL_EXTENSIONS_HPP
#pragma once

#include <boost/multi/detail/index_range.hpp>

#include <tuple>

#if defined(__NVCC__)
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

namespace boost::multi::detail {

template<class... Exts>
class extensions;

template<>
class extensions<> {
	public:
	extensions() = default;
};

template<class Ex, class... Exts>
class extensions<Ex, Exts...> : private Ex {
	extensions<Exts...> rest_;

 public:
	extensions() = default;

	BOOST_MULTI_HD explicit constexpr extensions(Ex ex, Exts... rest) : Ex{ex}, rest_{rest...} {}

	template<std::size_t I>
	BOOST_MULTI_HD constexpr auto get() const {
		if constexpr(I == 0) {
			return static_cast<Ex const&>(*this); }
		else {
			return rest_.template get<I - 1>();
		}
	}

	BOOST_MULTI_HD constexpr auto extension() const { return static_cast<Ex const&>(*this); }
	BOOST_MULTI_HD constexpr auto sub() const { return rest_; }
};

template<class... Exts> extensions(Exts...) -> extensions<Exts...>;

template<std::size_t I, class... Exts>
auto get(::boost::multi::detail::extensions<Exts...> const& exts)
->decltype(exts.template get<I>()) {
	return exts.template get<I>(); }

template<class Self>
struct tyid {
	using type = Self;
};

}  // end namespace boost::multi::detail

template<class... Exts>
struct std::tuple_size<::boost::multi::detail::extensions<Exts...> > {
	static constexpr std::size_t value = sizeof...(Exts);
};

template<std::size_t I, class Ex, class... Exts>
struct std::tuple_element<I, ::boost::multi::detail::extensions<Ex, Exts...> > {
	using type = typename std::conditional_t<
		I == 0,
		::boost::multi::detail::tyid<Ex>,
		::std::tuple_element<I - 1, ::boost::multi::detail::extensions<Exts...>>
	>::type;
};

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_DETAIL_EXTENSIONS_HPP
