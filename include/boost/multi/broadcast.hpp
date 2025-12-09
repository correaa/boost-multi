// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_BROADCAST_HPP
#define BOOST_MULTI_BROADCAST_HPP

#include <boost/multi/array_ref.hpp>
#include <boost/multi/utility.hpp>  // for multi::detail::apply_square

// #include <boost/multi/detail/tuple_zip.hpp>
// #include <boost/multi/utility.hpp>  // IWYU pragma: export

// #include <cmath>
// #include <type_traits>

// #if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<ranges>)
// #include <ranges>  // IWYU pragma: keep
// #endif

// #ifdef _MSC_VER
// #pragma warning(push)
// #pragma warning(disable : 4623)  // assignment operator was implicitly defined as deleted
// #pragma warning(disable : 4626)  // assignment operator was implicitly defined as deleted
// #pragma warning(disable : 4625)  // copy constructor was implicitly defined as deleted
// #endif

// #include <boost/multi/detail/adl.hpp>  // TODO(correaa) remove instantiation of force_element_trivial in this header
// #include <boost/multi/detail/config/ASSERT.hpp>
// #include <boost/multi/detail/layout.hpp>          // IWYU pragma: export
// #include <boost/multi/detail/memory.hpp>          // for pointer_traits
// #include <boost/multi/detail/operators.hpp>       // for random_iterable
// #include <boost/multi/detail/pointer_traits.hpp>  // IWYU pragma: export
// #include <boost/multi/detail/serialization.hpp>
// #include <boost/multi/detail/types.hpp>  // for dimensionality_type  // IWYU pragma: export

// #include <algorithm>  // fpr copy_n
// #include <array>
// #include <cstring>     // for std::memset in reinterpret_cast
// #include <functional>  // for std::invoke
// #include <iterator>    // for std::next
// #include <memory>      // for std::pointer_traits
// #include <new>         // for std::launder

namespace boost::multi {

template<class A> struct bind_category {
	using type = A;
};

template<class T, dimensionality_type D, class... Ts>
struct bind_category<::boost::multi::array<T, D, Ts...>> {
	using type = ::boost::multi::array<T, D, Ts...>;
};

template<class T, dimensionality_type D, class... Ts>
struct bind_category<::boost::multi::array<T, D, Ts...>&> {
	using type = ::boost::multi::array<T, D, Ts...>&;
};

template<class T, dimensionality_type D, class... Ts>
struct bind_category<::boost::multi::array<T, D, Ts...> const&> {
	using type = ::boost::multi::array<T, D, Ts...> const&;
};

template<class T, dimensionality_type D, class... Ts>
struct bind_category<::boost::multi::subarray<T, D, Ts...>> {
	using type = ::boost::multi::subarray<T, D, Ts...>;
};

template<class T, dimensionality_type D, class... Ts>
struct bind_category<::boost::multi::subarray<T, D, Ts...>&> {
	using type = ::boost::multi::subarray<T, D, Ts...>&;
};

template<class T, dimensionality_type D, class... Ts>
struct bind_category<::boost::multi::subarray<T, D, Ts...> const&> {
	using type = ::boost::multi::subarray<T, D, Ts...> const&;
};

namespace broadcast {

// #if __cplusplus >= 202302L

template<class F, class A, class... Arrays, typename = decltype(std::declval<F&&>()(std::declval<typename std::decay_t<A>::reference>(), std::declval<typename std::decay_t<Arrays>::reference>()...))>
constexpr auto apply_front(F&& fun, A&& arr, Arrays&&... arrs) {
	return [fun = std::forward<F>(fun), &arr, &arrs...](auto is) { return fun(arr[is], arrs[is]...); } ^ multi::extensions_t<1>({arr.extension()});
}

template<class F, class... A> struct apply_bind_t;

template<class F, class A>
struct apply_bind_t<F, A> {
	F fun_;
	A a_;

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		return fun_(detail::invoke_square(a_, is...));  // a_[is...] in. C++23
	}
};

template<class F, class A, class B>
struct apply_bind_t<F, A, B> {
	F fun_;
	A a_;
	B b_;

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		return fun_(
			multi::detail::invoke_square(a_, is...),  // a_[is...] in C++23
			multi::detail::invoke_square(b_, is...)   // b_[is...] in C++23
		);
	}
};

template<class F, class A, class... As, typename = decltype(std::declval<F&&>()(std::declval<typename std::decay_t<A>::element>(), std::declval<typename std::decay_t<As>::element>()...))>
constexpr auto apply(F&& fun, A&& arr, As&&... arrs) {
	auto xs = arr.extensions();  // TODO(correaa) consider storing home() cursor only
	return apply_bind_t<F, std::decay_t<A>, std::decay_t<As>...>{std::forward<F>(fun), std::forward<A>(arr), std::forward<As>(arrs)...} ^ xs;
	//	return [fun = std::forward<F>(fun), &arr, &arrs...](auto... is) { return fun(arr[is...], arrs[is...]...); } ^ arr.extensions();
}

template<class F, class A, class B>
constexpr auto apply_broadcast(F&& fun, A&& alpha, B&& omega) {
	if constexpr(!multi::has_dimensionality<std::decay_t<A>>::value) {
		return apply_broadcast(std::forward<F>(fun), [alpha = std::forward<A>(alpha)]() { return alpha; } ^ multi::extensions_t<0>{}, std::forward<B>(omega));
	} else if constexpr(!multi::has_dimensionality<std::decay_t<B>>::value) {
		return apply_broadcast(std::forward<F>(fun), std::forward<A>(alpha), [omega = std::forward<B>(omega)]() { return omega; } ^ multi::extensions_t<0>{});
	} else {
		if constexpr(std::decay_t<A>::dimensionality < std::decay_t<B>::dimensionality) {
			return apply_broadcast(std::forward<F>(fun), alpha.repeated(omega.size()), omega);
		} else if constexpr(std::decay_t<B>::dimensionality < std::decay_t<A>::dimensionality) {
			return apply_broadcast(std::forward<F>(fun), alpha, omega.repeated(alpha.size()));
		} else {
			return apply(std::forward<F>(fun), std::forward<A>(alpha), std::forward<B>(omega));
		}
	}
}

// remember that you need C++23 to use the broadcast feature
template<class A, class B>
constexpr auto operator+(A&& alpha, B&& omega) { return broadcast::apply_broadcast(std::plus<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A, class B>
constexpr auto operator-(A&& alpha, B&& omega) { return broadcast::apply_broadcast(std::minus<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A>
constexpr auto operator-(A&& alpha) { return broadcast::apply(std::negate<>{}, std::forward<A>(alpha)); }

template<class A, class B>
constexpr auto operator*(A&& alpha, B&& omega) { return broadcast::apply_broadcast(std::multiplies<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A, class B>
constexpr auto operator/(A&& alpha, B&& omega) {
	return broadcast::apply_broadcast(std::divides<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

template<class A, class B>
constexpr auto operator&&(A&& alpha, B&& omega) { return broadcast::apply_broadcast(std::logical_and<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A, class B>
constexpr auto operator||(A&& a, B&& b) { return broadcast::apply_broadcast(std::logical_or<>{}, std::forward<A>(a), std::forward<B>(b)); }

template<class F, class A, std::enable_if_t<true, decltype(std::declval<F&&>()(std::declval<typename std::decay_t<A>::element>()))*> = nullptr>
constexpr auto operator|(A&& a, F fun) {
	return std::forward<A>(a).element_transformed(fun);
}

template<class F, class A, std::enable_if_t<true, decltype(std::declval<F>()(std::declval<typename std::decay_t<A>::reference>()))*> = nullptr>
constexpr auto operator|(A&& a, F fun) {
	return std::forward<A>(a).transformed(fun);
}

template<class A>
struct exp_bind_t {
	A a_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) TODO(correaa) consider saving .home() cursor

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		using ::std::exp;
		return exp(multi::detail::invoke_square(a_, is...));  // a_[is...] in C++23
	}
};

template<class A>
constexpr auto exp(A&& alpha) {
	auto xs = alpha.extensions();  // TODO(correaa) consider using .home() cursor
	return exp_bind_t<typename bind_category<A>::type>{std::forward<A>(alpha)} ^ xs;
}

template<class T> constexpr auto exp(std::initializer_list<T> il) { return exp(multi::array<T, 1>{il}); }
template<class T> constexpr auto exp(std::initializer_list<std::initializer_list<T>> il) { return exp(multi::array<T, 2>{il}); }

template<class A>
struct abs_bind_t {
	A a_;

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		using ::std::abs;
		return abs(multi::detail::invoke_square(a_, is...));  // a_[is...] in C++23
	}
};

template<class A>
constexpr auto abs(A&& a) { return abs_bind_t<typename bind_category<A>::type>{std::forward<A>(a)} ^ a.extensions(); }

template<class T> constexpr auto abs(std::initializer_list<T> il) { return abs(multi::array<T, 1>{il}); }
template<class T> constexpr auto abs(std::initializer_list<std::initializer_list<T>> il) { return abs(multi::array<T, 2>{il}); }

// #endif
}  // end namespace broadcast
}  // end namespace boost::multi

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_BROADCAST_HPP
