// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_BROADCAST_HPP
#define BOOST_MULTI_BROADCAST_HPP
#pragma once

#include "boost/multi/array_ref.hpp"
#include "boost/multi/utility.hpp"  // for multi::detail::apply_square

#include <type_traits>

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

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

template<class F, class A, class... Arrays, typename = decltype(std::declval<F&&>()(std::declval<typename std::decay_t<A>::reference>(), std::declval<typename std::decay_t<Arrays>::reference>()...))>
constexpr auto apply_front(F&& fun, A&& arr, Arrays&&... arrs) {
	return [fun_ = std::forward<F>(fun), &arr, &arrs...](auto is) { return fun_(arr[is], arrs[is]...); } ^ multi::extensions_t<1>({arr.extension()});
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
	// if constexpr(std::decay_t<A>::dimensionality > 1) {
	// 	using std::get;
	// 	std::cout << "exts: " << get<1>(arr.extensions()).size() << " vs ";
	// 	((std::cout << get<1>(arrs.extensions()).size() << " "), ...);
	// 	std::cout << '\n';
	// 	std::cout << std::flush;
	// }
	assert(((xs == arrs.extensions()) && ...));
	// std::cout << ... << arrs.extensions() << '\n';
	return apply_bind_t<F, std::decay_t<A>, std::decay_t<As>...>{std::forward<F>(fun), std::forward<A>(arr), std::forward<As>(arrs)...} ^ xs;
	//	return [fun = std::forward<F>(fun), &arr, &arrs...](auto... is) { return fun(arr[is...], arrs[is...]...); } ^ arr.extensions();
}

template<class T>
class identity_bind {
	T val_;

 public:
	template<class TT>
	explicit constexpr identity_bind(TT&& val) : val_{std::forward<TT>(val)} {}  // NOLINT(bugprone-forwarding-reference-overload)

	BOOST_MULTI_HD constexpr auto operator()() const -> auto& { return val_; }
};

template<class F, class A, class B>
constexpr auto map(F&& fun, A&& alpha, B&& omega) {
	if constexpr(!multi::has_dimensionality<std::decay_t<A>>::value) {
		return map(std::forward<F>(fun), identity_bind<A>{std::forward<A>(alpha)} ^ multi::extensions_t<0>{}, std::forward<B>(omega));
	} else if constexpr(!multi::has_dimensionality<std::decay_t<B>>::value) {
		return map(std::forward<F>(fun), std::forward<A>(alpha), identity_bind<B>{std::forward<B>(omega)} ^ multi::extensions_t<0>{});
	} else {
		using std::get;
		if constexpr(std::decay_t<A>::dimensionality < std::decay_t<B>::dimensionality) {
			return map(std::forward<F>(fun), std::forward<A>(alpha).repeated(get<std::decay_t<B>::dimensionality - std::decay_t<A>::dimensionality - 1>(omega.sizes())), std::forward<B>(omega));
		} else if constexpr(std::decay_t<B>::dimensionality < std::decay_t<A>::dimensionality) {
			return map(std::forward<F>(fun), std::forward<A>(alpha), std::forward<B>(omega).repeated(get<std::decay_t<A>::dimensionality - std::decay_t<B>::dimensionality - 1>(alpha.sizes())));
		} else {
			return apply(std::forward<F>(fun), std::forward<A>(alpha), std::forward<B>(omega));
		}
	}
}

template<class A, class B>
class apply_plus_t {
	A a_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
	B b_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

 public:
	template<class AA, class BB>
	apply_plus_t(AA&& a, BB&& b) : a_{std::forward<AA>(a)}, b_{std::forward<BB>(b)} {}

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		return multi::detail::invoke_square(a_, is...)  // like a_[is...] in C++23
			 +
			   multi::detail::invoke_square(b_, is...)  // like b_[is...] in C++23
			;
	}
};

template<class A, class B>
constexpr auto operator+(A&& alpha, B&& omega) noexcept {
	// if constexpr(!multi::has_dimensionality<std::decay_t<A>>::value) {
	// 	return broadcast::operator+([alpha_ = std::forward<A>(alpha)]() { return alpha_; } ^ multi::extensions_t<0>{}, omega);
	// } else if constexpr(!multi::has_dimensionality<std::decay_t<B>>::value) {
	// 	return broadcast::operator+(alpha, [omega_ = std::forward<B>(omega)]() { return omega_; } ^ multi::extensions_t<0>{});
	// } else if constexpr(std::decay_t<A>::dimensionality < std::decay_t<B>::dimensionality) {
	// 	return broadcast::operator+(alpha.repeated(omega.size()), omega);
	// } else if constexpr(std::decay_t<B>::dimensionality < std::decay_t<A>::dimensionality) {
	// 	return broadcast::operator+(alpha, omega.repeated(alpha.size()));
	// } else {
	// 	// return apply(std::forward<F>(fun), std::forward<A>(alpha), std::forward<B>(omega));
	// 	// auto ah = alpha.home();
	// 	// auto oh = omega.home();
	// 	// return broadcast::apply_plus_t<decltype(ah), decltype(oh)>(ah, oh) ^ axs;
	// 	auto axs = alpha.extensions();
	// 	assert(axs == omega.extensions());
	// 	return broadcast::apply_plus_t<A, B>(std::forward<A>(alpha), std::forward<B>(omega)) ^ axs;
	// }
	return broadcast::map(std::plus<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

template<class A, class B>
constexpr auto operator-(A&& alpha, B&& omega) { return broadcast::map(std::minus<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A>
constexpr auto operator-(A&& alpha) { return broadcast::apply(std::negate<>{}, std::forward<A>(alpha)); }

template<class A, class B>
constexpr auto operator*(A&& alpha, B&& omega) { return broadcast::map(std::multiplies<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A, class B>
constexpr auto operator/(A&& alpha, B&& omega) {
	return broadcast::map(std::divides<>{}, std::forward<A>(alpha), std::forward<B>(omega));
}

template<class A, class B>
constexpr auto operator&&(A&& alpha, B&& omega) { return broadcast::map(std::logical_and<>{}, std::forward<A>(alpha), std::forward<B>(omega)); }

template<class A, class B>
constexpr auto operator||(A&& a, B&& b) { return broadcast::map(std::logical_or<>{}, std::forward<A>(a), std::forward<B>(b)); }

template<class F, class A, std::enable_if_t<true, decltype(std::declval<F&&>()(std::declval<typename std::decay_t<A>::element>()))*> = nullptr>  // NOLINT(modernize-use-constraints) for C++23
constexpr auto operator|(A&& a, F fun) {
	return std::forward<A>(a).element_transformed(fun);
}

template<class F, class A, std::enable_if_t<true, decltype(std::declval<F>()(std::declval<typename std::decay_t<A>::reference>()))*> = nullptr>  // NOLINT(modernize-use-constraints) for C++23
constexpr auto operator|(A&& a, F fun) {
	return std::forward<A>(a).transformed(fun);
}

template<class A>
class exp_bind_t {
	A a_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) TODO(correaa) consider saving .home() cursor

 public:
	template<class AA>                                                                          // , std::enable_if_t<!std::is_base_v<exp_bind_t<A>, std::decay_t<AA> >, int> =0>
	BOOST_MULTI_HD constexpr explicit exp_bind_t(AA&& a) noexcept : a_{std::forward<AA>(a)} {}  // NOLINT(bugprone-forwarding-reference-overload)

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		using ::std::exp;
		return exp(multi::detail::invoke_square(a_, is...));  // a_[is...] in C++23
	}
};

template<class A> exp_bind_t(A) -> exp_bind_t<A>;

template<class A, std::enable_if_t<multi::has_extensions<std::decay_t<A>>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++23
BOOST_MULTI_HD constexpr auto exp(A&& alpha) {
	auto xs = alpha.extensions();  // shouldn't get to this point for scalars
	return exp_bind_t<A>(std::forward<A>(alpha)) ^ xs;
}

template<class T> constexpr auto exp(std::initializer_list<T> il) { return exp(multi::inplace_array<T, 1, 16>(il)); }
template<class T> constexpr auto exp(std::initializer_list<std::initializer_list<T>> il) { return exp(multi::inplace_array<T, 2, 16>(il)); }

template<class A>
struct abs_bind_t {
	A a_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		using ::std::abs;
		return abs(multi::detail::invoke_square(a_, is...));  // a_[is...] in C++23
	}
};

template<class A>
constexpr auto abs(A const& a) { return abs_bind_t<decltype(a.home())>{a.home()} ^ a.extensions(); }

template<class T> constexpr auto abs(std::initializer_list<T> il) { return abs(multi::array<T, 1>{il}); }
template<class T> constexpr auto abs(std::initializer_list<std::initializer_list<T>> il) { return abs(multi::array<T, 2>{il}); }

// #endif
}  // end namespace broadcast
}  // end namespace boost::multi

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_BROADCAST_HPP
