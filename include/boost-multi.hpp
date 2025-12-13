// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_HPP
#define BOOST_MULTI_HPP

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <cstddef>      // for size_t
#include <tuple>        // for deprecated functions  // for make_index_sequence, index_sequence, tuple_element, tuple_size, apply, tuple
#include <type_traits>  // for declval, decay_t, conditional_t, true_type
#include <utility>      // for forward, move

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4514)  // boost::multi::detail::tuple<>::operator <': unreferenced inline function has been removed
#pragma warning(disable : 4623)  // default constructor was implicitly defined as deleted
#pragma warning(disable : 4626)  // assignment operator was implicitly defined as deleted
#endif

namespace boost::multi {  // NOLINT(modernize-concat-nested-namespaces) keep c++14 compat
namespace detail {

// we need a custom tuple type, so some fundamental types (e.g. iterators) are trivially constructible
template<class... Ts> class tuple;  // TODO(correaa) consider renaming it to `tpl`

template<> class tuple<> {  // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
 public:
	tuple()             = default;
	tuple(tuple const&) = default;

	auto operator=(tuple const&) -> tuple& = default;

	BOOST_MULTI_HD constexpr auto operator==(tuple const& /*other*/) const { return true; }
	BOOST_MULTI_HD constexpr auto operator!=(tuple const& /*other*/) const  { return false; }

	BOOST_MULTI_HD constexpr auto operator<(tuple const& /*other*/) const { return false; }
	BOOST_MULTI_HD constexpr auto operator>(tuple const& /*other*/) const { return false; }

	template<class F>
	BOOST_MULTI_HD constexpr friend auto apply(F&& fn, tuple<> const& /*self*/) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return std::forward<F>(fn)();
	}
};

template<class T0, class... Ts> class tuple<T0, Ts...> : tuple<Ts...> {  // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
	T0 head_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) can be a reference
	using head_type = T0;
	using tail_type = tuple<Ts...>;

 public:
	BOOST_MULTI_HD constexpr auto head() const& -> T0 const& { return head_; }
	BOOST_MULTI_HD constexpr auto head() && -> T0&& { return std::move(head_); }
	BOOST_MULTI_HD constexpr auto head() & -> T0& { return head_; }

	BOOST_MULTI_HD constexpr auto tail() const& -> tail_type const& { return static_cast<tail_type const&>(*this); }
	BOOST_MULTI_HD constexpr auto tail() && -> decltype(auto) { return static_cast<tail_type&&>(*this); }
	BOOST_MULTI_HD constexpr auto tail() & -> tail_type& { return static_cast<tail_type&>(*this); }

	constexpr tuple()             = default;
	constexpr tuple(tuple const&) = default;

	// this is horrible hack and can produce ODR reported by Circle
	// operator std::tuple<T0&, Ts&...>() & {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	// 	using std::apply;
	// 	return apply([](auto&&... ts) {return std::tuple<T0&, Ts&...>{std::forward<decltype(ts)>(ts)...}; }, *this);
	// }
	// operator std::tuple<T0&, Ts&...>() && {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	// 	using std::apply;
	// 	return apply([](auto&&... ts) {return std::tuple<T0&, Ts&...>{std::forward<decltype(ts)>(ts)...}; }, *this);
	// }
	// operator std::tuple<T0&, Ts&...>() const& {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	// 	using std::apply;
	// 	return apply([](auto&&... ts) {return std::tuple<T0&, Ts&...>{std::forward<decltype(ts)>(ts)...}; }, *this);
	// }

	// TODO(correaa) make conditional explicit constructor depending on the conversions for T0, Ts...
	BOOST_MULTI_HD constexpr explicit tuple(T0 head, tuple<Ts...> tail) : tail_type{std::move(tail)}, head_{std::move(head)} {}
	// cppcheck-suppress noExplicitConstructor ; allow bracket init in function argument // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr tuple(T0 head, Ts... tail) : tail_type{tail...}, head_{head} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) to allow bracket function calls

	// cppcheck-suppress noExplicitConstructor ; allow bracket init in function argument // NOLINTNEXTLINE(runtime/explicit)
	template<class TT0 = T0,
		std::enable_if_t<!std::is_same_v<TT0, tuple>, int> =0,  // NOLINT(modernize-use-constraints,modernize-type-traits) for C++20
		std::enable_if_t<sizeof(TT0*) && (sizeof...(Ts) == 0), int> =0  // NOLINT(modernize-use-constraints) for C++20
	>
	// cppcheck-suppress noExplicitConstructor ; see below
	BOOST_MULTI_HD constexpr tuple(TT0 head) : tail_type{}, head_{head} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) to allow bracket function calls

	// cppcheck-suppress noExplicitConstructor ; allow bracket init in function argument // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr explicit tuple(::std::tuple<T0, Ts...> other) : tuple(::std::apply([](auto... es) {return tuple(es...);}, other)) {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

	constexpr auto operator=(tuple const&) -> tuple& = default;

	template<class... TTs>
	constexpr auto operator=(tuple<TTs...> const& other)  // NOLINT(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator) signature used for SFINAE
		-> decltype(std::declval<head_type&>() = other.head(), std::declval<tail_type&>() = other.tail(), std::declval<tuple&>()) {
		head_ = other.head(), tail() = other.tail();
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator==(tuple const& other) const -> bool {
		return
			head_ == other.head_
			&& 
			tail() == other.tail()
		;
	}
	BOOST_MULTI_HD constexpr auto operator!=(tuple const& other) const -> bool { return head_ != other.head_ || tail() != other.tail(); }

	BOOST_MULTI_HD constexpr auto operator<(tuple const& other) const {
		if(head_ < other.head_) {
			return true;
		}
		if(other.head_ < head_) {
			return false;
		}
		return tail() < other.tail();
	}
	BOOST_MULTI_HD constexpr auto operator>(tuple const& other) const {
		if(head_ > other.head_) {
			return true;
		}
		if(other.head_ > head_) {
			return false;
		}
		return tail() > other.tail();
	}

 private:

	template<class F, std::size_t... I>
	BOOST_MULTI_HD constexpr auto apply_impl_(F&& fn, std::index_sequence<I...> /*012*/) const& -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return std::forward<F>(fn)(this->get<I>()...);
	}

	template<class F, std::size_t... I>
	BOOST_MULTI_HD constexpr auto apply_impl_(F&& fn, std::index_sequence<I...> /*012*/) & -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return std::forward<F>(fn)(this->get<I>()...);
	}

	template<class F, std::size_t... I>
	BOOST_MULTI_HD constexpr auto apply_impl_(F&& fn, std::index_sequence<I...> /*012*/) && -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return std::forward<F>(fn)(std::move(*this).template get<I>()...);
	}

 public:
	template<class F>
	BOOST_MULTI_HD constexpr auto apply(F&& fn) const& -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return apply_impl_(std::forward<F>(fn), std::make_index_sequence<sizeof...(Ts) + 1>{});
	}
	template<class F>
	BOOST_MULTI_HD constexpr auto apply(F&& fn) & -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return apply_impl_(std::forward<F>(fn), std::make_index_sequence<sizeof...(Ts) + 1>{});
	}
	template<class F>
	BOOST_MULTI_HD constexpr auto apply(F&& fn) && -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return std::move(*this).apply_impl_(std::forward<F>(fn), std::make_index_sequence<sizeof...(Ts) + 1>{});
	}

	template<class F>
	friend BOOST_MULTI_HD constexpr auto apply(F&& fn, tuple<T0, Ts...> const& self) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return self.apply(std::forward<F>(fn));
	}

	template<class F>
	friend BOOST_MULTI_HD constexpr auto apply(F&& fn, tuple<T0, Ts...> & self) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return self.apply(std::forward<F>(fn));
	}

	template<class F>
	friend BOOST_MULTI_HD constexpr auto apply(F&& fn, tuple<T0, Ts...> && self) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
		return std::move(self).apply(std::forward<F>(fn));
	}

 private:
	template<std::size_t N> struct priority : std::conditional_t<N == 0, std::true_type, priority<N - 1>> {};

	template<class Index>
	constexpr auto at_aux_(priority<0> /*prio*/, Index idx) const
		-> decltype(ht_tuple(std::declval<head_type const&>(), std::declval<tail_type const&>()[idx])) {
		return ht_tuple(head(), tail()[idx]);
	}

	template<class Index>
	constexpr auto at_aux_(priority<1> /*prio*/, Index idx) const
		-> decltype(ht_tuple(std::declval<head_type const&>()[idx], std::declval<tail_type const&>())) {
		return ht_tuple(head()[idx], tail());
	}

 public:
	template<class Index>
	constexpr auto operator[](Index idx) const
		-> decltype(std::declval<tuple<T0, Ts...> const&>().at_aux_(priority<1>{}, idx)) {
		return this->at_aux_(priority<1>{}, idx);
	}

	template<std::size_t N, std::enable_if_t<(N==0), int> =0>  // NOLINT(modernize-use-constraints) for C++20
	BOOST_MULTI_HD constexpr auto get() const& -> T0 const& {  // NOLINT(readability-identifier-length) std naming
		return head();
	}

	template<std::size_t N, std::enable_if_t<(N!=0), int> =0>  // NOLINT(modernize-use-constraints) for C++20
	BOOST_MULTI_HD constexpr auto get() const& -> auto const& {  // NOLINT(readability-identifier-length) std naming
		return this->tail().template get<N - 1>();  // this-> for msvc 19.14 compilation
	}

#ifdef __NVCC__  // in place of global -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\"
	#pragma nv_diagnostic push
	#pragma nv_diag_suppress = implicit_return_from_non_void_function
#endif

#ifdef __NVCOMPILER
#pragma diagnostic push
#pragma diag_suppress = implicit_return_from_non_void_function
#endif

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#endif

	template<std::size_t N>
	BOOST_MULTI_HD constexpr auto get() & -> decltype(auto) {  // NOLINT(readability-identifier-length) std naming
		if constexpr(N == 0) {
			return head();
		} else {
			return tail().template get<N - 1>();
		}
	}

	template<std::size_t N>
	BOOST_MULTI_HD constexpr auto get() && -> decltype(auto) {  // NOLINT(readability-identifier-length) std naming
		if constexpr(N == 0) {
			return std::move(*this).head();
		} else {
			return std::move(*this).tail().template get<N - 1>();
		}
	}
};

#ifdef __NVCC__
#pragma nv_diagnostic pop
#elif defined(__NVCOMPILER)
#pragma diagnostic pop
#endif

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef __INTEL_COMPILER  // this instance is necessary due to a bug in intel compiler icpc
//  TODO(correaa) : this class can be collapsed with the general case with [[no_unique_address]] in C++20
template<class T0> class tuple<T0> {  // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
	T0      head_;
	tuple<> tail_;

 public:
	constexpr auto head() const& -> T0 const& { return head_; }
	constexpr auto head() && -> T0&& { return std::move(head_); }
	constexpr auto head() & -> T0& { return head_; }

	constexpr auto tail() const& -> tuple<> const& { return tail_; }
	constexpr auto tail() && -> tuple<>&& { return std::move(tail_); }
	constexpr auto tail() & -> tuple<>& { return tail_; }

	constexpr tuple()             = default;
	constexpr tuple(tuple const&) = default;  // cppcheck-suppress noExplicitConstructor ; workaround cppcheck 2.11

	// cppcheck-suppress noExplicitConstructor ; allow bracket init in function argument // NOLINTNEXTLINE(runtime/explicit)
	constexpr tuple(T0 t0, tuple<> sub) : head_{std::move(t0)}, tail_{sub} {}
	constexpr explicit tuple(T0 t0) : head_{std::move(t0)}, tail_{} {}

	constexpr auto operator=(tuple const& other) -> tuple& = default;

	constexpr auto operator==(tuple const& other) const { return head_ == other.head_; }
	constexpr auto operator!=(tuple const& other) const { return head_ != other.head_; }

	constexpr auto operator<(tuple const& other) const { return head_ < other.head_; }
	constexpr auto operator>(tuple const& other) const { return head_ > other.head_; }
};
#endif

#if defined(__cpp_deduction_guides) && (__cpp_deduction_guides >= 201703L) 
template<class T0, class... Ts> tuple(T0, tuple<Ts...>) -> tuple<T0, Ts...>;
#endif

template<class T0, class... Ts> constexpr auto mk_tuple(T0 head, Ts... tail) {
	return tuple<T0, Ts...>(std::move(head), std::move(tail)...);
}

template<class T0, class... Ts> constexpr auto tie(T0& head, Ts&... tail) {
	return tuple<T0&, Ts&...>(head, tail...);
}

template<class T0, class... Ts> BOOST_MULTI_HD constexpr auto ht_tuple(T0 head, tuple<Ts...> tail)
-> tuple<T0, Ts...> {
	return tuple<T0, Ts...>(std::move(head), std::move(tail));
}

template<class T0, class Tuple> struct tuple_prepend;

template<class T0, class... Ts>
struct tuple_prepend<T0, tuple<Ts...>> {
	using type = tuple<T0, Ts...>;
};

template<class T0, class Tuple>
using tuple_prepend_t = typename tuple_prepend<T0, Tuple>::type;

template<class T0, class... Ts>
constexpr auto head(tuple<T0, Ts...> const& t) -> decltype(auto) {  // NOLINT(readability-identifier-length) std naming
	return t.head();
}

template<class T0, class... Ts>
constexpr auto head(tuple<T0, Ts...>&& t) -> decltype(auto) {  // NOLINT(readability-identifier-length) std naming
	return std::move(t).head();
}

template<class T0, class... Ts>
constexpr auto head(tuple<T0, Ts...>& t) -> decltype(auto) {  // NOLINT(readability-identifier-length) std naming
	return t.head();
}

template<class T0, class... Ts>
BOOST_MULTI_HD constexpr auto tail(tuple<T0, Ts...> const& t) -> decltype(t.tail()) { return t.tail(); }  // NOLINT(readability-identifier-length) std naming

template<class T0, class... Ts>
BOOST_MULTI_HD constexpr auto tail(tuple<T0, Ts...>&& t) -> decltype(std::move(t).tail()) { return std::move(t).tail(); }  // NOLINT(readability-identifier-length) std naming

template<class T0, class... Ts>
BOOST_MULTI_HD constexpr auto tail(tuple<T0, Ts...>& t) -> decltype(t.tail()) { return t.tail(); }  // NOLINT(readability-identifier-length) std naming

#ifdef __NVCC__  // in place of global -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\"
	#pragma nv_diagnostic push
	#pragma nv_diag_suppress = implicit_return_from_non_void_function
#endif

#ifdef __NVCOMPILER
#pragma diagnostic push
#pragma diag_suppress = implicit_return_from_non_void_function
#endif

#ifndef _MSC_VER
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#endif

template<std::size_t N, class T0, class... Ts>
BOOST_MULTI_HD constexpr auto get(tuple<T0, Ts...> const& t) -> auto const& {  // NOLINT(readability-identifier-length) std naming
	if constexpr(N == 0) {
		return t.head();
	} else {
		using std::get;
		return get<N - 1>(t.tail());
	}
}

template<std::size_t N, class T0, class... Ts>
BOOST_MULTI_HD constexpr auto get(tuple<T0, Ts...>& tup) -> auto& {
	if constexpr(N == 0) {
		return tup.head();
	} else {
		return get<N - 1>(tup.tail());
	}
}

template<std::size_t N, class T0, class... Ts>
BOOST_MULTI_HD constexpr auto get(tuple<T0, Ts...>&& tup) -> auto&& {
	if constexpr(N == 0) {
		return std::move(std::move(tup)).head();
	} else {
		return get<N - 1>(std::move(std::move(tup).tail()));
	}
}

#ifdef __NVCC__
#pragma nv_diagnostic pop
#elif defined(__NVCOMPILER)
#pragma diagnostic pop
#endif

#ifndef _MSC_VER
#pragma GCC diagnostic pop
#endif

}  // end namespace detail
}  // end namespace boost::multi

// Some versions of Clang throw warnings that stl uses class std::tuple_size instead
// of struct std::tuple_size like it should be
#ifdef __clang__
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

template<class... Ts>
struct std::tuple_size<boost::multi::detail::tuple<Ts...>> : std::integral_constant<std::size_t, sizeof...(Ts)> {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) for structured bindings
	// // cppcheck-suppress unusedStructMember
	// static constexpr std::size_t value = sizeof...(Ts);
};

template<>
struct std::tuple_element<0, boost::multi::detail::tuple<>> {  // NOLINT(cert-dcl58-cpp) to have structured bindings
	using type = void;
};

template<class T0>
struct std::tuple_element<0, boost::multi::detail::tuple<T0>> {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) for structured bindings
	using type = T0;
};

template<class T0, class... Ts>
struct std::tuple_element<0, boost::multi::detail::tuple<T0, Ts...>> {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) for structured bindings
	using type = T0;
};

template<std::size_t N, class T0, class... Ts>
struct std::tuple_element<N, boost::multi::detail::tuple<T0, Ts...>> {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) for structured bindings
	using type = tuple_element_t<N - 1, boost::multi::detail::tuple<Ts...>>;
};

template<class F, class Tuple, std::size_t... I>
BOOST_MULTI_HD constexpr auto std_apply_timpl(F&& fn, Tuple&& tp, std::index_sequence<I...> /*012*/) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get
	(void)tp;  // fix "error #827: parameter "t" was never referenced" in NVC++ and "error #869: parameter "t" was never referenced" in oneAPI-ICPC
	return std::forward<F>(fn)(boost::multi::detail::get<I>(std::forward<Tuple>(tp))...);  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) use forward_as?
}

namespace std {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) to implement structured bindings

template<class F, class... Ts>
BOOST_MULTI_HD constexpr auto apply(F&& fn, boost::multi::detail::tuple<Ts...> const& tp) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal to define tuple get
	return std_apply_timpl(
		std::forward<F>(fn), tp,
		std::make_index_sequence<sizeof...(Ts)>{}
	);
}

template<class F, class... Ts>
BOOST_MULTI_HD constexpr auto apply(F&& fn, boost::multi::detail::tuple<Ts...>& tp) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal to define tuple get
	return std_apply_timpl(
		std::forward<F>(fn), tp,
		std::make_index_sequence<sizeof...(Ts)>{}
	);
}

template<class F, class... Ts>
BOOST_MULTI_HD constexpr auto apply(F&& fn, boost::multi::detail::tuple<Ts...>&& tp) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get
	return std_apply_timpl(
		std::forward<F>(fn), std::move(tp),
		std::make_index_sequence<sizeof...(Ts)>{}
	);
}

}  // end namespace std

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace boost::multi {  // NOLINT(modernize-concat-nested-namespaces) keep c++14 compat
namespace detail {

template<class Tuple1, class Tuple2, std::size_t... Is>
constexpr auto tuple_zip_impl(Tuple1&& tup1, Tuple2&& tup2, std::index_sequence<Is...> /*012*/) {
	using boost::multi::detail::get;
	return boost::multi::detail::mk_tuple(
		boost::multi::detail::mk_tuple(
			get<Is>(std::forward<Tuple1>(tup1)),
			get<Is>(std::forward<Tuple2>(tup2))
		)...
	);
}

template<class Tuple1, class Tuple2, class Tuple3, std::size_t... Is>
constexpr auto tuple_zip_impl(Tuple1&& tup1, Tuple2&& tup2, Tuple3&& tup3, std::index_sequence<Is...> /*012*/) {
	using boost::multi::detail::get;
	return boost::multi::detail::mk_tuple(
		boost::multi::detail::mk_tuple(
			get<Is>(std::forward<Tuple1>(tup1)),
			get<Is>(std::forward<Tuple2>(tup2)),
			get<Is>(std::forward<Tuple3>(tup3))
		)...
	);
}

template<class Tuple1, class Tuple2, class Tuple3, class Tuple4, std::size_t... Is>
constexpr auto tuple_zip_impl(Tuple1&& tup1, Tuple2&& tup2, Tuple3&& tup3, Tuple4&& tup4, std::index_sequence<Is...> /*012*/) {
	using boost::multi::detail::get;
	return boost::multi::detail::mk_tuple(
		boost::multi::detail::mk_tuple(
			get<Is>(std::forward<Tuple1>(tup1)),  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) use forward_as
			get<Is>(std::forward<Tuple2>(tup2)),  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) use forward_as
			get<Is>(std::forward<Tuple3>(tup3)),  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) use forward_as
			get<Is>(std::forward<Tuple4>(tup4))   // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) use forward_as
		)...
	);
}

template<class Tuple1, class Tuple2, class Tuple3, class Tuple4, class Tuple5, std::size_t... Is>
constexpr auto tuple_zip_impl(Tuple1&& tup1, Tuple2&& tup2, Tuple3&& tup3, Tuple4&& tup4, Tuple5&& tup5, std::index_sequence<Is...> /*012*/) {
	using boost::multi::detail::get;
	return boost::multi::detail::mk_tuple(
		boost::multi::detail::mk_tuple(
			get<Is>(std::forward<Tuple1>(tup1)),
			get<Is>(std::forward<Tuple2>(tup2)),
			get<Is>(std::forward<Tuple3>(tup3)),
			get<Is>(std::forward<Tuple4>(tup4)),
			get<Is>(std::forward<Tuple4>(tup5))
		)...
	);
}

template<class T1, class T2>
constexpr auto tuple_zip(T1&& tup1, T2&& tup2) {
	return detail::tuple_zip_impl(
		std::forward<T1>(tup1), std::forward<T2>(tup2),
		std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>()
	);
}

template<class T1, class T2, class T3>
constexpr auto tuple_zip(T1&& tup1, T2&& tup2, T3&& tup3) {
	return detail::tuple_zip_impl(
		std::forward<T1>(tup1), std::forward<T2>(tup2), std::forward<T3>(tup3),
		std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>()
	);
}

template<class T1, class T2, class T3, class T4>
constexpr auto tuple_zip(T1&& tup1, T2&& tup2, T3&& tup3, T4&& tup4) {
	return detail::tuple_zip_impl(
		std::forward<T1>(tup1), std::forward<T2>(tup2), std::forward<T3>(tup3), std::forward<T4>(tup4),
		std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>()
	);
}

template<class T1, class T2, class T3, class T4, class T5>
constexpr auto tuple_zip(T1&& tup1, T2&& tup2, T3&& tup3, T4&& tup4, T5&& tup5) {
	return detail::tuple_zip_impl(
		std::forward<T1>(tup1), std::forward<T2>(tup2), std::forward<T3>(tup3), std::forward<T4>(tup4), std::forward<T5>(tup5),
		std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>()
	);
}

}  // end namespace detail

using detail::tie;

}  // end namespace boost::multi

#undef BOOST_MULTI_HD

#endif

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Copyright 2023-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <type_traits>
#include <utility>

namespace boost::multi::detail {  // this library requires C++17 and above !!!

template<class From, class To> constexpr bool is_implicitly_convertible_v = std::is_convertible_v<From, To>;  // this library needs C++17 or higher (e.g. -std=c++17)
template<class From, class To> constexpr bool is_explicitly_convertible_v = std::is_constructible_v<To, From>;

template<class To, class From, std::enable_if_t<std::is_convertible_v<From, To>, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto implicit_cast(From&& ref) -> To {return static_cast<To>(std::forward<From&&>(ref));}

template<class To, class From, std::enable_if_t<std::is_constructible_v<To, From> &&  ! std::is_convertible_v<From, To>, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto explicit_cast(From&& ref) -> To {return static_cast<To>(std::forward<From&&>(ref));}

}  // end namespace boost::multi::detail
#endif

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

namespace boost::multi::detail {
template<class T> auto              what(T&&) -> T&&                     = delete;  // NOLINT(cppcoreguidelines-missing-std-forward)
template<class... Ts> auto          what(Ts&&...) -> std::tuple<Ts&&...> = delete;  // NOLINT(cppcoreguidelines-missing-std-forward)
template<class T, class... Ts> auto what() -> std::tuple<Ts&&...>        = delete;

template<int V> auto what_value() -> std::integral_constant<int, V> = delete;
template<int V> struct what_value_t;
}  // namespace boost::multi::detail

// Copyright 2019-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// clang-format off

#ifdef __has_cpp_attribute

// No discard first
#  ifdef __NVCC__
#    define BOOST_MULTI_NODISCARD(MsG)
#  elif __has_cpp_attribute(nodiscard)
#    if (__has_cpp_attribute(nodiscard) >= 201907L) && (__cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L))
#      define BOOST_MULTI_NODISCARD(MsG) [[nodiscard]]  // [[nodiscard(MsG)]] in c++20 empty message is not allowed with paren
#    else
#      define BOOST_MULTI_NODISCARD(MsG) [[nodiscard]]  // NOLINT(cppcoreguidelines-macro-usage) TODO(correaa) check if this is needed in C++17
#    endif
#  elif __has_cpp_attribute(gnu::warn_unused_result)
#    define BOOST_MULTI_NODISCARD(MsG) [[gnu::warn_unused_result]]
#  endif 

// No discard class
#  if(__has_cpp_attribute(nodiscard) && !defined(__NVCC__) && (!defined(__clang__) || (defined(__clang__) && (__cplusplus >= 202002L))))
#    if (__has_cpp_attribute(nodiscard) >= 201907L) && (__cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L))
#      define BOOST_MULTI_NODISCARD_CLASS(MsG) [[nodiscard_(MsG)]]
#    else
#      define BOOST_MULTI_NODISCARD_CLASS(MsG) [[nodiscard]]
#    endif
#  endif

#endif

#ifndef BOOST_MULTI_NODISCARD
#  define BOOST_MULTI_NODISCARD(MsG)
#endif

#ifndef BOOST_MULTI_NODISCARD_CLASS
#  define BOOST_MULTI_NODISCARD_CLASS(MsG)
#endif

// clang-format on

// Copyright 2019-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifdef __has_cpp_attribute
	#if __has_cpp_attribute(no_unique_address) >= 201803L && !defined(__NVCC__) && !defined(__PGI) && (__cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L))
		// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) this macro will be needed until C++20
		#define BOOST_MULTI_NO_UNIQUE_ADDRESS [[no_unique_address]]
	#endif
#endif

#ifndef BOOST_MULTI_NO_UNIQUE_ADDRESS
	#ifdef _MSC_VER
		#define BOOST_MULTI_NO_UNIQUE_ADDRESS  // [[msvc::no_unique_address]]
	#else
		#define BOOST_MULTI_NO_UNIQUE_ADDRESS
	#endif
#endif

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <algorithm>    // for std::for_each  // IWYU pragma: keep  // bug in iwyu 0.18
#include <cstddef>      // for size_t, byte
#include <cstdint>      // for uint32_t
#include <iterator>     // for next
#include <type_traits>  // for enable_if_t, decay_t
#include <utility>      // for forward

#if defined(__cpp_lib_byte) && (__cpp_lib_byte >= 201603L )
using BOOST_MULTI_BYTE = std::byte;
#else
using BOOST_MULTI_BYTE = unsigned char;
#endif

namespace boost::archive::detail { template <class Ar> class common_iarchive; }  // lines 24-24
namespace boost::archive::detail { template <class Ar> class common_oarchive; }  // lines 25-25

namespace boost::serialization { struct binary_object; }
namespace boost::serialization { template <class T> class array_wrapper; }
namespace boost::serialization { template <class T> class nvp; }

namespace cereal { template <class ArchiveType, std::uint32_t Flags> struct InputArchive; }
namespace cereal { template <class ArchiveType, std::uint32_t Flags> struct OutputArchive; }
namespace cereal { template <class T> class NameValuePair; }  // if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point  // IWYU pragma: keep  // bug in iwyu 0.18

namespace boost {  // NOLINT(modernize-concat-nested-namespaces) keep c++14 compat
namespace multi {

template<class Ar, class Enable = void>
struct archive_traits {
	template<class T>
	/*inline*/ static auto make_nvp(char const* /*n*/, T&& value) noexcept { return std::forward<T>(value); }  // match original boost declaration
};

template<class Archive, class MA, std::enable_if_t<std::is_same_v<MA, std::decay_t<MA>> && (MA::dimensionality > -1), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
auto operator>>(Archive& arxiv, MA&& self)  // this is for compatibility with Archive type
	-> decltype(arxiv >> static_cast<MA&>(std::forward<MA>(self))) {
	return arxiv >> static_cast<MA&>(std::forward<MA>(self));
}

template<class Archive, class MA, std::enable_if_t<std::is_same_v<MA, std::decay_t<MA>> && (MA::dimensionality > -1), int> = 0>  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
auto operator<<(Archive& arxiv, MA&& self)  // this is for compatibility with Archive type
->decltype(arxiv << static_cast<MA&>(std::forward<MA>(self))) {
	return arxiv << static_cast<MA&>(std::forward<MA>(self)); }

template<class Archive, class MA, std::enable_if_t<std::is_same_v<MA, std::decay_t<MA>> && (MA::dimensionality > -1), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
auto operator&(Archive& arxiv, MA&& self)  // this is for compatibility with Archive type
	-> decltype(arxiv & static_cast<MA&>(std::forward<MA>(self))) {
	return arxiv & static_cast<MA&>(std::forward<MA>(self));
}

template<class Ar>
struct archive_traits<Ar, typename std::enable_if_t<std::is_base_of_v<boost::archive::detail::common_oarchive<Ar>, Ar> || std::is_base_of_v<boost::archive::detail::common_iarchive<Ar>, Ar>>> {
	template<class T> using nvp           = boost::serialization::nvp<T>;
	template<class T> using array_wrapper = boost::serialization::array_wrapper<T>;
	template<class T> struct binary_object_t {
		using type = boost::serialization::binary_object;
	};

	template<class T> /*inline*/ static auto make_nvp(char const* name, T& value) noexcept -> nvp<T> const { return nvp<T>{name, value}; }  // NOLINT(readability-const-return-type) match original boost declaration
	template<class T> /*inline*/ static auto make_nvp(char const* name, T&& value) noexcept -> nvp<T> const { return nvp<T>{name, value /*static_cast<T&>(std::forward<T>(value))*/}; }  // NOLINT(readability-const-return-type,cppcoreguidelines-missing-std-forward) match original boost declaration

	template<class T>        /*inline*/ static auto make_array(T* first, std::size_t size) noexcept -> array_wrapper<T> const { return array_wrapper<T>{first, size}; }  // NOLINT(readability-const-return-type) original boost declaration
	template<class T = void> /*inline*/ static auto make_binary_object(BOOST_MULTI_BYTE const* first, std::size_t size) noexcept -> const typename binary_object_t<T>::type { return typename binary_object_t<T>::type(first, size); }  // if you get an error here you need to eventually `#include<boost/serialization/binary_object.hpp>`// NOLINT(readability-const-return-type,clang-diagnostic-ignored-qualifiers) original boost declaration
};

template<class Ar>
struct archive_traits<
	Ar,
	typename std::enable_if_t<
		std::is_base_of_v<cereal::OutputArchive<Ar, 0>, Ar> || std::is_base_of_v<cereal::OutputArchive<Ar, 1>, Ar> || std::is_base_of_v<cereal::InputArchive<Ar, 0>, Ar> || std::is_base_of_v<cereal::InputArchive<Ar, 1>, Ar>>> {
	using self_t = archive_traits<Ar, typename std::enable_if_t<
		                                  std::is_base_of_v<cereal::OutputArchive<Ar, 0>, Ar> || std::is_base_of_v<cereal::OutputArchive<Ar, 1>, Ar> || std::is_base_of_v<cereal::InputArchive<Ar, 0>, Ar> || std::is_base_of_v<cereal::InputArchive<Ar, 1>, Ar>>>;

	//  template<class T>
	//  inline static auto make_nvp  (char const* name, T const& value) noexcept {return cereal::NameValuePair<T const&>{name, value};}  // if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point  // TODO(correaa) replace by cereal::make_nvp from cereal/cereal.hpp
	// template<class T>
	// inline static auto make_nvp  (std::string const& name, T&& value) noexcept {return cereal::NameValuePair<T>{name.c_str(), std::forward<T>(value)};}  // if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point
	template<class T>
	/*inline*/ static auto make_nvp(char const* name, T&& value) noexcept { return cereal::NameValuePair<T>{name, std::forward<T>(value)}; }  // if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point
	//  template<class T>
	//  inline static auto make_nvp  (char const* name, T&  value) noexcept {return cereal::NameValuePair<T&>{name,                 value};}  // if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point

	template<class T>
	struct array_wrapper {
		T*          p_;
		std::size_t c_;

		template<class Archive>
		void serialize(Archive& arxiv, unsigned int const /*version*/) {
			std::for_each(  // std::for_each_n is absent in GCC 7
				p_, std::next(p_, c_),
				[&arxiv](auto& item) { arxiv& make_nvp("item", item); }
			);
			// for(std::size_t i = 0; i != c_; ++i) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			//  auto& item = p_[i];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			//  arxiv &                                        make_nvp("item"   , item   );  // "item" is the name used by Boost.Serialization XML make_array
			//  // arxiv & boost::multi::archive_traits<Archive>::make_nvp("element", element);
			//  // arxiv &                                cereal::make_nvp("element", element);
			//  // arxiv &                                      CEREAL_NVP(           element);
			//  // arxiv &                                                            element ;
			// }
		}
	};

	template<class T>
	/*inline*/ static auto make_array(T* ptr, std::size_t count) -> array_wrapper<T> { return array_wrapper<T>{ptr, count}; }

	template<class T>
	/*inline*/ static auto make_nvp(char const* name, array_wrapper<T>&& value) noexcept { return make_nvp(name, /*static_cast<array_wrapper<T>&>(std::move(*/ value /*))*/); }  // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
};

}  // end namespace multi
}  // end namespace boost

namespace boost {  // NOLINT(modernize-concat-nested-namespaces) keep c++14 compat

namespace serialization {

// workaround for rvalue subarrays
template<class T, class = std::enable_if_t<std::is_rvalue_reference_v<T&&> > >  // NOLINT(modernize-use-constraints) for C++20
inline auto make_nvp(char const* name, T&& value) noexcept -> ::boost::serialization::nvp<T> {  // NOLINT(cppcoreguidelines-missing-std-forward) workaround legacy interface
	return ::boost::serialization::nvp<T>(name, value);
}

}  // end namespace serialization

using ::boost::serialization::make_nvp;

}  // end namespace boost

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <cstddef>      // for std::size_t
#include <type_traits>  // for make_signed_t

namespace boost::multi {

using size_t    = std::make_signed_t<std::size_t>;
using size_type = std::make_signed_t<std::size_t>;

using index           = std::make_signed_t<size_type>;
using difference_type = std::make_signed_t<index>;

using dimensionality_t    = index;
using dimensionality_type = dimensionality_t;

}  // end namespace boost::multi

#include <algorithm>    // for min, max
#include <cassert>
#include <cstddef>      // for ptrdiff_t
#include <functional>   // for minus, plus
#include <iterator>     // for reverse_iterator, random_access_iterator_tag
#include <limits>       // for numeric_limits
#include <memory>       // for pointer_traits
#include <type_traits>  // for declval, true_type, decay_t, enable_if_t
#include <utility>      // for forward

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

namespace boost::multi {

using boost::multi::detail::tuple;

template<
	class Self,
	class ValueType, class AccessCategory,
	class Reference = ValueType&, class DifferenceType = typename std::pointer_traits<ValueType*>::difference_type, class Pointer = ValueType*>
class iterator_facade {
 protected:
	iterator_facade() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 private:
	using self_type = Self;
	[[nodiscard]] constexpr auto self_() & { return static_cast<self_type&>(*this); }
	[[nodiscard]] constexpr auto self_() const& { return static_cast<self_type const&>(*this); }

 public:
	using value_type        = ValueType;
	using reference         = Reference;
	using pointer           = Pointer;  // NOSONAR(cpp:S5008) false positive
	using difference_type   = DifferenceType;
	using iterator_category = AccessCategory;

	// friend constexpr auto operator!=(self_type const& self, self_type const& other) { return !(self == other); }

	friend constexpr auto operator<=(self_type const& self, self_type const& other) { return (self < other) || (self == other); }
	friend constexpr auto operator>(self_type const& self, self_type const& other) { return !(self <= other); }
	friend constexpr auto operator>=(self_type const& self, self_type const& other) { return !(self < other); }

	constexpr auto        operator-(difference_type n) const { return self_type{self_()} -= n; }
	constexpr auto        operator+(difference_type n) const { return self_type{self_()} += n; }

	template<class = void>  // nvcc workaround 
	friend constexpr auto operator+(difference_type n, self_type const& self) { return self + n; }

	friend constexpr auto operator++(self_type& self, int) -> self_type {
		self_type ret = self;
		++self;
		return ret;
	}
	friend constexpr auto operator--(self_type& self, int) -> self_type {
		self_type ret = self;
		--self;
		return ret;
	}

	constexpr auto operator[](difference_type n) const { return *(self_() + n); }
};

template<typename IndexType = std::true_type, typename IndexTypeLast = IndexType, class Plus = std::plus<>, class Minus = std::minus<>>
class range {
	#ifdef _MSC_VER
	#pragma warning(push)
	#pragma warning(disable : 4820)	// 'boost::multi::range<IndexType,IndexTypeLast,std::plus<void>,std::minus<void>>': '3' bytes padding added after data member 'boost::multi::range<IndexType,IndexTypeLast,std::plus<void>,std::minus<void>>::first_'
	#endif

	BOOST_MULTI_NO_UNIQUE_ADDRESS
	IndexType first_;  //  = {};

	#ifdef _MSC_VER
	#pragma warning(pop)
	#endif

	#ifdef __clang__
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wpadded"
	#endif

	IndexTypeLast last_;  // = first_;  // TODO(correaa) check how to do partially initialzed

	#ifdef __clang__
	#pragma clang diagnostic pop
	#endif

 public:
	template<class Archive>  // , class ArT = multi::archive_traits<Ar>>
	void serialize(Archive& arxiv, unsigned /*version*/) {
		arxiv & multi::archive_traits<Archive>::make_nvp("first", first_);
		// arxiv &               BOOST_SERIALIZATION_NVP(         first_);
		// arxiv &                     cereal:: make_nvp("first", first_);
		// arxiv &                            CEREAL_NVP(         first_);
		// arxiv &                                                first_ ;

		arxiv & multi::archive_traits<Archive>::make_nvp("last", last_);
		// arxiv &                  BOOST_SERIALIZATION_NVP(         last_ );
		// arxiv &                        cereal:: make_nvp("last" , last_ );
		// arxiv &                               CEREAL_NVP(         last_ );
		// arxiv &                                                   last_  ;
	}

	using value_type      = decltype(IndexTypeLast{} + IndexType{});
	using difference_type = decltype(IndexTypeLast{} - IndexType{});  // std::make_signed_t<value_type>;
	using size_type       = difference_type;
	using const_reference = value_type;
	using reference       = const_reference;
	using const_pointer   = value_type;
	using pointer         = value_type;

	range() = default;  // cppcheck-suppress uninitMemberVar ;

	// range(range const&) = default;

	template<class Range,
	         std::enable_if_t<!std::is_base_of_v<range, std::decay_t<Range>>, int> = 0,  // NOLINT(modernize-type-traits) for C++20
	         decltype(detail::implicit_cast<IndexType>(std::declval<Range&&>().first()),
	                  detail::implicit_cast<IndexTypeLast>(std::declval<Range&&>().last())
	         )*                                                                    = nullptr>
	// cppcheck-suppress noExplicitConstructor ;  // NOLINTNEXTLINE(runtime/explicit)
	constexpr /*implicit*/ range(Range&& other)  // NOLINT(bugprone-forwarding-reference-overload,google-explicit-constructor,hicpp-explicit-conversions) // NOSONAR(cpp:S1709) ranges are implicitly convertible if elements are implicitly convertible
	: first_{std::forward<Range>(other).first()}, last_{std::forward<Range>(other).last()} {}  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved)

	template<
		class Range,
		std::enable_if_t<!std::is_base_of_v<range, std::decay_t<Range>>, unsigned> = 0,
		decltype(detail::explicit_cast<IndexType>(std::declval<Range&&>().first()),
		         detail::explicit_cast<IndexTypeLast>(std::declval<Range&&>().last())
		)*                                                                    = nullptr>
	constexpr explicit range(Range&& other)  // NOLINT(bugprone-forwarding-reference-overload)
	: first_{std::forward<Range>(other).first()}, last_{std::forward<Range>(other).last()} {}

	BOOST_MULTI_HD constexpr range(IndexType first, IndexTypeLast last) : first_{first}, last_{last} {}

	// TODO(correaa) make this iterator SCARY
	class const_iterator : public boost::multi::iterator_facade<const_iterator, value_type, std::random_access_iterator_tag, const_reference, difference_type> {
		typename const_iterator::value_type curr_;
		constexpr explicit const_iterator(value_type current) : curr_{current} {}
		friend class range;

	 public:
		template<class T> using rebind = typename range<std::decay_t<T>>::const_iterator;
		using pointer = const_iterator;
		using element_type = IndexTypeLast;
 
		const_iterator() = default;

		template<class OtherConstIterator, class = decltype(std::declval<typename const_iterator::value_type&>() = *OtherConstIterator{})>
		// cppcheck-suppress noExplicitConstructor ; see below
		const_iterator(OtherConstIterator const& other) : curr_{*other} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

		BOOST_MULTI_HD constexpr auto operator==(const_iterator const& other) const -> bool { return curr_ == other.curr_; }
		BOOST_MULTI_HD constexpr auto operator!=(const_iterator const& other) const -> bool { return curr_ != other.curr_; }

		BOOST_MULTI_HD constexpr auto operator<(const_iterator const& other) const -> bool { return curr_ < other.curr_; }  // mull-ignore: cxx_lt_to_le

		constexpr auto operator++() -> const_iterator& {
			++curr_;
			return *this;
		}
		constexpr auto operator--() noexcept(noexcept(--curr_)) -> const_iterator& {
			--curr_;
			return *this;
		}

		constexpr auto operator-=(typename const_iterator::difference_type n) -> const_iterator& {
			curr_ -= n;
			return *this;
		}
		constexpr auto operator+=(typename const_iterator::difference_type n) -> const_iterator& {
			curr_ += n;
			return *this;
		}

		constexpr auto operator-(typename const_iterator::difference_type n) const -> const_iterator {
			return const_iterator{*this} -= n;
		}

		constexpr auto operator+(typename const_iterator::difference_type n) const -> const_iterator {
			return const_iterator{*this} += n;
		}

		constexpr auto operator-(const_iterator const& other) const { return curr_ - other.curr_; }
		constexpr auto operator*() const noexcept -> typename const_iterator::reference { return curr_; }
	};

	using iterator               = const_iterator;
	using reverse_iterator       = std::reverse_iterator<iterator>;
	using const_reverse_iterator = std::reverse_iterator<const_iterator>;

	[[nodiscard]] BOOST_MULTI_HD constexpr auto first() const { return first_; }
	[[nodiscard]] BOOST_MULTI_HD constexpr auto last() const { return last_; }

	constexpr auto operator[](difference_type n) const -> const_reference { return first() + n; }

	[[nodiscard]] BOOST_MULTI_HD constexpr auto front() const -> value_type { return first(); }
	[[nodiscard]] BOOST_MULTI_HD constexpr auto back() const -> value_type { return last() - 1; }

	[[nodiscard]] constexpr auto cbegin() const { return const_iterator{first_}; }
	[[nodiscard]] constexpr auto cend() const { return const_iterator{last_}; }

	[[nodiscard]] constexpr auto rbegin() const { return reverse_iterator{end()}; }
	[[nodiscard]] constexpr auto rend() const { return reverse_iterator{begin()}; }

	[[nodiscard]] constexpr auto begin() const -> const_iterator { return cbegin(); }
	[[nodiscard]] constexpr auto end() const -> const_iterator { return cend(); }

	BOOST_MULTI_HD constexpr auto        is_empty() const& noexcept { return first_ == last_; }

	[[nodiscard]] BOOST_MULTI_HD constexpr auto empty() const& noexcept { return is_empty(); }

	#ifdef __NVCC__
	#pragma nv_diagnostic push
	#pragma nv_diag_suppress = 20013  // calling a constexpr __host__ function("operator std::streamoff") from a __host__ __device__ function("size") is not allowed.  // TODO(correaa) implement HD integral_constant
	#endif

	BOOST_MULTI_HD constexpr auto        size() const& noexcept -> size_type { return last_ - first_; }

	#ifdef __NVCC__
	#pragma nv_diagnostic pop
	#endif

	friend BOOST_MULTI_HD constexpr auto operator==(range const& self, range const& other) {
		return (self.empty() && other.empty()) || (self.first_ == other.first_ && self.last_ == other.last_);
	}
	friend BOOST_MULTI_HD constexpr auto operator!=(range const& self, range const& other) { return !(self == other); }

	[[nodiscard]]  // ("find returns an iterator to the sequence, that is the only effect")]] for C++20
	constexpr auto find(value_type const& value) const -> const_iterator {
		if(value >= last_ || value < first_) {
			return end();
		}
		return begin() + (value - front());
	}
	template<class Value> [[nodiscard]] BOOST_MULTI_HD constexpr auto contains(Value const& value) const -> bool { return (first_ <= value) && (value < last_); }
	template<class Value> [[nodiscard]] BOOST_MULTI_HD constexpr auto count(Value const& value) const -> size_type { return contains(value); }

	friend constexpr auto intersection(range const& self, range const& other) {
		using std::max;
		using std::min;
		auto new_first = max(self.first(), other.first());
		auto new_last  = min(self.last(), other.last());
		new_first      = min(new_first, new_last);
		return range<decltype(new_first), decltype(new_last)>(new_first, new_last);
	}
};

#if defined(__cpp_deduction_guides) && (__cpp_deduction_guides >= 201703)
template<typename IndexType, typename IndexTypeLast = IndexType>     // , class Plus = std::plus<>, class Minus = std::minus<> >
range(IndexType, IndexTypeLast) -> range<IndexType, IndexTypeLast>;  // #3
#endif

template<class IndexType = std::true_type, typename IndexTypeLast = IndexType>
constexpr auto make_range(IndexType first, IndexTypeLast last) -> range<IndexType, IndexTypeLast> {
	return {first, last};
}

template<class IndexType = std::ptrdiff_t>
class intersecting_range {
	range<IndexType> impl_;
	
	constexpr intersecting_range() noexcept :  // MSVC 19.07 needs constexpr to initialize ALL later
		impl_{
			(std::numeric_limits<IndexType>::min)(),  // NOLINT(readability-redundant-parentheses) for MSVC min macros
			(std::numeric_limits<IndexType>::max)()   // NOLINT(readability-redundant-parentheses) for MSVC max macros
		}
	{}

	static constexpr auto make_(IndexType first, IndexType last) -> intersecting_range {
		intersecting_range ret;
		ret.impl_ = range<IndexType>{first, last};
		return ret;
	}
	friend constexpr auto intersection(intersecting_range const& self, range<IndexType> const& other) {
		return intersection(self.impl_, other);
	}
	friend constexpr auto intersection(range<IndexType> const& other, intersecting_range const& self) {
		return intersection(other, self.impl_);
	}
	friend constexpr auto operator<(intersecting_range const& self, IndexType end) {
		return intersecting_range::make_(self.impl_.first(), end);
	}
	friend constexpr auto operator<=(IndexType first, intersecting_range const& self) {
		return intersecting_range::make_(first, self.impl_.last());
	}

 public:
	constexpr auto        operator*() const& -> intersecting_range const& { return *this; }
	static constexpr auto all() noexcept { return intersecting_range{}; }
};

[[maybe_unused]] constexpr intersecting_range<> ALL = intersecting_range<>::all();
[[maybe_unused]] constexpr intersecting_range<> _   = ALL;  // NOLINT(readability-identifier-length)
[[maybe_unused]] constexpr intersecting_range<> U   = ALL;  // NOLINT(readability-identifier-length)
[[maybe_unused]] constexpr intersecting_range<> ooo = ALL;

[[maybe_unused]] constexpr intersecting_range<> V = U;  // NOLINT(readability-identifier-length)
[[maybe_unused]] constexpr intersecting_range<> A = V;  // NOLINT(readability-identifier-length)

#if !defined(__clang__) && !defined(CPPCHECK)
// cppcheck-suppress preprocessorErrorDirective ;  // unicode
// [[maybe_unused]] constexpr intersecting_range<> ∀ = V;  // not valid in clang or g++-15
// [[maybe_unused]] constexpr intersecting_range<> https://www.compart.com/en/unicode/U+2200 = V;
// [[maybe_unused]] constexpr intersecting_range<> ┄ = ALL;  // not valid in g++-15
#endif

template<class IndexType = std::ptrdiff_t, class IndexTypeLast = decltype(std::declval<IndexType>() + IndexType{1})>
struct extension_t : public range<IndexType, IndexTypeLast> {
	using range<IndexType, IndexTypeLast>::range;

	BOOST_MULTI_HD constexpr extension_t(IndexType first, IndexTypeLast last) noexcept
	: range<IndexType, IndexTypeLast>{first, last} {}

//	BOOST_MULTI_HD constexpr extension_t(extension_t::size_type size) : extensions_t(IndexType{}, IndexType{} + size) {}

	// cppcheck-suppress noExplicitConstructor ; because syntax convenience // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extension_t(IndexTypeLast last) noexcept  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) // NOSONAR(cpp:S1709) allow terse syntax
	: range<IndexType, IndexTypeLast>(IndexType{}, IndexType{} + last) {}

	template<
		class OtherExtension,
		decltype(
			detail::implicit_cast<IndexType>(std::declval<OtherExtension>().first()),
			detail::implicit_cast<IndexTypeLast>(std::declval<OtherExtension>().last())
		)* = nullptr
	>
	// cppcheck-suppress noExplicitConstructor ;  // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extension_t(OtherExtension const& other) noexcept  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: extension_t{other.first(), other.last()} {}

	// template<
	// class OtherExtension,
	// 	decltype(
	// 		detail::explicit_cast<IndexType>(std::declval<OtherExtension>().first()),
	// 		detail::explicit_cast<IndexTypeLast>(std::declval<OtherExtension>().last())
	// 	)* = nullptr
	// >
	// BOOST_MULTI_HD constexpr explicit extension_t(OtherExtension const& other) noexcept
	// : extension_t{other.first(), other.last()} {}

	template<class OtherExtension>
	BOOST_MULTI_HD constexpr auto operator=(OtherExtension const& other) -> extension_t& {
		(*this) = extension_t{other};
		return *this;
	}

	// BOOST_MULTI_HD constexpr extension_t() noexcept : range<IndexType, IndexTypeLast>() {}
	constexpr extension_t() = default;

	// friend constexpr auto size(extension_t const& self) -> typename extension_t::size_type { return self.size(); }

	friend constexpr auto intersection(extension_t const& ex1, extension_t const& ex2) -> extension_t {
		using std::max;
		using std::min;

		auto       first = max(ex1.first(), ex2.first());
		auto const last  = min(ex1.last(), ex2.last());

		first = min(first, last);

		return extension_t{first, last};
	}
};

#if defined(__cpp_deduction_guides) && (__cpp_deduction_guides >= 201703)
template<class IndexType, class IndexTypeLast>
extension_t(IndexType, IndexTypeLast) -> extension_t<IndexType, IndexTypeLast>;

template<class IndexType = multi::index>
extension_t(IndexType) -> extension_t<std::integral_constant<IndexType, 0>, IndexType>;
#endif

template<class IndexType = std::ptrdiff_t, class IndexTypeLast = decltype(std::declval<IndexType>() + 1)>
constexpr auto make_extension_t(IndexType first, IndexTypeLast last) {
	return extension_t<IndexType, IndexTypeLast>{first, last};
}

template<class IndexType = boost::multi::size_t>
constexpr auto make_extension_t(IndexType last) { return make_extension_t(std::integral_constant<IndexType, 0>{}, last); }

using index_range     = range<index>;
using index_extension = extension_t<index>;
using iextension      = index_extension;
using irange          = index_range;

namespace detail {

template<typename, typename>
struct append_to_type_seq {};

template<typename T, typename... Ts, template<typename...> class TT>
struct append_to_type_seq<T, TT<Ts...>> {
	using type = TT<Ts..., T>;
};

template<typename T, dimensionality_type N, template<typename...> class TT>
struct repeat {
	using type = typename append_to_type_seq<
		T,
		typename repeat<T, N - 1, TT>::type>::type;
};

template<typename T, template<typename...> class TT>
struct repeat<T, 0, TT> {
	using type = TT<>;
};

}  // end namespace detail

template<dimensionality_type D> using index_extensions = typename detail::repeat<index_extension, D, tuple>::type;

template<dimensionality_type D, class Tuple>
constexpr auto contains(index_extensions<D> const& iex, Tuple const& tup) {
	//  using detail::head;
	//  using detail::tail;
	return contains(head(iex), head(tup)) && contains(tail(iex), tail(tup));
}

}  // end namespace boost::multi

#undef BOOST_MULTI_HD

// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <cstddef>      // for ptrdiff_t
#include <iterator>     // for random_access_iterator_tag
#include <type_traits>  // for enable_if_t, is_base_of
#include <utility>      // for forward

#ifdef __NVCC__
	#define BOOST_MULTI_HD __host__ __device__
#else
	#define BOOST_MULTI_HD
#endif

namespace boost::multi {

struct empty_base {};

template<class Self> struct selfable {
 protected:
	selfable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using self_type = Self;
	constexpr auto        self() const -> self_type const& {
		static_assert(std::is_base_of_v<selfable<Self>, Self>);
		return static_cast<self_type const&>(*this);
	}
	constexpr auto        self() -> self_type& {
		static_assert(std::is_base_of_v<selfable<Self>, Self>);
		return static_cast<self_type&>(*this);
	}
	friend constexpr auto self(selfable const& self) -> self_type const& {
		static_assert(std::is_base_of_v<selfable<Self>, Self>);
		return self.self();
	}
};

template<class Self>
class ra_iterable : selfable<Self> {
	ra_iterable() = default;
	friend Self;

	template<class Self2 = Self>
	using difference_type_t = decltype(std::declval<Self2 const&>() - std::declval<Self2 const&>());

 public:
	using iterator_category = std::random_access_iterator_tag;

	template<class Self2 = Self, std::enable_if_t<std::is_same_v<Self2, Self>, int> =0>  // NOLINT(modernize-use-constraints) for C++20
	friend BOOST_MULTI_HD constexpr auto operator+(Self2 self, difference_type_t<Self2> const& n) { return self += n; }
	// template<class Self2 = Self>
	// friend auto operator+(difference_type<Self2> const& n, Self2 const& self) { return self + n; }
	BOOST_MULTI_HD constexpr auto operator++(int) { Self tmp{*this}; ++(this->self()); return tmp; }  // NOLINT(cert-dcl21-cpp)
	BOOST_MULTI_HD constexpr auto operator--(int) { Self tmp{*this}; --(this->self()); return tmp; }  // NOLINT(cert-dcl21-cpp)

	template<class Self2 = Self, std::enable_if_t<std::is_same_v<Self2, Self>, int> =0>  // NOLINT(modernize-use-constraints) for C++20
	BOOST_MULTI_HD constexpr auto friend operator!=(Self2 const& self, Self2 const& other) { return !(self == other); }
};

template<class Self, class U> struct equality_comparable2;

template<class Self>
struct equality_comparable2<Self, Self> : selfable<Self> {
	// friend constexpr auto operator==(equality_comparable2 const& self, equality_comparable2 const& other) {return     self.self() == other.self() ;}
	friend constexpr auto operator!=(equality_comparable2 const& self, equality_comparable2 const& other) { return !(self.self() == other.self()); }
};

template<class Self> struct equality_comparable : equality_comparable2<Self, Self> {
 protected:
	equality_comparable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;
};

template<class T, class V> struct totally_ordered2;

template<class Self>
struct totally_ordered2<Self, Self> : equality_comparable2<totally_ordered2<Self, Self>, totally_ordered2<Self, Self>> {
	using self_type = Self;
	BOOST_MULTI_HD constexpr auto self() const -> self_type const& { return static_cast<self_type const&>(*this); }

	// friend auto operator< (totally_ordered2 const& self, totally_ordered2 const& other) -> bool {return     self.self() < other.self() ;}
	friend BOOST_MULTI_HD constexpr auto operator==(totally_ordered2 const& self, totally_ordered2 const& other) -> bool { return !(self.self() < other.self()) && !(other.self() < self.self()); }
	// friend auto operator!=(totally_ordered2 const& self, totally_ordered2 const& other) {return    (s.self() < o.self()) or     (o.self() < s.self());}

	friend BOOST_MULTI_HD constexpr auto operator<=(totally_ordered2 const& self, totally_ordered2 const& other) -> bool { return !(other.self() < self.self()); }

	friend BOOST_MULTI_HD constexpr auto operator>(totally_ordered2 const& self, totally_ordered2 const& other) -> bool { return !(self.self() < other.self()) && !(self.self() == other.self()); }
	friend BOOST_MULTI_HD constexpr auto operator>=(totally_ordered2 const& self, totally_ordered2 const& other) -> bool { return !(self.self() < other.self()); }
};

template<class Self> using totally_ordered = totally_ordered2<Self, Self>;

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4820 )  // '3' bytes padding added after data member
#endif
template<class T>
struct totally_ordered2<T, void> {
	// template<class U>
	// friend constexpr auto operator<=(T const& self, U const& other) { return (self < other) || (self == other); }
	// template<class U>
	// friend constexpr auto operator>=(T const& self, U const& other) { return (other < self) || (self == other); }
	// template<class U>
	// friend constexpr auto operator>(T const& self, U const& other) { return other < self; }
};
#ifdef _MSC_VER
#pragma warning( pop )
#endif

template<class T>
struct copy_constructible {};

template<class T>
struct weakly_incrementable : selfable<T> {
 protected:
	weakly_incrementable() = default;

 public:
	constexpr auto operator++(int) -> T {
		auto ret{this->self()}; ++(this->self()); return ret;
	}
};

template<class T>
struct weakly_decrementable {
 protected:
	weakly_decrementable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend T;
	// friend T& operator--(weakly_decrementable& t){return --static_cast<T&>(t);}
};

template<class Self>
struct incrementable : totally_ordered<Self> {
 protected:
	incrementable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	friend BOOST_MULTI_HD constexpr auto operator++(incrementable& self, int) -> Self {
		static_assert(std::is_base_of_v<incrementable<Self>, Self>);
		Self tmp{self.self()};
		++self.self();
		assert(self.self() > tmp);
		return tmp;
	}
};

template<class T>
struct decrementable : weakly_decrementable<T> {
 protected:
	decrementable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend T;

 public:
	template<class U, typename = std::enable_if_t<!std::is_base_of_v<T, U>>>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto operator--(U& self, int) -> T {
		T tmp{self};
		--self;
		return tmp;
	}
};

template<class Self>
struct steppable : totally_ordered<Self> {
 protected:
	steppable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using self_type = Self;
	BOOST_MULTI_HD constexpr auto self() const -> self_type const& { return static_cast<self_type const&>(*this); }
	BOOST_MULTI_HD constexpr auto self() -> self_type& { return static_cast<self_type&>(*this); }

	friend BOOST_MULTI_HD constexpr auto operator++(steppable& self, int) -> Self {
		Self tmp{self.self()};
		++self.self();
		return tmp;
	}
	friend BOOST_MULTI_HD constexpr auto operator--(steppable& self, int) -> Self {
		Self tmp{self.self()};
		--self.self();
		return tmp;
	}
};

template<class Self, typename Difference>
struct affine_with_unit : steppable<Self> {
 protected:
	affine_with_unit() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using self_type = Self;
	BOOST_MULTI_HD constexpr auto cself() const -> self_type const& { return static_cast<self_type const&>(*this); }
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto self() const -> self_type const& { return static_cast<self_type const&>(*this); }
	BOOST_MULTI_HD constexpr auto self() -> self_type& { return static_cast<self_type&>(*this); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using difference_type = Difference;
	friend BOOST_MULTI_HD constexpr auto operator++(affine_with_unit& self) -> Self& { return self.self() += difference_type{1}; }
	friend BOOST_MULTI_HD constexpr auto operator--(affine_with_unit& self) -> Self& { return self.self() -= difference_type{1}; }

	BOOST_MULTI_HD constexpr auto operator+(difference_type const& diff) const -> Self {
		auto ret{cself()};
		ret += diff;
		return ret;
	}
	friend BOOST_MULTI_HD constexpr auto operator+(difference_type const& diff, affine_with_unit const& self) -> Self {
		auto ret{self.self()};
		ret += diff;
		return ret;
	}
	friend constexpr auto operator<(affine_with_unit const& self, affine_with_unit const& other) -> bool {
		return difference_type{0} < other.self() - self.self();
	}
};

template<class Self, typename Reference>
struct dereferenceable {
 protected:
	dereferenceable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using self_type = Self;
	constexpr auto self() const -> self_type const& { return static_cast<self_type const&>(*this); }
	constexpr auto self() -> self_type& { return static_cast<self_type&>(*this); }

	using reference = Reference;

	BOOST_MULTI_HD constexpr auto operator*() const -> reference { return *(self().operator->()); }
};

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4820 )  // '7' bytes padding added after base class
#endif

template<class Self, typename Difference, typename Reference>
struct random_accessable  // NOLINT(fuchsia-multiple-inheritance)
: affine_with_unit<Self, Difference>
, dereferenceable<Self, Reference> {
 protected:
	random_accessable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using difference_type   = Difference;
	using reference         = Reference;
	using iterator_category = std::random_access_iterator_tag;

	using self_type = Self;
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto self() const -> self_type const& { return static_cast<self_type const&>(*this); }
	BOOST_MULTI_HD constexpr auto self() -> self_type& { return static_cast<self_type&>(*this); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	BOOST_MULTI_HD constexpr auto operator[](difference_type idx) const -> reference { return *(self() + idx); }
};

#ifdef _MSC_VER
#pragma warning( pop )
#endif

template<class Self, class D>
class addable2 {
 protected:
	addable2() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using difference_type = D;

	template<class TT, typename = std::enable_if_t<std::is_base_of<Self, TT>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend BOOST_MULTI_HD constexpr auto operator+(TT&& self, difference_type const& diff) -> Self { return Self{std::forward<TT>(self)} += diff; }

	template<class TT, typename = std::enable_if_t<std::is_base_of<Self, TT>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend BOOST_MULTI_HD constexpr auto operator+(difference_type const& diff, TT&& self) -> Self { return std::forward<TT>(self) + diff; }
};

template<class T, class D>
class subtractable2 {
 protected:
	subtractable2() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend T;

 public:
	using difference_type = D;
	// TODO(correaa) clang 16 picks up this and converts the difference_type to TT !!
	// template<class TT, class = T>
	// friend auto operator-(TT&& self, difference_type const& diff) -> T {T tmp{std::forward<TT>(self)}; tmp -= diff; return tmp;}
};

template<class T, class Difference>
struct affine : addable2<T, Difference>
, subtractable2<T, Difference> {
 protected:
	affine() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend T;

 public:
	using difference_type = Difference;
};

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4820 )  // '3' bytes padding added after data member
#endif
template<class T>
class random_iterable {
 protected:
	random_iterable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend T;

 public:
	constexpr auto        cfront() const& -> decltype(auto) { return static_cast<T const&>(*this).front(); }
	constexpr auto        cback() const& -> decltype(auto) { return static_cast<T const&>(*this).back(); }
	friend constexpr auto cfront(T const& self) -> decltype(auto) { return self.cfront(); }
	friend constexpr auto cback(T const& self) -> decltype(auto) { return self.cback(); }
};
#ifdef _MSC_VER
#pragma warning( pop )
#endif

namespace detail {
template<class Self, class Value, class Reference = Value&, class Pointer = Value*, class Difference = std::ptrdiff_t>
struct random_access_iterator : equality_comparable2<Self, Self> {
 protected:
	random_access_iterator() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	friend Self;

 public:
	using difference_type   = Difference;
	using value_type        = Value;
	using pointer           = Pointer;
	using reference         = Reference;
	using iterator_category = std::random_access_iterator_tag;
	BOOST_MULTI_HD constexpr auto operator*() const -> Reference { return *static_cast<Self const&>(*this); }
};
}  // end namespace detail

}  // end namespace boost::multi

#undef BOOST_MULTI_HD

#include <algorithm>         // for max
#include <array>             // for array
#include <cassert>           // for assert
#include <cstddef>           // for size_t, ptrdiff_t, __GLIBCXX__
#include <cstdlib>           // for abs
#include <initializer_list>  // for initializer_list
#include <iostream>
#include <iterator>
#include <memory>       // for swap
#include <tuple>        // for tuple_element, tuple, tuple_size, tie, make_index_sequence, index_sequence
#include <type_traits>  // for enable_if_t, integral_constant, decay_t, declval, make_signed_t, common_type_t
#include <utility>      // for forward

#if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<ranges>)
#if !defined(__clang_major__) || !(__clang_major__ == 16)
#include <ranges>    // IWYU pragma: keep
#endif
#endif

// clang-format off
namespace boost::multi { template <boost::multi::dimensionality_type D, typename SSize = multi::size_type> struct layout_t; }
namespace boost::multi::detail { template <class ...Ts> class tuple; }
// clang-format on

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4514)  // inline function removed, in MSVC C++17 mode
#pragma warning(disable : 5045)  // Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
#endif

namespace boost::multi {

template<typename Stride>
struct stride_traits;

template<>
struct stride_traits<std::ptrdiff_t> {
	using category = std::random_access_iterator_tag;
};

template<typename Stride>
struct stride_traits {
	using category = typename Stride::category;
};

template<typename Integer>
struct stride_traits<std::integral_constant<Integer, 1>> {
#if defined(__cplusplus) && (__cplusplus >= 202002L) && (!defined(__clang__) || __clang_major__ != 10)
	using category = std::contiguous_iterator_tag;
#else
	using category = std::random_access_iterator_tag;
#endif
};

namespace detail {

template<class Tuple, std::size_t... Ns>
constexpr auto tuple_tail_impl(Tuple&& tup, std::index_sequence<Ns...> /*012*/) {
	(void)tup;  // workaround bug warning in nvcc
	using boost::multi::detail::get;
	return boost::multi::detail::tuple{std::forward<decltype(get<Ns + 1U>(std::forward<Tuple>(tup)))>(get<Ns + 1U>(std::forward<Tuple>(tup)))...};
}

template<class Tuple>
constexpr auto tuple_tail(Tuple&& t)  // NOLINT(readability-identifier-length) std naming
	-> decltype(tuple_tail_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>> - 1U>())) {
	return tuple_tail_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>> - 1U>());
}

}  // end namespace detail

// template<dimensionality_type D, typename SSize=multi::size_type> struct layout_t;

template<dimensionality_type D>
struct extensions_t;

template<typename T, dimensionality_type D, class Alloc = std::allocator<T> > struct array;

namespace detail {
struct non_copyable_base {
	non_copyable_base(non_copyable_base const&) = delete;
	non_copyable_base(non_copyable_base&&) = default;

	non_copyable_base() = default;

	auto operator=(non_copyable_base const&) -> non_copyable_base& = default;
	auto operator=(non_copyable_base&&) -> non_copyable_base& = default;

	~non_copyable_base() = default;
};

struct copyable_base {
	copyable_base(copyable_base const&) = default;
	copyable_base(copyable_base&&) = default;

	copyable_base() = default;

	auto operator=(copyable_base const&) -> copyable_base& = default;
	auto operator=(copyable_base&&) -> copyable_base& = default;

	~copyable_base() = default;
};
}  // end namespace detail

template<dimensionality_type D, class Proj>
class restriction
:
	std::conditional_t<
		std::is_reference_v<Proj>,
		detail::non_copyable_base,
		detail::copyable_base
	>
{
	extensions_t<D> xs_;
	Proj proj_;

	template<class Fun, class Tup>
	static BOOST_MULTI_HD constexpr auto std_apply_(Fun&& fun, Tup&& tup) -> decltype(auto) {
		using std::apply;
		return apply(std::forward<Fun>(fun), std::forward<Tup>(tup));
	}

 public:
	static constexpr dimensionality_type dimensionality = D;
	constexpr static dimensionality_type rank_v = D;

	using difference_type = typename extensions_t<D>::difference_type;
	using index = typename extensions_t<D>::index;

	BOOST_MULTI_HD constexpr restriction(extensions_t<D> xs, Proj proj) : xs_{xs}, proj_{std::move(proj)} {}

	using element = decltype(std_apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()));

	using value_type = std::conditional_t<
		(D == 1),
		element,
		array<element, D - 1>
	>;

 private:
	struct bind_front_t {
		multi::index idx_;
		Proj proj_;
		template<class... Args>
		BOOST_MULTI_HD constexpr auto operator()(Args&&... rest) const noexcept { return proj_(idx_, std::forward<Args>(rest)...); }
	};

	template<class Fun, class... Args>
	static BOOST_MULTI_HD constexpr auto apply_(Fun&& fun, Args&&... args) {
		using std::apply;
		return apply(std::forward<Fun>(fun), std::forward<Args>(args)...);
	}

 public:
	using reference = std::conditional_t<(D != 1),
		restriction<D - 1, bind_front_t>,
		decltype(apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()))  // (std::declval<index>()))
	>;

	#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	template<class... Indices>
	BOOST_MULTI_HD constexpr auto operator[](index idx, Indices... rest) const {
		return operator[](idx)[rest...];
	}
	BOOST_MULTI_HD constexpr auto operator[]() const -> decltype(auto) { return proj_() ; }
	#endif

	BOOST_MULTI_HD constexpr auto operator[](index idx) const -> decltype(auto) {
		// assert( extension().contains(idx) );
		if constexpr(D != 1) {
			// auto ll = [idx, proj = proj_](auto... rest) { return proj(idx, rest...); };
			// return restriction<D - 1, decltype(ll)>(extensions_t<D - 1>(xs_.base().tail()), ll);
			// return [idx, proj = proj_](auto... rest) noexcept { return proj(idx, rest...); } ^ extensions_t<D - 1>(xs_.base().tail());
			return bind_front_t{idx, proj_} ^ extensions_t<D - 1>(xs_.base().tail());
		} else {
			return proj_(idx);
		}
	}

	constexpr auto operator+() const { return multi::array<element, D>{*this}; }

	struct bind_transposed_t {
		Proj proj_;
		template<class T1, class T2, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, T2 jj, Ts... rest) const noexcept -> element { return proj_(jj, ii, rest...); }
	};

	BOOST_MULTI_HD constexpr auto transposed() const -> restriction<D, bind_transposed_t > {
		return bind_transposed_t{proj_} ^ layout_t<D>(extensions()).transpose().extensions();
		// return [proj = proj_](auto i, auto j, auto... rest) { return proj(j, i, rest...); } ^ layout_t<D>(extensions()).transpose().extensions();
	}

	struct bind_diagonal_t {
		Proj proj_;
		template<class T1, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ij, Ts... rest) const noexcept -> element { return proj_(ij, ij, rest...); }
	};

	BOOST_MULTI_HD constexpr auto diagonal() const -> restriction<D - 1, bind_diagonal_t > {
		static_assert( D > 1 );
		using std::get;  // needed for C++17
		return bind_diagonal_t{proj_} ^ (std::min( get<0>(sizes()), get<1>(sizes()) )*extensions().sub().sub());
		// return [proj = proj_](auto i, auto j, auto... rest) { return proj(j, i, rest...); } ^ layout_t<D>(extensions()).transpose().extensions();
	}

	BOOST_MULTI_HD constexpr auto operator~() const { return transposed(); }

	struct bind_repeat_t {
		Proj proj_;
		template<class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(multi::index /*unused*/, Ts... rest) const noexcept -> element { return proj_(rest...); }
	};

	BOOST_MULTI_HD auto repeated(multi::size_t n) const -> restriction<D + 1, bind_repeat_t> {
		return bind_repeat_t{proj_} ^ (n*extensions());
	}

	struct bind_partitioned_t {
		Proj proj_;
		size_type nn_;
		template<class T1, class T2, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, T2 jj, Ts... rest) const noexcept -> element { return proj_((ii * nn_) + jj, rest...); }
	};

	BOOST_MULTI_HD constexpr auto partitioned(size_type nn) const noexcept -> restriction<D + 1, bind_partitioned_t > {
		return bind_partitioned_t{proj_, size()/nn} ^ layout_t<D>(extensions()).partition(nn).extensions();
	}

	struct bind_reversed_t {
		Proj proj_;
		size_type size_m1;
		template<class T1, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, Ts... rest) const noexcept -> element { return proj_(size_m1 - ii, rest...); }
	};

	BOOST_MULTI_HD constexpr auto reversed() const { return bind_reversed_t{proj_, size() - 1} ^ extensions(); }

	struct bind_rotated_t {
		Proj proj_;
		size_type size_;
		template<class T1, class T2, class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(T1 ii, Ts... rest) const noexcept { return proj_(rest..., ii); }
	};

	BOOST_MULTI_HD constexpr auto rotated() const { return bind_rotated_t{proj_, size()} ^ extensions(); }

	template<class Proj2>
	struct bind_element_transformed_t {
		Proj proj_;
		Proj2 proj2_;
		template<class... Ts>
		BOOST_MULTI_HD constexpr auto operator()(Ts... rest) const noexcept -> element { return proj2_(proj_(rest...)); }
	};

	template<class Proj2>
	BOOST_MULTI_HD auto element_transformed(Proj2 proj2) const -> restriction<D, bind_element_transformed_t<Proj2> > {
		return bind_element_transformed_t<Proj2>{proj_, proj2} ^ extensions();
	}

	template<class Proj2>
	class bind_transform_t {
		restriction proj_;
		Proj2 proj2_;
		friend restriction;
		bind_transform_t(restriction proj, Proj2 proj2) : proj_{std::move(proj)}, proj2_{std::move(proj2)} {}

	 public:
		BOOST_MULTI_HD constexpr auto operator()(restriction::index idx) const noexcept { return proj2_(proj_[idx]); }
	};

	template<class Proj2, dimensionality_type One = 1  /*workaround for MSVC*/>
	BOOST_MULTI_HD auto transformed(Proj2 proj2) const -> restriction<1, bind_transform_t<Proj2> > {
		return bind_transform_t<Proj2>{*this, proj2} ^ multi::extensions_t<One>({extension()});
	}

	template<class Cursor, dimensionality_type DD = D>
	class cursor_t {
		Proj const* Pproj_;
		Cursor cur_;
		friend class restriction;
		explicit constexpr cursor_t(Proj const* Pproj, Cursor cur) : Pproj_{Pproj}, cur_{cur} {}

	 public:
		using difference_type = restriction::difference_type;

	 	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> decltype(auto) {
			if constexpr(DD != 1) {
				auto cur = cur_[n];
				return cursor_t<decltype(cur), DD - 1>{Pproj_, cur};
			} else {
				return apply_(*Pproj_, cur_[n]);
			}
		}
	};

	auto home() const {
		auto cur = extensions().home();
		return cursor_t<decltype(cur), D>{&proj_, cur};
	}

	class iterator {
		typename extensions_t<D>::iterator it_;
		Proj const* Pproj_;

		iterator(typename extensions_t<D>::iterator it, Proj const* Pproj) : it_{it}, Pproj_{Pproj} {}

		friend restriction;

		struct bind_front_t {
			multi::index idx_;
			Proj proj_;
			template<class... Args>
			BOOST_MULTI_HD constexpr auto operator()(Args&&... rest) const noexcept { return proj_(idx_, std::forward<Args>(rest)...); }
		};

	 public:
		constexpr iterator() = default;  // cppcheck-suppress uninitMemberVar ; partially formed
		// constexpr iterator() {}  // = default;  // NOLINT(hicpp-use-equals-default,modernize-use-equals-default) TODO(correaa) investigate workaround

		iterator(iterator const& other) = default;
		iterator(iterator&&) noexcept = default;

		auto operator=(iterator&&) noexcept -> iterator& = default;
		auto operator=(iterator const&) -> iterator&     = default;

		~iterator() = default;

		using value_type = std::conditional_t<(D != 1),
			restriction<D - 1, bind_front_t>,
			decltype(apply_(std::declval<Proj>(), std::declval<typename extensions_t<D>::element>()))  // (std::declval<index>()))
		>;

		using reference = std::conditional_t<(D != 1),
			restriction<D - 1, bind_front_t>,
			decltype(apply_(std::declval<Proj&>(), std::declval<typename extensions_t<D>::element>()))  // (std::declval<index>()))
		>;

		using iterator_category = std::random_access_iterator_tag;

		constexpr auto operator++() -> auto& { ++it_; return *this; }
		constexpr auto operator--() -> auto& { --it_; return *this; }

		constexpr auto operator+=(difference_type dd) -> auto& { it_+=dd; return *this; }
		constexpr auto operator-=(difference_type dd) -> auto& { it_-=dd; return *this; }

		constexpr auto operator++(int) -> iterator { iterator ret{*this}; ++(*this); return ret; }
		constexpr auto operator--(int) -> iterator { iterator ret{*this}; --(*this); return ret; }

		friend constexpr auto operator-(iterator const& self, iterator const& other) { return self.it_ - other.it_; }
		friend constexpr auto operator+(iterator const& self, difference_type n) { iterator ret{self}; return ret += n; }
		friend constexpr auto operator-(iterator const& self, difference_type n) { iterator ret{self}; return ret -= n; }

		friend constexpr auto operator+(difference_type n, iterator const& self) { return self + n; }

		friend constexpr auto operator==(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ == other.it_; }
		friend constexpr auto operator!=(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ != other.it_; }

		friend auto operator<=(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ <= other.it_; }
		friend auto operator< (iterator const& self, iterator const& other) noexcept -> bool { return self.it_ <  other.it_; }
		friend auto operator> (iterator const& self, iterator const& other) noexcept -> bool { return self.it_ >  other.it_; }
		friend auto operator>=(iterator const& self, iterator const& other) noexcept -> bool { return self.it_ >  other.it_; }

		BOOST_MULTI_HD constexpr auto operator*() const -> decltype(auto) {
			if constexpr(D != 1) {
				using std::get;
				// auto ll = [idx = get<0>(*it_), proj = proj_](auto... rest) { return proj(idx, rest...); };
				return restriction<D - 1, bind_front_t>(extensions_t<D - 1>((*it_).tail()), bind_front_t{get<0>(*it_), *Pproj_});
			} else {
				using std::get;
				return (*Pproj_)(get<0>(*it_));
			}
		}

		BOOST_MULTI_HD auto operator[](difference_type dd) const { return *((*this) + dd); }  // TODO(correaa) use ra_iterator_facade
	};

	constexpr auto begin() const { return iterator{xs_.begin(), &proj_}; }
	constexpr auto end() const { return iterator{xs_.end(), &proj_}; }

	constexpr auto size() const { return xs_.size(); }
	constexpr auto sizes() const { return xs_.sizes(); }

	constexpr auto extension() const { return xs_.extension(); }
	constexpr auto extensions() const { return xs_; }

	constexpr auto front() const { return *begin(); }
	constexpr auto back() const { return *(begin() + (size() - 1)); }

	class elements_t {
		typename extensions_t<D>::elements_t elems_;
		Proj proj_;

		elements_t(typename extensions_t<D>::elements_t elems, Proj proj) : elems_{elems}, proj_{std::move(proj)} {}
		friend class restriction;

	public:
		BOOST_MULTI_HD constexpr auto operator[](index idx) const -> decltype(auto) { using std::apply; return apply(proj_, elems_[idx]); }

		using difference_type = restriction::difference_type;

		class iterator : ra_iterable<iterator> {
			typename extensions_t<D>::elements_t::iterator it_;
			BOOST_MULTI_NO_UNIQUE_ADDRESS Proj proj_;

		 public:
			iterator(typename extensions_t<D>::elements_t::iterator it, Proj proj) : it_{it}, proj_{std::move(proj)} {}

			auto operator++() -> auto& { ++it_; return *this; }
			auto operator--() -> auto& { --it_; return *this; }

			constexpr auto operator+=(difference_type dd) -> auto& { it_+=dd; return *this; }
			constexpr auto operator-=(difference_type dd) -> auto& { it_-=dd; return *this; }

			friend constexpr auto operator-(iterator const& self, iterator const& other) { return self.it_ - other.it_; }

			constexpr auto operator*() const -> decltype(auto) { using std::apply; return apply(proj_, *it_); }

			using difference_type = elements_t::difference_type;
			using value_type = difference_type;
			using pointer = void;
			using reference = value_type;
			using iterator_category = std::random_access_iterator_tag;

			friend auto operator==(iterator const& self, iterator const& other) -> bool { return self.it_ == other.it_; }
			friend auto operator!=(iterator const& self, iterator const& other) -> bool { return self.it_ != other.it_; }

			friend auto operator<=(iterator const& self, iterator const& other) -> bool { return self.it_ <= other.it_; }
			friend auto operator< (iterator const& self, iterator const& other) -> bool { return self.it_ <  other.it_; }

			BOOST_MULTI_HD constexpr auto operator[](difference_type dd) const { return *((*this) + dd); }  // TODO(correaa) use ra_iterator_facade
		};

		auto begin() const { return iterator{elems_.begin(), proj_}; }
		auto end()   const { return iterator{elems_.end()  , proj_}; }

		auto size() const { return elems_.size(); }
	};

	constexpr auto elements() const { return elements_t{xs_.elements(), proj_}; }
	constexpr auto num_elements() const { return xs_.num_elements(); }
};

template<dimensionality_type D>
struct extensions_t : boost::multi::detail::tuple_prepend_t<index_extension, typename extensions_t<D - 1>::base_> {
	using base_ = boost::multi::detail::tuple_prepend_t<index_extension, typename extensions_t<D - 1>::base_>;

 public:
	static constexpr dimensionality_type dimensionality = D;
	constexpr static dimensionality_type rank_v = D;

	using difference_type = index_extension::difference_type;
	using nelems_type = multi::index;

	using element = boost::multi::detail::tuple_prepend_t<index_extension::value_type, typename extensions_t<D - 1>::element>;

	extensions_t() = default;

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(multi::size_t size)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : allow terse syntax
	: extensions_t{index_extension{size}} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(index_extension ext1)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{ext1} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 2, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	BOOST_MULTI_HD constexpr extensions_t(index_extension ext1, index_extension ext2)
	: base_{ext1, ext2} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 3, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	BOOST_MULTI_HD constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3)
	: base_{ext1, ext2, ext3} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 4, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	BOOST_MULTI_HD constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3, index_extension ext4) noexcept
	: base_{ext1, ext2, ext3, ext4} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 5, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	BOOST_MULTI_HD constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3, index_extension ext4, index_extension ext5)
	: base_{ext1, ext2, ext3, ext4, ext5} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 6, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	BOOST_MULTI_HD constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3, index_extension ext4, index_extension ext5, index_extension ext6)
	: base_{ext1, ext2, ext3, ext4, ext5, ext6} {}

	template<class T1, class T = void, class = decltype(base_{tuple<T1>{}}), std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(detail::tuple<T1> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_{std::move(extensions)} {}

	template<class T1, class T = void, class = decltype(base_{::std::tuple<T1>{}}), std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(::std::tuple<T1> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T = void, class = decltype(base_{tuple<T1, T2>{}}), std::enable_if_t<sizeof(T*) && D == 2, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(detail::tuple<T1, T2> const& extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{extensions} {}

	template<class T1, class T2, class T = void, class = decltype(base_{::std::tuple<T1, T2>{}}), std::enable_if_t<sizeof(T*) && D == 2, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(::std::tuple<T1, T2> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T = void, class = decltype(base_{tuple<T1, T2, T3>{}}), std::enable_if_t<sizeof(T*) && D == 3, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(tuple<T1, T2, T3> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T = void, class = decltype(base_{::std::tuple<T1, T2, T3>{}}), std::enable_if_t<sizeof(T*) && D == 3, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(::std::tuple<T1, T2, T3> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T = void, class = decltype(base_{tuple<T1, T2, T3, T4>{}}), std::enable_if_t<sizeof(T*) && D == 4, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(tuple<T1, T2, T3, T4> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T = void, class = decltype(base_{::std::tuple<T1, T2, T3, T4>{}}), std::enable_if_t<sizeof(T*) && D == 4, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(::std::tuple<T1, T2, T3, T4> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T5, class T = void, class = decltype(base_{tuple<T1, T2, T3, T4, T5>{}}), std::enable_if_t<sizeof(T*) && D == 5, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(tuple<T1, T2, T3, T4, T5> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
	: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T5, class T = void, class = decltype(base_{::std::tuple<T1, T2, T3, T4, T5>{}}), std::enable_if_t<sizeof(T*) && D == 5, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(::std::tuple<T1, T2, T3, T4, T5> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_{std::move(extensions)} {}

	template<class... Ts>
	BOOST_MULTI_HD constexpr explicit extensions_t(tuple<Ts...> const& tup)
	: extensions_t(tup, std::make_index_sequence<static_cast<std::size_t>(D)>()) {}

	// template<std::size_t I, class TU> static constexpr auto get_(TU const& tu) { using std::get; return get<I>(tu); }

	template<class OtherExtensions,
		decltype( multi::detail::implicit_cast<index_extension>(OtherExtensions{}.extension()) )* = nullptr,
		decltype( multi::detail::implicit_cast<typename layout_t<D - 1>::extensions_type>(OtherExtensions{}.sub()) )* = nullptr
	>
	// cppcheck-suppress noExplicitConstructor ;  // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(OtherExtensions const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: extensions_t(other.extension(), other.sub()) {}

	BOOST_MULTI_HD constexpr extensions_t(index_extension const& extension, typename layout_t<D - 1>::extensions_type const& other)
	: extensions_t(multi::detail::ht_tuple(extension, other.base())) {}

	BOOST_MULTI_HD constexpr auto base() const& -> base_ const& { return *this; }
	BOOST_MULTI_HD constexpr auto base() & -> base_& { return *this; }

	friend constexpr auto operator*(index_extension const& extension, extensions_t const& self) -> extensions_t<D + 1> {
		// return extensions_t<D + 1>(tuple(extension, self.base()));
		return extensions_t<D + 1>(extension, self);
	}

	friend BOOST_MULTI_HD auto operator==(extensions_t const& self, extensions_t const& other) { return self.base() == other.base(); }
	friend BOOST_MULTI_HD auto operator!=(extensions_t const& self, extensions_t const& other) { return self.base() != other.base(); }

	using index        = multi::index;
	using indices_type = multi::detail::tuple_prepend_t<index, typename extensions_t<D - 1>::indices_type>;

	template<class Func>
	friend BOOST_MULTI_HD constexpr auto operator^(Func fun, extensions_t const& xs) {
		return restriction<D, Func>(xs, std::move(fun));
	}
	template<class Func>
	friend constexpr auto operator->*(extensions_t const& xs, Func fun) {
		return restriction<D, Func>(xs, std::move(fun));
	}

	BOOST_MULTI_HD constexpr auto sub() const {
		return extensions_t<D - 1>{static_cast<base_ const&>(*this).tail()};
	}

	[[nodiscard]]
	BOOST_MULTI_HD constexpr auto from_linear(nelems_type const& n) const -> indices_type {
		auto const sub_num_elements = sub().num_elements();
		#if !(defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
		assert(sub_num_elements != 0);  // clang hip doesn't allow assert in host device functions
		#endif
		return multi::detail::ht_tuple(n / sub_num_elements, sub().from_linear(n % sub_num_elements));
	}

	friend constexpr auto operator%(nelems_type idx, extensions_t const& extensions) { return extensions.from_linear(idx); }

	constexpr explicit operator bool() const { return !layout_t<D>{*this}.empty(); }

	template<class... Indices>
	BOOST_MULTI_HD constexpr auto to_linear(index const& idx, Indices const&... rest) const {
		auto const sub_extensions = extensions_t<D - 1>{this->base().tail()};
		return (idx * sub_extensions.num_elements()) + sub_extensions.to_linear(rest...);
	}

	template<class... Indices>
	BOOST_MULTI_HD constexpr auto operator()(index idx, Indices... rest) const { return to_linear(idx, rest...); }

	template<class Before, dimensionality_type DD>
	class cursor_t {
		Before bef_;
		// missing start indices information
		template<class, dimensionality_type> friend class cursor_t;
		friend extensions_t;

	 public:
		cursor_t() = default;
		explicit cursor_t(Before const& bef) : bef_{bef} {}
		
		static constexpr dimensionality_type dimensionality = DD;

		constexpr auto operator[](difference_type n) const {
			using std::apply;
			if constexpr(DD != 1) {
				return cursor_t<typename multi::layout_t<std::tuple_size_v<Before> + 1>::indexes, DD - 1> {
					apply([n] (auto... es) {return detail::mk_tuple(es..., n);}, bef_) 
				};
			} else {
				return apply([n] (auto... es) {return detail::mk_tuple(es..., n);}, bef_); 
			}
		}
	};

	auto home() const {
		return cursor_t<tuple<>, D>{};
	}

	class iterator {  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) constructor does not initialize these fields: idx_
		index idx_;
		extensions_t<D - 1> rest_;
		friend extensions_t;
	
		constexpr iterator(index idx, extensions_t<D - 1> rest) : idx_{idx}, rest_{rest} {}

	 public:
		iterator() = default;

		using difference_type = index;
		using value_type = decltype(ht_tuple(std::declval<index>(), std::declval<extensions_t<D - 1>>().base()));
		using pointer = void;
		using reference = value_type;
		using iterator_category = std::random_access_iterator_tag;

		constexpr auto operator+=(difference_type d) -> iterator& { idx_ += d; return *this; }
		constexpr auto operator-=(difference_type d) -> iterator& { idx_ -= d; return *this; }

		constexpr auto operator+(difference_type d) const { return iterator{idx_ + d, rest_}; }
		constexpr auto operator-(difference_type d) const { return iterator{idx_ - d, rest_}; }

		friend constexpr auto operator-(iterator const& self, iterator const& other) -> difference_type { assert( self.rest_ == other.rest_ ); return self.idx_ - other.idx_; }

		friend constexpr auto operator+(difference_type n, iterator const& self) { return self + n; }

		constexpr auto operator++() -> auto& { ++idx_; return *this; }
		constexpr auto operator--() -> auto& { --idx_; return *this; }

		constexpr auto operator++(int) -> iterator { iterator ret{*this}; ++idx_; return ret; }
		constexpr auto operator--(int) -> iterator { iterator ret{*this}; --idx_; return ret; }

		constexpr auto operator*() const {
			// multi::detail::what(rest_);
			return ht_tuple(idx_, rest_.base());
		}

		BOOST_MULTI_HD constexpr auto operator[](difference_type const& n) const -> reference { return *((*this) + n); }

		friend constexpr auto operator==(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ == other.idx_; }
		friend constexpr auto operator!=(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ != other.idx_; }

		friend constexpr auto operator<(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ < other.idx_; }
		friend constexpr auto operator>(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ > other.idx_; }

		friend constexpr auto operator<=(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ <= other.idx_; }
		friend constexpr auto operator>=(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ >= other.idx_; }
	};

	constexpr auto begin() const { return iterator{this->base().head().first(), this->base().tail()}; }
	constexpr auto end()   const { return iterator{this->base().head().last() , this->base().tail()}; }

	BOOST_MULTI_HD constexpr auto operator[](index idx) const {
		return static_cast<base_ const&>(*this)[idx];
	}

	template<class... Indices>
	BOOST_MULTI_HD constexpr auto next_canonical(index& idx, Indices&... rest) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		if(extensions_t<D - 1>{this->base().tail()}.next_canonical(rest...)) {
			++idx;
		}
		if(idx == this->base().head().last()) {
			idx = this->base().head().first();
			return true;
		}
		return false;
	}
	template<class... Indices>
	constexpr auto prev_canonical(index& idx, Indices&... rest) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		if(extensions_t<D - 1>{this->base().tail()}.prev_canonical(rest...)) {
			--idx;
		}
		if(idx < static_cast<index>(this->base().head().first())) {
			idx = static_cast<index>(this->base().head().back());
			return true;
		}
		return false;
	}

	class elements_t {
		extensions_t xs_;
		explicit constexpr elements_t(extensions_t const& xs) : xs_{xs} {}

		friend struct extensions_t;

	 public:
		using difference_type = extensions_t::difference_type;

		class iterator {
			index_extension::iterator curr_;

			typename extensions_t<D - 1>::elements_t::iterator rest_it_;
			typename extensions_t<D - 1>::elements_t::iterator rest_begin_;
			typename extensions_t<D - 1>::elements_t::iterator rest_end_;

			BOOST_MULTI_HD constexpr iterator(
				index_extension::iterator curr,
				typename extensions_t<D - 1>::elements_t::iterator rest_it,
				typename extensions_t<D - 1>::elements_t::iterator rest_begin,
				typename extensions_t<D - 1>::elements_t::iterator rest_end
			)
			: curr_{curr}, rest_it_{rest_it}, rest_begin_{rest_begin}, rest_end_{rest_end} {}

			friend class elements_t;

		 public:		
			using difference_type   = elements_t::difference_type;
			using value_type        = indices_type;
			using pointer           = void;
			using reference         = value_type;
			using iterator_category = std::random_access_iterator_tag;

			template<class CUT>
			class mk_tup {
				CUT cu_;

			 public:
				constexpr explicit mk_tup(CUT cu) : cu_{cu} {}
				template<class... Ts>
				constexpr auto operator()(Ts... es) const { return detail::mk_tuple(cu_, es...); }
			};

			BOOST_MULTI_HD constexpr auto operator*() const {
				// printf("op* %ld ...\n", *curr_);
				using std::apply;
				return apply(mk_tup<decltype(*curr_)>{*curr_}, *rest_it_);
				// return apply([cu = *curr_] BOOST_MULTI_HD (auto... es) {return detail::mk_tuple(cu, es...);}, *rest_it_); 
			}

			BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> iterator& {
				if(n > 0) {  // mull-ignore: cxx_gt_to_ge
					curr_ += (rest_it_ - rest_begin_ + n) / (rest_end_ - rest_begin_);
					rest_it_ = rest_begin_ + ((rest_it_ - rest_begin_ + n) % (rest_end_ - rest_begin_));
				} else if(n < 0) {  // mull-ignore
					curr_ -= (rest_end_ - rest_it_ - n) / (rest_end_ - rest_begin_);
					rest_it_ = rest_end_ - ((rest_end_ - rest_it_ - n) % (rest_end_ - rest_begin_));
					if(rest_it_ == rest_end_) {
						rest_it_ = rest_begin_;
						++curr_;
					}
				}
				return *this;
			}

			BOOST_MULTI_HD constexpr auto operator-=(difference_type n) -> iterator& {
				if(n > 0) {  // mull-ignore: cxx_gt_to_ge
					curr_ -= (rest_end_ - rest_it_ + n) / (rest_end_ - rest_begin_);
					rest_it_ = rest_end_ - ((rest_end_ - rest_it_ + n) % (rest_end_ - rest_begin_));
					if(rest_it_ == rest_end_) {
						rest_it_ = rest_begin_;
						++curr_;
					}
				} else if(n < 0) {  // mull-ignore
					curr_ += (rest_it_ - rest_begin_ - n) / (rest_end_ - rest_begin_);
					rest_it_ = rest_begin_ + ((rest_it_ - rest_begin_ - n) % (rest_end_ - rest_begin_));
				}
				return *this;
			}

			friend BOOST_MULTI_HD constexpr auto operator-(iterator const& self, iterator const& other) -> difference_type {
				return ((self.curr_ - other.curr_) * (self.rest_end_ - self.rest_begin_)) + (self.rest_it_ - self.rest_begin_) - (other.rest_it_ - other.rest_begin_);
			}

			BOOST_MULTI_HD constexpr auto operator-(difference_type n) const {
				return iterator{*this} -= n;
			}

			BOOST_MULTI_HD constexpr auto operator+(difference_type n) const {
				return iterator{*this} += n;
			}

			BOOST_MULTI_HD constexpr auto operator++() -> auto& {
				// printf("++\n");
				++rest_it_;
				if( rest_it_ == rest_end_ ) {
					rest_it_ = rest_begin_;
					++curr_;
				}
				return *this;
			}

			BOOST_MULTI_HD constexpr auto operator--() -> auto& {
				// assert(0);
				// printf("--\n");
				if( rest_it_ == rest_begin_ ) {
					rest_it_ = rest_end_;
					--curr_;
				}
				--rest_it_;
				return *this;
			}

			BOOST_MULTI_HD constexpr auto operator[](difference_type dd) const { return *((*this) + dd); }

			friend BOOST_MULTI_HD constexpr auto operator==(iterator const& self, iterator const& other) { return (self.curr_ == other.curr_) && (self.rest_it_ == other.rest_it_); }
			friend BOOST_MULTI_HD constexpr auto operator!=(iterator const& self, iterator const& other) { return (self.curr_ != other.curr_) || (self.rest_it_ != other.rest_it_); }

			friend BOOST_MULTI_HD constexpr auto operator< (iterator const& self, iterator const& other) { return (self.curr_ <  other.curr_) || ((self.curr_ == other.curr_) && (self.rest_it_ < other.rest_it_)); }
			friend BOOST_MULTI_HD constexpr auto operator<=(iterator const& self, iterator const& other) { return (self < other) || (self == other); }
		};

		constexpr auto begin() const {
			return iterator{
				xs_.head().begin(),
				extensions_t<D - 1>{xs_.tail()}.elements().begin(),
				extensions_t<D - 1>{xs_.tail()}.elements().begin(),
				extensions_t<D - 1>{xs_.tail()}.elements().end(),
			};
		}

		constexpr auto end() const {
			return iterator{
				xs_.head().end(),
				extensions_t<D - 1>{xs_.tail()}.elements().begin(),
				extensions_t<D - 1>{xs_.tail()}.elements().begin(),
				extensions_t<D - 1>{xs_.tail()}.elements().end(),
			};
		}

		BOOST_MULTI_HD constexpr auto operator[](index idx) const { return begin()[idx]; }

		auto size() const { return xs_.num_elements(); }
	};

	constexpr auto elements() const { return elements_t{*this}; }

	template<class Func>
	BOOST_MULTI_HD constexpr auto element_transformed(Func fun) const { return [fun](auto const&... xs){ return fun(detail::mk_tuple(xs...)); } ^(*this); }

	BOOST_MULTI_HD constexpr auto extension() const { return this->get<0>(); }
	BOOST_MULTI_HD constexpr auto size() const { return this->get<0>().size(); }
	BOOST_MULTI_HD constexpr auto sizes() const {
		return this->apply([](auto const&... xs) { return multi::detail::mk_tuple(xs.size()...); });
	}

 private:
	template<class Archive, std::size_t... I>
	void serialize_impl_(Archive& arxiv, std::index_sequence<I...> /*unused012*/) {
		using boost::multi::detail::get;
		(void)std::initializer_list<unsigned>{(arxiv & multi::archive_traits<Archive>::make_nvp("extension", get<I>(this->base())), 0U)...};
	}

 public:
	template<class Archive>
	void serialize(Archive& arxiv, unsigned int const /*version*/) {
		serialize_impl_(arxiv, std::make_index_sequence<static_cast<std::size_t>(D)>());
	}

 private:
	template<class Array, std::size_t... I, typename = decltype(base_{boost::multi::detail::get<I>(std::declval<Array const&>())...})>
	BOOST_MULTI_HD constexpr extensions_t(Array const& tup, std::index_sequence<I...> /*unused012*/)
	: base_{boost::multi::detail::get<I>(tup)...} {}

	static BOOST_MULTI_HD constexpr auto multiply_fold_() -> size_type { return static_cast<size_type>(1U); }
	static BOOST_MULTI_HD constexpr auto multiply_fold_(size_type const& size) -> size_type { return size; }
	template<class... As>
	static BOOST_MULTI_HD constexpr auto multiply_fold_(size_type const& size, As const&... rest) -> size_type { return size * static_cast<size_type>(multiply_fold_(rest...)); }

	template<std::size_t... I>
	BOOST_MULTI_HD constexpr auto num_elements_impl_(std::index_sequence<I...> /*unused012*/) const -> size_type {
		using boost::multi::detail::get;
		return static_cast<size_type>(multiply_fold_(static_cast<size_type>(get<I>(this->base()).size())...));
	}

 public:
	BOOST_MULTI_HD constexpr auto num_elements() const -> size_type {
		return static_cast<size_type>(num_elements_impl_(std::make_index_sequence<static_cast<std::size_t>(D)>()));
	}
	friend constexpr auto intersection(extensions_t const& self, extensions_t const& other) -> extensions_t {
		using boost::multi::detail::get;
		return extensions_t{
			multi::detail::ht_tuple(
				index_extension{intersection(get<0>(self.base()), get<0>(other.base()))},
				intersection(extensions_t<D - 1>{self.base().tail()}, extensions_t<D - 1>{other.base().tail()}).base()
			)
		};
	}

	template<std::size_t Index, std::enable_if_t<(Index < D), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(extensions_t const& self) -> typename std::tuple_element_t<Index, base_> {
		using boost::multi::detail::get;
		return get<Index>(self.base());
	}

	template<std::size_t Index, std::enable_if_t<(Index < D), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr auto get() const -> std::tuple_element_t<Index, base_> {
		using boost::multi::detail::get;
		return get<Index>(this->base());
	}

	template<class Fn>
	constexpr auto apply(Fn&& fn) const -> decltype(auto) {
		return std::apply(std::forward<Fn>(fn), this->base());
	}
};

template<> struct extensions_t<0> : tuple<> {
	using base_ = tuple<>;

 private:
	// base_ impl_;

 public:
	static constexpr dimensionality_type dimensionality = 0;  // TODO(correaa): consider deprecation

	using rank = std::integral_constant<dimensionality_type, 0>;
	using element = tuple<>;

	using index = multi::index;

	using nelems_type = index;
	using difference_type = index;

	explicit BOOST_MULTI_HD constexpr extensions_t(tuple<> const& tup)
	: base_{tup} {}

	extensions_t() = default;

	BOOST_MULTI_HD constexpr auto base() const& -> base_ const& { return *this; }
	BOOST_MULTI_HD constexpr auto base() & -> base_& { return *this; }

	template<class Archive> static void serialize(Archive& /*ar*/, unsigned /*version*/) { /*noop*/ }

	static BOOST_MULTI_HD constexpr auto num_elements() /*const*/ -> size_type { return 1; }

	using indices_type = tuple<>;

	[[nodiscard]] static constexpr auto from_linear(nelems_type const& n) /*const*/ -> indices_type {
		assert(n == 0);
		(void)n;  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : constexpr function
		return indices_type{};
	}
	friend constexpr auto operator%(nelems_type const& n, extensions_t const& /*s*/) -> tuple<> { return /*s.*/ from_linear(n); }

	static BOOST_MULTI_HD constexpr auto to_linear() /*const*/ -> difference_type { return 0; }
	BOOST_MULTI_HD constexpr auto        operator()() const { return to_linear(); }

	constexpr void operator[](index) const = delete;

	static BOOST_MULTI_HD constexpr auto next_canonical() /*const*/ -> bool { return true; }
	static BOOST_MULTI_HD constexpr auto prev_canonical() /*const*/ -> bool { return true; }

	friend constexpr auto intersection(extensions_t const& /*x1*/, extensions_t const& /*x2*/) -> extensions_t { return {}; }

	constexpr BOOST_MULTI_HD auto operator==(extensions_t const& /*other*/) const { return true; }
	constexpr BOOST_MULTI_HD auto operator!=(extensions_t const& /*other*/) const { return false; }

	template<std::size_t Index>  // TODO(correaa) = detele ?
	friend constexpr auto get(extensions_t const& self) -> typename std::tuple_element_t<Index, base_> {
		using boost::multi::detail::get;
		return get<Index>(self.base());
	}

	template<std::size_t Index>  // TODO(correaa) = detele ?
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto get() const -> typename std::tuple_element_t<Index, base_> {
		using boost::multi::detail::get;
		return get<Index>(this->base());
	}

	template<class Fun>
	friend BOOST_MULTI_HD constexpr auto operator^(Fun&& fun, extensions_t const& xs) {
		return restriction<0, std::decay_t<Fun> >(xs, std::forward<Fun>(fun));
	}
};

template<> struct extensions_t<1> : tuple<multi::index_extension> {
	using base_ = tuple<multi::index_extension>;

	static constexpr auto dimensionality = 1;  // TODO(correaa): consider deprecation

	constexpr static dimensionality_type rank_v = 1;

	using size_type = multi::index_extension::size_type;
	using difference_type = multi::index_extension::difference_type;
	using element = tuple<multi::index_extension::value_type>;
	using index = multi::index;

	constexpr auto extension() const { using std::get; return get<0>(static_cast<base_ const&>(*this)); }

	constexpr auto sub() const { return extensions_t<0>{this->base().tail()}; }

	class cursor_t {
		index idx_;
		extensions_t<0> rest_;
		friend extensions_t;

		constexpr cursor_t(index idx, extensions_t<0> rest) : idx_{idx}, rest_{rest} {}

	 public:
		cursor_t() = default;
		using value_type = decltype(ht_tuple(std::declval<index>(), std::declval<extensions_t<0>>().base()));
		using reference = value_type;

		BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> reference { return ht_tuple(idx_ + n, rest_.base());; }
	};

	auto home() const -> cursor_t {
		return cursor_t{this->base().head().first(), extensions_t<0>{this->base().tail()}};
	}

	class iterator {  // : public weakly_incrementable<iterator> {
		index idx_;
		extensions_t<0> rest_;
		friend extensions_t;
	
		constexpr iterator(index idx, extensions_t<0> rest) : idx_{idx}, rest_{rest} {}

	 public:
		iterator() = default;

		using difference_type = index;
		using value_type = decltype(ht_tuple(std::declval<index>(), std::declval<extensions_t<0>>().base()));
		using pointer = void;
		using reference = value_type;
		using iterator_category = std::random_access_iterator_tag;

		constexpr auto operator+(difference_type d) const { return iterator{idx_ + d, rest_}; }
		constexpr auto operator-(difference_type d) const { return iterator{idx_ - d, rest_}; }

		friend constexpr auto operator-(iterator const& self, iterator const& other) -> difference_type { return self.idx_ - other.idx_; }
		friend constexpr auto operator+(difference_type n, iterator const& self) { return self + n; }

		constexpr auto operator+=(difference_type d) -> iterator& { idx_ += d; return *this; }
		constexpr auto operator-=(difference_type d) -> iterator& { idx_ -= d; return *this; }

		constexpr auto operator++() -> iterator& { ++idx_; return *this; }
		constexpr auto operator--() -> iterator& { --idx_; return *this; }

		constexpr auto operator++(int) -> iterator { iterator ret{*this}; operator++(); return ret; }  // NOLINT(cert-dcl21-cpp)
		constexpr auto operator--(int) -> iterator { iterator ret{*this}; operator--(); return ret; }  // NOLINT(cert-dcl21-cpp)

		constexpr auto operator*() const {
			// multi::detail::what(rest_);
			return ht_tuple(idx_, rest_.base());
		}

		BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> reference { return *((*this) + n); }

		friend constexpr auto operator==(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ == other.idx_; }
		friend constexpr auto operator!=(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ != other.idx_; }

		friend constexpr auto operator<(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ < other.idx_; }
		friend constexpr auto operator>(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ > other.idx_; }

		friend constexpr auto operator<=(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ <= other.idx_; }
		friend constexpr auto operator>=(iterator const& self, iterator const& other) { assert( self.rest_ == other.rest_ ); return self.idx_ >= other.idx_; }
	};

	constexpr auto begin() const { return iterator{this->base().head().first(), extensions_t<0>{this->base().tail()}}; }
	constexpr auto end()   const { return iterator{this->base().head().last() , extensions_t<0>{this->base().tail()}}; }

	class elements_t {
		multi::index_range rng_;

	 public:
		class iterator : multi::index_range::iterator {
			friend class elements_t;  // enclosing class is friend automatically?
			BOOST_MULTI_HD constexpr explicit iterator(multi::index_range::iterator it)
			: multi::index_range::iterator{it} {}

			BOOST_MULTI_HD constexpr auto base_() const -> multi::index_range::iterator const& { return *this; }
			BOOST_MULTI_HD constexpr auto base_() -> multi::index_range::iterator& { return *this; }

		 public:
			using value_type      = std::tuple<multi::index_range::iterator::value_type>;
			using difference_type = multi::index_range::iterator::difference_type;
			using reference = value_type;
			// using pointer = void;
			// using reference = value_type;

			BOOST_MULTI_HD constexpr auto operator*() const -> reference { return *base_(); }

			BOOST_MULTI_HD constexpr auto operator++() -> iterator& {
				++base_();
				return *this;
			}
			BOOST_MULTI_HD constexpr auto operator--() -> iterator& {
				--base_();
				return *this;
			}

			BOOST_MULTI_HD constexpr auto operator++(int) { iterator ret{*this}; ++(*this); return ret; }
			BOOST_MULTI_HD constexpr auto operator--(int) { iterator ret{*this}; --(*this); return ret; }

			BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> iterator& {
				base_() += n;
				return *this;
			}
			BOOST_MULTI_HD constexpr auto operator-=(difference_type n) -> iterator& {
				base_() -= n;
				return *this;
			}

			BOOST_MULTI_HD constexpr auto operator+(difference_type n) const -> iterator { return iterator{*this} += n; }
			BOOST_MULTI_HD constexpr auto operator-(difference_type n) const -> iterator { return iterator{*this} -= n; }

			friend BOOST_MULTI_HD constexpr auto operator-(iterator const& self, iterator const& other) -> difference_type {
				return self.base_() - other.base_();
			}

			BOOST_MULTI_HD constexpr auto operator==(iterator const& other) const { return base_() == other.base_(); }
			BOOST_MULTI_HD constexpr auto operator!=(iterator const& other) const { return base_() != other.base_(); }

			BOOST_MULTI_HD constexpr auto operator<(iterator const& other) const { return base_() < other.base_(); }
			BOOST_MULTI_HD constexpr auto operator<=(iterator const& other) const { return base_() <= other.base_(); }

			BOOST_MULTI_HD auto operator[](difference_type n) const { return *((*this) + n); }
		};
		// using const_iterator = iterator;

		BOOST_MULTI_HD constexpr auto begin() const -> iterator { return iterator{rng_.begin()}; }
		BOOST_MULTI_HD constexpr auto end() const -> iterator { return iterator{rng_.end()}; }

		using size_type = multi::index_extension::size_type;
		using difference_type = multi::index_extension::difference_type;
		using value_type      = iterator::value_type;
		using reference       = iterator::reference;

		BOOST_MULTI_HD constexpr auto operator[](difference_type n) const noexcept(noexcept(*(std::declval<iterator>()+n))) -> reference { return *(begin()+n); }

		BOOST_MULTI_HD constexpr auto size() const -> size_type { return end() - begin(); }

		BOOST_MULTI_HD constexpr explicit elements_t(multi::index_range rng)
		: rng_{rng} {}
	};

	auto elements() const {
		using std::get;
		// auto rng = get<0>(static_cast<tuple<multi::index_extension> const&>(*this));
		return elements_t{get<0>(static_cast<tuple<multi::index_extension> const&>(*this))};
	}

	template<class Fun>
	friend BOOST_MULTI_HD constexpr auto operator^(Fun&& fun, extensions_t const& xs) {
		return restriction<1, std::decay_t<Fun> >(xs, std::forward<Fun>(fun));
	}

	using nelems_type = index;

	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax (compatible with std::vector(int) constructor
	BOOST_MULTI_HD constexpr extensions_t(multi::size_t size)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_(multi::index_extension{0, size}) {}

	template<class T1>
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int>  // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(tuple<T1> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_{static_cast<multi::index_extension>(extensions.head())} {}

	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(multi::index_extension const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_{other} {}

	BOOST_MULTI_HD constexpr explicit extensions_t(base_ tup)
	: base_{tup} {}

	template<class OtherExtensions,
		decltype( multi::detail::implicit_cast<multi::index_extension>(OtherExtensions{}.extension()) )* = nullptr
	>
	// cppcheck-suppress noExplicitConstructor ;  // NOLINTNEXTLINE(runtime/explicit)
	BOOST_MULTI_HD constexpr extensions_t(OtherExtensions const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: base_{other.extension()} {}

	extensions_t() = default;

	BOOST_MULTI_HD constexpr auto base() const& -> base_ const& { return *this; }
	BOOST_MULTI_HD constexpr auto base() & -> base_& { return *this; }

	BOOST_MULTI_HD constexpr auto operator==(extensions_t const& other) const { return base() == other.base(); }
	BOOST_MULTI_HD constexpr auto operator!=(extensions_t const& other) const { return base() != other.base(); }

	BOOST_MULTI_HD constexpr auto size() const -> size_type { return this->base().head().size(); }

	BOOST_MULTI_HD constexpr auto num_elements() const { return size(); }

	using indices_type = multi::detail::tuple<multi::index>;

	[[nodiscard]] BOOST_MULTI_HD constexpr auto from_linear(nelems_type const& n) const -> indices_type {  // NOLINT(readability-convert-member-functions-to-static) TODO(correaa)
		return indices_type{n};
	}

	friend constexpr auto operator%(nelems_type idx, extensions_t const& extensions)
		-> multi::detail::tuple<multi::index> {
		return extensions.from_linear(idx);
	}

	static BOOST_MULTI_HD constexpr auto to_linear(index const& idx) -> difference_type { return idx; }

	BOOST_MULTI_HD constexpr auto operator[](index idx) const {
		using std::get;
		return multi::detail::tuple<multi::index>{get<0>(this->base())[idx]};
	}
	BOOST_MULTI_HD constexpr auto operator()(index idx) const { return idx; }

	template<class... Indices>
	BOOST_MULTI_HD constexpr auto next_canonical(index& idx) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		using boost::multi::detail::get;
		// if(idx == ::boost::multi::detail::get<0>(this->base()).back()) {
		// 	idx = ::boost::multi::detail::get<0>(this->base()).first();
		// 	return true;
		// }
		++idx;
		if(idx == get<0>(this->base()).last()) {
			idx = get<0>(this->base()).first();
			return true;
		}
		return false;
	}
	constexpr auto prev_canonical(index& idx) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		using boost::multi::detail::get;
		if(idx == get<0>(this->base()).first()) {
			// idx = 42;  // TODO(correaa) implement and test
			idx = get<0>(this->base()).back();
			return true;
		}
		--idx;
		return false;
	}

	friend auto intersection(extensions_t const& self, extensions_t const& other) {
		return extensions_t{
			intersection(
				boost::multi::detail::get<0>(self.base()),
				boost::multi::detail::get<0>(other.base())
			)
		};
	}
	template<class Archive>
	void serialize(Archive& arxiv, unsigned /*version*/) {
		using boost::multi::detail::get;
		auto&  extension_ = get<0>(this->base());
		arxiv& multi::archive_traits<Archive>::make_nvp("extension", extension_);
	}

	template<std::size_t Index, std::enable_if_t<(Index < 1), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto get() const -> decltype(auto) {  // -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(this->base());
	}

	template<std::size_t Index, std::enable_if_t<(Index < 1), int> = 0>      // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(extensions_t const& self) -> decltype(auto) {  // -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(self.base());
	}
};

template<dimensionality_type D> using iextensions = extensions_t<D>;

template<boost::multi::dimensionality_type D>
constexpr auto array_size_impl(boost::multi::extensions_t<D> const&)
	-> std::integral_constant<std::size_t, static_cast<std::size_t>(D)>;

extensions_t(multi::size_t) -> extensions_t<1>;
extensions_t(multi::size_t, multi::size_t) -> extensions_t<2>;
extensions_t(multi::size_t, multi::size_t, multi::size_t) -> extensions_t<3>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<4>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<5>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<6>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<7>;

}  // end namespace boost::multi

// Some versions of Clang throw warnings that stl uses class std::tuple_size instead
// of struct std::tuple_size like it should be
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

template<boost::multi::dimensionality_type D>
struct std::tuple_size<boost::multi::extensions_t<D>>  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) to implement structured binding
: std::integral_constant<std::size_t, static_cast<std::size_t>(D)> {};

template<>
struct std::tuple_element<0, boost::multi::extensions_t<0>> {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) to implement structured binding
	using type = void;
};

template<std::size_t Index, boost::multi::dimensionality_type D>
struct std::tuple_element<Index, boost::multi::extensions_t<D>> {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) to implement structured binding
	using type = typename std::tuple_element_t<Index, typename boost::multi::extensions_t<D>::base_>;
};

namespace std {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification)

// clang wants tuple_size to be a class, not a struct with -Wmismatched-tags
#if !defined(__GLIBCXX__) || (__GLIBCXX__ <= 20190406)
template<> struct tuple_size<boost::multi::extensions_t<0>> : std::integral_constant<boost::multi::dimensionality_type, 0> {};
template<> struct tuple_size<boost::multi::extensions_t<1>> : std::integral_constant<boost::multi::dimensionality_type, 1> {};
template<> struct tuple_size<boost::multi::extensions_t<2>> : std::integral_constant<boost::multi::dimensionality_type, 2> {};
template<> struct tuple_size<boost::multi::extensions_t<3>> : std::integral_constant<boost::multi::dimensionality_type, 3> {};
template<> struct tuple_size<boost::multi::extensions_t<4>> : std::integral_constant<boost::multi::dimensionality_type, 4> {};
template<> struct tuple_size<boost::multi::extensions_t<5>> : std::integral_constant<boost::multi::dimensionality_type, 5> {};
#else
template<> class tuple_size<boost::multi::extensions_t<0>> : public std::integral_constant<boost::multi::dimensionality_type, 0> {};
template<> class tuple_size<boost::multi::extensions_t<1>> : public std::integral_constant<boost::multi::dimensionality_type, 1> {};
template<> class tuple_size<boost::multi::extensions_t<2>> : public std::integral_constant<boost::multi::dimensionality_type, 2> {};
template<> class tuple_size<boost::multi::extensions_t<3>> : public std::integral_constant<boost::multi::dimensionality_type, 3> {};
template<> class tuple_size<boost::multi::extensions_t<4>> : public std::integral_constant<boost::multi::dimensionality_type, 4> {};
template<> class tuple_size<boost::multi::extensions_t<5>> : public std::integral_constant<boost::multi::dimensionality_type, 5> {};
#endif

#if !defined(_MSC_VER) && (!defined(__GLIBCXX__) || (__GLIBCXX__ <= 20240707))
template<std::size_t N, ::boost::multi::dimensionality_type D>
constexpr auto get(::boost::multi::extensions_t<D> const& tp)  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get, gcc workaround
	-> decltype(tp.template get<N>()) {
	return tp.template get<N>();
}

// template<std::size_t N>  // , boost::multi::dimensionality_type D>
// constexpr auto get(boost::multi::extensions_t<2> const& tp)  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get, gcc workaround
// // ->decltype(tp.template get<N>()) {
// -> decltype(auto) {
//  return tp.template get<N>(); }

template<std::size_t N, ::boost::multi::dimensionality_type D>
constexpr auto get(::boost::multi::extensions_t<D>& tp)  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get, gcc workaround
	-> decltype(tp.template get<N>()) {
	return tp.template get<N>();
}

template<std::size_t N, boost::multi::dimensionality_type D>
constexpr auto get(::boost::multi::extensions_t<D>&& tp)  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get, gcc workaround
	-> decltype(std::move(tp).template get<N>()) {
	return std::move(tp).template get<N>();
}
#endif

template<typename Fn, boost::multi::dimensionality_type D>
constexpr auto
apply(Fn&& fn, boost::multi::extensions_t<D> const& xs) noexcept -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) workaround
	return xs.apply(std::forward<Fn>(fn));
}

}  // end namespace std

namespace boost::multi {

struct monostate : equality_comparable<monostate> {
	friend BOOST_MULTI_HD constexpr auto operator==(monostate const& /*self*/, monostate const& /*other*/) { return true; }
};

template<typename SSize = multi::index>
class stride_t {
	difference_type stride_;

 public:
	BOOST_MULTI_HD constexpr auto operator()() const -> difference_type { return stride_; }

	template<class Ptr>
	BOOST_MULTI_HD constexpr auto operator()(Ptr ptr) const -> Ptr { return ptr + stride_; }

	using category = std::random_access_iterator_tag;
};

template<typename SSize = multi::index>
class contiguous_stride_t {
 public:
	// using difference_type = SSize;

	BOOST_MULTI_HD constexpr auto operator()() const -> SSize { return 1; }

	template<class Ptr>
	BOOST_MULTI_HD constexpr auto operator()(Ptr const& ptr) const -> Ptr { return ptr + 1; }

#if (__cplusplus >= 202002L)
	using category = std::random_access_iterator_tag;  // std::contiguous_iterator_tag;
#else
	using category = std::random_access_iterator_tag;
#endif
};

using multi::detail::tuple;

template<typename SSize = multi::index>
class contiguous_layout {

 public:
	using dimensionality_type                           = multi::dimensionality_t;
	using rank                                          = std::integral_constant<dimensionality_type, 1>;
	static constexpr auto                rank_v         = rank::value;
	static constexpr dimensionality_type dimensionality = rank_v;

	using size_type  = SSize;
	using sizes_type = typename boost::multi::detail::tuple<size_type>;

	using difference_type = SSize;

	using index           = size_type;
	using index_range     = multi::range<index>;
	using index_extension = multi::extension_t<index>;

	using indexes = tuple<index>;

	using extension_type  = multi::extension_t<index>;
	using extensions_type = multi::extensions_t<1>;

	using stride_type  = std::integral_constant<int, 1>;
	using strides_type = boost::multi::detail::tuple<stride_type>;

	using offset_type = std::integral_constant<int, 0>;

	using nelems_type = SSize;

	using sub_type = layout_t<0, SSize>;

 private:
	// BOOST_MULTI_NO_UNIQUE_ADDRESS sub_type sub_;
	// BOOST_MULTI_NO_UNIQUE_ADDRESS stride_type stride_;
	size_type nelems_;

	template<std::size_t N, class Tup>
	static constexpr auto get_(Tup&& tup) {
		using std::get;
		return get<N>(std::forward<Tup>(tup));
	}

 public:
	constexpr explicit contiguous_layout(multi::extensions_t<1> xs)
	: nelems_{get_<0>(xs).size()} {}

	BOOST_MULTI_HD constexpr contiguous_layout(
		sub_type /*sub*/,
		stride_type /*stride*/,
		offset_type /*offset*/,
		nelems_type nelems
	)
	: /*sub_{sub}, stride_{} offset_{},*/ nelems_{nelems} {}

 private:
	constexpr auto at_aux_(index /*idx*/) const {
		return sub_type{};  // sub_.sub_, sub_.stride_, sub_.offset_ + offset_ + (idx*stride_), sub_.nelems_}();
	}

 public:
	BOOST_MULTI_HD constexpr auto operator[](index idx) const { return at_aux_(idx); }

	template<typename... Indices>
	BOOST_MULTI_HD constexpr auto operator()(index idx, Indices... rest) const { return operator[](idx)(rest...); }
	BOOST_MULTI_HD constexpr auto operator()(index idx) const { return at_aux_(idx); }
	BOOST_MULTI_HD constexpr auto operator()() const { return *this; }

	BOOST_MULTI_HD constexpr auto stride() const { return std::integral_constant<int, 1>{}; }
	BOOST_MULTI_HD constexpr auto offset() const { return std::integral_constant<int, 0>{}; }
	BOOST_MULTI_HD constexpr auto extension() const { return extension_type{0, nelems_}; }

	BOOST_MULTI_HD constexpr auto num_elements() const { return nelems_; }

	BOOST_MULTI_HD constexpr auto size() const { return nelems_; }
	BOOST_MULTI_HD constexpr auto sizes() const { return sizes_type{size()}; }

	BOOST_MULTI_HD constexpr auto nelems() const { return nelems_; }

	BOOST_MULTI_HD constexpr auto extensions() const { return multi::extensions_t<1>{extension()}; }

	BOOST_MULTI_HD constexpr auto is_empty() const -> bool { return nelems_ == 0; }

	BOOST_MULTI_NODISCARD("empty checks for emptyness, it performs no action. Use `is_empty()` instead")
	BOOST_MULTI_HD constexpr auto empty() const { return is_empty(); }

	constexpr auto sub() const { return layout_t<0, SSize>{}; }

	constexpr auto is_compact() const { return std::true_type{}; }

	BOOST_MULTI_HD constexpr auto drop(difference_type count) const {
		assert(count <= this->size());

		return contiguous_layout{
			this->sub(),
			this->stride(),
			this->offset(),
			this->stride() * (this->size() - count)
		};
	}

	BOOST_MULTI_HD constexpr auto slice(index first, index last) const {
		return contiguous_layout{
			this->sub(),
			this->stride(),
			this->offset(),
			(this->is_empty()) ? 0 : this->nelems() / this->size() * (last - first)
		};
	}
};

template<typename Stride1, typename Stride2, typename Size1, typename Pointer = void*>
class bistride {
	using stride1_type = Stride1;
	using size1_type = Size1;
	using stride2_type = Stride2;
	using offset_type     = std::ptrdiff_t;

	stride1_type stride1_;
	stride2_type stride2_;
	size_type    nelems2_;
	Pointer ptr_;
	std::ptrdiff_t n_;

	public:
	using category = std::random_access_iterator_tag;

	BOOST_MULTI_HD constexpr explicit bistride(stride1_type stride1, stride2_type stride2, size_type size, Pointer ptr)  // NOLINT(bugprone-easily-swappable-parameters)
	: stride1_{stride1}, stride2_{stride2}, nelems2_{size}, ptr_{ptr}, n_{1} {}

	BOOST_MULTI_HD constexpr explicit bistride(stride1_type stride1, stride2_type stride2, size_type size, Pointer ptr, std::ptrdiff_t n)  // NOLINT(bugprone-easily-swappable-parameters)
	: stride1_{stride1}, stride2_{stride2}, nelems2_{size}, ptr_{ptr}, n_{n} {}

	BOOST_MULTI_HD constexpr auto operator*(std::ptrdiff_t nn) const { return bistride{stride1_, stride2_, nelems2_, ptr_, nn*n_}; }

	#if (defined(__clang__) && (__clang_major__ >= 16)) && !defined(__INTEL_LLVM_COMPILER)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
	#endif
	template<class Ptr>
	friend BOOST_MULTI_HD constexpr auto operator+=(Ptr& ptr, bistride const& self) -> Ptr& {
		ptr = ptr + self;
		return ptr;
	}

	template<class Ptr>
	friend BOOST_MULTI_HD constexpr auto operator+=(Ptr& ptr, bistride& self) -> Ptr& {
		if(self.n_ == 1) {
			ptr += self.stride2_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			if(ptr == static_cast<Ptr>(self.ptr_) + self.nelems2_) {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				self.ptr_ = static_cast<Ptr>(self.ptr_) + self.stride1_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				ptr = static_cast<Ptr>(self.ptr_);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
			}
		} else {
			ptr = ptr + self;
		}
		return ptr;
	}

	#if (defined(__clang__) && (__clang_major__ >= 16)) && !defined(__INTEL_LLVM_COMPILER)
	#pragma clang diagnostic pop
	#endif

	BOOST_MULTI_HD constexpr auto operator-(offset_type /*unused*/) const { return *this; }
	template<class Ptr>
	friend BOOST_MULTI_HD constexpr auto operator+(Ptr const& ptr, bistride const& self) {
		auto base = static_cast<Ptr>(self.ptr_);
		auto dist = ptr - base;
		auto i = dist / self.stride1_;
		auto j = (dist % self.stride1_) / self.stride2_;

		auto shift = j + self.n_;
		auto size2 = self.nelems2_/self.stride2_;

		auto j0 = shift % size2;
		auto i0 = (shift / size2) + i;

		auto ret = base + (i0 * self.stride1_) + (j0 * self.stride2_);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return ret;
	}

	template<class Ptr>
	BOOST_MULTI_HD constexpr auto segment_base(Ptr const& ptr) const {
		auto base = static_cast<Ptr>(ptr_);
		auto dist = ptr - base;
		auto i = dist / stride1_;
		auto ret = base + (i * stride1_);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return ret;
	}
};

template<dimensionality_type D>
struct bilayout {
	using size_type       = multi::size_t;  // SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index           = difference_type;

	using stride1_type = difference_type;
	using stride2_type = difference_type;
	// using bistride_type = std::pair<index, index>;
	using sub_type = layout_t<D - 1>;

	using dimensionality_type    = typename sub_type::dimensionality_type;
	using rank                   = std::integral_constant<dimensionality_type, sub_type::rank::value + 1>;
	constexpr static auto rank_v = rank::value;

	constexpr static auto dimensionality() { return rank_v; }

 private:
	stride1_type stride1_;
	size_type    nelems1_;
	stride2_type stride2_;
	size_type    nelems2_;
	sub_type     sub_;
	void* ptr_;

 public:
	bilayout(
		stride1_type stride1,  // NOLINT(bugprone-easily-swappable-parameters)
		size_type    nelems1,
		stride2_type stride2,  // NOLINT(bugprone-easily-swappable-parameters)
		size_type    nelems2,
		sub_type     sub,
		void* ptr
	)
	: stride1_{stride1}, nelems1_{nelems1}, stride2_{stride2}, nelems2_{nelems2}, sub_{std::move(sub)}, ptr_{ptr} {}

	using offset_type     = std::ptrdiff_t;
	// using stride_type     = void;  // std::pair<stride1_type, stride2_type>;

	using stride_type = bistride<stride1_type, stride2_type, size_type>;

	using index_range     = multi::range<index>;
	using extension_type  = void;
	using extensions_type = void;
	using sizes_type      = void;
	using indexes         = void;

	using strides_type = void;

	// auto stride() const = delete;
	BOOST_MULTI_HD constexpr auto stride() const {
		return stride_type{stride1_, stride2_, nelems2_, ptr_, 1};
	}
	auto num_elements() const = delete;

	BOOST_MULTI_HD constexpr auto offset() const { return offset_type{}; }
	BOOST_MULTI_HD constexpr auto size() const { return (nelems2_ / stride2_) * (nelems1_ / stride1_); }

	auto nelems() const     = delete;
	void extension() const  = delete;
	auto extensions() const = delete;
	auto is_empty() const   = delete;
	auto empty() const      = delete;
	BOOST_MULTI_HD constexpr auto sub() const { return sub_; }
	auto sizes() const      = delete;

	auto is_compact() const = delete;

	using index_extension = multi::index_extension;
};

template<class Ptr>
class segmented_ptr {
	Ptr ptr_;
	Ptr first_;
	Ptr last_;
	std::ptrdiff_t stride_;

 public:
	segmented_ptr(Ptr ptr, Ptr first, Ptr last, std::ptrdiff_t stride) : ptr_{ptr}, first_{first}, last_{last}, stride_{stride} {}
	auto operator++() -> segmented_ptr& {
		++ptr_;
		if(ptr_ == last_) {
			first_ += stride_;
			last_ += stride_;
			ptr_ = first_;
		}
		return *this;
	}

	auto operator--() -> segmented_ptr& {
		if(ptr_ == first_) {
			first_ -= stride_;
			last_ -= stride_;
			ptr_ = last_;
		}
		--ptr_;
		return *this;
	}
};

template<dimensionality_type D, typename SSize>
struct layout_t
	: multi::equality_comparable<layout_t<D, SSize>> {
	template<class Ptr = void*>
	auto flatten(Ptr ptr) const {
		return bilayout<D - 1>{
			stride(),
			nelems(),
			sub().stride(),
			sub().nelems(),
			sub().sub(),
			ptr
		};
	}

	using dimensionality_type = multi::dimensionality_type;
	using rank                = std::integral_constant<dimensionality_type, D>;

	using sub_type        = layout_t<D - 1>;
	using size_type       = SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index           = difference_type;

	using index_extension = multi::index_extension;
	using index_range     = multi::range<index>;

	using stride_type = index;
	using offset_type = index;
	using nelems_type = index;

	using strides_type = typename boost::multi::detail::tuple_prepend<stride_type, typename sub_type::strides_type>::type;
	using offsets_type = typename boost::multi::detail::tuple_prepend<offset_type, typename sub_type::offsets_type>::type;
	using nelemss_type = typename boost::multi::detail::tuple_prepend<nelems_type, typename sub_type::nelemss_type>::type;

	using extension_type = index_extension;  // not index_range!

	using extensions_type = extensions_t<rank::value>;
	using sizes_type      = typename boost::multi::detail::tuple_prepend<size_type, typename sub_type::sizes_type>::type;

	using indexes = typename boost::multi::detail::tuple_prepend<index, typename sub_type::indexes>::type;

	static constexpr dimensionality_type rank_v         = rank::value;
	static constexpr dimensionality_type dimensionality = rank_v;  // TODO(correaa): consider deprecation

	[[deprecated("for compatibility with Boost.MultiArray, use static `dimensionality` instead")]]
	static constexpr auto num_dimensions() { return dimensionality; }  // NOSONAR(cpp:S1133)

	friend constexpr auto dimensionality(layout_t const& /*self*/) { return rank_v; }

 private:
	sub_type    sub_;
	stride_type stride_;  // =  1;  // or std::numeric_limits<stride_type>::max()?
	offset_type offset_;
	nelems_type nelems_;

	template<dimensionality_type, typename> friend struct layout_t;

 public:
	layout_t() = default;

	template<
		class OtherLayout,
		class = decltype(sub_type{std::declval<OtherLayout const&>().sub()}),
		class = decltype(stride_type{std::declval<OtherLayout const&>().stride()}),
		class = decltype(offset_type{std::declval<OtherLayout const&>().offset()}),
		class = decltype(nelems_type{std::declval<OtherLayout const&>().nelems()})>
	BOOST_MULTI_HD constexpr explicit layout_t(OtherLayout const& other)
	: sub_{other.sub()}, stride_{other.stride()}, offset_{other.offset()}, nelems_{other.nelems()} {}

 private:
	template<class Fun, class Tup>
	static BOOST_MULTI_HD constexpr auto apply_(Fun&& fun, Tup&& tup) -> decltype(auto) {  // this is workaround for icc 2021
		using std::apply;
		return apply(std::forward<Fun>(fun), std::forward<Tup>(tup));
	}

 public:
	#ifdef __NVCC__
	#pragma nv_diagnostic push
	#pragma nv_diag_suppress = 20013  // TODO(correa) use multi::apply  // calling a constexpr __host__ function("apply") from a __host__ __device__ function("layout_t") is not allowed.
	#endif
 private:
	template<class... Args>
	static BOOST_MULTI_HD constexpr auto std_apply_(Args&&... args) ->decltype(auto) { using std::apply; return apply(std::forward<Args>(args)...); }

 public:
	BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& extensions)
	: sub_{apply_        ([](auto const&... subexts) { return multi::extensions_t<D - 1>{subexts...}; }, detail::tail(extensions.base()))}
	// : sub_{/*std::*/apply([](auto const&... subexts) { return multi::extensions_t<D - 1>{subexts...}; }, detail::tail(extensions.base()))}
	, stride_{sub_.num_elements() ? sub_.num_elements() : 1}
	, offset_{boost::multi::detail::get<0>(extensions.base()).first() * stride_}
	, nelems_{boost::multi::detail::get<0>(extensions.base()).size() * sub().num_elements()} {}

	BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& extensions, strides_type const& strides)
	: sub_{std::apply([](auto const&... subexts) { return multi::extensions_t<D - 1>{subexts...}; }, detail::tail(extensions.base())), detail::tail(strides)}, stride_{boost::multi::detail::get<0>(strides)}, offset_{boost::multi::detail::get<0>(extensions.base()).first() * stride_}, nelems_{boost::multi::detail::get<0>(extensions.base()).size() * sub().num_elements()} {}
	#ifdef __NVCC__
	#pragma nv_diagnostic pop
	#endif

	BOOST_MULTI_HD constexpr explicit layout_t(sub_type const& sub, stride_type stride, offset_type offset, nelems_type nelems)  // NOLINT(bugprone-easily-swappable-parameters)
	: sub_{sub}, stride_{stride}, offset_{offset}, nelems_{nelems} {}

	BOOST_MULTI_HD constexpr explicit layout_t(sub_type const& sub, stride_type stride, offset_type offset /*, nelems_type nelems*/)  // NOLINT(bugprone-easily-swappable-parameters)
	: sub_{sub}, stride_{stride}, offset_{offset} /*, nelems_{nelems}*/ {}                                                            // this leaves nelems_ uninitialized

	constexpr auto origin() const { return sub_.origin() - offset_; }

 private:
	#ifdef __clang__
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wlarge-by-value-copy"
	#endif

	BOOST_MULTI_HD constexpr auto at_aux_(index idx) const {
		return sub_type{sub_.sub_, sub_.stride_, sub_.offset_ + offset_ + (idx * stride_), sub_.nelems_}();
	}

 public:
	BOOST_MULTI_HD constexpr auto operator[](index idx) const { return at_aux_(idx); }

	template<typename... Indices>
	BOOST_MULTI_HD constexpr auto operator()(index idx, Indices... rest) const { return operator[](idx)(rest...); }
	BOOST_MULTI_HD constexpr auto operator()(index idx) const { return at_aux_(idx); }

	#ifdef __clang__
	#pragma clang diagnostic pop
	#endif

	#ifdef __clang__
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunknown-warning-option"
	#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) can it be returned by reference?
	#endif

	BOOST_MULTI_HD constexpr auto operator()() const { return *this; }

	#ifdef __clang__
	#pragma clang diagnostic pop
	#endif

	BOOST_MULTI_HD constexpr auto        sub() & -> sub_type& { return sub_; }
	BOOST_MULTI_HD constexpr auto        sub() const& -> sub_type const& { return sub_; }
	friend BOOST_MULTI_HD constexpr auto sub(layout_t const& self) -> sub_type const& { return self.sub(); }

	BOOST_MULTI_HD constexpr auto        nelems() & -> nelems_type& { return nelems_; }
	BOOST_MULTI_HD constexpr auto        nelems() const& -> nelems_type const& { return nelems_; }
	friend BOOST_MULTI_HD constexpr auto nelems(layout_t const& self) -> nelems_type const& { return self.nelems(); }

	constexpr BOOST_MULTI_HD auto nelems(dimensionality_type dim) const { return (dim != 0) ? sub_.nelems(dim - 1) : nelems_; }

	friend BOOST_MULTI_HD constexpr auto operator==(layout_t const& self, layout_t const& other) -> bool {
		return self.sub_ == other.sub_ && self.stride_ == other.stride_ && self.offset_ == other.offset_ && self.nelems_ == other.nelems_;
		// return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) == std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	friend BOOST_MULTI_HD constexpr auto operator!=(layout_t const& self, layout_t const& other) -> bool {
		return !(self == other);
		// return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) != std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	constexpr BOOST_MULTI_HD auto operator<(layout_t const& other) const -> bool {
		return std::tie(sub_, stride_, offset_, nelems_) < std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	constexpr auto reindex() const { return *this; }
	constexpr auto reindex(index idx) const {
		return layout_t{
			sub(),
			stride(),
			idx * stride(),
			nelems()
		};
	}
	template<class... Indexes>
	constexpr auto reindexed(index first, Indexes... idxs) const {
		return ((reindexed(first).rotate()).reindexed(idxs...)).unrotate();
	}

	BOOST_MULTI_HD constexpr auto        num_elements() const noexcept -> size_type { return size() * sub_.num_elements(); }  // TODO(correaa) investigate mutation * -> /
	friend BOOST_MULTI_HD constexpr auto num_elements(layout_t const& self) noexcept -> size_type { return self.num_elements(); }

	BOOST_MULTI_HD constexpr auto        is_empty() const noexcept { return nelems_ == 0; }  // mull-ignore: cxx_eq_to_ne
	friend BOOST_MULTI_HD constexpr auto is_empty(layout_t const& self) noexcept { return self.is_empty(); }

	BOOST_MULTI_HD constexpr auto empty() const noexcept { return is_empty(); }

	friend BOOST_MULTI_HD constexpr auto         size(layout_t const& self) noexcept -> size_type { return self.size(); }
	BOOST_MULTI_HD constexpr  auto size() const noexcept -> size_type {
		if(nelems_ == 0) {
			return 0;
		}
		// BOOST_MULTI_ACCESS_ASSERT(stride_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		// if(nelems_ != 0) {MULTI_ACCESS_ASSERT(stride_ != 0);}
		// return nelems_ == 0?0:nelems_/stride_;
		// assert(stride_ != 0);
		return nelems_ / stride_;
	}

	BOOST_MULTI_HD constexpr auto stride() -> stride_type& { return stride_; }
	BOOST_MULTI_HD constexpr auto stride() const -> stride_type const& { return stride_; }

	friend BOOST_MULTI_HD constexpr auto stride(layout_t const& self) -> index { return self.stride(); }

	BOOST_MULTI_HD constexpr auto        strides() const -> strides_type { return strides_type{stride(), sub_.strides()}; }
	friend BOOST_MULTI_HD constexpr auto strides(layout_t const& self) -> strides_type { return self.strides(); }

	constexpr BOOST_MULTI_HD auto        offset(dimensionality_type dim) const -> index { return (dim != 0) ? sub_.offset(dim - 1) : offset_; }
	BOOST_MULTI_HD constexpr auto        offset() const -> index { return offset_; }
	friend BOOST_MULTI_HD constexpr auto offset(layout_t const& self) -> index { return self.offset(); }
	constexpr BOOST_MULTI_HD auto        offsets() const { return boost::multi::detail::tuple{offset(), sub_.offsets()}; }
	constexpr BOOST_MULTI_HD auto        nelemss() const { return boost::multi::detail::tuple{nelems(), sub_.nelemss()}; }

	constexpr auto base_size() const {
		using std::max;
		return max(nelems_, sub_.base_size());
	}

	constexpr auto        is_compact() const& { return base_size() == num_elements(); }
	friend constexpr auto is_compact(layout_t const& self) { return self.is_compact(); }

	constexpr auto        shape() const& -> decltype(auto) { return sizes(); }
	friend constexpr auto shape(layout_t const& self) -> decltype(auto) { return self.shape(); }

	BOOST_MULTI_HD constexpr auto sizes() const noexcept { return multi::detail::ht_tuple(size(), sub_.sizes()); }

	friend BOOST_MULTI_HD constexpr auto        extension(layout_t const& self) { return self.extension(); }
	[[nodiscard]] BOOST_MULTI_HD constexpr auto extension() const -> extension_type {
		if(nelems_ == 0) {
			return index_extension{};
		}
		// assert(stride_ != 0);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		assert(offset_ % stride_ == 0);
		assert(nelems_ % stride_ == 0);
		return index_extension{offset_ / stride_, (offset_ + nelems_) / stride_};
	}

	BOOST_MULTI_HD constexpr auto        extensions() const {
		// auto fa = extension();
		// auto sa = sub_.extensions().base();
		// auto ht_tuple = multi::detail::ht_tuple(fa, sa);
		// auto ret = extensions_type{ht_tuple};
		// return ret;
		return extensions_type{multi::detail::ht_tuple(extension(), sub_.extensions().base())};
	}

	friend BOOST_MULTI_HD constexpr auto extensions(layout_t const& self) -> extensions_type { return self.extensions(); }

	[[deprecated("use get<d>(m.extensions()")]]  // TODO(correaa) redeprecate, this is commented to give a smaller CI output
	constexpr auto
	extension(dimensionality_type dim) const {
		return std::apply([](auto... extensions) { return std::array<index_extension, static_cast<std::size_t>(D)>{extensions...}; }, extensions().base()).at(static_cast<std::size_t>(dim));
	}  // cppcheck-suppress syntaxError ; bug in cppcheck 2.14
	   //  [[deprecated("use get<d>(m.strides())  ")]]  // TODO(correaa) redeprecate, this is commented to give a smaller CI output
	constexpr auto stride(dimensionality_type dim) const {
		return std::apply([](auto... strides) { return std::array<stride_type, static_cast<std::size_t>(D)>{strides...}; }, strides()).at(static_cast<std::size_t>(dim));
	}
	//  [[deprecated("use get<d>(m.sizes())    ")]]  // TODO(correaa) redeprecate, this is commented to give a smaller CI output
	//  constexpr auto size     (dimensionality_type dim) const {return std::apply([](auto... sizes     ) {return std::array<size_type      , static_cast<std::size_t>(D)>{sizes     ...};}, sizes     ()       ).at(static_cast<std::size_t>(dim));}

	BOOST_MULTI_HD constexpr auto drop(difference_type count) const {
		assert(count <= this->size());

		return layout_t{
			this->sub(),
			this->stride(),
			this->offset(),
			this->stride() * (this->size() - count)
		};
	}

	BOOST_MULTI_HD constexpr auto slice(index first, index last) const {
		return layout_t{
			this->sub(),
			this->stride(),
			this->offset(),
			(this->is_empty()) ? 0 : this->nelems() / this->size() * (last - first)
		};
	}

	// template<typename Size>
	// constexpr auto partition(Size const& count) -> layout_t& {
	// 	stride_ *= count;
	// 	nelems_ *= count;
	// 	sub_.partition(count);
	// 	return *this;
	// }

	constexpr auto partition(size_type n) const {
		assert(n != 0);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		// vvv TODO(correaa) should be size() here?
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) normal in a constexpr function
		assert((this->nelems() % n) == 0);  // if you get an assertion here it means that you are partitioning an array with an incommunsurate partition
		return multi::layout_t<D + 1>{
			multi::layout_t<D>{
				this->sub(),
				this->stride(),
				this->offset(),
				this->nelems() / n  // mull-ignore: cxx_div_to_mul
			},
			this->nelems() / n,  // mull-ignore: cxx_div_to_mul
			0,
			this->nelems()
		};
		// new_layout.sub().nelems() /= n;
	}

	template<class TT>
	constexpr static void ce_swap(TT& t1, TT& t2) {
		TT tmp = std::move(t1);
		t1     = std::move(t2);
		t2     = tmp;
	}

	BOOST_MULTI_HD constexpr auto transpose() const {
		return layout_t(
			sub_type(
				sub().sub(),
				stride(),
				offset(),
				nelems()
			),
			sub().stride(),
			sub().offset(),
			sub().nelems()
		);
	}

	constexpr auto reverse() const {
		auto ret = unrotate();
		return layout_t(
			ret.sub().reverse(),
			ret.stride(),
			ret.offset(),
			ret.nelems()
		);
	}

	BOOST_MULTI_HD constexpr auto rotate() const {
		if constexpr(D > 1) {
			auto const ret = transpose();
			return layout_t(
				ret.sub().rotate(),
				ret.stride(),
				ret.offset(),
				ret.nelems()
			);
		} else {
			return *this;
		}
	}

	BOOST_MULTI_HD constexpr auto unrotate() const {
		if constexpr(D > 1) {
			auto const ret = layout_t(
				sub().unrotate(),
				stride(),
				offset(),
				nelems()
			);
			return ret.transpose();
		} else {
			return *this;
		}
	}

	constexpr auto hull_size() const -> size_type {
		if(is_empty()) {
			return 0;
		}
		return std::abs(size() * stride()) > std::abs(sub_.hull_size()) ? size() * stride() : sub_.hull_size();
	}

	[[deprecated("use two arg version")]] constexpr auto scale(size_type factor) const {
		return layout_t{sub_.scale(factor), stride_ * factor, offset_ * factor, nelems_ * factor};
	}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) use checked span
#endif

	BOOST_MULTI_HD constexpr auto take(size_type n) const {
		return layout_t(
			this->sub(),
			this->stride(),
			this->offset(),
			this->stride() * n
		);
	}

	BOOST_MULTI_HD constexpr auto halve() const {
		assert(this->size() % 2 == 0);
		return layout_t<D + 1>(
			this->take(this->size() / 2),
			this->nelems() / 2,
			0,
			this->nelems()
		);
	}

	constexpr auto scale(size_type num, size_type den) const {
		assert((stride_ * num) % den == 0);
		assert(offset_ == 0);  // TODO(correaa) implement ----------------vvv
		return layout_t{sub_.scale(num, den), stride_ * num / den, offset_ /* *num/den */, nelems_ * num / den};
	}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
};

template<typename SSize>
struct layout_t<0, SSize>
: multi::equality_comparable<layout_t<0, SSize>> {
	using dimensionality_type = multi::dimensionality_type;
	using rank                = std::integral_constant<dimensionality_type, 0>;

	using size_type       = SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index           = difference_type;
	using index_extension = multi::index_extension;
	using index_range     = multi::range<index>;

	using sub_type    = monostate;
	using stride_type = monostate;
	using offset_type = index;
	using nelems_type = index;

	using strides_type = tuple<>;
	using offsets_type = tuple<>;
	using nelemss_type = tuple<>;

	using extension_type = void;

	using extensions_type = extensions_t<rank::value>;
	using sizes_type      = tuple<>;
	using indexes         = tuple<>;

	static constexpr dimensionality_type rank_v         = rank::value;
	static constexpr dimensionality_type dimensionality = rank_v;  // TODO(correaa) : consider deprecation

	friend constexpr auto dimensionality(layout_t const& /*self*/) { return rank_v; }

 private:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820)  // '6' bytes padding added after data member
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif
	BOOST_MULTI_NO_UNIQUE_ADDRESS sub_type    sub_;
	BOOST_MULTI_NO_UNIQUE_ADDRESS stride_type stride_;  // TODO(correaa) padding struct 'boost::multi::layout_t<0>' with 1 byte to align 'stride_' [-Werror,-Wpadded]

	offset_type offset_;

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	nelems_type nelems_;

	template<dimensionality_type, typename> friend struct layout_t;

 public:
	layout_t() = default;

	BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& /*nil*/)
	: offset_{0}, nelems_{1} {}

	// BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& /*nil*/, strides_type const& /*nil*/) {}

	BOOST_MULTI_HD constexpr layout_t(sub_type sub, stride_type stride, offset_type offset, nelems_type nelems)  // NOLINT(bugprone-easily-swappable-parameters)
	: sub_{sub}, stride_{stride}, offset_{offset}, nelems_{nelems} {}

	[[nodiscard]] BOOST_MULTI_HD constexpr auto extensions() const { return extensions_type{}; }
	friend BOOST_MULTI_HD constexpr auto        extensions(layout_t const& self) { return self.extensions(); }

	[[nodiscard]] BOOST_MULTI_HD constexpr auto num_elements() const { return nelems_; }
	friend BOOST_MULTI_HD constexpr auto        num_elements(layout_t const& self) { return self.num_elements(); }

	[[nodiscard]] BOOST_MULTI_HD constexpr auto sizes() const { return tuple<>{}; }
	friend BOOST_MULTI_HD constexpr auto        sizes(layout_t const& self) { return self.sizes(); }

	[[nodiscard]] BOOST_MULTI_HD constexpr auto strides() const { return strides_type{}; }
	[[nodiscard]] BOOST_MULTI_HD constexpr auto offsets() const { return offsets_type{}; }
	[[nodiscard]] BOOST_MULTI_HD constexpr auto nelemss() const { return nelemss_type{}; }

	BOOST_MULTI_HD constexpr auto operator()() const { return offset_; }
	// constexpr explicit operator offset_type() const {return offset_;}

	constexpr auto stride() const -> stride_type = delete;
	constexpr auto offset() const -> offset_type { return offset_; }
	constexpr auto nelems() const -> nelems_type { return nelems_; }
	constexpr auto sub() const -> sub_type = delete;

	constexpr auto size() const -> size_type           = delete;
	constexpr auto extension() const -> extension_type = delete;

	BOOST_MULTI_HD constexpr auto is_empty() const noexcept { return nelems_ == 0; }

	BOOST_MULTI_NODISCARD("empty checks for emptyness, it performs no action. Use `is_empty()` instead")
	constexpr auto empty() const noexcept { return nelems_ == 0; }

	friend constexpr auto empty(layout_t const& self) noexcept { return self.empty(); }

	[[deprecated("is going to be removed")]]
	constexpr auto is_compact() const -> bool = delete;

	constexpr auto base_size() const -> size_type { return 0; }
	constexpr auto origin() const -> offset_type { return 0; }

	constexpr auto reverse() const { return *this; }
	// constexpr auto reverse()          -> layout_t& {return *this;}

	BOOST_MULTI_HD constexpr auto take(size_type /*n*/) const {
		return layout_t<0, SSize>{};
	}

	BOOST_MULTI_HD constexpr auto halve() const {
		return layout_t<1, SSize>(*this, 0, 0, 0);
	}

	// [[deprecated("use two arg version")]] constexpr auto scale(size_type /*size*/) const {return *this;}
	constexpr auto scale(size_type /*num*/, size_type /*den*/) const { return *this; }

	//  friend constexpr auto operator!=(layout_t const& self, layout_t const& other) {return not(self == other);}
	friend BOOST_MULTI_HD constexpr auto operator==(layout_t const& self, layout_t const& other) {
		return 
			self.sub_ == other.sub_ &&
			self.stride_ == other.stride_ &&
			self.nelems_ == other.nelems_
		;
		// return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) == std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	friend BOOST_MULTI_HD constexpr auto operator!=(layout_t const& self, layout_t const& other) {
		return !(self==other);
		// return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) != std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	constexpr auto operator<(layout_t const& other) const -> bool {
		return std::tie(offset_, nelems_) < std::tie(other.offset_, other.nelems_);
	}

	BOOST_MULTI_HD constexpr auto rotate() const { return *this; }
	BOOST_MULTI_HD constexpr auto unrotate() const { return *this; }

	constexpr auto hull_size() const -> size_type { return num_elements(); }  // not in bytes
};

BOOST_MULTI_HD constexpr auto
operator*(layout_t<0>::index_extension const& extensions_0d, layout_t<0>::extensions_type const& /*zero*/)
	-> layout_t<1>::extensions_type {
	return layout_t<1>::extensions_type{tuple<layout_t<0>::index_extension>{extensions_0d}};
}

BOOST_MULTI_HD constexpr auto operator*(extensions_t<1> const& extensions_1d, extensions_t<1> const& self) {
	using boost::multi::detail::get;
	return extensions_t<2>({get<0>(extensions_1d.base()), get<0>(self.base())});
}

}  // end namespace boost::multi

namespace boost::multi::detail {

template<class Tuple>
struct convertible_tuple : Tuple {
	using Tuple::Tuple;
	BOOST_MULTI_HD explicit convertible_tuple(Tuple const& other)
	: Tuple(other) {}

 public:
	using array_type = std::array<std::ptrdiff_t, std::tuple_size_v<Tuple>>;
	auto to_array() const noexcept {
		return std::apply([](auto... es) noexcept {
			return std::array<std::common_type_t<decltype(es)...>, sizeof...(es)>{{static_cast<size_type>(es)...}};
		},
						  static_cast<Tuple const&>(*this));
	}

	/*explicit*/ operator array_type() const& noexcept { return to_array(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	/*explicit*/ operator array_type() && noexcept { return to_array(); }      // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
#endif
	[[deprecated("This is here for nominal compatiblity with Boost.MultiArray, this would be a dangling conversion")]]
	operator std::ptrdiff_t const*() const&&;  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
											   /*{ return to_array().data(); }*/
#ifdef __clang__
#pragma clang diagnostic pop
#endif

	template<std::size_t Index, std::enable_if_t<(Index < std::tuple_size_v<Tuple>), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend BOOST_MULTI_HD constexpr auto get(convertible_tuple const& self) -> std::tuple_element_t<Index, Tuple> {
		using std::get;
		return get<Index>(static_cast<Tuple const&>(self));
	}
};

template<class Array>
struct decaying_array : Array {
	using Array::Array;
	explicit decaying_array(Array const& other)
	: Array(other) {}

	[[deprecated("possible dangling conversion, use `std::array<T, D> p` instead of `auto* p`")]]
	constexpr operator std::ptrdiff_t const*() const { return Array::data(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

	template<std::size_t Index, std::enable_if_t<(Index < std::tuple_size_v<Array>), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(decaying_array const& self) -> std::tuple_element_t<Index, Array> {
		using std::get;
		return get<Index>(static_cast<Array const&>(self));
	}
};
}  // end namespace boost::multi::detail

template<class Tuple> struct std::tuple_size<boost::multi::detail::convertible_tuple<Tuple>> : std::integral_constant<std::size_t, std::tuple_size_v<Tuple>> {};  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal to define tuple size
template<class Array> struct std::tuple_size<boost::multi::detail::decaying_array<Array>> : std::integral_constant<std::size_t, std::tuple_size_v<Array>> {};     // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal to define tuple size

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
namespace std::ranges {  // NOLINT(cert-dcl58-cpp) to enable borrowed, nvcc needs namespace
template<>
[[maybe_unused]] constexpr bool enable_borrowed_range<::boost::multi::extensions_t<1>::elements_t> = true;  // NOLINT(misc-definitions-in-headers)

template<class Fun, ::boost::multi::dimensionality_type D>
[[maybe_unused]] constexpr bool enable_borrowed_range<::boost::multi::restriction<D, Fun> > = true;  // NOLINT(misc-definitions-in-headers)
}  // end namespace std::ranges
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#undef BOOST_MULTI_HD

#include <cassert>
#include <functional>   // for std::invoke
#include <iterator>     // for std::size (in c++17)
#include <memory>       // for allocator<>
#include <type_traits>  // for std::invoke_result

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4626)  // 'boost::multi::transform_ptr<main::complex,const main::<lambda_3>,main::complex *,std::complex<double>>': assignment operator was implicitly defined as deleted [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\test\element_transformed.cpp.x.vcxproj]
#endif

namespace boost::multi {

struct uninitialized_elements_t {
	explicit uninitialized_elements_t() = default;
};

inline constexpr uninitialized_elements_t uninitialized_elements{};

template<class T, class Ptr = T*>
struct move_ptr : private std::move_iterator<Ptr> {
	using difference_type   = typename std::iterator_traits<std::move_iterator<Ptr>>::difference_type;
	using value_type        = typename std::iterator_traits<std::move_iterator<Ptr>>::value_type;
	using pointer           = Ptr;
	using reference         = typename std::move_iterator<Ptr>::reference;
	using iterator_category = typename std::iterator_traits<std::move_iterator<Ptr>>::iterator_category;

	template<class U> using rebind = std::conditional_t<
		std::is_const_v<U>,
		typename std::pointer_traits<Ptr>::template rebind<U>,
		move_ptr<U, U*>>;

	using std::move_iterator<Ptr>::move_iterator;

	BOOST_MULTI_HD constexpr /**/ operator Ptr() const { return std::move_iterator<Ptr>::base(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) // NOSONAR(cpp:S1709) decay to lvalue should be easy

	BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> move_ptr& {
		static_cast<std::move_iterator<Ptr>&>(*this) += n;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator-=(difference_type n) -> move_ptr& {
		static_cast<std::move_iterator<Ptr>&>(*this) -= n;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator+(difference_type n) const -> move_ptr {
		move_ptr ret{*this};
		ret += n;
		return ret;
	}
	BOOST_MULTI_HD constexpr auto operator-(difference_type n) const -> move_ptr {
		move_ptr ret{*this};
		ret -= n;
		return ret;
	}

	BOOST_MULTI_HD constexpr auto operator-(move_ptr const& other) const -> difference_type { return static_cast<std::move_iterator<Ptr> const&>(*this) - static_cast<std::move_iterator<Ptr> const&>(other); }

	BOOST_MULTI_HD constexpr auto operator<(move_ptr const& other) const -> bool { return static_cast<std::move_iterator<Ptr> const&>(*this) < static_cast<std::move_iterator<Ptr> const&>(other); }

	constexpr auto                operator*() const -> decltype(auto) { return *static_cast<std::move_iterator<Ptr> const&>(*this); }
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> decltype(auto) { return *((*this) + n); }

	BOOST_MULTI_HD constexpr auto operator==(move_ptr const& other) const -> bool { return static_cast<std::move_iterator<Ptr> const&>(*this) == static_cast<std::move_iterator<Ptr> const&>(other); }
	BOOST_MULTI_HD constexpr auto operator!=(move_ptr const& other) const -> bool { return static_cast<std::move_iterator<Ptr> const&>(*this) != static_cast<std::move_iterator<Ptr> const&>(other); }
};

template<class T> struct ref_add_const {
	using type = T const;
};  // this is not the same as std::add_const

template<class T> struct ref_add_const<T const> {
	using type = T const;
};
template<class T> struct ref_add_const<T const&> {
	using type = T const&;
};
template<class T> struct ref_add_const<T&> {
	using type = T const&;
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, class UF, class Ptr, class Ref = std::invoke_result_t<UF const&, typename std::iterator_traits<Ptr>::reference>>
struct transform_ptr {
	using difference_type   = typename std::iterator_traits<Ptr>::difference_type;
	using value_type        = std::decay_t<Ref>;  // typename std::iterator_traits<std::move_iterator<Ptr>>::value_type;
	using pointer           = Ptr;
	using reference         = Ref;
	using iterator_category = typename std::iterator_traits<Ptr>::iterator_category;

	template<class U> using rebind =
		transform_ptr<
			std::remove_cv_t<U>,
			UF, Ptr,
			std::conditional_t<
				std::is_const_v<U>,
				typename ref_add_const<Ref>::type,
				Ref>
			// typename std::conditional<
			//  std::is_const_v<U>,
			//  typename ref_add_const<Ref>::type,
			//  Ref
			// >::type
			>;

#if defined(__GNUC__) && (__GNUC__ < 9)
	constexpr explicit transform_ptr(std::nullptr_t nil) : p_{nil} /*, f_{}*/ {}  // seems to be necessary for gcc 7
#endif
#if defined(__NVCC__) || defined(__NVCOMPILER)
	constexpr transform_ptr() {}
#else
	constexpr transform_ptr();  // : p_{}, f_{} {}
#endif
	template<class UFF>
	constexpr transform_ptr(pointer ptr, UFF&& fun) : p_{ptr}, f_{std::forward<UFF>(fun)} {}

	template<class Other, class P = typename Other::pointer, decltype(detail::implicit_cast<pointer>(std::declval<P>()))* = nullptr>
	// cppcheck-suppress noExplicitConstructor
	constexpr /*mplc*/ transform_ptr(Other const& other) : p_{other.p_}, f_{other.f_} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) // NOSONAR(cpp:S1709)

	template<class Other, class P = typename Other::pointer, decltype(detail::explicit_cast<pointer>(std::declval<P>()))* = nullptr>
	constexpr explicit transform_ptr(Other const& other) : p_{other.p_}, f_{other.f_} {}

	// constexpr auto functor() const -> UF {return f_;}
	constexpr auto base() const -> Ptr const& { return p_; }
	constexpr auto operator*() const -> reference {  // NOLINT(readability-const-return-type) in case synthesis reference is a `T const`
		// invoke allows for example to use .transformed( &member) instead of .transformed( std::mem_fn(&member) )
		return std::invoke(f_, *p_);  // NOLINT(readability-const-return-type) in case synthesis reference is a `T const`
	}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	constexpr auto operator+=(difference_type n) -> transform_ptr& {
		p_ += n;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}
	constexpr auto operator-=(difference_type n) -> transform_ptr& {
		p_ -= n;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	constexpr auto operator+(difference_type n) const -> transform_ptr { return transform_ptr{*this} += n; }
	constexpr auto operator-(difference_type n) const -> transform_ptr { return transform_ptr{*this} -= n; }

	constexpr auto friend operator+(difference_type n, transform_ptr const& self) { return self + n; }

	constexpr auto operator-(transform_ptr const& other) const -> difference_type { return p_ - other.p_; }

	constexpr auto operator[](difference_type n) const -> reference { return *((*this) + n); }  // NOLINT(readability-const-return-type) transformed_view might return by const value.

	constexpr auto operator==(transform_ptr const& other) const -> bool { return p_ == other.p_; }
	constexpr auto operator!=(transform_ptr const& other) const -> bool { return p_ != other.p_; }

	constexpr auto operator==(std::nullptr_t const& nil) const -> bool { return p_ == nil; }
	constexpr auto operator!=(std::nullptr_t const& nil) const -> bool { return p_ != nil; }

	constexpr auto operator<(transform_ptr const& other) const -> bool { return p_ < other.p_; }

	transform_ptr(transform_ptr const&)     = default;
	transform_ptr(transform_ptr&&) noexcept = default;

	~transform_ptr() = default;

	auto operator=(transform_ptr&&) noexcept -> transform_ptr& = default;

	constexpr auto operator=(transform_ptr const& other) -> transform_ptr& {  // NOLINT(cert-oop54-cpp) self-assignment is ok
		// assert(f_ == other.f_);
		p_ = other.p_;
		return *this;
	}

 private:
	Ptr p_;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820)  // '7' bytes padding added after data member 'boost::multi::transform_ptr<main::complex,const main::<lambda_3>,main::complex *,std::complex<double>>::f_'
#pragma warning(disable : 4371)  // layout of class may have changed from a previous version of the compiler due to better packing of member 'boost::multi::transform_ptr<int,int main::S::* ,main::S *,int &>::f_'
#endif
	BOOST_MULTI_NO_UNIQUE_ADDRESS
	UF f_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) technically this type can be const
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	template<class, class, class, class> friend struct transform_ptr;
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class Array, typename Reference = void, typename Element = void>
struct array_traits;

template<class Array, typename Reference, typename Element>
struct array_traits {
	using reference              = typename Array::reference;
	using element                = typename Array::element;
	using element_ptr            = typename Array::element_ptr;
	using decay_type             = typename Array::decay_type;
	using default_allocator_type = typename Array::default_allocator_type;
};

template<class T, typename = typename T::rank>
auto        has_rank_aux(T const&) -> std::true_type;
inline auto has_rank_aux(...) -> std::false_type;

template<class T> struct has_rank : decltype(has_rank_aux(std::declval<T>())){};

template<typename T> struct rank;

template<class T, typename = decltype(std::declval<T&>().move())>
auto        has_member_move_aux(T const&) -> std::true_type;
inline auto has_member_move_aux(...) -> std::false_type;

template<class T> struct has_member_move : decltype(has_member_move_aux(std::declval<T>())){};

template<typename T, typename = std::enable_if_t<has_rank<T>{}>>
constexpr auto rank_aux(T const&) -> typename T::rank;

template<typename T, typename = std::enable_if_t<!has_rank<T>::value>>
constexpr auto rank_aux(T const&) -> std::integral_constant<size_t, std::rank_v<T>>;

template<typename T> struct rank : decltype(rank_aux(std::declval<T>())){};

template<class Pointer, std::enable_if_t<std::is_pointer<Pointer>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) special sfinae trick
constexpr auto stride(Pointer /*ptr*/) -> std::ptrdiff_t { return 1; }

template<class Pointer, std::enable_if_t<std::is_pointer<Pointer>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) special sfinae trick
constexpr auto base(Pointer ptr) -> Pointer { return ptr; }

template<class TPointer, class U>
constexpr auto reinterpret_pointer_cast(U* other)                                                 // name taken from thrust::reinterpret_pointer_cast, which is difference from std::reinterpret_pointer_cast(std::shared_ptr<T>)
	-> decltype(reinterpret_cast<TPointer>(other)) { return reinterpret_cast<TPointer>(other); }  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) : unavoidalbe implementation?

template<class T, std::size_t N>
constexpr auto size(T const (& /*array*/)[N]) noexcept { return static_cast<multi::size_type>(N); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, typename = typename T::get_allocator>
auto        has_get_allocator_aux(T const&) -> std::true_type;
inline auto has_get_allocator_aux(...) -> std::false_type;

template<class T, std::size_t N>
constexpr auto get_allocator(T (& /*array*/)[N]) noexcept -> std::allocator<std::decay_t<std::remove_all_extents_t<T[N]>>> { return {}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T>
constexpr auto get_allocator(T* const& /*t*/)
	-> decltype(std::allocator<typename std::iterator_traits<T*>::value_type>{}) {
	return std::allocator<typename std::iterator_traits<T*>::value_type>{};
}

template<class T>
constexpr auto default_allocator_of(T* /*unused*/) {
	return std::allocator<typename std::iterator_traits<T*>::value_type>{};
}

template<class T>
constexpr auto to_address(T* const& ptr)
	-> decltype(ptr) {
	return ptr;
}

template<class T>
auto has_get_allocator_aux(T const& cont) -> decltype(cont.get_allocator(), std::true_type{});

template<class T> struct has_get_allocator : decltype(has_get_allocator_aux(std::declval<T>())){};

// template<class T1, class T2, typename Ret = T1>  // std::common_type_t<T1, T2>>
// auto common(T1 const& val1, T2 const& val2) -> Ret {
//  return val1 == val2?
//      val1:
//      Ret{}
//  ;
// }

template<class T>
auto        has_num_elements_aux(T const& /*array*/) -> decltype(std::declval<T const&>().num_elements() + 1, std::true_type{});
inline auto has_num_elements_aux(...) -> decltype(std::false_type{});
template<class T> struct has_num_elements : decltype(has_num_elements_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class A, typename = std::enable_if_t<has_num_elements<A>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
constexpr auto num_elements(A const& arr)
	-> std::make_signed_t<decltype(arr.num_elements())> {
	return static_cast<std::make_signed_t<decltype(arr.num_elements())>>(arr.num_elements());
}

template<class T>
auto        has_size_aux(T const& cont) -> decltype(std::size(cont), std::true_type{});
inline auto has_size_aux(...) -> decltype(std::false_type{});
template<class T> struct has_size : decltype(has_size_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T>
auto        has_data_elements_aux(T&& array) -> decltype(array.data_elements() + 1, std::true_type{});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) why +1?
inline auto has_data_elements_aux(...) -> decltype(std::false_type{});
template<class T> struct has_data_elements : decltype(has_data_elements_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T>
auto        has_base_aux(T&& array) -> decltype(array.base() + 1, std::true_type{});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) why +1?
inline auto has_base_aux(...) -> decltype(std::false_type{});
template<class T> struct has_base : decltype(has_base_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

namespace detail {
template<class T>
auto        has_data_aux(T&& cont) -> decltype(cont.data_elements() + 1, std::true_type{});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) why +1?
inline auto has_data_aux(...) -> decltype(std::false_type{});
}  // end namespace detail
template<class T> struct has_data : decltype(detail::has_data_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class Array, std::enable_if_t<has_data<std::decay_t<Array>>::value && !has_data_elements<std::decay_t<Array>>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto data_elements(Array& arr) { return std::data(arr); }

template<class Array, std::enable_if_t<has_data<std::decay_t<Array>>::value && !has_data_elements<std::decay_t<Array>>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto data_elements(Array const& arr) { return std::data(arr); }                                                                           // .data();}

template<class A, std::enable_if_t<!has_num_elements<A>::value && has_size<A>::value && has_data<A>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
constexpr auto num_elements(A const& arr) -> std::make_signed_t<decltype(std::size(arr))> {
	return static_cast<std::make_signed_t<decltype(std::size(arr))>>(std::size(arr));  // (arr.size());
}

template<class A, std::enable_if_t<has_data_elements<A>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
constexpr auto data_elements(A const& arr)
	-> decltype(arr.data_elements()) {
	return arr.data_elements();
}

template<class T, std::enable_if_t<!std::is_array_v<std::decay_t<T>> && !has_data_elements<std::decay_t<T>>::value && !has_data<std::decay_t<T>>::value, int> = 0>  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
constexpr auto data_elements(T& value) -> decltype(&value) { return &value; }

template<class A> struct num_elements_t : std::integral_constant<std::ptrdiff_t, 1> {};

template<class T, std::size_t N> struct num_elements_t<T[N]> : std::integral_constant<std::ptrdiff_t, (N * num_elements_t<T>{})> {};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, std::size_t N> struct num_elements_t<T (&)[N]> : num_elements_t<T[N]> {};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, std::size_t N>
constexpr auto num_elements(T const (& /*array*/)[N]) noexcept { return num_elements_t<T[N]>{}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class Vector, typename = std::enable_if_t<std::is_same_v<typename Vector::pointer, decltype(std::declval<Vector>().data())>>, class = decltype(Vector{}.resize(1))>
auto data_elements(Vector const& vec)
	-> decltype(vec.data()) {
	return vec.data();
}

template<class T, std::size_t N>
constexpr auto stride(T const (& /*array*/)[N]) noexcept -> std::ptrdiff_t { return num_elements_t<T>{}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, std::size_t N>
constexpr auto is_compact(T const (& /*t*/)[N]) noexcept -> bool { return true; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, std::size_t N>
constexpr auto offset(T const (& /*t*/)[N]) noexcept -> std::ptrdiff_t { return 0; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, std::size_t N>
[[deprecated("use data_elements instead")]]  // this name is bad because when the element belongs to std:: then std::data is picked up by ADL and the
constexpr auto
data(T (&array)[N]) noexcept { return data(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, std::size_t N>
constexpr auto data_elements(T (&array)[N]) noexcept { return data_elements(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T>
auto        has_dimensionality_aux(T const& /*array*/) -> decltype(T::rank_v, std::true_type{});
inline auto has_dimensionality_aux(...) -> decltype(std::false_type{});
template<class T> struct has_dimensionality : decltype(has_dimensionality_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class Container, std::enable_if_t<has_dimensionality<Container>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
constexpr auto dimensionality(Container const& /*container*/)
	-> std::decay_t<decltype(typename Container::rank{} + 0)> {
	return Container::rank_v;
}

template<class T>
auto        has_dimensionaliy_member_aux(T const& /*array*/) -> decltype(static_cast<void>(static_cast<boost::multi::dimensionality_type>(T::rank_v)), std::true_type{});
inline auto has_dimensionaliy_member_aux(...) -> decltype(std::false_type{});
template<class T> struct has_dimensionality_member : decltype(has_dimensionaliy_member_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T, typename = std::enable_if_t<!has_dimensionality_member<T>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto dimensionality(T const& /*, void* = nullptr*/) { return 0; }

template<class T, std::size_t N>
constexpr auto dimensionality(T const (&array)[N]) { return 1 + dimensionality(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, class Ret = decltype(std::declval<T const&>().sizes())>
constexpr auto sizes(T const& arr) noexcept -> Ret { return arr.sizes(); }

template<class T, std::enable_if_t<!has_dimensionality<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
constexpr auto sizes(T const& /*unused*/) noexcept { return tuple<>{}; }

template<class T, std::size_t N>
constexpr auto sizes(T const (&array)[N]) noexcept {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility
													  //  using std::size; // this line needs c++17
	return multi::detail::ht_tuple(multi::size(array), multi::sizes(array[0]));
}

template<class T, std::size_t N>
constexpr auto base(T (&array)[N]) noexcept {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility
	return data_elements(array);
}

template<class T, std::size_t N>
constexpr auto base(T (*&array)[N]) noexcept { return base(*array); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, typename = std::enable_if_t<!std::is_array_v<T>>>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto base(T const* ptr) noexcept { return ptr; }

template<class T, typename = std::enable_if_t<!std::is_array_v<T>>>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto base(T* ptr) noexcept { return ptr; }

template<class T>
constexpr auto corigin(T const& value) { return &value; }

template<class T, std::size_t N>
constexpr auto corigin(T const (&array)[N]) noexcept { return corigin(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, typename = decltype(std::declval<T>().extension())>
auto        has_extension_aux(T const&) -> std::true_type;
inline auto has_extension_aux(...) -> std::false_type;
template<class T> struct has_extension : decltype(has_extension_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class Container, class = std::enable_if_t<!has_extension<Container>::value>>  // NOLINT(modernize-use-constraints) TODO(correaa)
auto extension(Container const& cont)                                                  // TODO(correaa) consider "extent"
	-> decltype(multi::extension_t<std::make_signed_t<decltype(size(cont))>>(0, static_cast<std::make_signed_t<decltype(size(cont))>>(size(cont)))) {
	return multi::extension_t<std::make_signed_t<decltype(size(cont))>>(0, static_cast<std::make_signed_t<decltype(size(cont))>>(size(cont)));
}

template<dimensionality_type Rank, class Container, std::enable_if_t<!has_extension<Container>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
auto extensions(Container const& cont) {
	if constexpr(Rank == 0) {
		return multi::extensions_t<0>{};
	} else {
		using std::size;
		return multi::extension_t<std::make_signed_t<decltype(size(cont))>>(0, static_cast<std::make_signed_t<decltype(size(cont))>>(size(cont))) * extensions<Rank - 1>(cont.front());
	}
}

// template<class T, typename = decltype(std::declval<T>().shape())>
//        auto has_shape_aux(T const&) -> std::true_type;
// inline auto has_shape_aux(...     ) -> std::false_type;

template<class T> struct has_shape : decltype(has_shape_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) trick

template<class T, typename = decltype(std::declval<T const&>().elements())>
auto        has_elements_aux(T const&) -> std::true_type;
inline auto has_elements_aux(...) -> std::false_type;

template<class T> struct has_elements : decltype(has_elements_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) trick

template<class T, typename = decltype(std::declval<T const&>().extensions())>
auto        has_extensions_aux(T const&) -> std::true_type;
inline auto has_extensions_aux(...) -> std::false_type;

template<class T> struct has_extensions : decltype(has_extensions_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) trick

template<class T, std::enable_if_t<has_extensions<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
[[nodiscard]] constexpr auto extensions(T const& array) -> std::decay_\
t<decltype(array.extensions())> {
	return array.extensions();
}

template<class BoostMultiArray, std::size_t... I>
constexpr auto extensions_aux2(BoostMultiArray const& arr, std::index_sequence<I...> /*012*/) {
	return boost::multi::extensions_t<BoostMultiArray::dimensionality>(
		boost::multi::iextension{static_cast<multi::index>(arr.index_bases()[I]), static_cast<multi::index>(arr.index_bases()[I]) + static_cast<multi::index>(arr.shape()[I])}...  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	);
}

template<class Element, class T, std::enable_if_t<has_extensions<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
[[nodiscard]] auto extensions_of(T const& array) {
	if constexpr(std::is_convertible_v<T const&, Element>) {
		return boost::multi::extensions_t<0>{};
	}
	if constexpr(std::is_convertible_v<typename T::reference, Element>) {
		return boost::multi::extensions_t<1>{array.extension()};
	}
}

template<class Arr2D>
auto transposed(Arr2D&& arr)
	-> decltype(std::forward<Arr2D>(arr).transposed()) {
	return std::forward<Arr2D>(arr).transposed();
}

// template<class BoostMultiArray, std::enable_if_t<has_shape<BoostMultiArray>::value && !has_extensions<BoostMultiArray>::value, int> =0>
// constexpr auto extensions(BoostMultiArray const& array) {
//  return extensions_aux2(array, std::make_index_sequence<BoostMultiArray::dimensionality>{});
// }

template<class T, std::enable_if_t<!has_extensions<T>::value /*&& !has_shape<T>::value*/, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
constexpr auto extensions(T const& /*unused*/) -> multi::layout_t<0>::extensions_type { return {}; }

template<class T, std::size_t N>
constexpr auto extensions(T (&array)[N]) {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	return index_extension{N} * extensions(array[0]);
}

template<dimensionality_type D>
struct extensions_aux {
	template<class T>
	static auto call(T const& array) {
		return array.extension() * extensions<D - 1>(array);
		// return tuple_cat(std::make_tuple(array.extension()), extensions<D-1>(array));
	}
};

template<> struct extensions_aux<0> {
	template<class T> static auto call(T const& /*unused*/) { return multi::extensions_t<0>{}; }  // std::make_tuple();}
};

template<dimensionality_type D, class T, std::enable_if_t<has_extension<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto extensions(T const& array) {
	return extensions_aux<D>::call(array);
}

template<class T1> struct extensions_t_aux;

template<class T1, class T2> auto extensions_me(T2 const& array) {
	return extensions_t_aux<T1>::call(array);
}

template<class T1> struct extension_t_aux {
	static auto call(T1 const& /*unused*/) { return std::make_tuple(); }
	template<class T2>
	static auto call(T2 const& array) { return tuple_cat(std::make_tuple(array.extension()), extensions_me<T1>(*begin(array))); }
};

template<class T, typename = decltype(std::declval<T const&>().layout())>
auto        has_layout_member_aux(T const&) -> std::true_type;
inline auto has_layout_member_aux(...) -> std::false_type;

template<class T>
struct has_layout_member : decltype(has_layout_member_aux(std::declval<T const&>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T, typename = std::enable_if_t<has_layout_member<T const&>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
auto layout(T const& array)
	-> decltype(array.layout()) {
	return array.layout();
}

template<class T, typename = std::enable_if_t<!has_layout_member<T const&>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
auto layout(T const& /*unused*/) -> layout_t<0> { return {}; }

template<class T, std::size_t N>
constexpr auto layout(T (&array)[N]) { return multi::layout_t<std::rank_v<T[N]>>{multi::extensions(array)}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): for backward compatibility

template<class T, std::size_t N>
constexpr auto strides(T (&array)[N]) { return layout(array).strides(); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): for backward compatibility

template<class T, std::size_t N>
struct array_traits<std::array<T, N>> {
	static constexpr auto dimensionality() -> dimensionality_type { return 1; }

	using reference   = T&;
	using value_type  = std::decay_t<T>;
	using pointer     = T*;
	using element     = value_type;
	using element_ptr = pointer;
	using decay_type  = std::array<value_type, N>;
};

template<class T, std::size_t N, std::size_t M>
struct array_traits<std::array<std::array<T, M>, N>> {
	static constexpr auto dimensionality() -> dimensionality_type { return 1 + array_traits<std::array<T, M>>::dimensionality(); }

	using reference   = std::array<T, M>&;
	using value_type  = std::array<std::decay_t<T>, M>;
	using pointer     = std::array<T, M>*;
	using element     = typename array_traits<std::array<T, M>>::element;
	using element_ptr = typename array_traits<std::array<T, M>>::element;
	using decay_type  = std::array<value_type, M>;
};

template<class T, std::size_t N> constexpr auto                data_elements(std::array<T, N>& arr) noexcept { return arr.data(); }
template<class T, std::size_t M, std::size_t N> constexpr auto data_elements(std::array<std::array<T, M>, N>& arr) noexcept { return data_elements(arr[0]); }

template<class T, std::size_t N> constexpr auto                data_elements(std::array<T, N> const& arr) noexcept { return arr.data(); }
template<class T, std::size_t M, std::size_t N> constexpr auto data_elements(std::array<std::array<T, M>, N> const& arr) noexcept { return data_elements(arr[0]); }

template<class T, std::size_t N> constexpr auto data_elements(std::array<T, N>&& arr) noexcept { return std::move(arr).data(); }

template<class T, std::size_t M, std::size_t N>
constexpr auto data_elements(std::array<std::array<T, M>, N>&& arr) noexcept { return data_elements(std::move(arr)[0]); }

template<class T, std::size_t N> constexpr auto num_elements(std::array<T, N> const& /*unused*/) noexcept
	-> std::ptrdiff_t { return N; }

template<class T, std::size_t M, std::size_t N>
constexpr auto num_elements(std::array<std::array<T, M>, N> const& arr)
	-> std::ptrdiff_t { return static_cast<std::ptrdiff_t>(N) * num_elements(arr[0]); }

template<class T, std::size_t N>
constexpr auto dimensionality(std::array<T, N> const& /*unused*/) -> boost::multi::dimensionality_type { return 1; }

template<class T, std::size_t M, std::size_t N>
constexpr auto dimensionality(std::array<std::array<T, M>, N> const& arr) -> boost::multi::dimensionality_type {
	return 1 + dimensionality(arr[0]);
}

template<class T, std::size_t N>
constexpr auto extensions(std::array<T, N> const& /*arr*/) {
	return multi::extensions_t<1>{multi::index_extension(0, N)};
}

template<class T, std::size_t N, std::size_t M>
auto extensions(std::array<std::array<T, N>, M> const& arr) {
	return multi::iextension{M} * extensions(arr[0]);
}

template<class T, std::size_t N>
constexpr auto stride(std::array<T, N> const& /*arr*/) {
	return static_cast<multi::size_type>(1U);  // multi::stride_type?
}

template<class T, std::size_t N, std::size_t M>
constexpr auto stride(std::array<std::array<T, N>, M> const& arr) {
	return num_elements(arr[0]);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) use checked span
#endif

template<class T, std::size_t N>
constexpr auto layout(std::array<T, N> const& arr) {
	return multi::layout_t<multi::array_traits<std::array<T, N>>::dimensionality()>{multi::extensions(arr)};
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace detail {
inline auto valid_mull(int age) -> bool {
	return age >= 21;
}
}  // end namespace detail

}  // end namespace boost::multi

namespace boost::multi::detail {

template<class F>
BOOST_MULTI_HD constexpr auto invoke_square(F&& fn) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get
	return std::forward<F>(fn);
}

template<class F, class Arg>
BOOST_MULTI_HD constexpr auto invoke_square(F&& fn, Arg&& arg) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get
	return std::forward<F>(fn)[std::forward<Arg>(arg)];
}

template<class F, class Arg, class... Args>
BOOST_MULTI_HD constexpr auto invoke_square(F&& fn, Arg&& arg, Args&&... args) -> decltype(auto) {  // NOLINT(cert-dcl58-cpp,bugprone-std-namespace-modification) normal idiom to defined tuple get
	return invoke_square(std::forward<F>(fn)[std::forward<Arg>(arg)], std::forward<Args>(args)...);
	// return            std::forward<F>(fn)[std::forward<Arg>(arg),  std::forward<Arg>(args)...];  // will not work with iterators or cursors in the current state, it is also a C++23-only feature
}

}  // end namespace boost::multi::detail

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#undef BOOST_MULTI_HD

#include <cmath>
#include <type_traits>

#if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<ranges>)
#include <ranges>  // IWYU pragma: keep
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4623)  // assignment operator was implicitly defined as deleted
#pragma warning(disable : 4626)  // assignment operator was implicitly defined as deleted
#pragma warning(disable : 4625)  // copy constructor was implicitly defined as deleted
#endif

namespace boost::multi {

template<class Element>
inline constexpr bool force_element_trivial = false;

template<class Element>
inline constexpr bool force_element_trivial_destruction = force_element_trivial<Element>;

template<class Element>
inline constexpr bool force_element_trivial_default_construction = force_element_trivial<Element>;

#ifdef _BOOST_MULTI_FORCE_TRIVIAL_STD_COMPLEX
template<class T>
inline constexpr bool force_element_trivial<std::complex<T>> = std::is_trivial_v<T>;

template<class T>
inline constexpr bool force_element_trivial_destruction<std::complex<T>> = std::is_trivially_default_constructible_v<T>;

template<class T>
inline constexpr bool force_element_trivial_default_construction<std::complex<T>> = std::is_trivially_destructible_v<T>;

template<> inline constexpr bool force_element_trivial<std::complex<double>>                      = true;
template<> inline constexpr bool force_element_trivial_default_construction<std::complex<double>> = true;
template<> inline constexpr bool force_element_trivial_destruction<std::complex<double>>          = true;

template<> inline constexpr bool force_element_trivial<std::complex<float>>                      = true;
template<> inline constexpr bool force_element_trivial_default_construction<std::complex<float>> = true;
template<> inline constexpr bool force_element_trivial_destruction<std::complex<float>>          = true;
#endif

}  // end namespace boost::multi

// Copyright 2020-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__CUDA__) || defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)

#ifdef __NVCC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress = 20011  // deep inside Thrust: calling a __host__ function("std::vector<double, ::std::allocator<double> > ::vector(const ::std::vector<double, ::std::allocator<double> > &)") from a __host__ __device__ function("thrust::system::detail::generic::detail::uninitialized_copy_functor<    ::std::vector<double, ::std::allocator<double> > ,     ::std::vector<double, ::std::allocator<double> > > ::operator ()< ::thrust::detail::tuple_of_iterator_references<    ::std::vector<double, ::std::allocator<double> >  &,     ::std::vector<double, ::std::allocator<double> >  & > > ") is not allowed
#pragma nv_diag_suppress = 20014  // deep inside Thrust: calling a __host__ function from a __host__ __device__ function is not allowed
#pragma nv_diag_suppress = 20015  // deep inside Thrust: calling a constexpr __host__ function from a __host__ __device__ function is not allowed
#endif

#include <thrust/copy.h>
#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/detail/memory_algorithms.h>
#include <thrust/equal.h>
#include <thrust/uninitialized_copy.h>

#ifdef __NVCC__
#pragma nv_diagnostic pop  // nv_diagnostics pop
#endif

#endif

#include <algorithm>    // for for_each, copy_n, fill, fill_n, lexicographical_compare, swap_ranges  // IWYU pragma: keep  // bug in iwyu 0.18
#include <cstddef>      // for size_t
#include <functional>   // for equal_to
#include <iterator>     // for iterator_traits, distance, size
#include <memory>       // for allocator_traits, allocator, pointer_traits
#include <type_traits>  // for decay_t, enable_if_t, conditional_t, declval, is_pointer, true_type
#include <utility>      // for forward, addressof

#ifdef _MULTI_FORCE_TRIVIAL_STD_COMPLEX
#include<complex>
#endif

#define BOOST_MULTI_DEFINE_ADL(FuN)  /*NOLINT(cppcoreguidelines-macro-usage) TODO(correaa) consider replacing for all ADL'd operations*/ \
namespace boost { \
namespace multi { \
namespace adl { \
	namespace custom {template<class...> struct FuN##_t;}   __attribute__((unused))  \
	static constexpr class FuN##_t { \
		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... args) const = delete; \
		template<class... As>          auto _(priority<1>,        As&&... args) const BOOST_MULTI_DECLRETURN(std::FuN(std::forward<As>(args)...)) \
		template<class... As>          auto _(priority<2>,        As&&... args) const BOOST_MULTI_DECLRETURN(     FuN(std::forward<As>(args)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(t).FuN(std::forward<As>(args)...))     \
		template<class... As>          auto _(priority<4>,        As&&... args) const BOOST_MULTI_DECLRETURN(custom::FuN##_t<As&&...>::_(std::forward<As>(args)...)) \
	public: \
		template<class... As> auto operator()(As&&... args) const-> decltype(_(priority<4>{}, std::forward<As>(args)...)) {return _(priority<4>{}, std::forward<As>(args)...);} \
	} (FuN); \
}  /* end namespace adl   */ \
}  /* end namespace multi */ \
}  /* end namespace boost */

#define BOOST_MULTI_DECLRETURN(ExpR) -> decltype(ExpR) {return ExpR;}  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing
#define BOOST_MULTI_JUSTRETURN(ExpR)                   {return ExpR;}  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing

namespace boost::multi {

template<std::size_t N> struct priority : std::conditional_t<N == 0, std::true_type, priority<N-1>> {};

class adl_copy_n_t {
	template<class... As>          constexpr auto _(priority<0>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(std::                copy_n(                      std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(::thrust::           copy_n(                      std::forward<As>(args)...))
#endif
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     copy_n(                      std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    copy_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).copy_n(                      std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_copy_n_t adl_copy_n;

// there is no move_n (std::move_n), use copy_n(std::make_move_iterator(first), count) instead

class adl_move_t {
	template<class... As>           constexpr auto _(priority<0>/**/,                      As&&... args) const BOOST_MULTI_DECLRETURN(              std::    move(                      std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)  // there is no thrust::move algorithm
	template<class It, class... As> constexpr auto _(priority<1>/**/, It first, It last, As&&... args) const BOOST_MULTI_DECLRETURN(           thrust::copy(std::make_move_iterator(first), std::make_move_iterator(last), std::forward<As>(args)...))
#endif
	template<class... As>           constexpr auto _(priority<2>/**/,                      As&&... args) const BOOST_MULTI_DECLRETURN(                     move(                      std::forward<As>(args)...))
	template<class T, class... As>  constexpr auto _(priority<3>/**/, T&& arg,             As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    move(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As>  constexpr auto _(priority<4>/**/, T&& arg,             As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).move(                      std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_move_t adl_move;

class adl_fill_n_t {
	template<         class... As> constexpr auto _(priority<0>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  fill_n              (std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<         class... As> constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(           thrust::  fill_n              (std::forward<As>(args)...))
#endif
	template<         class... As> constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     fill_n              (std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    fill_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).fill_n              (std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_fill_n_t adl_fill_n;

class adl_fill_t {
	template<         class... As> constexpr auto _(priority<0>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  fill              (std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<         class... As> constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(           thrust::  fill              (std::forward<As>(args)...))
#endif
	template<         class... As> constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     fill              (std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    fill(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).fill              (std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_fill_t adl_fill;

class adl_equal_t {
	template<         class...As> constexpr auto _(priority<1>/**/,          As&&...args) const BOOST_MULTI_DECLRETURN(               std::  equal(                      std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<         class...As> constexpr auto _(priority<2>/**/,          As&&...args) const BOOST_MULTI_DECLRETURN(          ::thrust::  equal(                      std::forward<As>(args)...))
#endif
	template<         class...As> constexpr auto _(priority<3>/**/,          As&&...args) const BOOST_MULTI_DECLRETURN(                      equal(                      std::forward<As>(args)...))
	template<         class...As> constexpr auto _(priority<4>/**/,          As&&...args) const BOOST_MULTI_DECLRETURN(                      equal(                      std::forward<As>(args)..., std::equal_to<>{}))  // WORKAROUND makes syntax compatible with boost::ranges::equal if, for some reason, it is included.
	template<class T, class...As> constexpr auto _(priority<5>/**/, T&& arg, As&&...args) const BOOST_MULTI_DECLRETURN( std::decay_t<T>::    equal(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class...As> constexpr auto _(priority<6>/**/, T&& arg, As&&...args) const BOOST_MULTI_DECLRETURN( std::forward<T>(arg).equal(                      std::forward<As>(args)...))

 public:
	template<class...As>          constexpr auto operator()(As&&...args) const BOOST_MULTI_DECLRETURN(_(priority<6>{}, std::forward<As>(args)...))
};
inline constexpr adl_equal_t adl_equal;

#ifndef _MSC_VER
template<class... As, class = std::enable_if_t<sizeof...(As) == 0> > void copy(As...) = delete;  // NOLINT(modernize-use-constraints) TODO(correaa)
#endif

class adl_copy_t {
	template<class InputIt, class OutputIt,
		class=std::enable_if_t<std::is_assignable_v<typename std::iterator_traits<OutputIt>::reference, typename std::iterator_traits<InputIt>::reference>>  // NOLINT(modernize-use-constraints) TODO(correaa)
	>
	                               constexpr auto _(priority<1>/**/, InputIt first, InputIt last, OutputIt d_first) const BOOST_MULTI_DECLRETURN(std::copy(first, last, d_first))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(         ::thrust::copy(std::forward<As>(args)...))
#endif
	template<         class... As> constexpr auto _(priority<3>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                   copy(std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::copy(std::forward<T>(arg), std::forward<As>(args)...))
//  template<class... As         > constexpr auto _(priority<5>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(boost::multi::adl_custom_copy<std::decay_t<As>...>::copy(std::forward<As>(as)...))
	template<class T, class... As> constexpr auto _(priority<6>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).copy(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN( _(priority<6>{}, std::forward<As>(args)...) ) \
};
inline constexpr adl_copy_t adl_copy;

namespace adl {
	// namespace custom {template<class...> struct fill_t;}
	class fill_t {
		template<class... As>          auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  fill              (std::forward<As>(args)...))
		template<class... As>          auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     fill              (std::forward<As>(args)...))
		template<class T, class... As> auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).fill              (std::forward<As>(args)...))
		// template<class... As>          auto _(priority<4>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(custom::             fill_t<As&&...>::_(std::forward<As>(args)...))
	
	 public:
		template<class... As> auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<5>{}, std::forward<As>(args)...))
	};
	inline constexpr fill_t fill;
}  // end namespace adl

namespace xtd {

template<class T>  // this one goes last!!!
constexpr auto to_address(T const& ptr) noexcept;

template<class T>
constexpr auto me_to_address(priority<0> /**/, T const& ptr) noexcept
	-> decltype(to_address(ptr.operator->())) {
	return to_address(ptr.operator->());
}

template<class T>
constexpr auto me_to_address(priority<1> /**/, T const& ptr) noexcept
	-> decltype(std::pointer_traits<T>::to_address(ptr)) {
	return std::pointer_traits<T>::to_address(ptr);
}

template<class T, std::enable_if_t<std::is_pointer<T>{}, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto me_to_address(priority<2>/**/, T const& ptr) noexcept -> T {
    static_assert(! std::is_function_v<T>);
    return ptr;
}

template<class T>  // this one goes last!!!
constexpr auto to_address(T const& ptr) noexcept
->decltype(me_to_address(priority<2>{}/**/, ptr)) {
	return me_to_address(priority<2>{}    , ptr); }

template<class Alloc, class ForwardIt, class Size,
	typename Value = typename std::iterator_traits<ForwardIt>::value_type, typename = decltype(std::addressof(*ForwardIt{})),
	typename = decltype(Value())
>
auto alloc_uninitialized_value_construct_n(Alloc& alloc, ForwardIt first, Size count) -> ForwardIt {
// ->std::decay_t<decltype(std::allocator_traits<Alloc>::construct(alloc, std::addressof(*first), Value()), first)>
	ForwardIt current = first;
	try {
		for (; count > 0 ; ++current, --count) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current), Value());  // !!!!!!!!!!!!!! if you are using std::complex type consider making complex default constructible (e.g. by type traits)
		}
		//  ::new (static_cast<void*>(std::addressof(*current))) Value();
		return current;
	} catch(...) {
		for(; current != first; ++first) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*first));
		}
		throw;
	}
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

template<class Alloc, class ForwardIt, class Size, class T = typename std::iterator_traits<ForwardIt>::value_type>
auto alloc_uninitialized_default_construct_n(Alloc& alloc, ForwardIt first, Size count)
-> std::decay_t<decltype(std::allocator_traits<Alloc>::construct(alloc, std::addressof(*first)), first)> {
	if(std::is_trivially_default_constructible_v<T>) {
		std::advance(first, count);
		return first;
	}
	using alloc_traits = std::allocator_traits<Alloc>;
	ForwardIt current  = first;

	try {
		//  return std::for_each_n(first, count, [&](T& elem) { alloc_traits::construct(alloc, std::addressof(elem)); ++current; });
		//  workadoung for gcc 8.3.1 in Lass
		std::for_each(first, first + count, [&](T& elem) { alloc_traits::construct(alloc, std::addressof(elem)); ++current; });  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return first + count;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	} catch(...) {
		// LCOV_EXCL_START  // TODO(correaa) add test
		std::for_each(first, current, [&](T& elem) { alloc_traits::destroy(alloc, std::addressof(elem)); });
		throw;
		// LCOV_EXCL_STOP
	}

	// return current;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

}  // end namespace xtd

template<class BidirIt, class Size, class T = typename std::iterator_traits<BidirIt>::value_type>
constexpr auto destroy_n(BidirIt first, Size count)
->std::decay_t<decltype(std::addressof(*(first - 1)), first)> {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	first += count;
	for(; count != 0; --first, --count) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
		std::addressof(*(first-1))->~T();  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
	return first;
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

template<class Alloc, class BidirIt, class Size, class T = typename std::iterator_traits<BidirIt>::value_type>
constexpr auto alloc_destroy_n(Alloc& alloc, BidirIt first, Size count)
->std::decay_t<decltype(std::addressof(*(first-1)), first)> {  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	first += count;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	for (; count != 0; --first, --count) {  // NOLINT(altera-unroll-loops,cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) consider using an algorithm
		std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*(first - 1)));  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}
	return first;
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

class adl_uninitialized_copy_t {
	template<class InIt, class FwdIt, class = decltype(std::addressof(*FwdIt{}))>                 // sfinae friendy std::uninitialized_copy
	[[nodiscard]] constexpr auto _(priority<1> /**/, InIt first, InIt last, FwdIt d_first) const  // N_O_L_I_N_T(performance-unnecessary-value-param) bug in clang-tidy
	// BOOST_MULTI_DECLRETURN(       std::uninitialized_copy(first, last, d_first))
	{
#if __cplusplus >= 202002L
		using ValueType = typename std::iterator_traits<FwdIt>::value_type;
		if(
			std::is_constant_evaluated() && (std::is_trivially_default_constructible_v<ValueType> || multi::force_element_trivial_default_construction<ValueType>)
		) {
			return std::copy(std::move(first), std::move(last), std::move(d_first));
		}
#endif
		return std::uninitialized_copy(std::move(first), std::move(last), std::move(d_first));
	}
#if defined(__CUDACC__) || defined(__HIPCC__)
	template<class InIt, class FwdIt, class ValueType = typename std::iterator_traits<FwdIt>::value_type>
	constexpr auto _(priority<2>/**/, InIt first, InIt last, FwdIt d_first) const -> decltype(::thrust::uninitialized_copy(first, last, d_first))  // doesn't work with culang 17, cuda 12 ?
	{
		if constexpr(std::is_trivially_default_constructible_v<ValueType> || multi::force_element_trivial_default_construction<ValueType>) {
			return ::thrust::copy(first, last, d_first);
		} else {
			return ::thrust::uninitialized_copy(first, last, d_first);
		}
	}
#endif
	template<class TB, class... As       > constexpr auto _(priority<3>/**/, TB&& first, As&&... args       ) const BOOST_MULTI_DECLRETURN(                        uninitialized_copy(                 std::forward<TB>(first) , std::forward<As>(args)...))
	template<class TB, class TE, class DB> constexpr auto _(priority<4>/**/, TB&& first, TE&& last, DB&& d_first) const BOOST_MULTI_DECLRETURN(std::decay_t<DB>      ::uninitialized_copy(                 std::forward<TB>(first) , std::forward<TE>(last), std::forward<DB>(d_first)            ))
	template<class TB, class... As       > constexpr auto _(priority<5>/**/, TB&& first, As&&... args       ) const BOOST_MULTI_DECLRETURN(std::decay_t<TB>      ::uninitialized_copy(std::forward<TB>(first), std::forward<As>(args)...))
	template<class TB, class... As       > constexpr auto _(priority<6>/**/, TB&& first, As&&... args       ) const BOOST_MULTI_DECLRETURN(std::forward<TB>(first).uninitialized_copy(                         std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<6>{}, std::forward<As>(args)...))
};
inline constexpr adl_uninitialized_copy_t adl_uninitialized_copy;

#ifdef __NVCC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress = implicit_return_from_non_void_function
#endif

class adl_uninitialized_copy_n_t {
	template<class... As>          constexpr auto _(priority<1>/**/,        As&&... args) const BOOST_MULTI_DECLRETURN(                  std::uninitialized_copy_n(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,        As&&... args) const BOOST_MULTI_DECLRETURN(                       uninitialized_copy_n(std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<
		class It, class Size, class ItFwd,
		class ValueType = typename std::iterator_traits<ItFwd>::value_type,
		class = std::enable_if_t<! std::is_rvalue_reference_v<typename std::iterator_traits<It>::reference> >
	>
	constexpr auto _(priority<3>/**/, It first, Size count, ItFwd d_first) const -> decltype(::thrust::uninitialized_copy_n(first, count, d_first)) {
		if constexpr(std::is_trivially_default_constructible_v<ValueType> || multi::force_element_trivial_default_construction<ValueType>) {
			return ::thrust::copy_n(first, count, d_first);
		} else {
			return ::thrust::uninitialized_copy_n(first, count, d_first);
		}
	}
#endif
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    uninitialized_copy_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<5>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).uninitialized_copy_n(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<5>{}, std::forward<As>(args)...))  // TODO(correaa) this might trigger a compiler crash with g++ 7.5 because of operator&() && overloads
};
inline constexpr adl_uninitialized_copy_n_t adl_uninitialized_copy_n;

#ifdef __NVCC__
#pragma nv_diagnostic pop
#endif

class adl_uninitialized_move_n_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  uninitialized_move_n(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     uninitialized_move_n(std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    uninitialized_move_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).uninitialized_move_n(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const {return _(priority<4>{}, std::forward<As>(args)...);}
};
inline constexpr auto adl_uninitialized_move_n = adl_uninitialized_move_n_t{};

namespace xtd {

template<class T, class InputIt, class Size, class ForwardIt>
constexpr auto alloc_uninitialized_copy_n(std::allocator<T>& /*alloc*/, InputIt first, Size count, ForwardIt d_first) {
	return adl_uninitialized_copy_n(first, count, d_first);}

// template<class T, class InputIt, class Size, class ForwardIt>
// constexpr auto alloc_uninitialized_move_n(std::allocator<T>& /*alloc*/, InputIt first, Size count, ForwardIt d_first) {
//  return adl_uninitialized_move_n(first, count, d_first);}

template<class Alloc, class InputIt, class Size, class ForwardIt, class = decltype(std::addressof(*ForwardIt{}))>
auto alloc_uninitialized_copy_n(Alloc& alloc, InputIt first, Size count, ForwardIt d_first) {
	ForwardIt current = d_first;
	try {
		for(; count > 0; ++first, ++current, --count) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current), *first);
		}
		return current;
	} catch(...) {
		for(; d_first != current; ++d_first) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*d_first));
		}
		throw;
	}
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

template<class Alloc, class InputIt, class Size, class ForwardIt>
// [[deprecated("check")]]
auto alloc_uninitialized_move_n(Alloc& alloc, InputIt first, Size count, ForwardIt d_first) {
	ForwardIt current = d_first;
	try {
		// NOLINTNEXTLINE(altera-unroll-loops,cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) consider using an algorithm
		for(; count > 0; ++first, ++current, --count) {  // mull-ignore: cxx_gt_to_ge
			std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current), std::move(*first));
		}
		return current;
	} catch(...) {
		for(; d_first != current; ++d_first) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch,cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*d_first));
		}
		throw;
	}
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class T, class InputIt, class ForwardIt>
constexpr auto alloc_uninitialized_copy(std::allocator<T>&/*allocator*/, InputIt const& first, InputIt const& last, ForwardIt d_first) {
	return adl_uninitialized_copy(first, last, d_first);
}

template<class Alloc, class InputIt, class ForwardIt, class=decltype(std::addressof(*std::declval<ForwardIt>())),
	class=std::enable_if_t<std::is_constructible_v<typename std::iterator_traits<ForwardIt>::value_type, typename std::iterator_traits<InputIt>::reference>>  // NOLINT(modernize-use-constraints) TODO(correaa)
>
#if __cplusplus >= 202002L
constexpr
#endif
auto alloc_uninitialized_copy(Alloc& alloc, InputIt first, InputIt last, ForwardIt d_first) {
// ->std::decay_t<decltype(a.construct(std::addressof(*d_first), *first), d_first)> // problematic in clang-11 + gcc-9
	ForwardIt current = d_first;
	using alloc_traits = std::allocator_traits<Alloc>;
	try {
		std::for_each(first, last, [&](auto const& elem) {  // TODO(correaa) replace by adl_for_each
			alloc_traits::construct(alloc, std::addressof(*current), elem);
			++current;
		});
		return current;
	} catch(...) {
		std::for_each(d_first, current, [&](auto const& elem) {
			std::allocator_traits<Alloc>::destroy(alloc, std::addressof(elem));
		});
		throw;
	}
}

template<class Alloc, class ForwardIt, class Size, class T>
auto alloc_uninitialized_fill_n(Alloc& alloc, ForwardIt first, Size n, T const& value)
->std::decay_t<decltype(std::allocator_traits<Alloc>::construct(alloc, std::addressof(*first), value), first)> {
	ForwardIt current = first;  // using std::to_address;
	try {
		for(; n > 0; ++current, --n) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current), value);
		}
		return current;
	} catch(...) {
		for(; first != current; ++first) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*first));
		}
		throw;
	}
}
}  // end namespace xtd

class adl_distance_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::    distance(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                       distance(std::forward<As>(args)...))
//	template<class It1, class It2> constexpr auto _(priority<3>/**/, It1 it1, It2 it2     ) const BOOST_MULTI_DECLRETURN(it2 - it1)
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  distance(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<5>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).distance(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<5>{}, std::forward<As>(args)...))
};
inline constexpr adl_distance_t adl_distance;

class adl_begin_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                std::begin(std::forward<As>(args)...))
//  template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     begin(std::forward<As>(args)...))  // this is catching boost::range_iterator if Boost 1.53 is included
// #if defined(__NVCC__)  // this is no thrust::begin
//  template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(::thrust::           begin(                      std::forward<As>(args)...))
// #endif
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(    std::decay_t<T>::begin(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).begin(std::forward<As>(args)...))

 public:
	template<class... As> [[nodiscard]] constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_begin_t adl_begin;

class adl_end_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  end(std::forward<As>(args)...))
	// template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     end(std::forward<As>(args)...))
// #if defined(__NVCC__)  // there is no thrust::end
//  template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(::thrust::           end(                      std::forward<As>(args)...))
// #endif
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  end(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).end(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_end_t adl_end;

class adl_size_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                std::size(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     size(std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(    std::decay_t<T>::size(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).size(std::forward<As>(args)...))

 public:
	template<class... As> [[nodiscard]] constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_size_t adl_size;

class adl_swap_ranges_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  swap_ranges(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     swap_ranges(std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  swap_ranges(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).swap_ranges(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_swap_ranges_t adl_swap_ranges;

class adl_lexicographical_compare_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              std::  lexicographical_compare(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     lexicographical_compare(std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  lexicographical_compare(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).lexicographical_compare(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_lexicographical_compare_t adl_lexicographical_compare;

class adl_uninitialized_value_construct_n_t {
	template<class... As>              constexpr auto _(priority<1>/**/,                      As&&... args) const BOOST_MULTI_DECLRETURN(              std::  uninitialized_value_construct_n(std::forward<As>(args)...))  // TODO(correaa) use boost alloc_X functions?
	template<class... As>              constexpr auto _(priority<2>/**/,                      As&&... args) const BOOST_MULTI_DECLRETURN(                     uninitialized_value_construct_n(std::forward<As>(args)...))
	template<class T, class... As>     constexpr auto _(priority<3>/**/, T&& arg,             As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::uninitialized_value_construct_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As>     constexpr auto _(priority<4>/**/, T&& arg,             As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).uninitialized_value_construct_n(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const {return (_(priority<4>{}, std::forward<As>(args)...));}
};
inline constexpr adl_uninitialized_value_construct_n_t adl_uninitialized_value_construct_n;

class adl_alloc_uninitialized_value_construct_n_t {
	template<class Alloc, class... As> constexpr auto _(priority<1>/**/, Alloc&& /*alloc*/, As&&... args) const BOOST_MULTI_DECLRETURN(                       adl_uninitialized_value_construct_n(std::forward<As>(args)...))  // NOLINT(cppcoreguidelines-missing-std-forward)
//  template<class... As>              constexpr auto _(priority<2>/**/,                    As&&... args) const BOOST_MULTI_DECLRETURN(              xtd::  alloc_uninitialized_value_construct_n(std::forward<As>(args)...))  // TODO(correaa) use boost alloc_X functions?
	template<class... As>              constexpr auto _(priority<3>/**/,                    As&&... args) const BOOST_MULTI_DECLRETURN(                     alloc_uninitialized_value_construct_n(std::forward<As>(args)...))
	template<class T, class... As>     constexpr auto _(priority<4>/**/, T&& arg,           As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  alloc_uninitialized_value_construct_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As>     constexpr auto _(priority<5>/**/, T&& arg,           As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).alloc_uninitialized_value_construct_n(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const {return (_(priority<5>{}, std::forward<As>(args)...));}
};
inline constexpr adl_alloc_uninitialized_value_construct_n_t adl_alloc_uninitialized_value_construct_n;

class adl_uninitialized_default_construct_n_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const {return                  std::  uninitialized_default_construct_n(                      std::forward<As>(args)...);}
	// #if defined(__NVCC__)
	// template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(             thrust::uninitialized_default_construct_n(                      std::forward<As>(args)...))
	// #endif
	template<class... As>          constexpr auto _(priority<3>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     uninitialized_default_construct_n(                      std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  uninitialized_default_construct_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<5>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).uninitialized_default_construct_n(                      std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const {return (_(priority<5>{}, std::forward<As>(args)...));}
};
inline constexpr adl_uninitialized_default_construct_n_t adl_uninitialized_default_construct_n;

class adl_alloc_uninitialized_default_construct_n_t {
	template<class Alloc, class... As>          constexpr auto _(priority<1>/**/, Alloc&&/*unused*/, As&&... args) const BOOST_MULTI_JUSTRETURN(                      adl_uninitialized_default_construct_n(                      std::forward<As>(args)...))  // NOLINT(cppcoreguidelines-missing-std-forward)
	template<class... As>                       constexpr auto _(priority<2>/**/,                    As&&... args) const BOOST_MULTI_DECLRETURN(               xtd::alloc_uninitialized_default_construct_n(                      std::forward<As>(args)...))  // TODO(correaa) use boost alloc_X functions?
#if defined(__CUDACC__) || defined(__HIPCC__)
#if defined(THRUST_VERSION) && (THRUST_VERSION < 200700)  // 200800)
	// boost::multi::detail::what_value_t<THRUST_VERSION> a;
	template<class Alloc, class It, class Size> constexpr auto _(priority<3>/**/, Alloc&& alloc, It first, Size n) const BOOST_MULTI_DECLRETURN(         (thrust::detail::default_construct_range(std::forward<Alloc>(alloc), first, n)) )
#else
	// boost::multi::detail::what_value_t<THRUST_VERSION> b;
	template<class Alloc, class It, class Size> constexpr auto _(priority<3>/**/, Alloc&& alloc, It first, Size n) const BOOST_MULTI_DECLRETURN(         (thrust::detail::value_initialize_range(std::forward<Alloc>(alloc), first, n)) )
#endif
#endif
	template<class... As>                       constexpr auto _(priority<4>/**/,          As&&... args          ) const BOOST_MULTI_DECLRETURN(                     alloc_uninitialized_default_construct_n(                      std::forward<As>(args)...))  
	template<class T, class... As>              constexpr auto _(priority<5>/**/, T&& arg, As&&... args          ) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  alloc_uninitialized_default_construct_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As>              constexpr auto _(priority<6>/**/, T&& arg, As&&... args          ) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).alloc_uninitialized_default_construct_n(                      std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const {return (_(priority<6>{}, std::forward<As>(args)...));}
};
inline constexpr adl_alloc_uninitialized_default_construct_n_t adl_alloc_uninitialized_default_construct_n;

class adl_destroy_n_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(            multi::  destroy_n              (std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     destroy_n              (std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(  std::decay_t<T>::  destroy_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).destroy_n              (std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<As>(args)...))
};
inline constexpr adl_destroy_n_t adl_destroy_n;

class adl_alloc_destroy_n_t {
	template<class Alloc, class... As> constexpr auto _(priority<1>/**/, Alloc&&/*unused*/, As&&... args) const BOOST_MULTI_DECLRETURN(             adl_destroy_n              (std::forward<As>(args)...))  // NOLINT(cppcoreguidelines-missing-std-forward)
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<class Alloc, class It, class Size> constexpr auto _(priority<2>/**/, Alloc& alloc, It first, Size n) const BOOST_MULTI_DECLRETURN(   (thrust::detail::destroy_range(alloc, first, first + n)))
#endif
	template<             class... As> constexpr auto _(priority<3>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(multi::              alloc_destroy_n              (std::forward<As>(args)...))  // TODO(correaa) use boost alloc_X functions?
	template<             class... As> constexpr auto _(priority<4>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     alloc_destroy_n              (std::forward<As>(args)...))
	template<class T,     class... As> constexpr auto _(priority<5>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::decay_t<T>::    alloc_destroy_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T,     class... As> constexpr auto _(priority<6>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).alloc_destroy_n              (std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<6>{}, std::forward<As>(args)...))
};
inline constexpr adl_alloc_destroy_n_t adl_alloc_destroy_n;

class adl_alloc_uninitialized_copy_t {
	template<class Alloc, class... As> constexpr auto _(priority<1>/**/, Alloc&&/*ll*/, As&&... args) const BOOST_MULTI_DECLRETURN(                             adl_uninitialized_copy(                            std::forward<As>(args)...))  // NOLINT(cppcoreguidelines-missing-std-forward)
	template<class Alloc, class... As> constexpr auto _(priority<2>/**/, Alloc&& alloc, As&&... args) const BOOST_MULTI_DECLRETURN(                      xtd::alloc_uninitialized_copy(std::forward<Alloc>(alloc), std::forward<As>(args)...))
	template<class Alloc, class... As> constexpr auto _(priority<3>/**/, Alloc&& alloc, As&&... args) const BOOST_MULTI_DECLRETURN(                           alloc_uninitialized_copy(std::forward<Alloc>(alloc), std::forward<As>(args)...))
	template<class Alloc, class... As> constexpr auto _(priority<4>/**/, Alloc&& alloc, As&&... args) const BOOST_MULTI_DECLRETURN(      std::decay_t<Alloc>::alloc_uninitialized_copy(std::forward<Alloc>(alloc), std::forward<As>(args)...))
	template<class Alloc, class... As> constexpr auto _(priority<5>/**/, Alloc&& alloc, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<Alloc>(alloc).alloc_uninitialized_copy(                            std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<5>{}, std::forward<As>(args)...))
};
inline constexpr adl_alloc_uninitialized_copy_t adl_alloc_uninitialized_copy;

class adl_alloc_uninitialized_copy_n_t {
	template<class Alloc, class... As> constexpr auto _(priority<1>/**/, Alloc&& /*alloc*/, As&&... args) const BOOST_MULTI_DECLRETURN(                       adl_uninitialized_copy_n(std::forward<As>(args)...))  // NOLINT(cppcoreguidelines-missing-std-forward)
	template<class... As>              constexpr auto _(priority<2>/**/,                    As&&... args) const BOOST_MULTI_DECLRETURN(                     alloc_uninitialized_copy_n(std::forward<As>(args)...))
//  template<class... As>              constexpr auto _(priority<3>/**/,                    As&&... args) const BOOST_MULTI_DECLRETURN(                xtd::alloc_uninitialized_copy_n(std::forward<As>(args)...))
// #if defined(__NVCC__)
//  there is no thrust alloc uninitialized copy 
// #endif
	template<class T, class... As>     constexpr auto _(priority<5>/**/, T&& arg,           As&&... args) const BOOST_MULTI_DECLRETURN(    std::decay_t<T>::alloc_uninitialized_copy_n(std::forward<T>(arg), std::forward<As>(args)...))
	template<class T, class... As>     constexpr auto _(priority<6>/**/, T&& arg,           As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).alloc_uninitialized_copy_n(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const {return _(priority<6>{}, std::forward<As>(args)...);}
};
inline constexpr adl_alloc_uninitialized_copy_n_t adl_alloc_uninitialized_copy_n;

class alloc_uninitialized_move_n_t {
// TODO(correaa) : fallback to no alloc version
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const {return(                             xtd::  alloc_uninitialized_move_n(std::forward<As>(args)...));}
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                     alloc_uninitialized_move_n(std::forward<As>(args)...))
	template<class T, class... As> constexpr auto _(priority<3>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(std::forward<T>(arg).alloc_uninitialized_move_n(std::forward<As>(args)...))

 public:
	template<class... As> constexpr auto operator()(As&&... args) const { return _(priority<3>{}, std::forward<As>(args)...); }
};
inline constexpr alloc_uninitialized_move_n_t adl_alloc_uninitialized_move_n;

class uninitialized_fill_n_t {
	template<class... As>          constexpr auto _(priority<1>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(               std::    uninitialized_fill_n(std::forward<As>(args)...))
	template<class... As>          constexpr auto _(priority<2>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(                        uninitialized_fill_n(std::forward<As>(args)...))
#if defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
	template<class... As>          constexpr auto _(priority<3>/**/,          As&&... args) const BOOST_MULTI_DECLRETURN(              ::thrust::uninitialized_fill_n(std::forward<As>(args)...))
#endif
	template<class T, class... As> constexpr auto _(priority<4>/**/, T&& arg, As&&... args) const BOOST_MULTI_DECLRETURN( std::forward<T>(arg).uninitialized_fill_n(std::forward<As>(args)...))

 public:
	template<class T1, class... As> constexpr auto operator()(T1&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<T1>(arg), std::forward<As>(args)...))
};
inline constexpr uninitialized_fill_n_t adl_uninitialized_fill_n;

class alloc_uninitialized_fill_n_t {
	template<             class... As> constexpr auto _(priority<1>/**/,                   As&&... args) const BOOST_MULTI_DECLRETURN(                       xtd::alloc_uninitialized_fill_n(std::forward<As>(args)...))
	template<class Alloc, class... As> constexpr auto _(priority<2>/**/, Alloc&&/*alloc*/, As&&... args) const BOOST_MULTI_DECLRETURN(                              adl_uninitialized_fill_n(std::forward<As>(args)...))  // NOLINT(cppcoreguidelines-missing-std-forward)
	template<             class... As> constexpr auto _(priority<3>/**/,                   As&&... args) const BOOST_MULTI_DECLRETURN(                            alloc_uninitialized_fill_n(std::forward<As>(args)...))
	template<class Alloc, class... As> constexpr auto _(priority<4>/**/, Alloc&&  alloc  , As&&... args) const BOOST_MULTI_DECLRETURN( std::forward<Alloc>(alloc).alloc_uninitialized_fill_n(std::forward<As>(args)...))

 public:
	template<class T1, class... As> constexpr auto operator()(T1&& arg, As&&... args) const BOOST_MULTI_DECLRETURN(_(priority<4>{}, std::forward<T1>(arg), std::forward<As>(args)...))
};
inline constexpr alloc_uninitialized_fill_n_t adl_alloc_uninitialized_fill_n;

// template<dimensionality_type N>
// struct recursive {
//  template<class Alloc, class InputIt, class ForwardIt>
//  static constexpr auto alloc_uninitialized_copy(Alloc& alloc, InputIt first, InputIt last, ForwardIt dest){
//      using std::begin; using std::end;
//      while(first!=last) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
//          recursive<N-1>::alloc_uninitialized_copy(alloc, begin(*first), end(*first), begin(*dest));
//          ++first;
//          ++dest;
//      }
//      return dest;
//  }
// };

// template<> struct recursive<1> {
//  template<class Alloc, class InputIt, class ForwardIt>
//  static auto alloc_uninitialized_copy(Alloc& alloc, InputIt first, InputIt last, ForwardIt dest){
//      return adl_alloc_uninitialized_copy(alloc, first, last, dest);
//  }
// };

}  // end namespace boost::multi

#undef BOOST_MULTI_DECLRETURN
#undef BOOST_MULTI_JUSTRETURN

#endif

// Copyright 2019-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include<cassert>

#if defined(BOOST_MULTI_ASSERT_DISABLE)  // to activate bounds check compile in debug mode (default) with -DBOOST_MULTI_ACCESS_DEBUG
	#define BOOST_MULTI_ASSERT(Expr) /*empty*/ // NOLINT(cppcoreguidelines-macro-usage
#else
	#define BOOST_MULTI_ASSERT(Expr) assert(Expr)  // NOLINT(cppcoreguidelines-macro-usage)
#endif

// #if defined(BOOST_MULTI_NDEBUG) || defined(__CUDACC__)
//  #define BOOST_MULTI_ASSERT(Expr)  // NOLINT(cppcoreguidelines-macro-usage
// #else
//  // #include<stacktrace>
//  // // NOLINTNEXTLINE(cppcoreguidelines-macro-usage) this is for very inefficient asserts
//  // #if defined(__cpp_lib_stacktrace) && (__cpp_lib_stacktrace >= 202011L)
//  // #define BOOST_MULTI_ASSERT(Expr) assert((std::cerr<<std::stacktrace()<<std::endl) && (Expr))
//  // #else
//  #define BOOST_MULTI_ASSERT(Expr) assert(Expr)  // NOLINT(cppcoreguidelines-macro-usage)
//  // #endif
// #endif

// Copyright 2019-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <iterator>                 // for copy, iterator_traits
#include <memory>                   // for allocator_traits
#include <type_traits>              // for declval, enable_if_t, false_type, is_trivially_default_constructible, true_type, void_t
#include <utility>                  // for addressof, forward

namespace boost::multi {

template<class Alloc>
struct allocator_traits : std::allocator_traits<Alloc> {};

// https://en.cppreference.com/w/cpp/memory/destroy
template<
	class Alloc, class ForwardIt,
	std::enable_if_t<!has_rank<ForwardIt>::value, int> =0  // NOLINT(modernize-use-constraints) TODO(correaa)
>
void destroy(Alloc& alloc, ForwardIt first, ForwardIt last) {
	for(; first != last; ++first) {allocator_traits<Alloc>::destroy(alloc, std::addressof(*first));}  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
}

template<class Alloc, class ForwardIt, std::enable_if_t<has_rank<ForwardIt>::value && ForwardIt::rank_v == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
void destroy(Alloc& alloc, ForwardIt first, ForwardIt last) {
	//  using multi::to_address;
	std::for_each(first, last, [&](auto& elem) {alloc.destroy(addressof(elem));});
	// for(; first != last; ++first) {alloc.destroy(to_address(first));}  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
}

template<class Alloc, class ForwardIt, std::enable_if_t<has_rank<ForwardIt>::value && ForwardIt::rank_v != 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
void destroy(Alloc& alloc, ForwardIt first, ForwardIt last) {
	for(; first != last; ++first) {destroy(alloc, begin(*first), end(*first));} // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
}

template<class Alloc, class InputIt, class Size, class ForwardIt>
auto uninitialized_move_n(Alloc& alloc, InputIt first, Size n, ForwardIt dest) -> ForwardIt {
	ForwardIt curr = dest;
//  using std::addressof;
	try {
		for(; n > 0; ++first, ++curr, --n) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			alloc.construct(std::addressof(*curr), std::move(*first));
		}
		return curr;
	} catch(...) {destroy(alloc, dest, curr); throw;}
}

template<class Alloc, class ForwardIt, class Size>
auto uninitialized_default_construct_n(Alloc& alloc, ForwardIt first, Size n) -> ForwardIt {
	ForwardIt current = first;
	try {
		for(; n > 0; ++current, --n) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
		//  allocator_traits<Alloc>::construct(a, to_address(current));
			alloc.construct(to_address(current));
		}
		return current;
	} catch(...) {destroy(alloc, first, current); throw;}
}

template<
	class Alloc, class ForwardIt, class Size,
	typename T = typename std::iterator_traits<ForwardIt>::value_type,
	typename = std::enable_if_t<! std::is_trivially_default_constructible<T>{}>  // NOLINT(modernize-use-constraints) TODO(correaa)
>
auto uninitialized_value_construct_n(Alloc& alloc, ForwardIt first, Size n) -> ForwardIt {
	ForwardIt current = first;  // using std::addressof;
	try {
		for(; n > 0; ++current, --n) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
			allocator_traits<Alloc>::construct(alloc, to_address(current), T{});
		}
		//  a.construct(to_address(current), T());  //  a.construct(std::pointer_traits<Ptr>::pointer_to(*current), T());  //   AT::construct(a, to_address(current), T());  // AT::construct(a, addressof(*current), T()); //  a.construct(addressof(*current), T());
		return current;
	} catch(...) {destroy(alloc, first, current); throw;}
}

template<class... Args> auto std_copy(Args&&... args) {
	using std::copy;
	return copy(std::forward<Args>(args)...);
}

namespace xtd {

template<class Alloc, class InputIt, class MIt, typename = std::enable_if_t<! has_rank<MIt>{}> >  // NOLINT(modernize-use-constraints) TODO(correaa)
auto alloc_uninitialized_copy(Alloc& alloc, InputIt first, InputIt last, MIt dest) -> MIt {
	MIt current = dest;
//  using multi::to_address;
	try {
		for(; first != last; ++first, ++current) {alloc.construct(to_address(current), *first);}  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
		return current;
	} catch(...) {destroy(alloc, dest, current); throw;}
}

}  // end namespace xtd

template<class, class = void>
struct is_allocator : std::false_type {};

template<class Alloc>
struct is_allocator<Alloc, std::void_t<decltype(
	std::declval<Alloc const&>() == Alloc{std::declval<Alloc const&>()},
	std::declval<Alloc&>().deallocate(typename Alloc::pointer{std::declval<Alloc&>().allocate(std::declval<typename Alloc::size_type>())}, std::declval<typename Alloc::size_type>())
)>> : std::true_type {};

template<class Alloc> constexpr bool is_allocator_v = is_allocator<Alloc>::value;

// template<dimensionality_type N, class InputIt, class ForwardIt>
// auto uninitialized_copy(InputIt first, InputIt last, ForwardIt dest) {
//  while(first!=last) {  // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
//    uninitialized_copy<N-1>(begin(*first), end(*first), begin(*dest));
//    ++first;
//    ++dest;
//  }
//  return dest;
// }

}  // end namespace boost::multi

// Copyright 2020-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <cstddef>      // for size_t
#include <iterator>     // for iterator_traits
#include <memory>       // for allocator, pointer_traits
#include <type_traits>  // for conditional_t, declval, true_type

namespace boost::multi {

template<std::size_t N> struct priority_me : std::conditional_t<N == 0, std::true_type, priority_me<N-1>>{};

template<class Pointer>  auto dat_aux(priority_me<0>, Pointer ) -> std::allocator<typename std::iterator_traits<Pointer>::value_type>;
template<class T>        auto dat_aux(priority_me<1>, T*      ) -> std::allocator<typename std::iterator_traits<T*>::value_type>;
template<class FancyPtr> auto dat_aux(priority_me<2>, FancyPtr) -> typename FancyPtr::default_allocator_type;

template<class Pointer>
struct pointer_traits/*, typename Pointer::default_allocator_type>*/ : std::pointer_traits<Pointer>{
	using default_allocator_type = decltype(dat_aux(priority_me<2>{}, std::declval<Pointer>()));
};

}  // end namespace boost::multi

#include <algorithm>  // fpr copy_n
#include <array>
#include <cstring>     // for std::memset in reinterpret_cast
#include <functional>  // for std::invoke
#include <iterator>    // for std::next
#include <memory>      // for std::pointer_traits
#include <new>         // for std::launder

#if __has_include(<span>)
#if !defined(_MSVC_LANG) || (_MSVC_LANG > 202002L)
#include <span>
#endif
#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L && !defined(_MSVC_LANG)
#define BOOST_MULTI_HAS_SPAN
#endif
#endif

#include <utility>  // for forward

#ifdef __NVCC__
#define BOOST_MULTI_FRIEND_CONSTEXPR template<class = void> friend constexpr  // workaround nvcc
#else
#define BOOST_MULTI_FRIEND_CONSTEXPR friend constexpr
#endif

#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#define BOOST_MULTI_IGNORED_UNSAFE_BUFFER_USAGE_PUSH() \
	_Pragma("clang diagnostic push")                   \
		_Pragma("clang diagnostic ignored \"-Wunsafe-buffer-usage\"")
#else
#define BOOST_MULTI_IGNORED_UNSAFE_BUFFER_USAGE_PUSH()
#endif

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#define BOOST_MULTI_IGNORED_UNSAFE_BUFFER_USAGE_POP() _Pragma("clang diagnostic pop")
#else
#define BOOST_MULTI_IGNORED_UNSAFE_BUFFER_USAGE_POP()
#endif

namespace boost::multi {

template<typename T, dimensionality_type D, typename ElementPtr = T const*, class Layout = layout_t<D>>
struct const_subarray;

namespace detail {
template<class T, dimensionality_type D, class... Ts>
auto is_const_subarray_aux(multi::const_subarray<T, D, Ts...> const&) -> std::true_type;
auto is_const_subarray_aux(...) -> std::false_type;
}  // end namespace detail

template<class T> struct is_const_subarray : decltype(detail::is_const_subarray_aux(std::declval<T>())){};
template<class T>
constexpr bool is_const_subarray_v = is_const_subarray<T>::value;

template<typename T, dimensionality_type D, typename ElementPtr = T*, class Layout = layout_t<D, typename std::pointer_traits<ElementPtr>::difference_type>>
class subarray;

template<typename T, dimensionality_type D, typename ElementPtr = T*, class Layout = layout_t<D, typename std::pointer_traits<ElementPtr>::difference_type>>
class move_subarray;

template<typename T, dimensionality_type D, typename ElementPtr, class Layout>
constexpr auto is_subarray_aux(const_subarray<T, D, ElementPtr, Layout> const&) -> std::true_type;
constexpr auto is_subarray_aux(...) -> std::false_type;

template<class A> struct is_subarray : decltype(is_subarray_aux(std::declval<A>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)

template<dimensionality_type D>
struct of_dim {
	template<typename T, class ElementPtr, class Layout>
	static constexpr auto is_subarray_of_dim_aux(subarray<T, D, ElementPtr, Layout> const&) -> std::true_type;
	static constexpr auto is_subarray_of_dim_aux(...) -> std::false_type;

	template<class A> struct is_subarray_of_dim : decltype(is_subarray_of_dim_aux(std::declval<A>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
};

// template<typename T, dimensionality_type D, class A = std::allocator<T>> struct array;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<typename T, dimensionality_type D, typename ElementPtr = T*, class Layout = layout_t<D, std::make_signed_t<typename std::pointer_traits<ElementPtr>::size_type>>>
struct array_types : private Layout {  // cppcheck-suppress syntaxError ; false positive in cppcheck
	using element      = T;
	using element_type = element;  // this follows more closely https://en.cppreference.com/w/cpp/memory/pointer_traits

	using element_ptr       = ElementPtr;
	using element_const_ptr = typename std::pointer_traits<ElementPtr>::template rebind<element const>;
	using element_move_ptr  = multi::move_ptr<element, element_ptr>;

	using element_ref = typename std::iterator_traits<element_ptr>::reference;

	using layout_t = Layout;

	using rank = typename layout_t::rank;

	using layout_t::rank_v;

	using dimensionality_type = typename layout_t::dimensionality_type;
	using layout_t::dimensionality;

	using layout_t::stride;
	using typename layout_t::stride_type;

	using layout_t::num_elements;
	using layout_t::offset;

	using typename layout_t::index;
	using typename layout_t::index_extension;
	using typename layout_t::index_range;

	using typename layout_t::strides_type;

	BOOST_MULTI_HD constexpr auto strides() const { return detail::convertible_tuple<decltype(layout_t::strides())>(layout_t::strides()); }

	using typename layout_t::difference_type;

	using layout_t::size;
	using typename layout_t::size_type;

	using layout_t::nelems;

	using layout_t::extension;
	using typename layout_t::extension_type;

	using layout_t::extensions;
	using typename layout_t::extensions_type;

	BOOST_MULTI_HD constexpr auto extensions() const -> extensions_type { return static_cast<layout_t const&>(*this).extensions(); }

	using layout_t::empty;
	using layout_t::is_empty;

	using layout_t::sub;

	using layout_t::sizes;
	using typename layout_t::sizes_type;

	using typename layout_t::indexes;

	[[deprecated("This is for compatiblity with Boost.MultiArray, you can use `rank` member type or `dimensionality` static member variable")]]
	static constexpr auto num_dimensions() { return dimensionality; }

	[[deprecated("This is for compatiblity with Boost.MultiArray, you can use `offsets` member function")]]
	auto index_bases() const -> std::ptrdiff_t const*;  // = delete;  this function is not implemented, it can give a linker error

	[[deprecated("This is for compatiblity with Boost.MultiArray, you can use `offsets` member function")]]
	constexpr auto shape() const { return detail::convertible_tuple<decltype(this->sizes())>(this->sizes()); }

	using layout_t::is_compact;

	friend constexpr auto                size(array_types const& self) noexcept -> size_type { return self.size(); }
	friend BOOST_MULTI_HD constexpr auto extension(array_types const& self) noexcept -> extension_type { return self.extension(); }
	friend constexpr auto                is_empty(array_types const& self) noexcept -> bool { return self.is_empty(); }
	friend constexpr auto                num_elements(array_types const& self) noexcept -> size_type { return self.num_elements(); }

	friend constexpr auto extensions(array_types const& self) noexcept -> extensions_type { return self.extensions(); }
	friend constexpr auto sizes(array_types const& self) noexcept -> sizes_type { return self.sizes(); }

	// TODO(correaa) [[deprecated("use member syntax for non-salient properties")]]
	friend constexpr auto stride(array_types const& self) noexcept -> stride_type { return self.stride(); }

	// TODO(correaa) [[deprecated("use member syntax for non-salient properties")]]
	friend constexpr auto strides(array_types const& self) noexcept /*-> strides_type*/ { return self.strides(); }

 protected:
	constexpr auto layout_mutable() -> layout_t& { return static_cast<layout_t&>(*this); }

 public:
	using value_type = typename std::conditional_t<
		(D > 1),
		array<element, D - 1, typename multi::pointer_traits<element_ptr>::default_allocator_type>,
		element>;

	using reference = typename std::conditional_t<
		(D > 1),
		subarray<element, D - 1, element_ptr>,
		typename std::iterator_traits<element_ptr>::reference>;

	using const_reference = typename std::conditional_t<
		(D > 1),
		const_subarray<element, D - 1, element_ptr>,
		typename std::iterator_traits<element_const_ptr>::reference>;

	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto base() const -> element_const_ptr { return base_; }

	BOOST_MULTI_HD constexpr auto mutable_base() const -> element_ptr { return base_; }

	BOOST_MULTI_HD constexpr auto cbase() const -> element_const_ptr { return base_; }
	BOOST_MULTI_HD constexpr auto mbase() const& -> element_ptr& { return base_; }

	BOOST_MULTI_HD constexpr auto layout() const -> layout_t const& { return *this; }
	friend constexpr auto         layout(array_types const& self) -> layout_t const& { return self.layout(); }

	BOOST_MULTI_IGNORED_UNSAFE_BUFFER_USAGE_PUSH()
	// [[clang::unsafe_buffer_usage]]
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto origin() const& -> decltype(auto) { return base_ + Layout::origin(); }  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	BOOST_MULTI_IGNORED_UNSAFE_BUFFER_USAGE_POP()

	friend constexpr auto origin(array_types const& self) -> decltype(auto) { return self.origin(); }

 protected:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820)  // warning C4820:  '7' bytes padding added after data member 'boost::multi::array_types<T,2,ElementPtr,Layout>::base_' [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\test\array_fancyref.cpp.x.vcxproj]
#endif
	BOOST_MULTI_NO_UNIQUE_ADDRESS
	element_ptr base_;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes,misc-non-private-member-variables-in-classes) : TODO(correaa) try to make it private, [static_]array needs mutation
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	template<class, ::boost::multi::dimensionality_type, typename, bool, bool, typename, class> friend struct array_iterator;

	using derived = subarray<T, D, ElementPtr, Layout>;
	BOOST_MULTI_HD constexpr explicit array_types(std::nullptr_t) : Layout{}, base_(nullptr) {}

 public:
	array_types() = default;  // cppcheck-suppress uninitMemberVar ; base_ not initialized

	// BOOST_MULTI_HD constexpr array_types(layout_t const& lyt, element_ptr const& data)
	// : Layout{lyt}, base_{data} {}
	BOOST_MULTI_HD constexpr array_types(layout_t const& lyt, element_ptr data)
	: Layout{lyt}, base_{std::move(data)} {}

 protected:
	template<
		class ArrayTypes,
		typename = std::enable_if_t<!std::is_base_of<array_types, std::decay_t<ArrayTypes>>{}>, decltype(multi::detail::explicit_cast<element_ptr>(std::declval<ArrayTypes const&>().base_))* = nullptr>
	// underlying pointers are explicitly convertible
	BOOST_MULTI_HD constexpr explicit array_types(ArrayTypes const& other)
	: Layout{other.layout()}, base_{other.base_} {}

	template<
		class ArrayTypes,
		typename                                                                                      = std::enable_if_t<!std::is_base_of<array_types, std::decay_t<ArrayTypes>>{}>,
		decltype(multi::detail::implicit_cast<element_ptr>(std::declval<ArrayTypes const&>().base_))* = nullptr>
	// cppcheck-suppress noExplicitConstructor ; because underlying pointers are implicitly convertible
	BOOST_MULTI_HD constexpr /*implt*/ array_types(ArrayTypes const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : inherit behavior of underlying pointer
	: Layout{other.layout()}, base_{other.base_} {}

	template<
		typename ElementPtr2,
		typename = decltype(Layout{std::declval<array_types<T, D, ElementPtr2, Layout> const&>().layout()}),
		typename = decltype(element_ptr{std::declval<array_types<T, D, ElementPtr2, Layout> const&>().base_})>
	BOOST_MULTI_HD constexpr explicit array_types(array_types<T, D, ElementPtr2, Layout> const& other)
	: Layout{other.layout()}, base_{other.base_} {}

	template<class T2, ::boost::multi::dimensionality_type D2, class E2, class L2> friend struct array_types;
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<typename T, multi::dimensionality_type D, typename ElementPtr, class Layout, bool IsConst>
struct subarray_ptr;

template<typename T, multi::dimensionality_type D, typename ElementPtr = T*, class Layout = multi::layout_t<D>>
using const_subarray_ptr = subarray_ptr<T, D, ElementPtr, Layout, true>;

template<typename T, multi::dimensionality_type D, typename ElementPtr = T*, class Layout = multi::layout_t<D>, bool IsConst = false>
struct subarray_ptr  // NOLINT(fuchsia-multiple-inheritance) : to allow mixin CRTP
: boost::multi::iterator_facade<
	  subarray_ptr<T, D, ElementPtr, Layout, IsConst>, void, std::random_access_iterator_tag,
	  subarray<T, D, ElementPtr, Layout> const&, typename Layout::difference_type> {

 private:
	Layout layout_;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820)  //'boost::multi::subarray_ptr<double,1,fancy::ptr<double>,boost::multi::layout_t<1,boost::multi::size_type>,true>': '7' bytes padding added after data member 'boost::multi::subarray_ptr<double,1,fancy::ptr<double>,boost::multi::layout_t<1,boost::multi::size_type>,true>::base_'
#endif

	ElementPtr                                                 base_;
	typename std::iterator_traits<ElementPtr>::difference_type offset_;  // = []() { assert(0); return 0; } ();

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

 public:
	template<typename, multi::dimensionality_type, typename, class, bool> friend struct subarray_ptr;
	template<typename, multi::dimensionality_type, typename, bool, bool, typename, class> friend struct array_iterator;

	// ~subarray_ptr() = default;  // lints(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)

	using pointer         = subarray<T, D, ElementPtr, Layout>*;
	using element_type    = typename subarray<T, D, ElementPtr, Layout>::decay_type;
	using difference_type = typename Layout::difference_type;

	using value_type = element_type;

	using reference = std::conditional_t<
		IsConst,
		const_subarray<T, D, ElementPtr, Layout>,
		subarray<T, D, ElementPtr, Layout>>;

	using iterator_category = std::random_access_iterator_tag;

	// cppcheck-suppress noExplicitConstructor
	BOOST_MULTI_HD constexpr subarray_ptr(std::nullptr_t nil) : layout_{}, base_{nil}, offset_{0} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) terse syntax and functionality by default

	subarray_ptr() = default;  // cppcheck-suppress uninitMemberVar ; base_ is not initialized

	template<typename, multi::dimensionality_type, typename, class, bool> friend struct subarray_ptr;

	BOOST_MULTI_HD constexpr subarray_ptr(typename reference::element_ptr base, layout_t<typename reference::rank{} - 1> lyt) : layout_{lyt}, base_{base}, offset_{0} {}

	template<bool OtherIsConst, std::enable_if_t<!OtherIsConst, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	// cppcheck-suppress noExplicitConstructor ; see below
	BOOST_MULTI_HD constexpr /*mplct*/ subarray_ptr(subarray_ptr<T, D, ElementPtr, Layout, OtherIsConst> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : propagate implicitness of pointer
	: layout_{other.layout_}, base_{other.base_}, offset_{other.offset_} {}

	template<
		typename OtherT, multi::dimensionality_type OtherD, typename OtherEPtr, class OtherLayout, bool OtherIsConst,
		decltype(multi::detail::implicit_cast<typename reference::element_ptr>(std::declval<OtherEPtr>()))* = nullptr  // propagate implicitness of pointer
		>
	// cppcheck-suppress noExplicitConstructor ; because underlying pointer is implicitly convertible
	BOOST_MULTI_HD constexpr /*mplct*/ subarray_ptr(subarray_ptr<OtherT, OtherD, OtherEPtr, OtherLayout, OtherIsConst> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: layout_{other.layout_}, base_{other.base_} {}

	template<
		class ElementPtr2,
		std::enable_if_t<std::is_same_v<ElementPtr2, ElementPtr> && (D == 0), int> = 0  // NOLINT(modernize-use-constraints) for C++20
		>
	BOOST_MULTI_HD constexpr explicit subarray_ptr(ElementPtr2 const& other) : layout_{}, base_{other} {}

	BOOST_MULTI_HD constexpr explicit operator bool() const { return static_cast<bool>(base()); }

	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto operator*() const -> reference { return reference(layout_, base_); }

	BOOST_MULTI_HD constexpr auto operator->() const {
		class proxy {
			reference ref_;

		 public:
			BOOST_MULTI_HD constexpr explicit proxy(reference&& ref) : ref_{std::move(ref)} {}
			BOOST_MULTI_HD constexpr auto operator->() && -> reference* { return std::addressof(this->ref_); }
		};
		return proxy{operator*()};
	}

	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> reference { return *(*this + n); }

	BOOST_MULTI_HD constexpr auto operator<(subarray_ptr const& other) const -> bool { return distance_to(other) > 0; }

	BOOST_MULTI_HD constexpr subarray_ptr(typename reference::element_ptr base, Layout const& lyt) : layout_{lyt}, base_{std::move(base)} {}

	template<typename, multi::dimensionality_type, typename, class> friend struct const_subarray;

	BOOST_MULTI_HD constexpr auto base() const -> typename reference::element_ptr { return base_; }

	friend BOOST_MULTI_HD constexpr auto base(subarray_ptr const& self) { return self.base(); }

	template<class OtherSubarrayPtr, std::enable_if_t<!std::is_base_of_v<subarray_ptr, OtherSubarrayPtr>, int> = 0>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	constexpr auto operator==(OtherSubarrayPtr const& other) const
		-> decltype((base_ == other.base_) && (layout_ == other.layout_)) {
		return (base_ == other.base_) && (layout_ == other.layout_);
	}

	template<class OtherSubarrayPtr, std::enable_if_t<!std::is_base_of_v<subarray_ptr, OtherSubarrayPtr>, int> = 0>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	constexpr auto operator!=(OtherSubarrayPtr const& other) const
		-> decltype((base_ != other.base_) || (layout_ != other.layout_)) {
		return (base_ != other.base_) || (layout_ != other.layout_);
	}

	constexpr auto operator==(subarray_ptr const& other) const -> bool {
		return (base_ == other.base_) && (layout_ == other.layout_);
	}

	constexpr auto operator!=(subarray_ptr const& other) const -> bool {
		return (base_ != other.base_) || (layout_ != other.layout_);
	}

	template<
		typename OtherT, multi::dimensionality_type OtherD, typename OtherEPtr, class OtherL, bool OtherIsConst,
		std::enable_if_t<!std::is_base_of_v<subarray_ptr, subarray_ptr<OtherT, OtherD, OtherEPtr, OtherL, OtherIsConst>>, int> = 0  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
		>
	friend BOOST_MULTI_HD constexpr auto operator==(subarray_ptr const& self, subarray_ptr<OtherT, OtherD, OtherEPtr, OtherL, OtherIsConst> const& other) -> bool {
		BOOST_MULTI_ASSERT((!self || !other) || (self->layout() == other->layout()));  // comparing array ptrs of different provenance is undefined
		return self->base() == other->base();
	}

	template<
		typename OtherT, multi::dimensionality_type OtherD, typename OtherEPtr, class OtherL, bool OtherIsConst,
		std::enable_if_t<!std::is_base_of_v<subarray_ptr, subarray_ptr<OtherT, OtherD, OtherEPtr, OtherL, OtherIsConst>>, int> = 0  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
		>
	friend BOOST_MULTI_HD constexpr auto operator!=(subarray_ptr const& self, subarray_ptr<OtherT, OtherD, OtherEPtr, OtherL, OtherIsConst> const& other) -> bool {
		BOOST_MULTI_ASSERT((!self || !other) || (self->layout() == other->layout()));  // comparing array ptrs of different provenance is undefined
		return self->base() != other->base();
	}

 protected:
	BOOST_MULTI_HD constexpr void increment() { base_ += layout_.nelems(); }
	BOOST_MULTI_HD constexpr void decrement() { base_ -= layout_.nelems(); }

	BOOST_MULTI_HD constexpr void advance(difference_type n) { base_ += layout_.nelems() * n; }
	BOOST_MULTI_HD constexpr auto distance_to(subarray_ptr const& other) const -> difference_type {
		BOOST_MULTI_ASSERT(layout_.nelems() == other.layout_.nelems());
		// assert( Ref::nelems() == other.Ref::nelems() && Ref::nelems() != 0 );
		// assert( (other.base() - base())%Ref::nelems() == 0);
		BOOST_MULTI_ASSERT(layout_ == other.layout_);
		return (other.base_ - base_) / layout_.nelems();
	}

 public:
	BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> subarray_ptr& {
		advance(n);
		return *this;
	}
};

template<class Element, dimensionality_type D, typename ElementPtr, bool IsConst = false, bool IsMove = false, typename Stride = typename std::iterator_traits<ElementPtr>::difference_type, class SubLayout = layout_t<D - 1>>
struct array_iterator;

template<class Element, ::boost::multi::dimensionality_type D, typename ElementPtr, bool IsConst, bool IsMove, typename Stride, class SubLayout>
struct array_iterator  // NOLINT(fuchsia-multiple-inheritance) for facades
: boost::multi::iterator_facade<
	  array_iterator<Element, D, ElementPtr, IsConst, IsMove, Stride>, void, std::random_access_iterator_tag,
	  subarray<Element, D - 1, ElementPtr> const&, typename layout_t<D - 1>::difference_type>
, multi::decrementable<array_iterator<Element, D, ElementPtr, IsConst, IsMove, Stride>>
, multi::incrementable<array_iterator<Element, D, ElementPtr, IsConst, IsMove, Stride>>
, multi::affine<array_iterator<Element, D, ElementPtr, IsConst, IsMove, Stride>, multi::difference_type>
, multi::totally_ordered2<array_iterator<Element, D, ElementPtr, IsConst, IsMove, Stride>, void> {
	~array_iterator() = default;  // lints(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)

	constexpr auto operator=(array_iterator&&)  // lints(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
		noexcept                                // lints(hicpp-noexcept-move,performance-noexcept-move-constructor)
		-> array_iterator& = default;

	array_iterator(array_iterator&&) noexcept  // lints(hicpp-noexcept-move,performance-noexcept-move-constructor)
		= default;                             // lints(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)

	using difference_type   = typename layout_t<D>::difference_type;
	using element           = Element;
	using element_ptr       = ElementPtr;
	using element_const_ptr = typename std::pointer_traits<ElementPtr>::template rebind<element const>;
	using value_type        = typename subarray<element, D - 1, element_ptr>::decay_type;

	using pointer   = subarray<element, D - 1, element_ptr>*;
	using reference = std::conditional_t<
		IsConst,
		const_subarray<element, D - 1, element_ptr>,
		subarray<element, D - 1, element_ptr>>;
	using const_reference = const_subarray<element, D - 1, element_ptr>;

	template<class Element2>
	using rebind = array_iterator<std::decay_t<Element2>, D, typename std::pointer_traits<ElementPtr>::template rebind<Element2>, IsConst, IsMove, Stride>;

	using iterator_category = std::random_access_iterator_tag;

	constexpr static dimensionality_type rank_v = D;

	using rank = std::integral_constant<dimensionality_type, D>;  // TODO(correaa) make rank a function for compat with mdspan?

	using ptr_type = subarray_ptr<element, D - 1, element_ptr, layout_t<D - 1>, true>;

	using stride_type = Stride;
	using layout_type = typename reference::layout_type;  // layout_t<D - 1>

	BOOST_MULTI_HD constexpr array_iterator() : ptr_{}, stride_{} {}  // = default;  // TODO(correaa) make = default, now it is not compiling

	template<class, dimensionality_type, class, bool, bool, typename, class> friend struct array_iterator;

	template<
		class EElement, typename PPtr, bool B, typename S, class L,
		decltype(multi::detail::explicit_cast<ElementPtr>(std::declval<array_iterator<EElement, D, PPtr, false, B, S, L>>().base()))* = nullptr>
	BOOST_MULTI_HD constexpr explicit array_iterator(array_iterator<EElement, D, PPtr, false, B, S, L> const& other)
	: ptr_{element_ptr{other.base()}, other.ptr_->layout()}, stride_{other.stride_} {}

	template<
		class EElement, typename PPtr, bool B, typename S, class L,
		decltype(multi::detail::implicit_cast<ElementPtr>(std::declval<array_iterator<EElement, D, PPtr, false, B, S, L>>().base()))* = nullptr>  // propagate implicitness of pointer
	// cppcheck-suppress noExplicitConstructor ; because underlying pointer is implicitly convertible
	BOOST_MULTI_HD constexpr /*mplct*/ array_iterator(array_iterator<EElement, D, PPtr, false, B, S, L> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: ptr_(other.ptr_), stride_{other.stride_} {}

	array_iterator(array_iterator const&)                    = default;
	auto operator=(array_iterator const&) -> array_iterator& = default;

	BOOST_MULTI_HD constexpr explicit operator bool() const { return ptr_->base(); }  // TODO(correaa) implement bool conversion for subarray_ptr
	BOOST_MULTI_HD constexpr auto     operator*() const -> reference { return *ptr_; }

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) can/should it be returned by reference?
#endif

	BOOST_MULTI_HD constexpr auto operator->() const -> decltype(auto) { return ptr_; }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	BOOST_MULTI_HD constexpr auto operator+(difference_type n) const -> array_iterator {
		array_iterator ret{*this};
		ret += n;
		return ret;
	}
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> subarray<element, D - 1, element_ptr> { return *((*this) + n); }

	template<bool OtherIsConst, std::enable_if_t<(IsConst != OtherIsConst), int> = 0>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	BOOST_MULTI_HD constexpr auto operator==(array_iterator<Element, D, ElementPtr, OtherIsConst> const& other) const -> bool {
		// BOOST_MULTI_ASSERT( this->stride_ == other.stride_ );
		// BOOST_MULTI_ASSERT( this->ptr_->layout() == other.ptr_->layout() );
		return (this->ptr_ == other.ptr_) && (this->stride_ == other.stride_) && ((*(this->ptr_)).layout() == (*(other.ptr_)).layout());
	}

	BOOST_MULTI_HD constexpr auto operator==(array_iterator const& other) const -> bool {
		BOOST_MULTI_ASSERT(this->stride_ == other.stride_);
		BOOST_MULTI_ASSERT(this->ptr_->layout() == other.ptr_->layout());
		return (this->ptr_ == other.ptr_);  // && (this->stride_ == other.stride_) && ( (*(this->ptr_)).layout() == (*(other.ptr_)).layout() );
	}

	BOOST_MULTI_HD constexpr auto operator!=(array_iterator const& other) const -> bool {
		return !operator==(other);
	}

	BOOST_MULTI_HD constexpr auto operator<(array_iterator const& other) const -> bool {
		return 0 < other - *this;
	}

	BOOST_MULTI_HD constexpr explicit array_iterator(typename subarray<element, D - 1, element_ptr>::element_ptr base, layout_t<D - 1> const& lyt, stride_type stride)
	: ptr_(base, lyt), stride_{stride} {}

	template<class, dimensionality_type, class, class> friend struct const_subarray;

	template<class... As>
	BOOST_MULTI_HD constexpr auto operator()(index idx, As... args) const -> decltype(auto) { return this->operator[](idx)(args...); }
	BOOST_MULTI_HD constexpr auto operator()(index idx) const -> decltype(auto) { return this->operator[](idx); }

 private:
	template<class Self, typename Tuple, std::size_t... I>
	static BOOST_MULTI_HD constexpr auto apply_impl_(Self&& self, Tuple const& tuple, std::index_sequence<I...> /*012*/) -> decltype(auto) {
		using std::get;  // for C++17 compatibility
		return std::forward<Self>(self)(get<I>(tuple)...);
	}

 public:
	template<typename Tuple> BOOST_MULTI_HD constexpr auto apply(Tuple const& tpl) const -> decltype(auto) { return apply_impl_(*this, tpl, std::make_index_sequence<std::tuple_size_v<Tuple>>()); }

 private:
	ptr_type    ptr_;
	stride_type stride_;

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	BOOST_MULTI_HD constexpr void decrement_() { ptr_.base_ -= stride_; }
	BOOST_MULTI_HD constexpr void advance_(difference_type n) { ptr_.base_ += stride_ * n; }  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

 public:
	BOOST_MULTI_HD constexpr auto base() const -> element_ptr { return ptr_.base_; }
	BOOST_MULTI_HD constexpr auto stride() const -> stride_type { return stride_; }

	friend /*constexpr*/ auto base(array_iterator const& self) -> element_ptr { return self.base(); }
	friend constexpr auto     stride(array_iterator const& self) -> stride_type { return self.stride_; }  // TODO(correaa) remove

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	constexpr auto operator++() -> array_iterator& {
		ptr_.base_ += stride_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}

	constexpr auto operator--() -> array_iterator& {
		ptr_.base_ -= stride_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

	friend constexpr auto operator-(array_iterator const& self, array_iterator const& other) -> difference_type {
		BOOST_MULTI_ASSERT(self.stride_ == other.stride_);  // LCOV_EXCL_LINE
		BOOST_MULTI_ASSERT(self.stride_ != 0);              // LCOV_EXCL_LINE
		return (self.ptr_.base() - other.ptr_.base()) / self.stride_;
	}

	constexpr auto operator+=(difference_type n) -> array_iterator& {
		advance_(+n);
		return *this;
	}

	constexpr auto operator-=(difference_type n) -> array_iterator& {
		advance_(-n);
		return *this;
	}
};

template<typename ElementPtr, dimensionality_type D, class StridesType>
struct cursor_t {
	using difference_type = typename std::iterator_traits<ElementPtr>::difference_type;
	using strides_type    = StridesType;

	using element_ptr  = ElementPtr;
	using element_ref  = typename std::iterator_traits<element_ptr>::reference;
	using element_type = typename std::iterator_traits<element_ptr>::value_type;

	using pointer   = element_ptr;
	using reference = element_ref;

	using indices_type = typename extensions_t<D>::indices_type;

	cursor_t() = default;

 private:
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820)  // '7' bytes padding added after data member 'boost::multi::array_types<T,2,ElementPtr,Layout>::base_' [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\test\array_fancyref.cpp.x.vcxproj]
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

	strides_type strides_;
	element_ptr  base_;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

	template<class, dimensionality_type, class, class> friend struct const_subarray;
	template<class, dimensionality_type, class> friend struct cursor_t;

	BOOST_MULTI_HD constexpr cursor_t(element_ptr base, strides_type const& strides) : strides_{strides}, base_{base} {}

	template<class OtherCursor, class = decltype(multi::detail::implicit_cast<element_ptr>(std::declval<OtherCursor>().base()))>
	// cppcheck-suppress noExplicitConstructor
	BOOST_MULTI_HD constexpr cursor_t(OtherCursor const& other) : strides_{other.strides()}, base_{other.base()} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	template<class OtherCursor>
	BOOST_MULTI_HD constexpr explicit cursor_t(OtherCursor const& other) : strides_{other.strides()}, base_{other.base()} {}

 public:
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> decltype(auto) {
		using std::get;  // for C++17 compatibility
		if constexpr(D != 1) {
			return cursor_t<
				ElementPtr,
				D - 1,
				std::decay_t<decltype(strides_.tail())>>{
				base_ + get<0>(strides_) * n,  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
				strides_.tail()
			};
		} else {
			return base_[get<0>(strides_) * n];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		}
	}

	BOOST_MULTI_HD constexpr auto operator()(difference_type n) const -> decltype(auto) {
		return operator[](n);
	}
	template<class... Ns>
	BOOST_MULTI_HD constexpr auto operator()(difference_type n, Ns... rest) const -> decltype(auto) {
		return operator[](n)(rest...);
	}

 private:
	template<class Tuple, std::size_t... I>
	BOOST_MULTI_HD constexpr auto apply_impl_(Tuple const& tup, std::index_sequence<I...> /*012*/) const -> decltype(auto) {
		using std::get;  // for C++17 compatibility
		return ((get<I>(tup) * get<I>(strides_)) + ...);
	}

 public:
	template<class Tuple = indices_type>
	BOOST_MULTI_HD constexpr auto operator+=(Tuple const& tup) -> cursor_t& {
		base_ += apply_impl_(tup, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
		return *this;
	}
	BOOST_MULTI_HD constexpr auto operator*() const -> reference { return *base_; }
	BOOST_MULTI_HD constexpr auto operator->() const -> pointer { return base_; }

	BOOST_MULTI_HD constexpr auto base() const -> pointer { return base_; }
	BOOST_MULTI_HD constexpr auto strides() const -> strides_type { return strides_; }
	template<multi::dimensionality_type DD = 0>
	BOOST_MULTI_HD constexpr auto stride() const {
		using std::get;
		return get<DD>(strides_);
	}
};

template<typename Pointer, class LayoutType>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
struct elements_iterator_t
// : boost::multi::random_accessable<elements_iterator_t<Pointer, LayoutType>, typename std::iterator_traits<Pointer>::difference_type, typename std::iterator_traits<Pointer>::reference>
{
	using difference_type   = typename std::iterator_traits<Pointer>::difference_type;
	using value_type        = typename std::iterator_traits<Pointer>::value_type;
	using pointer           = Pointer;
	using reference         = std::remove_const_t<typename std::iterator_traits<Pointer>::reference>;  // TODO(correaa) investigate why top-level const reaches here
	using iterator_category = std::random_access_iterator_tag;

	using const_pointer = typename std::pointer_traits<pointer>::template rebind<value_type const>;

	using layout_type = LayoutType;

 private:
	pointer                                   base_;
	layout_type                               l_;
	difference_type                           n_ = 0;
	extensions_t<layout_type::dimensionality> xs_;

	using indices_type = typename extensions_t<layout_type::dimensionality>::indices_type;
	indices_type ns_   = {};

	template<typename, class> friend struct elements_iterator_t;
	template<typename, class> friend struct elements_range_t;

	BOOST_MULTI_HD constexpr elements_iterator_t(pointer base, layout_type const& lyt, difference_type n)
	: base_{std::move(base)}, l_{lyt}, n_{n}, xs_{l_.extensions()}, ns_{lyt.is_empty() ? indices_type{} : xs_.from_linear(n)} {}

 public:
	elements_iterator_t() = default;

	BOOST_MULTI_HD constexpr auto base() -> pointer { return base_; }
	BOOST_MULTI_HD constexpr auto base() const -> const_pointer { return base_; }

	BOOST_MULTI_HD constexpr auto layout() const -> layout_type { return l_; }

	template<class Other, decltype(multi::detail::implicit_cast<pointer>(std::declval<Other>().base_))* = nullptr>
	// cppcheck-suppress noExplicitConstructor
	BOOST_MULTI_HD constexpr /*impl*/ elements_iterator_t(Other const& other) : elements_iterator_t{other.base_, other.l_, other.n_} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	template<class Other>
	BOOST_MULTI_HD constexpr explicit elements_iterator_t(Other const& other) : elements_iterator_t{other.base_, other.l_, other.n_} {}

	elements_iterator_t(elements_iterator_t const&) = default;

	BOOST_MULTI_HD constexpr auto operator=(elements_iterator_t const& other) -> elements_iterator_t& {  // fixes (?) warning: definition of implicit copy assignment operator for 'elements_iterator_t<boost::multi::array<double, 3> *, boost::multi::layout_t<1>>' is deprecated because it has a user-declared copy constructor [-Wdeprecated-copy]
		if(&other == this) {
			return *this;
		}  // for cert-oop54-cpp
		base_ = other.base_;
		xs_   = other.xs_;
		n_    = other.n_;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator++() -> elements_iterator_t& {
		apply([&xs = this->xs_](auto&... idxs) { return xs.next_canonical(idxs...); }, ns_);
		// std::apply([&xs = this->xs_](auto&... idxs) { return xs.next_canonical(idxs...); }, ns_);
		++n_;
		return *this;
	}
	BOOST_MULTI_HD constexpr auto operator--() -> elements_iterator_t& {
		std::apply([&xs = this->xs_](auto&... idxs) { return xs.prev_canonical(idxs...); }, ns_);
		--n_;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> elements_iterator_t& {
		auto const nn = apply(xs_, ns_);
		ns_           = xs_.from_linear(nn + n);
		n_ += n;
		return *this;
	}
	BOOST_MULTI_HD constexpr auto operator-=(difference_type n) -> elements_iterator_t& {
		// auto const nn = std::apply(xs_, ns_);
		// ns_ = xs_.from_linear(nn - n);
		n_ -= n;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator-(elements_iterator_t const& other) const -> difference_type {
		BOOST_MULTI_ASSERT(base_ == other.base_ && l_ == other.l_);
		return n_ - other.n_;
	}

	// BOOST_MULTI_HD constexpr auto n() const { return n_; }

	BOOST_MULTI_HD constexpr auto operator<(elements_iterator_t const& other) const -> bool {
		BOOST_MULTI_ASSERT(base_ == other.base_ && l_ == other.l_);
		return n_ < other.n_;
	}

	BOOST_MULTI_HD constexpr auto operator<=(elements_iterator_t const& other) const -> bool { return ((*this) < other) || ((*this) == other); }

	BOOST_MULTI_HD constexpr auto operator>(elements_iterator_t const& other) const -> bool { return other < (*this); }
	BOOST_MULTI_HD constexpr auto operator>=(elements_iterator_t const& other) const -> bool { return !((*this) < other); }

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	BOOST_MULTI_HD constexpr auto current() const -> pointer { return base_ + std::apply(l_, ns_); }

	// BOOST_MULTI_HD constexpr auto operator->() const -> pointer { return base_ + std::apply(l_, ns_); }

	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto operator*() const -> reference /*decltype(base_[0])*/ {
		return base_[apply(l_, ns_)];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}

	BOOST_MULTI_HD constexpr auto operator[](difference_type const& n) const -> reference {
		auto const nn = apply(xs_, ns_);
		return base_[apply(l_, xs_.from_linear(nn + n))];
	}  // explicit here is necessary for nvcc/thrust

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

	BOOST_MULTI_HD constexpr auto operator+(difference_type n) const -> elements_iterator_t {
		auto ret{*this};
		ret += n;
		return ret;
	}
	BOOST_MULTI_HD constexpr auto operator-(difference_type n) const -> elements_iterator_t {
		auto ret{*this};
		ret -= n;
		return ret;
	}

	BOOST_MULTI_HD constexpr auto operator==(elements_iterator_t const& other) const -> bool {
		BOOST_MULTI_ASSERT(base_ == other.base_ && l_ == other.l_);  // TODO(correaa) calling host function from host device
		return n_ == other.n_;                                       // and base_ == other.base_ and l_ == other.l_;
	}
	BOOST_MULTI_HD constexpr auto operator!=(elements_iterator_t const& other) const -> bool {
		BOOST_MULTI_ASSERT(base_ == other.base_ && l_ == other.l_);  // TODO(correaa) calling host function from host device
		return n_ != other.n_;
	}
};

template<typename Pointer, class LayoutType>
struct elements_range_t {
	using pointer     = Pointer;
	using layout_type = LayoutType;

	using value_type    = typename std::iterator_traits<pointer>::value_type;
	using const_pointer = typename std::pointer_traits<pointer>::template rebind<value_type const>;

	using reference       = typename std::iterator_traits<pointer>::reference;
	using const_reference = typename std::iterator_traits<const_pointer>::reference;

	using size_type       = typename std::iterator_traits<pointer>::difference_type;
	using difference_type = typename std::iterator_traits<pointer>::difference_type;

	using iterator       = elements_iterator_t<pointer, layout_type>;
	using const_iterator = elements_iterator_t<const_pointer, layout_type>;

	using element = value_type;

 private:
	pointer     base_;
	layout_type l_;

 public:
	template<class OtherRange, decltype(multi::detail::implicit_cast<pointer>(std::declval<OtherRange>().base_))* = nullptr>
	// cppcheck-suppress noExplicitConstructor ; because underlying pointer is implicitly convertible  // NOLINTNEXTLINE(runtime/explicit)
	constexpr /*impl*/ elements_range_t(OtherRange const& other) : base_{other.base}, l_{other.l_} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) to reproduce the implicitness of the argument
	template<class OtherRange, decltype(multi::detail::explicit_cast<pointer>(std::declval<OtherRange>().base_))* = nullptr>
	constexpr explicit elements_range_t(OtherRange const& other) : elements_range_t{other} {}

	constexpr elements_range_t(pointer base, layout_type const& lyt) : base_{std::move(base)}, l_{lyt} {}

	constexpr auto base() -> pointer { return base_; }
	constexpr auto base() const -> const_pointer { return base_; }

	constexpr auto layout() const -> layout_type { return l_; }

 private:
#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	constexpr auto at_aux_(difference_type n) const -> reference {
		BOOST_MULTI_ASSERT(!is_empty());
		return base_[std::apply(l_, l_.extensions().from_linear(n))];  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	}

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

 public:
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const& -> const_reference { return at_aux_(n); }
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) && -> reference { return at_aux_(n); }
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) & -> reference { return at_aux_(n); }

	constexpr auto size() const -> size_type { return l_.num_elements(); }

	using extension_type = multi::extension_t<index>;

	BOOST_MULTI_HD constexpr auto extension() const { return extension_type{0, size()}; }

	[[nodiscard]]
	constexpr auto empty() const -> bool { return l_.empty(); }
	constexpr auto is_empty() const -> bool { return l_.is_empty(); }

	elements_range_t(elements_range_t const&) = delete;
	elements_range_t(elements_range_t&&)      = delete;

	// template<typename OP, class OL> auto operator==(elements_range_t<OP, OL> const& other) const -> bool {
	// 	return size() == other.size() && adl_equal(other.begin(), other.end(), begin());  // mull-ignore: cxx_eq_to_ne  // false positive bug in mull-18
	// }
	// template<typename OP, class OL> auto operator!=(elements_range_t<OP, OL> const& other) const -> bool {
	// 	// if(is_empty() && other.is_empty()) { return false; }
	// 	return size() != other.size() || !adl_equal(other.begin(), other.end(), begin());
	// }

	template<class Range> auto operator==(Range const& other) const -> bool {
		return size() == other.size() && adl_equal(other.begin(), other.end(), begin());  // mull-ignore: cxx_eq_to_ne  // false positive bug in mull-18
	}
	template<class Range> auto operator!=(Range const& other) const -> bool {
		return size() != other.size() || !adl_equal(other.begin(), other.end(), begin());
	}

	template<typename OP, class OL> void swap(elements_range_t<OP, OL>& other) & noexcept {
		BOOST_MULTI_ASSERT(size() == other.size());
		adl_swap_ranges(begin(), end(), other.begin());
	}
	template<typename OP, class OL> void swap(elements_range_t<OP, OL>& other) && noexcept {
		BOOST_MULTI_ASSERT(size() == other.size());
		adl_swap_ranges(begin(), end(), other.begin());
	}
	template<typename OP, class OL> void swap(elements_range_t<OP, OL>&& other) & noexcept {
		BOOST_MULTI_ASSERT(size() == other.size());
		adl_swap_ranges(begin(), end(), std::move(other).begin());
	}
	template<typename OP, class OL> void swap(elements_range_t<OP, OL>&& other) && noexcept {
		BOOST_MULTI_ASSERT(size() == other.size());
		adl_swap_ranges(begin(), end(), std::move(other).begin());
	}

	~elements_range_t() = default;

 private:
	BOOST_MULTI_HD constexpr auto begin_aux_() const { return iterator{base_, l_, 0}; }
	BOOST_MULTI_HD constexpr auto end_aux_() const { return iterator{base_, l_, l_.num_elements()}; }

 public:
	BOOST_MULTI_HD constexpr auto begin() const& -> const_iterator { return begin_aux_(); }
	BOOST_MULTI_HD constexpr auto end() const& -> const_iterator { return end_aux_(); }

	BOOST_MULTI_HD constexpr auto begin() && -> iterator { return begin_aux_(); }
	BOOST_MULTI_HD constexpr auto end() && -> iterator { return end_aux_(); }

	BOOST_MULTI_HD constexpr auto begin() & -> iterator { return begin_aux_(); }
	BOOST_MULTI_HD constexpr auto end() & -> iterator { return end_aux_(); }

	BOOST_MULTI_HD constexpr auto front() const& -> const_reference { return *begin(); }
	BOOST_MULTI_HD constexpr auto back() const& -> const_reference { return *std::prev(end(), 1); }

	BOOST_MULTI_HD constexpr auto front() && -> reference { return *begin(); }
	BOOST_MULTI_HD constexpr auto back() && -> reference { return *std::prev(end(), 1); }

	BOOST_MULTI_HD constexpr auto front() & -> reference { return *begin(); }
	BOOST_MULTI_HD constexpr auto back() & -> reference { return *std::prev(end(), 1); }

	auto operator=(elements_range_t const&) -> elements_range_t& = delete;

	auto operator=(elements_range_t&& other) noexcept -> elements_range_t& {  // cannot be =delete in NVCC?
		if(!is_empty()) {
			adl_copy(other.begin(), other.end(), this->begin());
		}
		return *this;
	}

	template<class OtherElementRange, class = decltype(adl_copy(std::begin(std::declval<OtherElementRange&&>()), std::end(std::declval<OtherElementRange&&>()), std::declval<iterator>()))>
	auto operator=(OtherElementRange&& other) & -> elements_range_t& {  // NOLINT(cppcoreguidelines-missing-std-forward) std::forward<OtherElementRange>(other) creates a problem with move-only elements
		BOOST_MULTI_ASSERT(size() == other.size());
		if(!is_empty()) {
			adl_copy(std::begin(other), std::end(other), begin());
		}
		return *this;
	}

	template<class OtherElementRange, class = decltype(adl_copy(std::begin(std::declval<OtherElementRange&&>()), std::end(std::declval<OtherElementRange&&>()), std::declval<iterator>()))>
	constexpr auto operator=(OtherElementRange&& other) && -> elements_range_t& {  // NOLINT(cppcoreguidelines-missing-std-forward) std::forward<OtherElementRange>(other) creates a problem with move-only elements
		BOOST_MULTI_ASSERT(size() == other.size());
		if(!is_empty()) {
			adl_copy(std::begin(other), std::end(other), begin());
		}
		return *this;
	}

	auto operator=(std::initializer_list<value_type> values) && -> elements_range_t& {
		operator=(values);
		return *this;
	}
	auto operator=(std::initializer_list<value_type> values) & -> elements_range_t& {
		BOOST_MULTI_ASSERT(static_cast<size_type>(values.size()) == size());
		adl_copy_n(values.begin(), values.size(), begin());
		return *this;
	}
};

template<class It>
[[deprecated("remove")]] BOOST_MULTI_HD constexpr auto ref(It begin, It end)
	-> multi::subarray<typename It::element, It::rank_v, typename It::element_ptr> {
	return multi::subarray<typename It::element, It::rank_v, typename It::element_ptr>{begin, end};
}

template<typename, ::boost::multi::dimensionality_type, class Alloc> struct dynamic_array;  // this might be needed by MSVC 14.3 in c++17 mode

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<typename T, ::boost::multi::dimensionality_type D, typename ElementPtr, class Layout>
struct const_subarray : array_types<T, D, ElementPtr, Layout> {
	using types = array_types<T, D, ElementPtr, Layout>;
	using ref_  = const_subarray;

	using array_types<T, D, ElementPtr, Layout>::rank_v;

	friend struct const_subarray<typename types::element, D + 1, typename types::element_ptr>;

	using typename types::element_type;
	using types::layout;

	using layout_type = Layout;

	// cppcheck-suppress-begin duplInheritedMember ; TODO(correaa) eliminate array_types base
	BOOST_MULTI_HD constexpr auto layout() const -> decltype(auto) { return array_types<T, D, ElementPtr, Layout>::layout(); }

	using basic_const_array = subarray<T, D, typename std::pointer_traits<ElementPtr>::template rebind<element_type const>, Layout>;

	const_subarray()                                         = default;
	auto operator=(const_subarray const&) -> const_subarray& = delete;
	auto operator=(const_subarray&&) -> const_subarray&      = delete;

	BOOST_MULTI_HD constexpr const_subarray(layout_type const& layout, ElementPtr const& base)
	: array_types<T, D, ElementPtr, Layout>{layout, base} {}

 protected:
	// using types::types;
	BOOST_MULTI_HD constexpr explicit const_subarray(std::nullptr_t nil) : types{nil} {}

	template<typename, ::boost::multi::dimensionality_type, class Alloc> friend struct dynamic_array;

	template<typename, multi::dimensionality_type, typename, class, bool> friend struct subarray_ptr;

	// TODO(correaa) vvv consider making it explicit (seems that in C++23 it can prevent auto s = a[0];)
	// const_subarray(const_subarray const&) = default;  // NOTE: reference type cannot be copied. perhaps you want to return by std::move or std::forward if you got the object from a universal reference argument

 public:
	const_subarray(const_subarray const&) = delete;

	using element           = typename types::element;
	using element_ptr       = typename types::element_ptr;
	using element_const_ptr = typename types::element_const_ptr;
	using element_ref       = typename types::element_ref;
	using element_cref      = typename std::iterator_traits<element_const_ptr>::reference;

	using elements_iterator  = elements_iterator_t<element_ptr, layout_type>;
	using celements_iterator = elements_iterator_t<element_const_ptr, layout_type>;

	using elements_range       = elements_range_t<element_ptr, layout_type>;
	using const_elements_range = elements_range_t<element_const_ptr, layout_type>;

	using index_gen [[deprecated("here to fulfill MultiArray concept")]]    = char*;
	using extent_gen [[deprecated("here to fulfill MultiArray concept")]]   = void;
	using extent_range [[deprecated("here to fulfill MultiArray concept")]] = void;

 private:
	constexpr auto elements_aux_() const { return elements_range(this->base_, this->layout()); }

 public:
	const_subarray(const_subarray&&) noexcept = default;  // lints(readability-redundant-access-specifiers)

	constexpr auto elements() const& { return const_elements_range(this->base(), this->layout()); }
	constexpr auto const_elements() const -> const_elements_range { return elements_aux_(); }

	constexpr auto hull() const -> std::pair<element_const_ptr, size_type> {
		return {this->base(), std::abs(this->hull_size())};
	}

	~const_subarray() = default;  // this lints(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)

	BOOST_MULTI_FRIEND_CONSTEXPR auto sizes(const_subarray const& self) noexcept -> typename const_subarray::sizes_type { return self.sizes(); }  // needed by nvcc
	BOOST_MULTI_FRIEND_CONSTEXPR auto size(const_subarray const& self) noexcept -> typename const_subarray::size_type { return self.size(); }     // needed by nvcc

	//  template<class T2> friend constexpr auto reinterpret_array_cast(const_subarray     && self) {return std::move(self).template reinterpret_array_cast<T2, typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2>>();}
	//  template<class T2> friend constexpr auto reinterpret_array_cast(const_subarray const& self) {return           self .template reinterpret_array_cast<T2, typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2>>();}

	friend constexpr auto dimensionality(const_subarray const& /*self*/) { return D; }

	// using typename types::reference;

	using default_allocator_type = typename multi::pointer_traits<const_subarray::element_ptr>::default_allocator_type;

	BOOST_MULTI_HD constexpr auto get_allocator() const -> default_allocator_type {
		using multi::get_allocator;
		return get_allocator(this->base());
	}

	BOOST_MULTI_FRIEND_CONSTEXPR auto get_allocator(const_subarray const& self) -> default_allocator_type { return self.get_allocator(); }

	using decay_type = array<typename types::element_type, D, typename multi::pointer_traits<typename const_subarray::element_ptr>::default_allocator_type>;

	friend constexpr auto decay(const_subarray const& self) -> decay_type { return self.decay(); }
	constexpr auto        decay() const& -> decay_type {
        decay_type ret{*this};
        return ret;
	}

	constexpr auto operator+() const -> decay_type { return decay(); }
	// using typename types::const_reference;

	using reference = typename std::conditional_t<
		(D > 1),
		const_subarray<element, D - 1, element_ptr>,
		typename std::iterator_traits<element_ptr>::reference>;

	using const_reference = typename std::conditional_t<
		(D > 1),
		const_subarray<element, D - 1, element_ptr>,
		typename std::iterator_traits<element_const_ptr>::reference>;

 private:
	template<typename, multi::dimensionality_type, typename, class> friend class subarray;

	BOOST_MULTI_HD constexpr auto at_aux_(index idx) const {
		BOOST_MULTI_ASSERT((this->stride() == 0 || (this->extension().contains(idx))) && ("out of bounds"));

		// clang-format off
	#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
	#endif
		return reference(
			this->layout().sub(),
			this->base_ + (idx * this->layout().stride() - this->layout().offset())  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		);  // cppcheck-suppress syntaxError ; bug in cppcheck 2.5
#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif
	}

 public:
	// clang-format off
	#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
	#endif
	// clang-format on

	BOOST_MULTI_HD constexpr auto operator[](index idx) const& -> const_reference {
		BOOST_MULTI_ASSERT((this->stride() == 0 || (this->extension().contains(idx))) && ("out of bounds"));  // N_O_L_I_N_T(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		return const_reference(
			this->layout().sub(),
			this->base_ + (idx * this->layout().stride() - this->layout().offset())  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		);                                                                           // cppcheck-suppress syntaxError ; bug in cppcheck 2.5
																					 // return at_aux_(idx);  // TODO(correaa) use at_aux
	}  // TODO(correaa) use return type to cast

	// clang-format off
	#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
	#pragma clang diagnostic pop
	#endif
	// clang-format on

	// template<class Tuple = std::array<index, static_cast<std::size_t>(D)>,
	// 		 typename    = std::enable_if_t<(std::tuple_size<Tuple>::value > 1)>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	// 		 >
	// BOOST_MULTI_HD constexpr auto operator[](Tuple const& tup) const
	// 	-> decltype(operator[](detail::head(tup))[detail::tuple_tail(tup)]) {
	// 	return operator[](detail::head(tup))[detail::tuple_tail(tup)];
	// }

	// template<class Tuple, typename = std::enable_if_t<(std::tuple_size<Tuple>::value == 1)>>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// BOOST_MULTI_HD constexpr auto operator[](Tuple const& tup) const
	// 	-> decltype(operator[](detail::head(tup))) {
	// 	return operator[](detail::head(tup));
	// }

	constexpr auto front() const& -> const_reference { return *begin(); }
	constexpr auto back() const& -> const_reference { return *(end() - 1); }  // std::prev(end(), 1);}

	using typename types::index;

	constexpr auto reindexed(index first) const& {
		// typename types::layout_t new_layout = this->layout();
		// new_layout.reindex(first);
		return const_subarray(this->layout().reindex(first), types::base_);
	}
	constexpr auto reindexed(index first) & {
		return const_subarray(this->layout().reindex(first), types::base_);
	}
	constexpr auto reindexed(index first) && { return const_subarray(this->layout().reindex(first), types::base_); }

	// TODO(correaa) : implement reindexed_aux
	template<class... Indexes>
	constexpr auto reindexed(index first, Indexes... idxs) const& -> const_subarray {
		return ((reindexed(first).rotated()).reindexed(idxs...)).unrotated();
	}

 private:
	constexpr auto taked_aux_(difference_type n) const {
		BOOST_MULTI_ASSERT(n <= this->size());
		return const_subarray(this->layout().take(n), this->base_);
	}

 public:
	constexpr auto taked(difference_type n) const& -> basic_const_array { return taked_aux_(n); }

 private:
	BOOST_MULTI_HD constexpr auto halved_aux_() const {
		auto new_layout = this->layout().halve();
		return subarray<T, D + 1, element_ptr>(new_layout, this->base_);
	}

 public:
	BOOST_MULTI_HD constexpr auto halved() const& -> const_subarray<T, D + 1, element_ptr> { return halved_aux_(); }

 private:
	constexpr auto dropped_aux_(difference_type n) const {
		BOOST_MULTI_ASSERT(n <= this->size());
		typename types::layout_t const new_layout{
			this->layout().sub(),
			this->layout().stride(),
			this->layout().offset(),
			this->stride() * (this->size() - n)
		};

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

		return const_subarray(new_layout, this->base_ + n * this->layout().stride() /*- this->layout().offset()*/);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif
	}

 public:
	constexpr auto dropped(difference_type n) const& -> basic_const_array { return dropped_aux_(n); }
	constexpr auto dropped(difference_type n) && -> const_subarray { return dropped_aux_(n); }
	constexpr auto dropped(difference_type n) & -> const_subarray { return dropped_aux_(n); }

 private:
	BOOST_MULTI_HD constexpr auto sliced_aux_(index first, index last) const {
		// TODO(correaa) remove first == last condition
		BOOST_MULTI_ASSERT(((first == last) || this->extension().contains(first)) && ("sliced first out of bounds"));     // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		BOOST_MULTI_ASSERT(((first == last) || this->extension().contains(last - 1)) && ("sliced last  out of bounds"));  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		typename types::layout_t new_layout = this->layout();
		new_layout.nelems()                 = this->stride() * (last - first);                                  // TODO(correaa) : reconstruct layout instead of mutating it
		BOOST_MULTI_ASSERT(this->base_ || ((first * this->layout().stride() - this->layout().offset()) == 0));  // it is UB to offset a nullptr

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

		return const_subarray(new_layout, this->base_ + (first * this->layout().stride() - this->layout().offset()));  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#ifdef __clang__
#pragma clang diagnostic pop
#endif
	}

 public:
	BOOST_MULTI_HD constexpr auto sliced(index first, index last) const& -> const_subarray { return sliced_aux_(first, last); }

	constexpr auto blocked(index first, index last) const& -> basic_const_array { return sliced(first, last).reindexed(first); }
	constexpr auto blocked(index first, index last) & -> const_subarray { return sliced(first, last).reindexed(first); }

	using iextension = typename const_subarray::index_extension;

	constexpr auto stenciled(iextension iex) & -> const_subarray { return blocked(iex.first(), iex.last()); }
	constexpr auto stenciled(iextension iex, iextension iex1) & -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1)).unrotated(); }
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2) & -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1, iex2)).unrotated(); }
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2, iextension iex3) & -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1, iex2, iex3)).unrotated(); }
	template<class... Xs>
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2, iextension iex3, Xs... iexs) & -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1, iex2, iex3, iexs...)).unrotated(); }

	constexpr auto stenciled(iextension iex) && -> const_subarray { return blocked(iex.first(), iex.last()); }
	constexpr auto stenciled(iextension iex, iextension iex1) && -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1)).unrotated(); }
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2) && -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1, iex2)).unrotated(); }
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2, iextension iex3) && -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1, iex2, iex3)).unrotated(); }
	template<class... Xs>
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2, iextension iex3, Xs... iexs) && -> const_subarray { return ((stenciled(iex).rotated()).stenciled(iex1, iex2, iex3, iexs...)).unrotated(); }

	constexpr auto stenciled(iextension iex) const& -> basic_const_array { return blocked(iex.first(), iex.last()); }
	constexpr auto stenciled(iextension iex, iextension iex1) const& -> basic_const_array { return ((stenciled(iex).rotated()).stenciled(iex1)).unrotated(); }
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2) const& -> basic_const_array { return ((stenciled(iex).rotated()).stenciled(iex1, iex2)).unrotated(); }
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2, iextension iex3) const& -> basic_const_array { return ((stenciled(iex).rotated()).stenciled(iex1, iex2, iex3)).unrotated(); }

	template<class... Xs>
	constexpr auto stenciled(iextension iex, iextension iex1, iextension iex2, iextension iex3, Xs... iexs) const& -> basic_const_array {
		return ((stenciled(iex).rotated()).stenciled(iex1, iex2, iex3, iexs...)).unrotated();
	}

	constexpr auto elements_at(size_type idx) const& -> decltype(auto) {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		auto const sub_num_elements = this->begin()->num_elements();
		return operator[](idx / sub_num_elements).elements_at(idx % sub_num_elements);
	}
	constexpr auto elements_at(size_type idx) && -> decltype(auto) {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		auto const sub_num_elements = this->begin()->num_elements();
		return operator[](idx / sub_num_elements).elements_at(idx % sub_num_elements);
	}
	constexpr auto elements_at(size_type idx) & -> decltype(auto) {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		auto const sub_num_elements = this->begin()->num_elements();
		return operator[](idx / sub_num_elements).elements_at(idx % sub_num_elements);
	}

 private:
	constexpr auto strided_aux_(difference_type diff) const {
		// auto new_layout = this->layout().do_stride();
		typename types::layout_t const new_layout{this->layout().sub(), this->layout().stride() * diff, this->layout().offset(), this->layout().nelems()};
		// template<typename T, ::boost::multi::dimensionality_type D, typename ElementPtr, class Layout>
		return subarray<T, D, ElementPtr, typename types::layout_t>(new_layout, types::base_);
	}

 public:
	constexpr auto strided(difference_type diff) const& { return strided_aux_(diff).as_const(); }
	// constexpr auto strided(difference_type diff) && -> const_subarray { return strided_aux_(diff); }
	// constexpr auto strided(difference_type diff) & -> const_subarray { return strided_aux_(diff); }

	constexpr auto sliced(
		typename types::index first, typename types::index last, typename types::index stride_
	) const& -> const_subarray {
		return sliced(first, last).strided(stride_);
	}

	using index_range = typename const_subarray::index_range;

	BOOST_MULTI_HD constexpr auto range(index_range irng) const& -> decltype(auto) { return sliced(irng.front(), irng.front() + irng.size()); }
	// constexpr auto range(index_range irng)     && -> decltype(auto) {return std::move(*this).sliced(irng.front(), irng.front() + irng.size());}
	// constexpr auto range(index_range irng)      & -> decltype(auto) {return                  sliced(irng.front(), irng.front() + irng.size());}

	[[deprecated("is_flattable will be a property of the layout soon")]]
	constexpr auto is_flattable() const -> bool {
		return (this->size() <= 1) || (this->stride() == this->layout().sub().nelems());
	}

	// friend constexpr auto flatted(const_subarray const& self) {return self.flatted();}
	constexpr auto flatted() const& {
		multi::layout_t<D - 1> new_layout{this->layout().sub()};
		new_layout.nelems() *= this->size();  // TODO(correaa) : use immutable layout
		return const_subarray<T, D - 1, ElementPtr>{new_layout, types::base_};
	}

 private:
	auto flattened_aux_() const {
		auto new_layout = this->layout().flatten(this->base_);
		return multi::subarray<T, D - 1, ElementPtr, decltype(new_layout)>(new_layout, this->base_);
	}

 public:
	auto flattened() const {
		auto new_layout = this->layout().flatten(this->base_);
		return multi::const_subarray<T, D - 1, ElementPtr, decltype(new_layout)>(new_layout, this->base_);
	}

	constexpr auto broadcasted() const& {
		// TODO(correaa) introduce a broadcasted_layout?
		multi::layout_t<D + 1> const new_layout(layout(), 0, 0);  //, (std::numeric_limits<size_type>::max)());  // paren for MSVC macros
		return const_subarray<T, D + 1, typename const_subarray::element_const_ptr>{new_layout, types::base_};
	}

 private:
	constexpr auto diagonal_aux_() const -> subarray<T, D - 1, typename const_subarray::element_ptr> {
		using boost::multi::detail::get;
		auto                   square_size = (std::min)(get<0>(this->sizes()), get<1>(this->sizes()));  // paren for MSVC macros
		multi::layout_t<D - 1> new_layout{(*this)({0, square_size}, {0, square_size}).layout().sub()};
		new_layout.nelems() += (*this)({0, square_size}, {0, square_size}).layout().nelems();  // TODO(correaa) : don't use mutation
		new_layout.stride() += (*this)({0, square_size}, {0, square_size}).layout().stride();  // TODO(correaa) : don't use mutation
		return {new_layout, types::base_};
	}

 public:
	// TODO(correaa) : define a diagonal_aux
	// constexpr auto diagonal()    && {return this->diagonal();}

	// constexpr auto diagonal()     & -> const_subarray<T, D-1, typename const_subarray::element_ptr> {
	//  using boost::multi::detail::get;
	//  auto square_size = (std::min)(get<0>(this->sizes()), get<1>(this->sizes()));  // paren for MSVC macros
	//  multi::layout_t<D-1> new_layout{(*this)({0, square_size}, {0, square_size}).layout().sub()};
	//  new_layout.nelems() += (*this)({0, square_size}, {0, square_size}).layout().nelems();  // TODO(correaa) : don't use mutation
	//  new_layout.stride() += (*this)({0, square_size}, {0, square_size}).layout().stride();  // TODO(correaa) : don't use mutation
	//  return {new_layout, types::base_};
	// }

	template<class Dummy = void, std::enable_if_t<(D > 1) && sizeof(Dummy*), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr auto diagonal() const& -> const_subarray<T, D - 1, typename const_subarray::element_ptr> {
		return this->diagonal_aux_();
	}

	// template<class Dummy = void, std::enable_if_t<(D > 1) && sizeof(Dummy*), int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// constexpr auto diagonal() const& -> const_subarray<T, D-1, typename const_subarray::element_const_ptr> {
	//  using std::get;  // for C++17 compatibility
	//  auto const square_size = (std::min)(get<0>(this->sizes()), get<1>(this->sizes()));  // parenthesis min for MSVC macros
	//  multi::layout_t<D-1> new_layout{(*this)({0, square_size}, {0, square_size}).layout().sub()};
	//  new_layout.nelems() += (*this)({0, square_size}, {0, square_size}).layout().nelems();
	//  new_layout.stride() += (*this)({0, square_size}, {0, square_size}).layout().stride();  // cppcheck-suppress arithOperationsOnVoidPointer ; false positive D == 1 doesn't happen here
	//  return {new_layout, types::base_};
	// }

	// friend constexpr auto diagonal(const_subarray const& self) {return           self .diagonal();}
	// friend constexpr auto diagonal(const_subarray&       self) {return           self .diagonal();}
	// friend constexpr auto diagonal(const_subarray&&      self) {return std::move(self).diagonal();}

	// using partitioned_type       = const_subarray<T, D+1, element_ptr      >;
	// using partitioned_const_type = const_subarray<T, D+1, element_const_ptr>;

 private:
	BOOST_MULTI_HD constexpr auto partitioned_aux_(size_type n) const {
		BOOST_MULTI_ASSERT(n != 0);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		// vvv TODO(correaa) should be size() here?
		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) normal in a constexpr function
		BOOST_MULTI_ASSERT((this->layout().nelems() % n) == 0);  // if you get an assertion here it means that you are partitioning an array with an incommunsurate partition
		multi::layout_t<D + 1> new_layout{this->layout(), this->layout().nelems() / n, 0, this->layout().nelems()};
		new_layout.sub().nelems() /= n;
		return subarray<T, D + 1, element_ptr>(new_layout, types::base_);
	}

 public:
	BOOST_MULTI_HD constexpr auto partitioned(size_type n) const& -> const_subarray<T, D + 1, element_ptr> { return partitioned_aux_(n); }

 private:
	BOOST_MULTI_HD constexpr auto chunked_aux_(size_type count) const {
		BOOST_MULTI_ASSERT(this->size() % count == 0);
		return partitioned_aux_(this->size() / count);
	}

 public:  // in Mathematica this is called Partition https://reference.wolfram.com/language/ref/Partition.html in RangesV3 it is called chunk
	BOOST_MULTI_HD constexpr auto chunked(size_type count) const& -> const_subarray<T, D + 1, element_ptr> { return chunked_aux_(count); }

	constexpr auto tiled(size_type count) const& {
		BOOST_MULTI_ASSERT(count != 0);
		struct divided_type {
			const_subarray<T, D + 1, element_ptr> quotient;
			const_subarray<T, D, element_ptr>     remainder;
		};
		return divided_type{
			this->taked(this->size() - (this->size() % count)).chunked(count),
			this->dropped(this->size() - (this->size() % count))
		};
	}

 private:
	constexpr auto reversed_aux_() const { return const_subarray(layout().reverse(), types::base_); }

 public:
	constexpr auto reversed() const& -> basic_const_array { return reversed_aux_(); }
	constexpr auto reversed() & -> const_subarray { return reversed_aux_(); }
	constexpr auto reversed() && -> const_subarray { return reversed_aux_(); }

 private:
	BOOST_MULTI_HD constexpr auto transposed_aux_() const {
		return const_subarray(layout().transpose(), types::base_);
	}

 public:
	BOOST_MULTI_HD constexpr auto transposed() const& -> const_subarray { return transposed_aux_(); }

	BOOST_MULTI_FRIEND_CONSTEXPR BOOST_MULTI_HD auto operator~(const_subarray const& self) -> const_subarray { return self.transposed(); }

 private:
	BOOST_MULTI_HD constexpr auto rotated_aux_() const {
		return const_subarray(layout().rotate(), types::base_);
	}
	BOOST_MULTI_HD constexpr auto unrotated_aux_() const {
		return const_subarray(layout().unrotate(), types::base_);
	}

 public:
	BOOST_MULTI_HD constexpr auto rotated() const& -> const_subarray { return rotated_aux_(); }
	BOOST_MULTI_HD constexpr auto unrotated() const& -> const_subarray { return unrotated_aux_(); }

 private:
	template<typename, ::boost::multi::dimensionality_type, typename, class> friend struct const_subarray;

	BOOST_MULTI_HD constexpr auto paren_aux_() const& { return const_subarray<T, D, ElementPtr, Layout>(this->layout(), this->base_); }

 public:
	BOOST_MULTI_HD constexpr auto operator()() const& -> const_subarray { return paren_aux_(); }

	// clang-format off
	#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	BOOST_MULTI_HD constexpr auto operator[]() const& -> const_subarray { return paren_aux_(); }
	#endif
	// clang-format on

	template<template<class...> class Container = std::vector, template<class...> class ContainerSub = std::vector, class... As>
	constexpr auto to(As&&... as) const& {
		using inner_value_type = typename const_subarray::value_type::value_type;
		using container_type   = Container<ContainerSub<inner_value_type>>;

		return container_type(this->begin(), this->end(), std::forward<As>(as)...);
	}

 private:
	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(index_range rng, As... args) const& { return range(rng).rotated().paren_aux_(args...).unrotated(); }
	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(intersecting_range<index> inr, As... args) const& -> decltype(auto) { return paren_aux_(intersection(this->extension(), inr), args...); }
	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(index idx, As... args) const& -> decltype(auto) { return operator[](idx).paren_aux_(args...); }

	template<class... As> BOOST_MULTI_HD constexpr auto brckt_aux_(index_range rng, As... args) const& { return range(rng).rotated().paren_aux_(args...).unrotated(); }
	template<class... As> BOOST_MULTI_HD constexpr auto brckt_aux_(intersecting_range<index> inr, As... args) const& -> decltype(auto) { return paren_aux_(intersection(this->extension(), inr), args...); }
	template<class... As> BOOST_MULTI_HD constexpr auto brckt_aux_(index idx, As... args) const& -> decltype(auto) { return operator[](idx).paren_aux_(args...); }

 public:
	// vvv DO NOT remove default parameter `= irange` : the default template parameters below help interpret the expression `{first, last}` syntax as index ranges
	template<class A1 = irange> BOOST_MULTI_HD constexpr auto                                                                       operator()(A1 arg1) const& -> decltype(auto) { return paren_aux_(arg1); }                                // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange> BOOST_MULTI_HD constexpr auto                                                    operator()(A1 arg1, A2 arg2) const& -> decltype(auto) { return paren_aux_(arg1, arg2); }                 // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange> BOOST_MULTI_HD constexpr auto                                 operator()(A1 arg1, A2 arg2, A3 arg3) const& -> decltype(auto) { return paren_aux_(arg1, arg2, arg3); }  // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange, class A4 = irange, class... As> BOOST_MULTI_HD constexpr auto operator()(A1 arg1, A2 arg2, A3 arg3, A4 arg4, As... args) const& -> decltype(auto) { return paren_aux_(arg1, arg2, arg3, arg4, args...); }

	// clang-format off
	#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	// vvv DO NOT remove default parameter `= irange` : the default template parameters below help interpret the expression `{first, last}` syntax as index ranges
	// template<class A1 = irange> BOOST_MULTI_HD constexpr auto                                                                    operator[](A1 arg1) const& -> decltype(auto) { return paren_aux_(arg1); }                             // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange> BOOST_MULTI_HD constexpr auto                                                    operator[](A1 arg1, A2 arg2) const& -> decltype(auto) { return brckt_aux_(arg1, arg2); }                 // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange> BOOST_MULTI_HD constexpr auto                                 operator[](A1 arg1, A2 arg2, A3 arg3) const& -> decltype(auto) { return brckt_aux_(arg1, arg2, arg3); }  // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange, class A4 = irange, class... As> BOOST_MULTI_HD constexpr auto operator[](A1 arg1, A2 arg2, A3 arg3, A4 arg4, As... args) const& -> decltype(auto) { return brckt_aux_(arg1, arg2, arg3, arg4, args...); }
	#endif
	// clang-format on

 private:
	template<typename Tuple, std::size_t... I> BOOST_MULTI_HD constexpr auto apply_impl_(Tuple const& tuple, std::index_sequence<I...> /*012*/) const& -> decltype(auto) {
		using std::get;
		return this->operator()(get<I>(tuple)...);
	}

 public:
	template<typename Tuple> BOOST_MULTI_HD constexpr auto apply(Tuple const& tuple) const& -> decltype(auto) { return apply_impl_(tuple, std::make_index_sequence<std::tuple_size_v<Tuple>>{}); }

	using iterator       = array_iterator<element, D, element_ptr, false, false, typename layout_type::stride_type, typename layout_type::sub_type>;
	using const_iterator = array_iterator<element, D, element_ptr, true, false, typename layout_type::stride_type, typename layout_type::sub_type>;
	using move_iterator  = array_iterator<element, D, element_ptr, false, true, typename layout_type::stride_type, typename layout_type::sub_type>;

	// using  move_iterator = array_iterator<element, D, element_move_ptr >;

	// using       reverse_iterator [[deprecated]] = std::reverse_iterator<      iterator>;
	// using const_reverse_iterator [[deprecated]] = std::reverse_iterator<const_iterator>;

	const_subarray(const_iterator first, const_iterator last)
	: const_subarray(layout_type(first->layout(), first.stride(), 0, (last - first) * first->size()), first.base()) {
		BOOST_MULTI_ASSERT(first->layout() == last->layout());
	}

 private:
	friend BOOST_MULTI_HD constexpr auto ref<iterator>(iterator begin, iterator end) -> multi::subarray<typename iterator::element, iterator::rank_v, typename iterator::element_ptr>;

 public:
	using ptr       = subarray_ptr<T, D, ElementPtr, Layout, false>;
	using const_ptr = const_subarray_ptr<T, D, ElementPtr, Layout>;  // TODO(correaa) add const_subarray_ptr

	using pointer       = ptr;
	using const_pointer = const_ptr;

 private:
	constexpr auto addressof_aux_() const { return ptr(this->base_, this->layout()); }

 public:
	constexpr auto addressof() && -> ptr { return addressof_aux_(); }
	constexpr auto addressof() & -> ptr { return addressof_aux_(); }
	constexpr auto addressof() const& -> const_ptr { return addressof_aux_(); }

	// NOLINTBEGIN(google-runtime-operator) //NOSONAR

	// operator& is not defined for r-values anyway
	constexpr auto operator&() && { return addressof(); }  // NOLINT(runtime/operator) //NOSONAR
	// [[deprecated("controversial")]]
	constexpr auto operator&() & { return addressof(); }  // NOLINT(runtime/operator) //NOSONAR
	// [[deprecated("controversial")]]
	constexpr auto operator&() const& { return addressof(); }  // NOLINT(runtime/operator) //NOSONAR

	// NOLINTEND(google-runtime-operator)

 private:
#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	BOOST_MULTI_HD constexpr auto begin_aux_() const { return iterator(types::base_, this->sub(), this->stride()); }
	BOOST_MULTI_HD constexpr auto end_aux_() const { return iterator(types::base_ + this->nelems(), this->sub(), this->stride()); }  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

 public:
	BOOST_MULTI_HD constexpr auto begin() const& -> const_iterator { return begin_aux_(); }
	BOOST_MULTI_HD constexpr auto end() const& -> const_iterator { return end_aux_(); }
	// friend /*constexpr*/ auto begin(const_subarray const& self) -> const_iterator { return self.begin(); }  // NOLINT(whitespace/indent) constexpr doesn't work with nvcc friend
	// friend /*constexpr*/ auto end(const_subarray const& self) -> const_iterator { return self.end(); }      // NOLINT(whitespace/indent) constexpr doesn't work with nvcc friend

	BOOST_MULTI_HD constexpr auto cbegin() const& { return begin(); }
	BOOST_MULTI_HD constexpr auto cend() const& { return end(); }
	// friend constexpr auto         cbegin(const_subarray const& self) { return self.cbegin(); }
	// friend constexpr auto         cend(const_subarray const& self) { return self.cend(); }

	using cursor       = cursor_t<typename const_subarray::element_ptr, D, typename const_subarray::strides_type>;
	using const_cursor = cursor_t<typename const_subarray::element_const_ptr, D, typename const_subarray::strides_type>;

 private:
	BOOST_MULTI_HD constexpr auto home_aux_() const { return cursor(this->base_, this->strides()); }

 public:
	BOOST_MULTI_HD constexpr auto home() const& -> const_cursor { return home_aux_(); }

	template<
		class Range,
		std::enable_if_t<!has_extensions<std::decay_t<Range>>::value, int> = 0,
		//  std::enable_if_t<not multi::is_implicitly_convertible_v<subarray, Range>, int> =0,
		class = decltype(Range(std::declval<typename const_subarray::const_iterator>(), std::declval<typename const_subarray::const_iterator>()))>
	constexpr explicit operator Range() const { return Range(begin(), end()); }  // NOLINT(fuchsia-default-arguments-calls) for example std::vector(it, ti, alloc = {})

	template<class TT, class... As>
	friend constexpr auto operator==(const_subarray const& self, const_subarray<TT, D, As...> const& other) -> bool {
		return (self.extension() == other.extension()) && (self.elements() == other.elements());
	}
	template<class TT, class... As>
	friend constexpr auto operator!=(const_subarray const& self, const_subarray<TT, D, As...> const& other) -> bool {
		return (self.extension() != other.extension()) || (self.elements() != other.elements());
	}

	constexpr auto operator==(const_subarray const& other) const -> bool {
		return (this->extension() == other.extension()) && (this->elements() == other.elements());
	}
	constexpr auto operator!=(const_subarray const& other) const -> bool {
		return (this->extension() != other.extension()) || (this->elements() != other.elements());
	}

	friend constexpr auto lexicographical_compare(const_subarray const& self, const_subarray const& other) -> bool {
		if(self.extension().first() > other.extension().first()) {
			return true;
		}
		if(self.extension().first() < other.extension().first()) {
			return false;
		}
		return adl_lexicographical_compare(
			self.begin(), self.end(),
			other.begin(), other.end()
		);
	}

	constexpr auto operator<(const_subarray const& other) const& -> bool { return lexicographical_compare(*this, other); }
	constexpr auto operator<=(const_subarray const& other) const& -> bool { return *this == other || lexicographical_compare(*this, other); }
	constexpr auto operator>(const_subarray const& other) const& -> bool { return other < *this; }

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, std::enable_if_t<std::is_const_v<typename std::pointer_traits<P2>::element_type>, int> = 0  // NOLINT(modernize-use-constraints) TODO(correaa)
			 >
	constexpr auto static_array_cast() const& {                                    // name taken from std::static_pointer_cast
		return subarray<T2, D, P2>(this->layout(), static_cast<P2>(this->base_));  // TODO(correaa) might violate constness
	}

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, std::enable_if_t<!std::is_const_v<typename std::pointer_traits<P2>::element_type>, int> = 0  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
			 >
	[[deprecated("violates constness")]]
	constexpr auto static_array_cast() const& {                                    // name taken from std::static_pointer_cast
		return subarray<T2, D, P2>(this->layout(), static_cast<P2>(this->base_));  // TODO(correaa) might violate constness
	}

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>>
	constexpr auto static_array_cast() && {  // name taken from std::static_pointer_cast
		return subarray<T2, D, P2>(this->layout(), static_cast<P2>(this->base_));
	}

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>>
	constexpr auto static_array_cast() & {  // name taken from std::static_pointer_cast
		return subarray<T2, D, P2>(this->layout(), static_cast<P2>(this->base_));
	}

 private:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, class... Args>
	constexpr auto static_array_cast_(Args&&... args) const& {  // name taken from std::static_pointer_cast
		return subarray<T2, D, P2>(this->layout(), P2{this->base_, std::forward<Args>(args)...});
	}

 public:
	template<class UF>
	BOOST_MULTI_HD constexpr auto element_transformed(UF&& fun) const& {
		return static_array_cast_<
			//  std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<UF const&, element_cref>>>,
			std::decay_t<std::invoke_result_t<UF const&, element_cref>>,
			transform_ptr<
				//  std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<UF const&, element_cref>>>,
				std::decay_t<std::invoke_result_t<UF const&, element_cref>>,
				UF, element_const_ptr, std::invoke_result_t<UF const&, element_cref>>>(std::forward<UF>(fun));
	}
	template<class UF>
	BOOST_MULTI_HD constexpr auto element_transformed(UF&& fun) & {
		return static_array_cast_<
			std::decay_t<std::invoke_result_t<UF const&, element_ref>>,
			transform_ptr<
				std::decay_t<std::invoke_result_t<UF const&, element_ref>>,
				UF, element_ptr, std::invoke_result_t<UF const&, element_ref>>>(std::forward<UF>(fun));
	}
	template<class UF>
	BOOST_MULTI_HD constexpr auto element_transformed(UF&& fun) && { return element_transformed(std::forward<UF>(fun)); }

	template<
		class T2, class P2 = typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2 const>,
		class Element = typename const_subarray::element,
		class PM      = T2 Element::*>
	constexpr auto member_cast(PM member) const& -> subarray<T2, D, P2> {
		static_assert(sizeof(T) % sizeof(T2) == 0, "array_member_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. "
												   "Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements.");

		return subarray<T2, D, P2>{this->layout().scale(sizeof(T), sizeof(T2)), static_cast<P2>(&(this->base_->*member))};
	}

	template<
		class T2, class P2 = typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2>,
		class Element = typename const_subarray::element,
		class PM      = T2 Element::*>
	constexpr auto member_cast(PM member) & -> subarray<T2, D, P2> {
		static_assert(sizeof(T) % sizeof(T2) == 0, "array_member_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. "
												   "Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements");

		return subarray<T2, D, P2>{this->layout().scale(sizeof(T), sizeof(T2)), static_cast<P2>(&(this->base_->*member))};
	}

	template<
		class T2, class P2 = typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2>,
		class Element = typename const_subarray::element,
		class PM      = T2 Element::*>
	constexpr auto member_cast(PM member) && -> subarray<T2, D, P2> {
		return this->member_cast<T2, P2, Element, PM>(member);
	}

	template<class T2, class P2 = typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2>>
	using rebind = subarray<std::decay_t<T2>, D, P2>;

	template<
		class T2 = std::remove_const_t<T>,
		class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>,
		std::enable_if_t<    // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
			std::is_same_v<  // check that pointer family is not changed
				typename std::pointer_traits<P2>::template rebind<T2>,
				typename std::pointer_traits<element_ptr>::template rebind<T2>> &&
				std::is_same_v<  // check that only constness is changed
					std::remove_const_t<typename std::pointer_traits<P2>::element_type>, std::remove_const_t<typename const_subarray::element_type>>,
			int> = 0>
	constexpr auto const_array_cast() const {
		if constexpr(std::is_pointer_v<P2>) {
			return rebind<T2, P2>(this->layout(), const_cast<P2>(this->base_));  // NOLINT(cppcoreguidelines-pro-type-const-cast)
		} else {
			return rebind<T2, P2>(this->layout(), reinterpret_cast<P2 const&>(this->base_));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)  //NOSONAR
		}
	}

	constexpr auto as_const() const {
		return rebind<element, element_const_ptr>{this->layout(), this->base()};
	}

 private:
	template<class T2, class P2>
	constexpr auto reinterpret_array_cast_aux_() const -> rebind<T2, P2> {
		// static_assert( sizeof(T)%sizeof(T2) == 0,
		//  "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );

		return {
			this->layout().scale(sizeof(T), sizeof(T2)),  // NOLINT(bugprone-sizeof-expression) : sizes are compatible according to static assert above
			reinterpret_pointer_cast<P2>(this->base_)     // if ADL gets confused here (e.g. multi:: and thrust::) then adl_reinterpret_pointer_cast will be necessary
		};
	}

 public:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2 const>>
	constexpr auto reinterpret_array_cast() const& { return reinterpret_array_cast_aux_<T2, P2>().as_const(); }

	template<
		class T2,
		class P2 =
			std::conditional_t<
				std::is_const_v<typename std::pointer_traits<typename const_subarray::element_ptr>::element_type>,
				typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2 const>,
				typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<T2>>>
	constexpr auto reinterpret_array_cast(size_type count) const& {
		static_assert(sizeof(T) % sizeof(T2) == 0, "error: reinterpret_array_cast is limited to integral stride values");

		BOOST_MULTI_ASSERT(sizeof(T) == sizeof(T2) * static_cast<std::size_t>(count));  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : checck implicit size compatibility

		if constexpr(std::is_pointer_v<ElementPtr>) {
			using void_ptr_like = std::conditional_t<
				std::is_const_v<typename std::pointer_traits<decltype(this->base_)>::element_type>,
				typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<void const>,
				typename std::pointer_traits<typename const_subarray::element_ptr>::template rebind<void>>;
			return const_subarray<T2, D + 1, P2>(
				layout_t<D + 1>(this->layout().scale(sizeof(T), sizeof(T2)), 1, 0, count).rotate(),
				static_cast<P2>(static_cast<void_ptr_like>(this->base_))  // NOLINT(bugprone-casting-through-void) direct reinterepret_cast doesn't work here for some exotic pointers (e.g. thrust::pointer)
			);
		} else {  // TODO(correaa) try to unify both if-branches
			return const_subarray<T2, D + 1, P2>(
				layout_t<D + 1>(this->layout().scale(sizeof(T), sizeof(T2)), 1, 0, count).rotate(),
				reinterpret_cast<P2 const&>(this->base_)  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,bugprone-casting-through-void) direct reinterepret_cast doesn't work here
			);
		}
	}

	template<class Archive>
	auto serialize(Archive& arxiv, unsigned int /*version*/) {
		using AT = multi::archive_traits<Archive>;
		// if(version == 0) {
		//  std::for_each(this->begin(), this->end(), [&](reference&& item) {arxiv & AT    ::make_nvp("item", std::move(item));});
		// } else {
		std::for_each(this->elements().begin(), this->elements().end(), [&](element const& elem) { arxiv& AT ::make_nvp("elem", elem); });
		// }
		//  std::for_each(this->begin(), this->end(), [&](auto&& item) {arxiv & cereal::make_nvp("item", item);});
		//  std::for_each(this->begin(), this->end(), [&](auto&& item) {arxiv &                          item ;});
	}
};

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

template<class T>
BOOST_MULTI_HD constexpr auto move(T&& val) noexcept -> decltype(auto) {
	if constexpr(has_member_move<T>::value) {
		return std::forward<T>(val).move();
	} else {
		return std::move(std::forward<T>(val));
	}
}

template<typename T, multi::dimensionality_type D, typename ElementPtr, class Layout>
class move_subarray : public subarray<T, D, ElementPtr, Layout> {
	// cppcheck-suppress noExplicitConstructor ; see below
	BOOST_MULTI_HD constexpr move_subarray(subarray<T, D, ElementPtr, Layout>& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  TODO(correa) check if this is necessary
	: subarray<T, D, ElementPtr, Layout>(other.layout(), other.mutable_base()) {}

	friend class subarray<T, D, ElementPtr, Layout>;

 public:
	using subarray<T, D, ElementPtr, Layout>::operator[];
	BOOST_MULTI_HD constexpr auto operator[](index idx) && -> decltype(auto) {
		return multi::move(subarray<T, D, ElementPtr, Layout>::operator[](idx));
	}

	using subarray<T, D, ElementPtr, Layout>::begin;
	using subarray<T, D, ElementPtr, Layout>::end;

	BOOST_MULTI_HD constexpr auto begin() && { return this->mbegin(); }
	BOOST_MULTI_HD constexpr auto end() && { return this->mend(); }
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<typename T, multi::dimensionality_type D, typename ElementPtr, class Layout>
class subarray : public const_subarray<T, D, ElementPtr, Layout> {
	// cppcheck-suppress noExplicitConstructor ; see below
	BOOST_MULTI_HD constexpr subarray(const_subarray<T, D, ElementPtr, Layout> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  TODO(correa) check if this is necessary
	: subarray(other.layout(), other.mutable_base()) {}

	template<typename, multi::dimensionality_type, typename, class> friend class subarray;
	template<typename, multi::dimensionality_type, typename, class, bool> friend struct subarray_ptr;

	template<class, multi::dimensionality_type, class, bool, bool, typename, class> friend struct array_iterator;

 public:
	subarray(subarray const&) = delete;

	BOOST_MULTI_HD constexpr auto        move() { return move_subarray<T, D, ElementPtr, Layout>(*this); }
	friend BOOST_MULTI_HD constexpr auto move(subarray& self) { return self.move(); }
	friend BOOST_MULTI_HD constexpr auto move(subarray&& self) { return std::move(self).move(); }

	using move_iterator = array_iterator<T, D, ElementPtr, false, true>;

	using typename const_subarray<T, D, ElementPtr, Layout>::element;
	using typename const_subarray<T, D, ElementPtr, Layout>::element_ptr;
	using typename const_subarray<T, D, ElementPtr, Layout>::element_const_ptr;

	using reference = typename std::conditional_t<
		(D > 1),
		subarray<element, D - 1, element_ptr>,
		typename std::iterator_traits<element_ptr>::reference>;

	using const_reference = typename std::conditional_t<
		(D > 1),
		const_subarray<element, D - 1, element_ptr>,
		typename std::iterator_traits<element_const_ptr>::reference>;

	subarray(subarray&&) noexcept = default;
	~subarray()                   = default;

	using ptr = subarray_ptr<T, D, ElementPtr, Layout, false>;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) use checked span
#endif

	// NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() && { return subarray_ptr<T, D, ElementPtr, Layout, false>(this->base_, this->layout()); }  // NOLINT(google-runtime-operator) : taking address of a reference-like object should be allowed  //NOSONAR
	// cppcheck-suppress duplInheritedMember ; to overwrite  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() & { return subarray_ptr<T, D, ElementPtr, Layout, false>(this->base_, this->layout()); }  // NOLINT(google-runtime-operator) : taking address of a reference-like object should be allowed  //NOSONAR

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	using const_subarray<T, D, ElementPtr, Layout>::operator&;
	// NOLINTNEXTLINE(runtime/operator)
	// BOOST_MULTI_HD constexpr auto operator&() const& {return subarray_ptr<const_subarray, Layout>{this->base_, this->layout()};}  // NOLINT(google-runtime-operator) extend semantics  //NOSONAR

	using const_subarray<T, D, ElementPtr, Layout>::const_subarray;

	using const_subarray<T, D, ElementPtr, Layout>::elements;
	constexpr auto elements() & { return this->elements_aux_(); }
	constexpr auto elements() && { return this->elements_aux_(); }

	using const_subarray<T, D, ElementPtr, Layout>::begin;
	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto begin() && noexcept { return this->begin_aux_(); }
	BOOST_MULTI_HD constexpr auto begin() & noexcept { return this->begin_aux_(); }

	using const_subarray<T, D, ElementPtr, Layout>::end;
	BOOST_MULTI_HD constexpr auto end() && noexcept { return this->end_aux_(); }
	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto end() & noexcept { return this->end_aux_(); }

	BOOST_MULTI_HD constexpr auto mbegin() { return move_iterator{this->begin()}; }
	BOOST_MULTI_HD constexpr auto mend() { return move_iterator{this->end()}; }

	using const_subarray<T, D, ElementPtr, Layout>::home;
	BOOST_MULTI_HD constexpr auto home() && { return this->home_aux_(); }
	BOOST_MULTI_HD constexpr auto home() & { return this->home_aux_(); }

	template<class It> constexpr auto assign(It first) & -> It {
		adl_copy_n(first, this->size(), begin());
		std::advance(first, this->size());
		return first;
	}
	template<class It> BOOST_MULTI_HD constexpr auto assign(It first) && -> It { return assign(first); }

	template<class TT = typename subarray::element_type>
	constexpr auto fill(TT const& value) & -> decltype(auto) {
		return adl_fill_n(this->begin(), this->size(), value), *this;
	}
	constexpr auto fill() & -> decltype(auto) { return fill(typename subarray::element_type{}); }

	template<class TT = typename subarray::element_type>
	[[deprecated]] constexpr auto fill(TT const& value) && -> decltype(auto) { return std::move(this->fill(value)); }
	[[deprecated]] constexpr auto fill() && -> decltype(auto) {
		return std::move(*this).fill(typename subarray::element_type{});
	}

	using const_subarray<T, D, ElementPtr, Layout>::strided;
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto strided(difference_type diff) && { return this->strided_aux_(diff); }
	constexpr auto strided(difference_type diff) & { return this->strided_aux_(diff); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using const_subarray<T, D, ElementPtr, Layout>::taked;
	constexpr auto taked(difference_type count) && -> subarray { return this->taked_aux_(count); }
	constexpr auto taked(difference_type count) & -> subarray { return this->taked_aux_(count); }

	using const_subarray<T, D, ElementPtr, Layout>::dropped;
	// cppcheck-suppress-begin duplInheritedMember ; to ovewrite
	constexpr auto dropped(difference_type count) && -> subarray { return this->dropped_aux_(count); }
	constexpr auto dropped(difference_type count) & -> subarray { return this->dropped_aux_(count); }
	// cppcheck-suppress-end duplInheritedMember ; to ovewrite

	using const_subarray<T, D, ElementPtr, Layout>::rotated;
	// cppcheck-suppress-begin duplInheritedMember ; to ovewrite
	BOOST_MULTI_HD constexpr auto rotated() && -> subarray { return const_subarray<T, D, ElementPtr, Layout>::rotated(); }
	BOOST_MULTI_HD constexpr auto rotated() & -> subarray { return const_subarray<T, D, ElementPtr, Layout>::rotated(); }
	// cppcheck-suppress-end duplInheritedMember ; to ovewrite

	using const_subarray<T, D, ElementPtr, Layout>::unrotated;
	BOOST_MULTI_HD constexpr auto unrotated() && -> subarray { return const_subarray<T, D, ElementPtr, Layout>::unrotated(); }
	BOOST_MULTI_HD constexpr auto unrotated() & -> subarray { return const_subarray<T, D, ElementPtr, Layout>::unrotated(); }

	using const_subarray<T, D, ElementPtr, Layout>::transposed;
	BOOST_MULTI_HD constexpr auto transposed() && -> subarray { return const_subarray<T, D, ElementPtr, Layout>::transposed(); }
	BOOST_MULTI_HD constexpr auto transposed() & -> subarray { return const_subarray<T, D, ElementPtr, Layout>::transposed(); }

	// BOOST_MULTI_FRIEND_CONSTEXPR BOOST_MULTI_HD
	// auto operator~ (subarray const& self) { return self.transposed(); }
	BOOST_MULTI_FRIEND_CONSTEXPR BOOST_MULTI_HD auto operator~(subarray& self) { return self.transposed(); }
	BOOST_MULTI_FRIEND_CONSTEXPR BOOST_MULTI_HD auto operator~(subarray&& self) { return std::move(self).transposed(); }

	using const_subarray<T, D, ElementPtr, Layout>::reindexed;

	template<class... Indexes>
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto reindexed(index first, Indexes... idxs) & -> subarray {
		return const_subarray<T, D, ElementPtr, Layout>::reindexed(first, idxs...);
		// return ((this->reindexed(first).rotated()).reindexed(idxs...)).unrotated();
	}
	template<class... Indexes>
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto reindexed(index first, Indexes... idxs) && -> subarray {
		return const_subarray<T, D, ElementPtr, Layout>::reindexed(first, idxs...);
		// return ((std::move(*this).reindexed(first).rotated()).reindexed(idxs...)).unrotated();
	}

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto base() const& -> typename subarray::element_const_ptr { return this->base_; }
	BOOST_MULTI_HD constexpr auto base() & -> ElementPtr { return this->base_; }
	BOOST_MULTI_HD constexpr auto base() && -> ElementPtr { return this->base_; }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto operator=(const_subarray<T, D, ElementPtr, Layout> const& other) & -> subarray& {
		if(this == std::addressof(other)) {
			return *this;
		}
		BOOST_MULTI_ASSERT(this->extension() == other.extension());
		this->elements() = other.elements();
		return *this;
	}

	constexpr void swap(subarray&& other) && noexcept {
		BOOST_MULTI_ASSERT(this->extension() == other.extension());
		adl_swap_ranges(this->elements().begin(), this->elements().end(), std::move(other).elements().begin());
	}
	friend constexpr void swap(subarray&& self, subarray&& other) noexcept { std::move(self).swap(std::move(other)); }

	// template<class A, typename = std::enable_if_t<!std::is_base_of_v<subarray, std::decay_t<A>>>> friend constexpr void swap(subarray&& self, A&& other) noexcept { std::move(self).swap(std::forward<A>(other)); }
	// template<class A, typename = std::enable_if_t<!std::is_base_of_v<subarray, std::decay_t<A>>>> friend constexpr void swap(A&& other, subarray&& self) noexcept { std::move(self).swap(std::forward<A>(other)); }

	// template<class Array> constexpr void swap(Array&& other) && noexcept {
	//  assert( std::move(*this).extension() == other.extension() );  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
	//  this->elements().swap(std::forward<Array>(other).elements());
	// //  adl_swap_ranges(this->begin(), this->end(), adl_begin(std::forward<Array>(o)));
	// }
	// template<class A> constexpr void swap(A&& other) & noexcept {return swap(std::forward<A>(other));}

	// friend constexpr void swap(subarray&& self, subarray&& other) noexcept {std::move(self).swap(std::move(other));}

	// template<class Array> constexpr void swap(subarray& self, Array&& other) noexcept {self.swap(std::forward<Array>(other));}  // TODO(correaa) remove
	// template<class Array> constexpr void swap(Array&& other, subarray& self) noexcept {self.swap(std::forward<Array>(other));}

	// fix mutation
	// template<class TT, class... As> constexpr auto operator=(const_subarray<TT, 1L, As...> const& other) && -> decltype(auto) {operator=(          other ); return *this;}
	template<class TT, class... As> constexpr auto operator=(const_subarray<TT, D, As...> const& other) & -> subarray& {
		BOOST_MULTI_ASSERT(other.extensions() == this->extensions());
		this->elements() = other.elements();
		return *this;
	}

	// fix mutation
	template<class TT, class... As> constexpr auto operator=(const_subarray<TT, D, As...>&& other) && -> subarray& {
		operator=(std::move(other));
		return *this;
	}
	template<class TT, class... As> constexpr auto operator=(const_subarray<TT, D, As...>&& other) & -> subarray& {
		BOOST_MULTI_ASSERT(this->extensions() == other.extensions());
		this->elements() = std::move(other).elements();
		return *this;
	}

	template<
		class Range,
		class                                              = std::enable_if_t<!std::is_base_of_v<subarray, Range>>,  // NOLINT(modernize-type-traits)  TODO(correaa) in C++20
		class                                              = std::enable_if_t<!is_subarray<Range>::value>,           // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
		std::enable_if_t<!has_elements<Range>::value, int> = 0>                                                      // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	constexpr auto operator=(Range const& rng) & -> subarray& {                                                      // lints(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
		BOOST_MULTI_ASSERT(this->size() == static_cast<size_type>(adl_size(rng)));                                   // TODO(correaa) or use std::cmp_equal?
		adl_copy_n(adl_begin(rng), adl_size(rng), this->begin());
		return *this;
	}

	template<
		class MultiRange,
		class                                                  = std::enable_if_t<!std::is_base_of_v<subarray, MultiRange>>,  // NOLINT(modernize-type-traits)  TODO(correaa) in C++20
		class                                                  = std::enable_if_t<!is_subarray<MultiRange>::value>,           // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
		std::enable_if_t<has_elements<MultiRange>::value, int> = 0>                                                           // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	constexpr auto operator=(MultiRange const& mrng) & -> subarray& {                                                         // lints(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
		BOOST_MULTI_ASSERT(this->extensions() == mrng.extensions());                                                          // TODO(correaa) or use std::cmp_equal?
		adl_copy_n(mrng.elements().begin(), this->num_elements(), this->elements().begin());
		return *this;
	}

	template<
		class Range,
		class = std::enable_if_t<!std::is_base_of_v<subarray, Range>>,  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
		class = std::enable_if_t<!is_subarray<Range>::value>            // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
		>
	constexpr auto operator=(Range const& rng) && -> subarray& {
		operator=(rng);
		return *this;
	}

	template<class TT, class... As>
	constexpr auto operator=(const_subarray<TT, D, As...> const& other) && -> subarray& {
		BOOST_MULTI_ASSERT(this->extension() == other.extension());  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		this->elements() = other.elements();
		return *this;
	}

	template<class TT, class... As>
	constexpr auto operator=(subarray<TT, D, As...>&& other) & -> subarray& {
		BOOST_MULTI_ASSERT(this->extension() == other.extension());  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		this->elements() = std::move(other).elements();
		return *this;
	}

	constexpr auto operator=(const_subarray<T, D, ElementPtr, Layout> const& other) const&& -> subarray&;  // for std::indirectly_writable

	constexpr auto operator=(subarray const& other) & -> subarray& {
		if(this == std::addressof(other)) {
			return *this;
		}
		BOOST_MULTI_ASSERT(this->extension() == other.extension());
		this->elements() = other.elements();
		return *this;
	}
	constexpr auto operator=(subarray&& other) & noexcept -> subarray& {  // TODO(correaa) make conditionally noexcept
		// if(this == std::addressof(other)) { return *this; }
		BOOST_MULTI_ASSERT(this->extension() == other.extension());
		this->elements() = std::move(other).elements();
		return *this;
	}

	auto operator=(std::initializer_list<typename subarray::value_type> values) && -> subarray& {
		operator=(values);
		return *this;
	}
	auto operator=(std::initializer_list<typename subarray::value_type> values) & -> subarray& {
		BOOST_MULTI_ASSERT(static_cast<size_type>(values.size()) == this->size());
		if(values.size() != 0) {
			adl_copy_n(values.begin(), values.size(), this->begin());
		}
		return *this;
	}

	// BOOST_MULTI_HD constexpr auto operator[](index idx) const&    { return static_cast<typename subarray::const_reference>(this->at_aux_(idx)); }  // TODO(correaa) use return type to cast
	using const_subarray<T, D, ElementPtr, Layout>::operator[];
	// BOOST_MULTI_HD constexpr auto operator[](index idx) const& { return const_subarray<T, D, ElementPtr, Layout>::operator[](idx); }

	// cppcheck-suppress-start duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto operator[](index idx) && -> typename subarray::reference { return this->at_aux_(idx); }
	BOOST_MULTI_HD constexpr auto operator[](index idx) & -> typename subarray::reference { return this->at_aux_(idx); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using const_subarray<T, D, ElementPtr, Layout>::diagonal;

	// template<class Dummy = void, std::enable_if_t<(D > 1) && sizeof(Dummy*), int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress-begin duplInheritedMember ; to override
	constexpr auto diagonal() & { return this->diagonal_aux_(); }
	constexpr auto diagonal() && { return this->diagonal_aux_(); }
	// cppcheck-suppress-end duplInheritedMember ; to override

	using const_subarray<T, D, ElementPtr, Layout>::sliced;
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto sliced(index first, index last) && -> subarray { return const_subarray<T, D, ElementPtr, Layout>::sliced(first, last); }
	BOOST_MULTI_HD constexpr auto sliced(index first, index last) & -> subarray { return const_subarray<T, D, ElementPtr, Layout>::sliced(first, last); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using const_subarray<T, D, ElementPtr, Layout>::range;
	BOOST_MULTI_HD constexpr auto range(index_range irng) && -> decltype(auto) { return std::move(*this).sliced(irng.front(), irng.front() + irng.size()); }
	BOOST_MULTI_HD constexpr auto range(index_range irng) & -> decltype(auto) { return sliced(irng.front(), irng.front() + irng.size()); }

 private:
	using const_subarray<T, D, ElementPtr, Layout>::paren_aux_;

	BOOST_MULTI_HD constexpr auto paren_aux_() & { return subarray<T, D, ElementPtr, Layout>(this->layout(), this->base_); }
	BOOST_MULTI_HD constexpr auto paren_aux_() && { return subarray<T, D, ElementPtr, Layout>(this->layout(), this->base_); }

	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(index idx) & -> decltype(auto) { return operator[](idx); }
	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(index idx) && -> decltype(auto) { return operator[](idx); }

	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(index idx, As... args) & -> decltype(auto) { return operator[](idx).paren_aux_(args...); }
	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(index idx, As... args) && -> decltype(auto) { return operator[](idx).paren_aux_(args...); }

	template<class... As>
	BOOST_MULTI_HD constexpr auto paren_aux_(index_range irng, As... args) & {
		return this->range(irng).rotated().paren_aux_(args...).unrotated();
	}
	template<class... As>
	BOOST_MULTI_HD constexpr auto paren_aux_(index_range irng, As... args) && {
		return std::move(*this).range(irng).rotated().paren_aux_(args...).unrotated();
	}

	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(intersecting_range<index> inr, As... args) & -> decltype(auto) { return paren_aux_(intersection(this->extension(), inr), args...); }
	template<class... As> BOOST_MULTI_HD constexpr auto paren_aux_(intersecting_range<index> inr, As... args) && -> decltype(auto) { return paren_aux_(intersection(this->extension(), inr), args...); }

 public:
	using const_subarray<T, D, ElementPtr, Layout>::operator();
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto operator()() & -> subarray { return this->paren_aux_(); }
	BOOST_MULTI_HD constexpr auto operator()() && -> subarray { return this->paren_aux_(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	template<class A1 = irange> BOOST_MULTI_HD constexpr auto                                                                       operator()(A1 arg1) & -> decltype(auto) { return this->paren_aux_(arg1); }
	template<class A1 = irange, class A2 = irange> BOOST_MULTI_HD constexpr auto                                                    operator()(A1 arg1, A2 arg2) & -> decltype(auto) { return this->paren_aux_(arg1, arg2); }
	template<class A1 = irange, class A2 = irange, class A3 = irange> BOOST_MULTI_HD constexpr auto                                 operator()(A1 arg1, A2 arg2, A3 arg3) & -> decltype(auto) { return this->paren_aux_(arg1, arg2, arg3); }
	template<class A1 = irange, class A2 = irange, class A3 = irange, class A4 = irange, class... As> BOOST_MULTI_HD constexpr auto operator()(A1 arg1, A2 arg2, A3 arg3, A4 arg4, As... args) & -> decltype(auto) { return this->paren_aux_(arg1, arg2, arg3, arg4, args...); }

	template<class A1 = irange> BOOST_MULTI_HD constexpr auto                                                                       operator()(A1 arg1) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1); }                                                                    // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange> BOOST_MULTI_HD constexpr auto                                                    operator()(A1 arg1, A2 arg2) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1, arg2); }                                                     // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange> BOOST_MULTI_HD constexpr auto                                 operator()(A1 arg1, A2 arg2, A3 arg3) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1, arg2, arg3); }                                      // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange, class A4 = irange, class... As> BOOST_MULTI_HD constexpr auto operator()(A1 arg1, A2 arg2, A3 arg3, A4 arg4, As... args) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1, arg2, arg3, arg4, args...); }  // NOLINT(whitespace/line_length) pattern line

	// clang-format off
	#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	// template<class A1 = irange> BOOST_MULTI_HD constexpr auto                                                                       operator[](A1 arg1) & -> decltype(auto) { return this->paren_aux_(arg1); }
	template<class A1 = irange, class A2 = irange> BOOST_MULTI_HD constexpr auto                                                    operator[](A1 arg1, A2 arg2) & -> decltype(auto) { return this->paren_aux_(arg1, arg2); }
	template<class A1 = irange, class A2 = irange, class A3 = irange> BOOST_MULTI_HD constexpr auto                                 operator[](A1 arg1, A2 arg2, A3 arg3) & -> decltype(auto) { return this->paren_aux_(arg1, arg2, arg3); }
	template<class A1 = irange, class A2 = irange, class A3 = irange, class A4 = irange, class... As> BOOST_MULTI_HD constexpr auto operator[](A1 arg1, A2 arg2, A3 arg3, A4 arg4, As... args) & -> decltype(auto) { return this->paren_aux_(arg1, arg2, arg3, arg4, args...); }

	// template<class A1 = irange> BOOST_MULTI_HD constexpr auto                                                                       operator[](A1 arg1) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1); }                                                                    // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange> BOOST_MULTI_HD constexpr auto                                                    operator[](A1 arg1, A2 arg2) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1, arg2); }                                                     // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange> BOOST_MULTI_HD constexpr auto                                 operator[](A1 arg1, A2 arg2, A3 arg3) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1, arg2, arg3); }                                      // NOLINT(whitespace/line_length) pattern line
	template<class A1 = irange, class A2 = irange, class A3 = irange, class A4 = irange, class... As> BOOST_MULTI_HD constexpr auto operator[](A1 arg1, A2 arg2, A3 arg3, A4 arg4, As... args) && -> decltype(auto) { return std::move(*this).paren_aux_(arg1, arg2, arg3, arg4, args...); }  // NOLINT(whitespace/line_length) pattern line
	#endif
	// clang-format on
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

 private:
	template<class Self, typename Tuple, std::size_t... I>
	static BOOST_MULTI_HD constexpr auto apply_impl_(Self&& self, Tuple const& tuple, std::index_sequence<I...> /*012*/) -> decltype(auto) {
		using std::get;  // for C++17 compatibility
		return std::forward<Self>(self)(get<I>(tuple)...);
	}

 public:
	using const_subarray<T, D, ElementPtr, Layout>::apply;
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	template<typename Tuple> BOOST_MULTI_HD constexpr auto apply(Tuple const& tpl) && -> decltype(auto) { return apply_impl_(std::move(*this), tpl, std::make_index_sequence<std::tuple_size_v<Tuple>>()); }
	template<typename Tuple> BOOST_MULTI_HD constexpr auto apply(Tuple const& tpl) & -> decltype(auto) { return apply_impl_(*this, tpl, std::make_index_sequence<std::tuple_size_v<Tuple>>()); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using const_subarray<T, D, ElementPtr, Layout>::partitioned;
	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto partitioned(size_type size) & -> subarray<T, D + 1, typename subarray::element_ptr> { return this->partitioned_aux_(size); }

	// cppcheck-suppress duplInheritedMember ; to overwrite
	BOOST_MULTI_HD constexpr auto partitioned(size_type size) && -> subarray<T, D + 1, typename subarray::element_ptr> { return this->partitioned_aux_(size); }

	using const_subarray<T, D, ElementPtr, Layout>::flatted;
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto flatted() & {
		// assert(is_flattable() && "flatted doesn't work for all layouts!");  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		multi::layout_t<D - 1> new_layout{this->layout().sub()};
		new_layout.nelems() *= this->size();  // TODO(correaa) : use immutable layout
		return subarray<T, D - 1, ElementPtr>(new_layout, this->base_);
	}
	constexpr auto flatted() && { return this->flatted(); }  // cppcheck-suppress duplInheritedMember ; to override

	using const_subarray<T, D, ElementPtr, Layout>::reinterpret_array_cast;

	template<class T2, class P2 = typename std::pointer_traits<ElementPtr>::template rebind<T2>>
	constexpr auto reinterpret_array_cast() & {  // cppcheck-suppress duplInheritedMember ; to override
		// static_assert( sizeof(T)%sizeof(T2) == 0,
		//  "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );

		return subarray<T2, D, P2>(
			this->layout().scale(sizeof(T), sizeof(T2)),  // NOLINT(bugprone-sizeof-expression) : sizes are compatible according to static assert above
			reinterpret_pointer_cast<P2>(this->base_)     // if ADL gets confused here (e.g. multi:: and thrust::) then adl_reinterpret_pointer_cast will be necessary
		);
	}

	template<class T2, class P2 = typename std::pointer_traits<ElementPtr>::template rebind<T2>>
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto reinterpret_array_cast() && {
		// static_assert( sizeof(T)%sizeof(T2) == 0,
		//  "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );

		return subarray<T2, D, P2>(
			this->layout().scale(sizeof(T), sizeof(T2)),  // NOLINT(bugprone-sizeof-expression) : sizes are compatible according to static assert above
			reinterpret_pointer_cast<P2>(this->base_)     // if ADL gets confused here (e.g. multi:: and thrust::) then adl_reinterpret_pointer_cast will be necessary
		);
	}

 private:
	template<typename P2>
	constexpr static auto reinterpret_pointer_cast_(ElementPtr const& ptr) -> decltype(auto) {
		if constexpr(std::is_pointer_v<ElementPtr>) {
			return static_cast<P2>(static_cast<void*>(ptr));  // NOLINT(bugprone-casting-through-void) direct reinterepret_cast doesn't work here
		} else {
			return reinterpret_cast<P2 const&>(ptr);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,bugprone-casting-through-void) direct reinterepret_cast doesn't work here
		}
	}

 public:
	template<class T2, class P2 = typename std::pointer_traits<ElementPtr>::template rebind<T2>>
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto reinterpret_array_cast(size_type count) & {
		static_assert(sizeof(T) % sizeof(T2) == 0, "error: reinterpret_array_cast is limited to integral stride values");

		BOOST_MULTI_ASSERT(sizeof(T) == sizeof(T2) * static_cast<std::size_t>(count));  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : checck implicit size compatibility

		layout_t<D + 1> const l1{this->layout().scale(sizeof(T), sizeof(T2)), 1, 0, count};
		auto const            l2 = l1.rotate();
		return subarray<T2, D + 1, P2>(
			l2,
			reinterpret_pointer_cast_<P2>(this->base_)  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,bugprone-casting-through-void) direct reinterepret_cast doesn't work here
		);
	}

	template<class T2, class P2 = typename std::pointer_traits<ElementPtr>::template rebind<T2>>
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto reinterpret_array_cast(size_type count) && {
		static_assert(sizeof(T) % sizeof(T2) == 0, "error: reinterpret_array_cast is limited to integral stride values");

		BOOST_MULTI_ASSERT(sizeof(T) == sizeof(T2) * static_cast<std::size_t>(count));  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : checck implicit size compatibility

		return subarray<T2, D + 1, P2>(
			layout_t<D + 1>(this->layout().scale(sizeof(T), sizeof(T2)), 1, 0, count).rotate(),
			reinterpret_pointer_cast_<P2>(this->base_)  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,bugprone-casting-through-void) direct reinterepret_cast doesn't work here
		);
	}

	using element_move_ptr = multi::move_ptr<typename subarray::element, typename subarray::element_ptr>;
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto element_moved() & { return subarray<T, D, typename subarray::element_move_ptr, Layout>(this->layout(), element_move_ptr{this->base_}); }
	constexpr auto element_moved() && { return element_moved(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	template<class Archive>
	auto serialize(Archive& arxiv, unsigned int /*version*/) {  // cppcheck-suppress duplInheritedMember ; to override
		using AT = multi::archive_traits<Archive>;
		// if(version == 0) {
		//  std::for_each(this->begin(), this->end(), [&](typename subarray::reference item) {arxiv & AT    ::make_nvp("item", item);});
		// } else {
		std::for_each(this->elements().begin(), this->elements().end(), [&](typename subarray::element& elem) { arxiv& AT ::make_nvp("elem", elem); });
		//}
		//  std::for_each(this->begin(), this->end(), [&](auto&& item) {arxiv & cereal::make_nvp("item", item);});
		//  std::for_each(this->begin(), this->end(), [&](auto&& item) {arxiv &                          item ;});
	}
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class Element, typename Ptr> struct array_iterator<Element, 0, Ptr> {};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class Element, typename Ptr, bool IsConst, bool IsMove, typename Stride>
struct array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>  // NOLINT(fuchsia-multiple-inheritance,cppcoreguidelines-pro-type-member-init,hicpp-member-init) stride_ is not initialized in some constructors
: boost::multi::iterator_facade<
	  array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>,
	  Element, std::random_access_iterator_tag,
	  std::conditional_t<
		  IsConst,
		  typename std::iterator_traits<typename std::pointer_traits<Ptr>::template rebind<Element const>>::reference,
		  std::conditional_t<
			  IsMove,
			  std::add_rvalue_reference_t<std::decay_t<typename std::iterator_traits<Ptr>::reference>>,
			  typename std::iterator_traits<Ptr>::reference>>,
	  multi::difference_type>
// , multi::affine<array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>, multi::difference_type>
// , multi::decrementable<array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>>
// , multi::incrementable<array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>>
// , multi::totally_ordered2<array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>, void>
{
	using affine = multi::affine<array_iterator<Element, 1, Ptr, IsConst, IsMove, Stride>, multi::difference_type>;

	using pointer = std::conditional_t<
		IsConst,
		typename std::pointer_traits<Ptr>::template rebind<Element const>,
		Ptr>;

 private:
	using reference_aux = std::conditional_t<
		IsConst,
		typename std::iterator_traits<typename std::pointer_traits<Ptr>::template rebind<Element const>>::reference,
		typename std::iterator_traits<Ptr>::reference>;

 public:
	using stride_type = Stride;  // multi::index;
	using reference   = std::conditional_t<
		  IsMove,
		  std::add_rvalue_reference_t<std::decay_t<reference_aux>>,
		  reference_aux>;

	using difference_type   = typename affine::difference_type;
	using iterator_category = typename stride_traits<Stride>::category;
	using iterator_concept  = typename stride_traits<Stride>::category;
	using element_type      = typename std::pointer_traits<Ptr>::element_type;  // workaround for clang 15 and libc++ in c++20 mode

	template<class Element2>
	using rebind = array_iterator<std::decay_t<Element2>, 1, typename std::pointer_traits<Ptr>::template rebind<Element2>, IsConst, IsMove, Stride>;

	static constexpr dimensionality_type dimensionality = 1;

#if defined(__cplusplus) && __cplusplus >= 202002L && (!defined(__clang__) || __clang_major__ != 10)
	// template<class T = void,
	//  std::enable_if_t<sizeof(T*) && std::is_base_of_v<std::contiguous_iterator_tag, iterator_category>, int> =0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	constexpr explicit operator Ptr() const& {
		static_assert(std::is_base_of_v<std::contiguous_iterator_tag, iterator_category>, "iterator must be continuous");
		return ptr_;
	}
#endif

	array_iterator()  = default;  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	using layout_type = multi::layout_t<0>;

	template<
		bool OtherIsConst, std::enable_if_t<!OtherIsConst, int> = 0  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
		>
	// NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	BOOST_MULTI_HD constexpr array_iterator(array_iterator<Element, 1, Ptr, OtherIsConst, IsMove, Stride> const& other)
	: ptr_{other.base()}, stride_{other.stride()} {}

	template<
		class Other,
		decltype(multi::detail::implicit_cast<Ptr>(typename Other::pointer{}))* = nullptr,
		decltype(std::declval<Other const&>().base())*                          = nullptr>
	// cppcheck-suppress noExplicitConstructor ; because underlying pointer is implicitly convertible
	BOOST_MULTI_HD constexpr /*mplct*/ array_iterator(Other const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to reproduce the implicitness of the argument
	: ptr_{other.base()}, stride_{other.stride()} {}

	template<
		class Other,
		decltype(multi::detail::explicit_cast<Ptr>(typename Other::pointer{}))* = nullptr,
		decltype(std::declval<Other const&>().data_)*                           = nullptr>
	constexpr explicit array_iterator(Other const& other)
	: ptr_{other.data_}, stride_{other.stride_} {}

	template<class, dimensionality_type, class, bool, bool, typename, class> friend struct array_iterator;

	template<
		class EElement, typename PPtr,
		typename = decltype(multi::detail::implicit_cast<Ptr>(std::declval<array_iterator<EElement, 1, PPtr>>().data_))>
	BOOST_MULTI_HD constexpr /*impl*/ array_iterator(array_iterator<EElement, 1, PPtr> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to reproduce the implicitness of original pointer
	: ptr_{other.base()}, stride_{other.stride_} {}

	BOOST_MULTI_HD constexpr explicit operator bool() const { return static_cast<bool>(this->ptr_); }

	BOOST_MULTI_HD constexpr auto operator[](typename array_iterator::difference_type n) const -> decltype(auto) {
		return *((*this) + n);
	}

	constexpr auto operator->() const { return static_cast<pointer>(ptr_); }

	constexpr auto segment() const {}

	using element     = Element;
	using element_ptr = Ptr;

	static constexpr dimensionality_type rank_v = 1;

	using rank = std::integral_constant<dimensionality_type, rank_v>;

	BOOST_MULTI_HD constexpr explicit array_iterator(typename subarray<element, 0, element_ptr>::element_ptr base, layout_t<0> const& /*lyt*/, Stride stride)
	: ptr_(std::move(base) /*, lyt*/), stride_{stride} {}

 private:
	friend struct const_subarray<Element, 1, Ptr>;  // TODO(correaa) fix template parameters

	element_ptr ptr_;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4820)  // warning C4820:  '7' bytes padding added after data member 'boost::multi::array_types<T,2,ElementPtr,Layout>::base_' [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\test\array_fancyref.cpp.x.vcxproj]
#endif
	BOOST_MULTI_NO_UNIQUE_ADDRESS
	stride_type stride_;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

 public:
	BOOST_MULTI_HD constexpr auto operator+(difference_type n) const { return array_iterator{*this} += n; }
	BOOST_MULTI_HD constexpr auto operator-(difference_type n) const { return array_iterator{*this} -= n; }

	BOOST_MULTI_HD constexpr auto base() const { return static_cast<pointer>(ptr_); }

	[[deprecated("use base() for iterator")]]
	BOOST_MULTI_HD constexpr auto data() const { return base(); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto base(array_iterator const& self) { return self.base(); }

	BOOST_MULTI_HD constexpr auto stride() const -> stride_type { return stride_; }
	friend constexpr auto         stride(array_iterator const& self) -> stride_type { return self.stride_; }

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif
	BOOST_MULTI_HD constexpr auto operator++() -> array_iterator& {
		ptr_ += stride_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}
	BOOST_MULTI_HD constexpr auto operator--() -> array_iterator& {
		ptr_ -= stride_;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> array_iterator& {
		ptr_ = ptr_ + stride_ * n;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}
	BOOST_MULTI_HD constexpr auto operator-=(difference_type n) -> array_iterator& {
		ptr_ -= stride_ * n;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}
#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

	BOOST_MULTI_HD constexpr auto operator-(array_iterator const& other) const -> difference_type {
		BOOST_MULTI_ASSERT(stride() != 0);
		BOOST_MULTI_ASSERT(stride() == other.stride());
		BOOST_MULTI_ASSERT((ptr_ - other.ptr_) % stride() == 0);
		return (ptr_ - other.ptr_) / stride();  // with struct-overflow=3 error: assuming signed overflow does not occur when simplifying `X - Y > 0` to `X > Y` [-Werror=strict-overflow]
	}

	BOOST_MULTI_HD constexpr auto operator==(array_iterator const& other) const noexcept {
		BOOST_MULTI_ASSERT(this->stride_ == other.stride_);
		return this->ptr_ == other.ptr_;
	}

	BOOST_MULTI_HD constexpr auto operator!=(array_iterator const& other) const noexcept {
		BOOST_MULTI_ASSERT(this->stride_ == other.stride_);
		return this->ptr_ != other.ptr_;
	}

	template<bool OtherIsConst, std::enable_if_t<OtherIsConst != IsConst, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	BOOST_MULTI_HD constexpr auto operator==(array_iterator<Element, 1, Ptr, OtherIsConst> const& other) const -> bool {
		BOOST_MULTI_ASSERT(this->stride_ == other.stride_);
		BOOST_MULTI_ASSERT(stride_ != 0);
		return this->ptr_ == other.ptr_;
	}

	BOOST_MULTI_HD constexpr auto operator<(array_iterator const& other) const -> bool {
		return 0 < other - *this;
	}

	BOOST_MULTI_HD constexpr auto operator*() const noexcept -> reference {
		return static_cast<reference>(*ptr_);
	}

	// BOOST_MULTI_HD constexpr auto segment() const -> segment_type {
	// }
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class Element, dimensionality_type D, typename... Ts>
using iterator = array_iterator<Element, D, Ts...>;

template<typename T, typename ElementPtr, class Layout>
class const_subarray<T, 0, ElementPtr, Layout>
: public array_types<T, 0, ElementPtr, Layout> {
 public:
	using types = array_types<T, 0, ElementPtr, Layout>;
	using types::types;

	using element      = typename types::element;
	using element_ref  = typename std::iterator_traits<typename const_subarray::element_ptr>::reference;
	using element_cref = typename std::iterator_traits<typename const_subarray::element_const_ptr>::reference;
	using iterator     = array_iterator<T, 0, ElementPtr>;

	using layout_type = Layout;

	constexpr auto operator=(element const& elem) & -> const_subarray& {
		//  MULTI_MARK_SCOPE(std::string{"multi::operator= D=0 from "}+typeid(T).name()+" to "+typeid(T).name() );
		adl_copy_n(&elem, 1, this->base_);
		return *this;
	}
	constexpr auto operator=(element const& elem) && -> const_subarray& {
		operator=(elem);
		return *this;  // lints(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
	}

	constexpr auto operator==(element const& elem) const -> bool {
		BOOST_MULTI_ASSERT(this->num_elements() == 1);
		return adl_equal(&elem, std::next(&elem, this->num_elements()), this->base());
	}
	constexpr auto operator!=(element const& elem) const { return !operator==(elem); }

	template<class Range0>
	constexpr auto operator=(Range0 const& rng) & -> const_subarray& {
		adl_copy_n(&rng, 1, this->base_);
		return *this;
	}

	constexpr auto elements_at(size_type idx [[maybe_unused]]) const& -> element_cref {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		return *(this->base_);
	}
	constexpr auto elements_at(size_type idx [[maybe_unused]]) && -> element_ref {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		return *(this->base_);
	}
	constexpr auto elements_at(size_type idx [[maybe_unused]]) & -> element_ref {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		return *(this->base_);
	}

	constexpr auto operator!=(const_subarray const& other) const { return !adl_equal(other.base_, other.base_ + 1, this->base_); }
	constexpr auto operator==(const_subarray const& other) const { return adl_equal(other.base_, other.base_ + 1, this->base_); }

	constexpr auto operator<(const_subarray const& other) const {
		return adl_lexicographical_compare(
			this->base_, this->base_ + this->num_elements(),
			other.base_, other.base_ + other.num_elements()
		);
	}

	using decay_type = typename types::element;

	BOOST_MULTI_HD constexpr auto operator()() const& -> element_ref { return *(this->base_); }  // NOLINT(hicpp-explicit-conversions)

	constexpr operator element_ref() && noexcept { return *(this->base_); }       // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax
	constexpr operator element_ref() & noexcept { return *(this->base_); }        // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax
	constexpr operator element_cref() const& noexcept { return *(this->base_); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax

	constexpr auto elements() const&;

	constexpr auto begin() const& = delete;
	constexpr auto end() const&   = delete;

#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	constexpr auto operator[]() const& = delete;
#else
	template<class IndexType>
	constexpr auto operator[](IndexType const&) const& = delete;
#endif

	auto           diagonal() const     = delete;
	constexpr auto sliced() const&      = delete;
	constexpr auto partitioned() const& = delete;

	constexpr auto strided(difference_type) const& = delete;

	constexpr auto taked(difference_type) const&   = delete;
	constexpr auto dropped(difference_type) const& = delete;

	BOOST_MULTI_HD constexpr auto reindexed() const& { return operator()(); }
	BOOST_MULTI_HD constexpr auto rotated() const& { return operator()(); }
	BOOST_MULTI_HD constexpr auto unrotated() const& { return operator()(); }

	auto transposed() const&              = delete;
	auto flatted() const&                 = delete;
	auto range() const& -> const_subarray = delete;

	using cursor       = cursor_t<typename const_subarray::element_ptr, 0, typename const_subarray::strides_type>;
	using const_cursor = cursor_t<typename const_subarray::element_const_ptr, 0, typename const_subarray::strides_type>;

 private:
	BOOST_MULTI_HD constexpr auto home_aux_() const { return cursor(this->base_, this->strides()); }

 public:
	BOOST_MULTI_HD constexpr auto home() const& -> const_cursor { return home_aux_(); }

 private:
	template<typename, multi::dimensionality_type, typename, class> friend class subarray;

	auto paren_aux_() const& { return operator()(); }

 public:
	template<class Tuple>
	BOOST_MULTI_HD constexpr auto apply(Tuple const& /*unused*/) const {
		static_assert(std::tuple_size_v<Tuple> == 0);
		return operator()();
	}

	BOOST_MULTI_HD constexpr auto operator&() const& {  // NOLINT(google-runtime-operator)
		return /*TODO(correaa) add const*/ subarray_ptr<T, 0, ElementPtr, Layout, false>(this->base_, this->layout());
	}  // NOLINT(google-runtime-operator) extend semantics  //NOSONAR

	template<class T2, class P2 = typename std::pointer_traits<ElementPtr>::template rebind<T2>>
	constexpr auto reinterpret_array_cast() const& {
		return const_subarray<T2, 0, P2>{
			typename const_subarray::layout_type{this->layout()},
			reinterpret_pointer_cast<P2>(this->base_)
		};
	}

	constexpr auto broadcasted() const& {
		multi::layout_t<1> const new_layout(this->layout(), 0, 0);  // , (std::numeric_limits<size_type>::max)());  // paren for MSVC macros
		return subarray<T, 1, typename const_subarray::element_const_ptr>(new_layout, types::base_);
	}

	template<class Archive>
	auto serialize(Archive& arxiv, unsigned int const /*version*/) const {
		using AT        = multi::archive_traits<Archive>;
		auto&  element_ = *(this->base_);
		arxiv& AT::make_nvp("element", element_);
		//  arxiv & cereal::make_nvp("element", element_);
		//  arxiv &                             element_ ;
	}
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<typename T, typename ElementPtr, class Layout>
struct const_subarray<T, 1, ElementPtr, Layout>  // NOLINT(fuchsia-multiple-inheritance) to define operators via CRTP
: multi::random_iterable<const_subarray<T, 1, ElementPtr, Layout>>
, array_types<T, 1, ElementPtr, Layout> {
	~const_subarray() = default;  // lints(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)

	template<class TT, std::enable_if_t<std::is_same_v<ElementPtr, TT const*>, int> = 0>   // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	explicit BOOST_MULTI_HD constexpr const_subarray(std::initializer_list<TT> const& il)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) this constructs a reference to the init list
	: array_types<T, 1, ElementPtr, Layout>(
		  layout_type(multi::extensions_t<1>(
			  {0, static_cast<size_type>(std::size(il))}
		  )),
		  std::data(il)
	  ) {
	}

	template<class TT, std::enable_if_t<std::is_same_v<ElementPtr, TT const*>, int> = 0>  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	/*explicit*/ BOOST_MULTI_HD constexpr const_subarray(std::initializer_list<TT>&& il)
	: array_types<T, 1, ElementPtr, Layout>(
		  layout_type(multi::extensions_t<1>(
			  {0, static_cast<size_type>(std::size(il))}
		  )),
		  std::data(il)
	  ) {
		(void)std::move(il);
	}

	// boost serialization needs `delete`. void boost::serialization::extended_type_info_typeid<T>::destroy(const void*) const [with T = boost::multi::subarray<double, 1, double*, boost::multi::layout_t<1> >]
	// void operator delete(void* ptr) noexcept = delete;
	// void operator delete(void* ptr, void* place ) noexcept = delete;  // NOLINT(bugprone-easily-swappable-parameters)

	static constexpr dimensionality_type rank_v = 1;

	using types = array_types<T, dimensionality_type{1}, ElementPtr, Layout>;
	using types::types;

	using rank        = std::integral_constant<dimensionality_type, rank_v>;
	using layout_type = Layout;
	using ref_        = const_subarray;

	using element_type = T;

	using element_ptr       = typename types::element_ptr;
	using element_const_ptr = typename std::pointer_traits<ElementPtr>::template rebind<element_type const>;
	using element_move_ptr  = multi::move_ptr<element_type, element_ptr>;
	using element_ref       = typename types::element_ref;
	using element_cref      = typename std::iterator_traits<element_const_ptr>::reference;

	using const_pointer   = element_const_ptr;
	using pointer         = element_ptr;
	using const_reference = typename array_types<T, dimensionality_type{1}, ElementPtr, Layout>::const_reference;
	using reference       = typename array_types<T, dimensionality_type{1}, ElementPtr, Layout>::reference;

	using default_allocator_type = typename multi::pointer_traits<typename const_subarray::element_ptr>::default_allocator_type;

	BOOST_MULTI_HD constexpr auto get_allocator() const -> default_allocator_type { return default_allocator_of(const_subarray::base()); }
	BOOST_MULTI_FRIEND_CONSTEXPR
	auto get_allocator(const_subarray const& self) -> default_allocator_type { return self.get_allocator(); }

	using decay_type = array<typename types::element, dimensionality_type{1}, typename multi::pointer_traits<typename const_subarray::element_ptr>::default_allocator_type>;

	constexpr auto                    decay() const -> decay_type { return decay_type{*this}; }
	BOOST_MULTI_FRIEND_CONSTEXPR auto decay(const_subarray const& self) -> decay_type { return self.decay(); }

	using basic_const_array = const_subarray<
		T, 1,
		typename std::pointer_traits<ElementPtr>::template rebind<typename const_subarray::element_type const>,
		Layout>;

 protected:
	template<class A> constexpr void intersection_assign(A&& other) && { intersection_assign(std::forward<A>(other)); }
	template<class A> constexpr void intersection_assign(A&& other) & {  // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved,cppcoreguidelines-missing-std-forward) false positive clang-tidy 17
		std::for_each(
			intersection(types::extension(), extension(other)).begin(),
			intersection(types::extension(), extension(other)).end(),
			[&](auto const idx) { operator[](idx) = std::forward<A>(other)[idx]; }
		);
	}

	template<typename, ::boost::multi::dimensionality_type, typename EP, class LLayout> friend struct const_subarray;
	template<typename, ::boost::multi::dimensionality_type, class Alloc> friend struct dynamic_array;  // TODO(correaa) check if this is necessary

	template<class T2, class P2, class TT, dimensionality_type DD, class PP>
	friend constexpr auto static_array_cast(subarray<TT, DD, PP> const&) -> decltype(auto);

 public:
	const_subarray(const_subarray const&) = delete;

	friend constexpr auto sizes(const_subarray const& self) noexcept -> typename const_subarray::sizes_type { return self.sizes(); }  // needed by nvcc
	friend constexpr auto size(const_subarray const& self) noexcept -> typename const_subarray::size_type { return self.size(); }     // needed by nvcc

	constexpr auto operator+() const { return decay(); }

	const_subarray(const_subarray&&) noexcept = default;  // in C++ 14 this was necessary to return array references from functions
	// in c++17 things changed and non-moveable non-copyable types can be returned from functions and captured by auto

 protected:
	template<typename, multi::dimensionality_type, typename, class, bool> friend struct subarray_ptr;
	template<class, dimensionality_type D, class, bool, bool, typename, class> friend struct array_iterator;

 public:
	friend constexpr auto dimensionality(const_subarray const& /*self*/) -> dimensionality_type { return 1; }

	BOOST_MULTI_HD constexpr auto operator&() const& { return const_subarray_ptr<T, 1, ElementPtr, Layout>{this->base_, this->layout()}; }  // NOLINT(google-runtime-operator) extend semantics  //NOSONAR

	BOOST_MULTI_HD constexpr void assign(std::initializer_list<typename const_subarray::value_type> values) const {
		BOOST_MULTI_ASSERT(values.size() == static_cast<std::size_t>(this->size()));
		if(values.size() != 0) {
			assign(values.begin(), values.end());
		}
	}
	template<class It>
	constexpr auto assign(It first) & -> It {
		adl_copy_n(first, this->size(), this->begin());
		std::advance(first, this->size());
		return first;
	}
	template<class It>
	constexpr auto assign(It first) && -> It { return assign(first); }
	template<class It>
	constexpr void assign(It first, It last) & {
		BOOST_MULTI_ASSERT(std::distance(first, last) == this->size());
		(void)last;  // N_O_L_I_N_T(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		assign(first);
	}
	template<class It>
	constexpr void assign(It first, It last) && { assign(first, last); }

	// constexpr auto operator=(const_subarray     &&) const& noexcept -> const_subarray const&;  // UNIMPLEMENTABLE! TO PASS THE viewable_range CONCEPT!!!, can't be = delete;
	constexpr auto operator=(const_subarray&&) & noexcept -> const_subarray&;  // UNIMPLEMENTABLE! TO PASS THE viewable_range CONCEPT!!!, can't be = delete;
	constexpr auto operator=(const_subarray const&) const -> const_subarray const& = delete;

	template<
		class ECPtr,
		class = std::enable_if_t<std::is_same_v<element_const_ptr, ECPtr> && !std::is_same_v<element_const_ptr, element_ptr>>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
		>
	constexpr auto operator=(const_subarray<T, 1L, ECPtr, Layout> const& other) const&& -> const_subarray& {
		assert(0);
		operator=(other);
		return *this;
	}  // required by https://en.cppreference.com/w/cpp/iterator/indirectly_writable for std::ranges::copy_n

	using cursor       = cursor_t<typename const_subarray::element_ptr, 1, typename const_subarray::strides_type>;
	using const_cursor = cursor_t<typename const_subarray::element_const_ptr, 1, typename const_subarray::strides_type>;

	auto diagonal() const = delete;

 private:
	BOOST_MULTI_HD constexpr auto home_aux_() const { return cursor(this->base_, this->strides()); }

 public:
	BOOST_MULTI_HD constexpr auto home() const& -> const_cursor { return home_aux_(); }

 private:
	template<typename, multi::dimensionality_type, typename, class> friend class subarray;

	BOOST_MULTI_HD constexpr auto at_aux_(index idx) const -> typename const_subarray::reference {  // NOLINT(readability-const-return-type) fancy pointers can deref into const values to avoid assignment
		if constexpr(std::is_integral_v<decltype(this->stride())>) {
			BOOST_MULTI_ASSERT((this->stride() == 0 || (this->extension().contains(idx))) && ("out of bounds"));
		}

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif
		return *(this->base_ + (this->stride() * idx - this->offset()));  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif
	}

 public:
	constexpr auto broadcasted() const& {
		// multi::layout_t<1> const self_layout{this->layout()};
		// TODO(correaa) introduce a broadcasted_layout?
		multi::layout_t<2> const new_layout(this->layout(), 0, 0, 1);  // , (std::numeric_limits<size_type>::max)()};
		return const_subarray<T, 2, ElementPtr, multi::layout_t<2>>(new_layout, types::base_);
	}

	constexpr auto repeated(multi::size_t n) const& {
		return [this](auto /*idx*/, auto... rest) { return detail::invoke_square(*this, rest...); } ^ (n * this->extensions());
	}

	template<template<class...> class Container = std::vector, class... As>
	constexpr auto to(As&&... as) const& {
		using inner_value_type = typename const_subarray::value_type;
		using container_type   = Container<inner_value_type>;
		return container_type(this->begin(), this->end(), std::forward<As>(as)...);
	}

	BOOST_MULTI_HD constexpr auto operator[](index idx) const& -> typename const_subarray::const_reference { return at_aux_(idx); }  // NOLINT(readability-const-return-type) fancy pointers can deref into const values to avoid assignment

	BOOST_MULTI_HD constexpr auto front() const& -> const_reference { return *begin(); }
	BOOST_MULTI_HD constexpr auto back() const& -> const_reference { return *std::prev(end(), 1); }

 private:
	template<class Self, typename Tuple, std::size_t... I, const_subarray* = nullptr>
	static constexpr auto apply_impl_(Self&& self, Tuple const& tuple, std::index_sequence<I...> /*012*/) -> decltype(auto) {
		using std::get;  // for C++17 compatibility
		return std::forward<Self>(self)(get<I>(tuple)...);
	}

 public:
	template<typename Tuple> BOOST_MULTI_HD constexpr auto apply(Tuple const& tuple) const& -> decltype(auto) { return apply_impl_(*this, tuple, std::make_index_sequence<std::tuple_size_v<Tuple>>()); }

	// template<class Tuple, std::enable_if_t<(std::tuple_size<Tuple>::value == 0), int> = 0> BOOST_MULTI_HD constexpr auto operator[](Tuple const& /*empty*/) const& -> decltype(auto) { return *this; }  // NOLINT(modernize-use-constraints) for C++20
	// template<class Tuple, std::enable_if_t<(std::tuple_size<Tuple>::value == 1), int> = 0> BOOST_MULTI_HD constexpr auto operator[](Tuple const& indices) const& -> decltype(auto) {                    // NOLINT(modernize-use-constraints) for C++20
	// 	using std::get;
	// 	return operator[](get<0>(indices));
	// }

	// template<class Tuple, std::enable_if_t<(std::tuple_size<Tuple>::value > 1), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	// BOOST_MULTI_HD constexpr auto operator[](Tuple const& indices) const& -> decltype(operator[](std::get<0>(indices))[detail::tuple_tail(indices)]) {
	// 	using std::get;  // for C++17 compatibility
	// 	return operator[](get<0>(indices))[detail::tuple_tail(indices)];
	// }

// Warning C4459 comes from boost::multi_array having a namespace indices which collides with the variable name?
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4459)
#endif

	[[deprecated("BMA compat, finish impl")]] constexpr auto operator[](std::tuple<irange> const& indices) const& {
		using std::get;
		return (*this)({get<0>(indices).front(), get<0>(indices).back() + 1});
	}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

	BOOST_MULTI_HD constexpr auto elements_at(size_type idx) const& -> decltype(auto) {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		return operator[](idx);
	}
	BOOST_MULTI_HD constexpr auto elements_at(size_type idx) && -> decltype(auto) {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		return operator[](idx);
	}
	BOOST_MULTI_HD constexpr auto elements_at(size_type idx) & -> decltype(auto) {
		BOOST_MULTI_ASSERT(idx < this->num_elements());
		return operator[](idx);
	}

	constexpr auto reindexed(index first) && { return reindexed(first); }
	constexpr auto reindexed(index first) & { return const_subarray{this->layout().reindex(first), types::base_}; }

 private:
	BOOST_MULTI_HD constexpr auto taked_aux_(difference_type count) const {
		BOOST_MULTI_ASSERT(count <= this->size());  // calculating size is expensive that is why
		typename types::layout_t const new_layout{
			this->layout().sub(),
			this->layout().stride(),
			this->layout().offset(),
			this->stride() * count
		};
		return const_subarray{new_layout, this->base_};
	}

 public:
	constexpr auto taked(difference_type count) const& -> const_subarray<T, 1, ElementPtr, Layout> { return taked_aux_(count); }

 private:
	BOOST_MULTI_HD constexpr auto dropped_aux_(difference_type count) const {

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

		return const_subarray(
			this->layout().drop(count), this->base_ + (count * this->layout().stride() /*- this->layout().offset()*/)  // TODO(correaa) fix need for offset  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		);

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif
	}

 public:
	constexpr auto dropped(difference_type count) const& -> const_subarray { return dropped_aux_(count); }

 private:
	BOOST_MULTI_HD constexpr auto sliced_aux_(index first, index last) const {

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

		// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return const_subarray{this->layout().slice(first, last), this->base_ + (first * this->layout().stride() /*- this->layout().offset()*/)};  // TODO(correaa) fix need for offset

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif
	}

 public:
	BOOST_MULTI_HD constexpr auto sliced(index first, index last) const& -> basic_const_array { return basic_const_array{sliced_aux_(first, last)}; }
	BOOST_MULTI_HD constexpr auto sliced(index first, index last) & -> const_subarray { return sliced_aux_(first, last); }
	BOOST_MULTI_HD constexpr auto sliced(index first, index last) && -> const_subarray { return sliced_aux_(first, last); }

	using elements_iterator  = elements_iterator_t<element_ptr, layout_type>;
	using celements_iterator = elements_iterator_t<element_const_ptr, layout_type>;

	using elements_range       = elements_range_t<element_ptr, layout_type>;
	using const_elements_range = elements_range_t<element_const_ptr, layout_type>;

 private:
	constexpr auto elements_aux_() const { return elements_range{this->base_, this->layout()}; }

 public:
	constexpr auto elements() & -> elements_range { return elements_aux_(); }
	constexpr auto elements() && -> elements_range { return elements_aux_(); }
	constexpr auto elements() const& -> const_elements_range { return const_elements_range{this->base(), this->layout()}; }  // TODO(correaa) simplify

	constexpr auto celements() const -> const_elements_range { return elements_aux_(); }

	constexpr auto hull() const -> std::pair<element_const_ptr, size_type> {
		return {(std::min)(this->base(), this->base() + this->hull_size()), std::abs(this->hull_size())};  // paren for MSVC macros
	}

	/*[[gnu::pure]]*/ constexpr auto blocked(index first, index last) & -> const_subarray {
		return sliced(first, last).reindexed(first);
	}
	/*[[gnu::pure]]*/ constexpr auto stenciled(typename const_subarray::index_extension ext) -> const_subarray {
		return blocked(ext.first(), ext.last());
	}

 private:
	constexpr auto strided_aux_(difference_type diff) const {
		auto const new_layout = typename types::layout_t{this->layout().sub(), this->layout().stride() * diff, this->layout().offset(), this->layout().nelems()};
		return subarray<T, 1, ElementPtr, Layout>(new_layout, types::base_);
	}

 public:
	constexpr auto strided(difference_type diff) const& -> const_subarray { return strided_aux_(diff); }

	BOOST_MULTI_HD constexpr auto sliced(index first, index last, difference_type stride) const& -> basic_const_array { return sliced(first, last).strided(stride); }

	BOOST_MULTI_HD constexpr auto range(index_range const& rng) const& { return sliced(rng.front(), rng.last()); }

 private:
	BOOST_MULTI_HD constexpr auto paren_aux_() const& { return const_subarray(this->layout(), this->base_); }

	BOOST_MULTI_HD constexpr auto paren_aux_(index idx) const& -> decltype(auto) { return operator[](idx); }

	BOOST_MULTI_HD constexpr auto paren_aux_(index_range const& rng) const& { return range(rng); }

 public:
	BOOST_MULTI_HD constexpr auto operator()() const& { return paren_aux_(); }
#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	BOOST_MULTI_HD constexpr auto operator[]() const& -> const_subarray { return paren_aux_(); }
#endif

	BOOST_MULTI_HD constexpr auto operator()(index idx) const -> decltype(auto) { return operator[](idx); }

	BOOST_MULTI_HD constexpr auto operator()(index_range const& rng) const& { return range(rng); }

 private:
	constexpr auto paren_aux_(intersecting_range<index> const& rng) const& -> decltype(auto) { return paren_aux_(intersection(this->extension(), rng)); }

 public:
	BOOST_MULTI_HD constexpr auto operator()(intersecting_range<index> const& isrange) const& -> decltype(auto) { return paren_aux_(isrange); }

	template<class... Args>
	BOOST_MULTI_HD constexpr auto operator()(Args&&... args) const& -> decltype(paren_(*this, std::forward<Args>(args)...)) {
		return paren_(*this, std::forward<Args>(args)...);
	}

 private:
	BOOST_MULTI_HD constexpr auto halved_aux_() const {
		auto new_layout = this->layout().halve();
		return subarray<T, 2, element_ptr>(new_layout, this->base_);
	}

 public:
	BOOST_MULTI_HD constexpr auto halved() const& -> const_subarray<T, 2, element_ptr> { return halved_aux_(); }

 private:
	BOOST_MULTI_HD constexpr auto partitioned_aux_(size_type size) const {
		BOOST_MULTI_ASSERT(size != 0);
		BOOST_MULTI_ASSERT((this->layout().nelems() % size) == 0);  // TODO(correaa) remove assert? truncate left over? (like mathematica) // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		multi::layout_t<2> new_layout{this->layout(), this->layout().nelems() / size, 0, this->layout().nelems()};
		new_layout.sub().nelems() /= size;  // TODO(correaa) : don't use mutation
		return subarray<T, 2, element_ptr>(new_layout, types::base_);
	}

 public:
	BOOST_MULTI_HD constexpr auto partitioned(size_type size) const& -> const_subarray<T, 2, element_ptr> { return partitioned_aux_(size); }

 private:
	BOOST_MULTI_HD constexpr auto chunked_aux_(size_type size) const {
		BOOST_MULTI_ASSERT(this->size() % size == 0);
		return partitioned_aux_(this->size() / size);
	}

 public:  // in Mathematica this is called Partition https://reference.wolfram.com/language/ref/Partition.html in RangesV3 it is called chunk
	BOOST_MULTI_HD constexpr auto chunked(size_type size) const& -> const_subarray<T, 2, element_ptr> { return chunked_aux_(size); }
	// BOOST_MULTI_HD constexpr auto chunked(size_type size)      & -> partitioned_type       {return chunked_aux_(size);}
	// BOOST_MULTI_HD constexpr auto chunked(size_type size)     && -> partitioned_type       {return chunked_aux_(size);}

	constexpr auto tiled(size_type count) const& {
		BOOST_MULTI_ASSERT(count != 0);
		struct divided_type {
			const_subarray<T, 2, element_ptr> quotient;
			const_subarray<T, 1, element_ptr> remainder;
		};
		return divided_type{
			this->taked(this->size() - (this->size() % count)).chunked(count),
			this->dropped(this->size() - (this->size() % count))
		};
	}

 private:
	constexpr auto reversed_aux_() const -> const_subarray {
		auto new_layout = this->layout();
		new_layout.reverse();
		return {new_layout, types::base_};
	}

 public:
	constexpr auto reversed() const& -> basic_const_array { return reversed_aux_(); }
	constexpr auto reversed() & -> const_subarray { return reversed_aux_(); }
	constexpr auto reversed() && -> const_subarray { return reversed_aux_(); }

	friend constexpr auto reversed(const_subarray const& self) -> basic_const_array { return self.reversed(); }
	friend constexpr auto reversed(const_subarray& self) -> const_subarray { return self.reversed(); }
	friend constexpr auto reversed(const_subarray&& self) -> const_subarray { return std::move(self).reversed(); }

	// friend constexpr auto   rotated(const_subarray const& self) -> decltype(auto) {return self.  rotated();}
	// friend constexpr auto unrotated(const_subarray const& self) -> decltype(auto) {return self.unrotated();}

	// constexpr auto   rotated()      & -> decltype(auto) {return operator()();}
	// constexpr auto   rotated()     && -> decltype(auto) {return operator()();}
	BOOST_MULTI_HD constexpr auto rotated() const& { return operator()(); }
	BOOST_MULTI_HD constexpr auto unrotated() const& { return operator()(); }

	auto transposed() const& = delete;
	auto flatted() const&    = delete;

	using iterator       = typename multi::array_iterator<element_type, 1, typename types::element_ptr, false, false, typename layout_type::stride_type>;
	using const_iterator = typename multi::array_iterator<element_type, 1, typename types::element_ptr, true, false, typename layout_type::stride_type>;
	using move_iterator  = typename multi::array_iterator<element_type, 1, typename types::element_ptr, false, true>;

	using reverse_iterator [[deprecated]]       = std::reverse_iterator<iterator>;
	using const_reverse_iterator [[deprecated]] = std::reverse_iterator<const_iterator>;

	struct [[deprecated("BMA compatibility")]] index_gen {
		auto operator[](irange const& rng) const { return std::make_tuple(rng); }
	};
	using extent_gen [[deprecated("BMA compatibility")]]   = std::array<irange, 1>;
	using extent_range [[deprecated("BMA compatibility")]] = irange;

	template<
		class Range,
		std::enable_if_t<!has_extensions<std::decay_t<Range>>::value, int> = 0,
		std::enable_if_t<!is_subarray<std::decay_t<Range>>::value, int>    = 0,
		class                                                              = decltype((void)std::declval<Range>().begin(), std::declval<Range>().end()),
		class                                                              = decltype(Range{std::declval<typename const_subarray::const_iterator>(), std::declval<typename const_subarray::const_iterator>()})>
	constexpr explicit operator Range() const {
		// vvv Range{...} needed by Windows GCC?
		return Range{begin(), end()};  // NOLINT(fuchsia-default-arguments-calls) e.g. std::vector(it, it, alloc = {})
	}

 private:
#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	BOOST_MULTI_HD constexpr auto begin_aux_() const { return iterator{this->base_, this->layout().sub(), this->stride()}; }
	BOOST_MULTI_HD constexpr auto end_aux_() const { return iterator{this->base_ + types::nelems(), this->layout().sub(), this->stride()}; }  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

 public:
	BOOST_MULTI_HD constexpr auto begin() const& -> const_iterator { return begin_aux_(); }
	BOOST_MULTI_HD constexpr auto end() const& -> const_iterator { return end_aux_(); }

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
	[[deprecated("implement as negative stride")]] constexpr auto rbegin() const& { return const_reverse_iterator(end()); }  // TODO(correaa) implement as negative stride?
	[[deprecated("implement as negative stride")]] constexpr auto rend() const& { return const_reverse_iterator(begin()); }  // TODO(correaa) implement as negative stride?
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

	BOOST_MULTI_HD constexpr auto cbegin() const& -> const_iterator { return begin(); }
	BOOST_MULTI_HD constexpr auto cend() const& -> const_iterator { return end(); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto cbegin(const_subarray const& self) { return self.cbegin(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto cend(const_subarray const& self) { return self.cend(); }

	template<class It> constexpr auto assign(It first) && -> decltype(adl_copy_n(first, std::declval<size_type>(), std::declval<iterator>()), void()) {
		return adl_copy_n(first, this->size(), std::move(*this).begin()), void();
	}

	friend constexpr auto operator==(const_subarray const& self, const_subarray const& other) -> bool {
		return self.extension() == other.extension() && self.elements() == other.elements();
	}

	friend constexpr auto operator!=(const_subarray const& self, const_subarray const& other) -> bool {
		return self.extension() != other.extension() || self.elements() != other.elements();
	}

	template<class OtherT, typename OtherEP, class OtherLayout>
	friend constexpr auto operator==(const_subarray const& self, const_subarray<OtherT, 1, OtherEP, OtherLayout> const& other) -> bool {
		return self.extension() == other.extension() && self.elements() == other.elements();
	}

	template<class TT, typename EEPP, class LL>
	friend constexpr auto operator!=(const_subarray const& self, const_subarray<TT, 1, EEPP, LL> const& other) -> bool {
		return self.extension() != other.extension() || self.elements() != other.elements();
	}

	friend constexpr auto operator<(const_subarray const& self, const_subarray const& other) -> bool { return lexicographical_compare_(self, other); }
	friend constexpr auto operator>(const_subarray const& self, const_subarray const& other) -> bool { return lexicographical_compare_(other, self); }  // NOLINT(readability-suspicious-call-argument)

	friend constexpr auto operator<=(const_subarray const& self, const_subarray const& other) -> bool { return lexicographical_compare_(self, other) || self == other; }
	friend constexpr auto operator>=(const_subarray const& self, const_subarray const& other) -> bool { return lexicographical_compare_(other, self) || self == other; }  // NOLINT(readability-suspicious-call-argument)

	template<class Range, typename = std::enable_if_t<!is_const_subarray_v<Range>>, typename = decltype(std::declval<Range const&>().extensions(), std::declval<Range const&>().elements())>
	friend constexpr auto operator==(const_subarray const& self, Range const& other) -> bool {
		return self.extensions() == other.extensions() && self.elements() == other.elements();
	}

	template<class Range, typename = std::enable_if_t<!is_const_subarray_v<Range>>, typename = decltype(std::declval<Range const&>().extensions(), std::declval<Range const&>().elements())>
	friend constexpr auto operator==(Range const& other, const_subarray const& self) -> bool {
		return self.extensions() == other.extensions() && self.elements() == other.elements();
	}

	template<class Range, typename = std::enable_if_t<!is_const_subarray_v<Range>>, typename = decltype(std::declval<Range const&>().extensions(), std::declval<Range const&>().elements())>
	friend constexpr auto operator!=(const_subarray const& self, Range const& other) -> bool {
		return self.extensions() != other.extensions() || self.elements() == other.elements();
	}

 private:
	template<class A1, class A2>
	static constexpr auto lexicographical_compare_(A1 const& self, A2 const& other) -> bool {  // NOLINT(readability-suspicious-call-argument)
		if(self.extension().first() > other.extension().first()) {
			return true;
		}
		if(self.extension().first() < other.extension().first()) {
			return false;
		}
		return adl_lexicographical_compare(adl_begin(self), adl_end(self), adl_begin(other), adl_end(other));
	}

 public:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>>
	constexpr auto static_array_cast() const -> subarray<T2, 1, P2, Layout> {  // name taken from std::static_pointer_cast
		return {this->layout(), static_cast<P2>(this->base_)};
	}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, class... Args>
	constexpr auto static_array_cast(Args&&... args) const -> subarray<T2, 1, P2, Layout> {  // name taken from std::static_pointer_cast
		return {
			this->layout(), P2{this->base_, std::forward<Args>(args)...}
		};
	}

	template<class UF>
	BOOST_MULTI_HD constexpr auto element_transformed(UF&& fun) const& {
		return static_array_cast<
			//  std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<UF const&, element_cref>>>,
			std::decay_t<std::invoke_result_t<UF const&, element_cref>>,
			transform_ptr<
				//  std::remove_cv_t<std::remove_reference_t<std::invoke_result_t<UF const&, element_cref>>>,
				std::decay_t<std::invoke_result_t<UF const&, element_cref>>,
				UF, element_const_ptr, std::invoke_result_t<UF const&, element_cref>>>(std::forward<UF>(fun));
	}
	template<class UF>
	BOOST_MULTI_HD constexpr auto element_transformed(UF&& fun) & {
		return static_array_cast<
			std::decay_t<std::invoke_result_t<UF const&, element_ref>>,
			transform_ptr<
				std::decay_t<std::invoke_result_t<UF const&, element_ref>>,
				UF, element_ptr, std::invoke_result_t<UF const&, element_ref>>>(std::forward<UF>(fun));
	}
	template<class UF>
	BOOST_MULTI_HD constexpr auto element_transformed(UF&& fun) && { return element_transformed(std::forward<UF>(fun)); }

	template<
		class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>,
		class Element = typename const_subarray::element,
		class PM      = T2 std::decay_t<Element>::*>
	constexpr auto member_cast(PM member) const {
		static_assert(sizeof(T) % sizeof(T2) == 0, "array_member_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. "
												   "Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements");

#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) reinterpret is what the function does. alternative for GCC/NVCC
		auto&& r1 = (*(reinterpret_cast<typename const_subarray::element_type* const&>(const_subarray::base_))).*member;  // ->*pm;
		auto*  p1 = &r1;
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) TODO(correaa) find a better way
		P2 p2 = reinterpret_cast<P2&>(p1);  // NOSONAR
#else
		auto p2 = static_cast<P2>(&(this->base_->*member));  // this crashes nvcc 11.2-11.4 and some? gcc compiler
#endif
		return subarray<T2, 1, P2>(this->layout().scale(sizeof(T), sizeof(T2)), p2);
	}

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>>
	constexpr auto reinterpret_array_cast() const& {
		BOOST_MULTI_ASSERT(this->layout().stride() * static_cast<size_type>(sizeof(T)) % static_cast<size_type>(sizeof(T2)) == 0);

		return const_subarray<T2, 1, P2>{
			layout_type{this->layout().sub(), this->layout().stride() * static_cast<size_type>(sizeof(T)) / static_cast<size_type>(sizeof(T2)), this->layout().offset() * static_cast<size_type>(sizeof(T)) / static_cast<size_type>(sizeof(T2)), this->layout().nelems() * static_cast<size_type>(sizeof(T)) / static_cast<size_type>(sizeof(T2))},
			reinterpret_pointer_cast<P2>(this->base_)
		};
	}

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2 const>>
	constexpr auto reinterpret_array_cast(size_type n) const& -> subarray<std::decay_t<T2>, 2, P2> {  // TODO(correaa) : use rebind for return type
		static_assert(sizeof(T) % sizeof(T2) == 0, "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases");

		return subarray<std::decay_t<T2>, 2, P2>{
			layout_t<2>{this->layout().scale(sizeof(T), sizeof(T2)), 1, 0, n},
			reinterpret_pointer_cast<P2>(this->base())
		}
			.rotated();
	}

	template<class Archive>
	void serialize(Archive& arxiv, unsigned /*version*/) {
		using AT = multi::archive_traits<Archive>;
		std::for_each(this->begin(), this->end(), [&](reference& item) { arxiv& AT ::make_nvp("item", item); });
		//  std::for_each(this->begin(), this->end(), [&](auto&&     item) {arxiv & cereal::make_nvp("item", item);});
		//  std::for_each(this->begin(), this->end(), [&](auto&&     item) {arxiv &                          item ;});
	}
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class T2, class P2, class Array, class... Args>
constexpr auto static_array_cast(Array&& self, Args&&... args) -> decltype(auto) {
	return std::forward<Array>(self).template static_array_cast<T2, P2>(std::forward<Args>(args)...);
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<
	typename T, dimensionality_type D, typename ElementPtr = T*,
	class Layout =
		std::conditional_t<
			(D == 1),
			// contiguous_layout<>,  // 1, typename std::pointer_traits<ElementPtr>::difference_type>,
			multi::layout_t<D, typename std::pointer_traits<ElementPtr>::difference_type>,
			multi::layout_t<D, typename std::pointer_traits<ElementPtr>::difference_type>>>
class array_ref : public subarray<T, D, ElementPtr, Layout> {
	using subarray_layout = Layout;

	using subarray_base = subarray<T, D, ElementPtr, Layout>;

 public:
	~array_ref() = default;  // lints(cppcoreguidelines-special-member-functions)

	using layout_type = typename subarray_base::layout_t;
	using iterator    = typename subarray_base::iterator;

	constexpr array_ref() = delete;  // because reference cannot be unbound

	// [[deprecated("references are not copyable, use auto&&")]]
	array_ref(array_ref const&) = delete;  // don't try to use `auto` for references, use `auto&&` or explicit value type
	array_ref(array_ref&&)      = delete;

	array_ref(iterator, iterator) = delete;

	// return type removed for MSVC
	friend constexpr auto sizes(array_ref const& self) noexcept /*-> typename array_ref::sizes_type*/ { return self.sizes(); }  // needed by nvcc
	friend constexpr auto size(array_ref const& self) noexcept /*-> typename array_ref::size_type*/ { return self.size(); }     // needed by nvcc

#if defined(BOOST_MULTI_HAS_SPAN) && !defined(__NVCC__)
	template<class Tconst = typename array_ref::element_type const, std::enable_if_t<std::is_convertible_v<typename array_ref::element_const_ptr, Tconst*> && (D == 1), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr explicit operator std::span<Tconst>() const& { return std::span<Tconst>(this->data_elements(), this->size()); }
#endif

	template<class OtherPtr, class = std::enable_if_t<!std::is_same_v<OtherPtr, ElementPtr>>, decltype(multi::detail::explicit_cast<ElementPtr>(std::declval<OtherPtr>()))* = nullptr>
	constexpr explicit array_ref(array_ref<T, D, OtherPtr>&& other)
	: subarray_base(other.layout(), ElementPtr{std::move(other).base()}) {}  // cppcheck-suppress internalAstError ; bug in cppcheck 2.13.0

	template<class OtherPtr, class = std::enable_if_t<!std::is_same_v<OtherPtr, ElementPtr>>, decltype(multi::detail::implicit_cast<ElementPtr>(std::declval<OtherPtr>()))* = nullptr>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	constexpr /*mplct*/ array_ref(array_ref<T, D, OtherPtr>&& other)         // NOLINT(google-explicit-constructor,hicpp-explicit-conversions,bugprone-use-after-move,hicpp-invalid-access-moved)
	: subarray_base(other.layout(), ElementPtr{std::move(other).base()}) {}  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved)

	constexpr array_ref(ElementPtr dat, ::boost::multi::extensions_t<D> const& xs) noexcept  // TODO(correa) eliminate this ctor
	: subarray_base(typename subarray_base::types::layout_t(xs), dat) {}

	constexpr array_ref(::boost::multi::extensions_t<D> exts, ElementPtr dat) noexcept
	: subarray_base{typename array_ref::types::layout_t(exts), dat} {}

	// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)  // compatibility with legacy c-arrays
	template<
		class Array,
		std::enable_if_t<  // NOLINT(modernize-use-constraints) for C++20
			!std::is_array_v<Array> && !std::is_base_of_v<array_ref, std::decay_t<Array>> && std::is_convertible_v<decltype(multi::data_elements(std::declval<Array&>())), ElementPtr>, int> = 0>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax and because a reference to c-array can be represented as an array_ref
	constexpr array_ref(Array& array)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax and because a reference to c-array can be represented as an array_ref
	: array_ref(multi::data_elements(array), extensions(array)) {}
	// NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

	template<class TT = void, std::enable_if_t<sizeof(TT*) && D == 0, int> = 0>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax and because a reference to c-array can be represented as an array_ref
	constexpr array_ref(  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax and because a reference to c-array can be represented as an array_ref
		T& elem           // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
	)
	: array_ref(&elem, {}) {}

	template<class TT, std::size_t N>
	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr array_ref(TT (&arr)[N])  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,google-explicit-constructor,hicpp-explicit-conversions) : for backward compatibility // NOSONAR
	: array_ref(
		  ::boost::multi::extensions(arr),
		  ::boost::multi::data_elements(arr)
	  ) {}

	template<class TT, std::size_t N>
	// cppcheck-suppress noExplicitConstructor ;  // NOLINTNEXTLINE(runtime/explicit)
	constexpr array_ref(std::array<TT, N>& arr) : array_ref(::boost::multi::extensions(arr), ::boost::multi::data_elements(arr)) {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) array_ptr is more general than pointer c-array support legacy c-arrays  // NOSONAR

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) bug in clang-tidy 19?
	template<class TT, std::enable_if_t<std::is_same_v<typename array_ref::value_type, TT>, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	// cppcheck-suppress noExplicitConstructor
	array_ref(std::initializer_list<TT> il)
	: array_ref(
		  (il.size() == 0) ? nullptr
						   : il.begin(),  // TODO(correaa) simplify conditional by still using a il pointer in empty case?
		  typename array_ref::extensions_type{static_cast<typename array_ref::size_type>(il.size())}
	  ) {}

	using subarray_base::operator=;

 private:
	template<class It> constexpr auto copy_elements_(It first) {
		return adl_copy_n(first, this->num_elements(), this->data_elements());
	}

 public:
	BOOST_MULTI_HD constexpr auto data_elements() const& { return static_cast<typename array_ref::element_const_ptr>(array_ref::base_); }

	template<class TT, class... As, std::enable_if_t<!std::is_base_of_v<array_ref, array_ref<TT, D, As...>>, int> = 0>  // NOLINT(modernize-use-constraints)  TODO(correaa) for C++20
	constexpr auto operator=(array_ref<TT, D, As...> const& other) && -> array_ref& {                                   // if MSVC complains here, it probably needs /EHsc /permissive- for C++17 mode
		BOOST_MULTI_ASSERT(this->extensions() == other.extensions());
		array_ref::copy_elements_(other.data_elements());
		return *this;
	}

	constexpr auto operator=(array_ref const& other) & -> array_ref& {
		if(this == std::addressof(other)) {
			return *this;
		}  // lints(cert-oop54-cpp)
		// TODO(correaa) assert on extensions, not on num elements
		BOOST_MULTI_ASSERT(this->num_elements() == other.num_elements());  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		array_ref::copy_elements_(other.data_elements());
		return *this;
	}

	constexpr auto operator=(array_ref const& other) && -> array_ref& {
		if(this == std::addressof(other)) {
			return *this;
		}  // lints(cert-oop54-cpp)
		operator=(other);
		return *this;
	}

	constexpr auto operator=(array_ref&& other) & noexcept(std::is_nothrow_copy_assignable_v<T>)  // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor,cppcoreguidelines-noexcept-move-operations)  //NOSONAR(cppS5018)
		-> array_ref& {
		if(this == std::addressof(other)) {
			return *this;
		}  // lints(cert-oop54-cpp)
		operator=(std::as_const(other));
		return *this;
	}

	constexpr auto operator=(array_ref&& other) && noexcept(std::is_nothrow_copy_assignable_v<T>)  // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor,cppcoreguidelines-noexcept-move-operations)
		-> array_ref& {
		if(this == std::addressof(other)) {
			return *this;
		}  // lints(cert-oop54-cpp)
		operator=(std::as_const(other));
		return *this;
	}

	template<typename TT, dimensionality_type DD = D, class... As>
	auto operator=(array_ref<TT, DD, As...> const& other) & -> array_ref& {
		BOOST_MULTI_ASSERT(this->extensions() == other.extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

	template<typename TT, dimensionality_type DD = D, class... As>
	constexpr auto operator=(array_ref<TT, DD, As...> const& other) && -> array_ref& {
		this->operator=(other);
		return *this;  // lints (cppcoreguidelines-c-copy-assignment-signature)
	}

	using elements_type  = array_ref<typename array_ref::element_type, 1, typename array_ref::element_ptr>;
	using celements_type = array_ref<typename array_ref::element_type, 1, typename array_ref::element_const_ptr>;

 private:
	constexpr auto elements_aux_() const {
		return elements_type{
			this->base_,
			typename elements_type::extensions_type{multi::iextension{this->num_elements()}}
		};
	}

 public:
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto elements() const& -> celements_type { return elements_aux_(); }
	constexpr auto elements() & -> elements_type { return elements_aux_(); }
	constexpr auto elements() && -> elements_type { return elements_aux_(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	friend constexpr auto elements(array_ref& self) -> elements_type { return self.elements(); }
	friend constexpr auto elements(array_ref&& self) -> elements_type { return std::move(self).elements(); }
	friend constexpr auto elements(array_ref const& self) -> celements_type { return self.elements(); }

	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto celements() const& { return celements_type{array_ref::data_elements(), array_ref::num_elements()}; }

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto element_moved() & { return array_ref<T, D, typename array_ref::element_move_ptr, Layout>(this->extensions(), typename array_ref::element_move_ptr{this->base_}); }
	constexpr auto element_moved() && { return element_moved(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	template<typename TT, class... As>
	friend constexpr auto operator==(array_ref const& self, array_ref<TT, D, As...> const& other) -> bool {
		if(self.extensions() != other.extensions()) {
			return false;
		}
		return adl_equal(
			other.data_elements(), other.data_elements() + other.num_elements(),  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) use span?
			self.data_elements()
		);
	}

	template<typename TT, class... As>
	friend constexpr auto operator!=(array_ref const& self, array_ref<TT, D, As...> const& other) -> bool {
		if(self.extensions() != other.extensions()) {
			return true;
		}
		return !adl_equal(
			other.data_elements(), other.data_elements() + other.num_elements(),  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) use span?
			self.data_elements()
		);
		// return ! operator==(self, other);  // commented due to bug in nvcc 22.11
	}

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

	BOOST_MULTI_HD constexpr auto data_elements() & -> typename array_ref::element_ptr { return array_ref::base_; }
	BOOST_MULTI_HD constexpr auto data_elements() && -> typename array_ref::element_ptr { return array_ref::base_; }

	friend constexpr auto data_elements(array_ref&& self) -> typename array_ref::element_ptr { return std::move(self).data_elements(); }

	// data() is here for compatibility with std::vector
	template<class Dummy = void, std::enable_if_t<(D == 1) && sizeof(Dummy*), int> = 0> constexpr auto data() const& { return data_elements(); }  // NOLINT(modernize-use-constraints) TODO(correaa)
	template<class Dummy = void, std::enable_if_t<(D == 1) && sizeof(Dummy*), int> = 0> constexpr auto data() && { return data_elements(); }      // NOLINT(modernize-use-constraints) TODO(correaa)
	template<class Dummy = void, std::enable_if_t<(D == 1) && sizeof(Dummy*), int> = 0> constexpr auto data() & { return data_elements(); }       // NOLINT(modernize-use-constraints) TODO(correaa)

	// TODO(correaa) : find a way to use [[deprecated("use data_elements()")]] for friend functions
	friend constexpr auto data(array_ref const& self) -> typename array_ref::element_ptr { return self.data_elements(); }
	friend constexpr auto data(array_ref& self) -> typename array_ref::element_ptr { return self.data_elements(); }
	friend constexpr auto data(array_ref&& self) -> typename array_ref::element_ptr { return std::move(self).data_elements(); }

	using decay_type = typename array_ref::decay_type;

	// cppcheck-suppress duplInheritedMember ; to override
	constexpr auto decay() const& -> decay_type const& { return static_cast<decay_type const&>(*this); }

 private:
	template<class TTN, std::size_t DD = 0>
	void check_sizes_() const {
		using std::get;  // for C++17 compatibility
		if(size_type{get<DD>(this->sizes())} != size_type{std::extent_v<TTN, unsigned{DD}>}) {
			throw std::bad_cast{};
		}
		if constexpr(DD + 1 != D) {
			check_sizes_<TTN, DD + 1>();
		}
	}

	template<class TT> static auto launder_(TT* pointer) -> TT* {
#if defined(__cpp_lib_launder) && (__cpp_lib_launder >= 201606L)
		return std::launder(pointer);
#else
		return pointer;
#endif
	}

	template<class, ::boost::multi::dimensionality_type, class> friend struct array;

	template<class TTN>
	constexpr auto to_carray_() & -> TTN& {
		check_sizes_<TTN>();
		return *launder_(reinterpret_cast<TTN*>(array_ref::base_));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
	}

	template<class TTN>
	constexpr auto to_carray_() const& -> TTN const& {
		check_sizes_<TTN>();
		return *launder_(reinterpret_cast<TTN const*>(array_ref::base_));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
	}

 public:
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>           // NOLINT(modernize-use-constraints) for C++20
	constexpr explicit operator TTN const&() const& { return to_carray_<TTN>(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN&() && { return to_carray_<TTN>(); }   // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	constexpr explicit operator TTN&() & { return to_carray_<TTN>(); }    // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

 private:
	template<class Ar>
	auto serialize_structured_(Ar& arxiv, unsigned int const version) {
		subarray_base::serialize(arxiv, version);
	}

	template<class Archive>
	auto serialize_flat_(Archive& arxiv, unsigned int const /*version*/) {
		using AT = multi::archive_traits<Archive>;
		arxiv& AT::make_nvp("elements", AT::make_array(this->data_elements(), static_cast<std::size_t>(this->num_elements())));
	}

	//  template<class Ar, class AT = multi::archive_traits<Ar>>
	//  auto serialize_binary_if(std::true_type, Ar& ar) {
	//      ar & AT::make_nvp("binary_data", AT::make_binary_object(this->data_elements(), static_cast<std::size_t>(this->num_elements())*sizeof(typename array_ref::element)));
	//  }
	//  template<class Ar>
	//  auto serialize_binary_if(std::false_type, Ar& ar) {return serialize_flat(ar);}

 public:
	template<class Archive>
	auto serialize(Archive& arxiv, unsigned int const version) {  // cppcheck-suppress duplInheritedMember ;
		serialize_flat_(arxiv, version);
		//      serialize_structured_(ar, version);
		//      switch(version) {
		//          case static_cast<unsigned int>( 0): return serialize_flat_(arxiv);
		//          case static_cast<unsigned int>(-1): return serialize_structured_(arxiv, version);
		//      //  case 2: return serialize_binary_if(std::is_trivially_copy_assignable<typename array_ref::element>{}, arxiv);
		//          default:
		//              if( this->num_elements() <= version ){serialize_structured_(arxiv, version);}
		//              else                                 {serialize_flat_       (arxiv         );}
		//      }
	}
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class T, dimensionality_type D, class Ptr = typename std::pointer_traits<T*>::template rebind<T const>>
using array_cref = array_ref<std::decay_t<T>, D, Ptr>;

template<class T, dimensionality_type D, class Ptr = T*>
using array_mref = array_ref<
	std::decay_t<T>, D,
	std::move_iterator<Ptr>>;

template<class TT, std::size_t N>
constexpr auto ref(
	TT (&arr)[N]  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) interact with legacy  // NOSONAR
) {
	return array_ref<std::remove_all_extents_t<TT[N]>, std::rank_v<TT[N]>>(arr);  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) interact with legacy
}

template<class T, dimensionality_type D, typename Ptr = T*>
struct array_ptr
: subarray_ptr<T, D, Ptr, typename array_ref<T, D, Ptr>::layout_t, false> {
	using basic_ptr = subarray_ptr<T, D, Ptr, typename array_ref<T, D, Ptr>::layout_t, false>;

	constexpr array_ptr(Ptr data, multi::extensions_t<D> extensions)
	: basic_ptr{data, typename array_ref<T, D, Ptr>::layout_t(extensions)} {}

	constexpr explicit array_ptr(std::nullptr_t nil) : array_ptr{nil, multi::extensions_t<D>{}} {}

	template<typename CArray>
	// cppcheck-suppress constParameterPointer ;  workaround cppcheck 2.11
	constexpr explicit array_ptr(CArray* data) : array_ptr{data_elements(*data), extensions(*data)} {}

	template<
		class TT, std::size_t N,
		std::enable_if_t<std::is_convertible_v<decltype(data_elements(std::declval<TT (&)[N]>())), Ptr>, int> = 0  // NOLINT(modernize-use-constraints,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) support legacy c-arrays TODO(correaa) for C++20
		>
	// cppcheck-suppress noExplicitConstructor ;  // NOLINTNEXTLINE(runtime/explicit)
	constexpr array_ptr(TT (*array)[N]) : array_ptr{data_elements(*array), extensions(*array)} {}  // NOLINT(modernize-use-constraints,google-explicit-constructor,hicpp-explicit-conversions,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) array_ptr is more general than pointer c-array support legacy c-arrays  TODO(correaa) for C++20  // NOSONAR

	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto operator*() const -> array_ref<T, D, Ptr> {
		return array_ref<T, D, Ptr>((*static_cast<subarray_ptr<T, D, Ptr, typename array_ref<T, D, Ptr>::layout_t, false> const&>(*this)).extensions(), this->base());
	}
};

template<class T, typename Ptr>
class [[deprecated("no good uses found")]] array_ptr<T, 0, Ptr> {  // TODO(correaa) make it private mutable member
	mutable multi::array_ref<T, 0, Ptr> ref_;                      // TODO(correaa) implement array_ptr like other cases

 public:
	~array_ptr() = default;

	constexpr array_ptr(array_ptr const&)     = default;
	constexpr array_ptr(array_ptr&&) noexcept = default;  // NOLINT(hicpp-noexcept-move,performance-noexcept-move-constructor) TODO(correaa) change the implementation like the other cases

	constexpr explicit array_ptr(Ptr dat, typename multi::array_ref<T, 0, Ptr>::extensions_type extensions) : ref_(dat, extensions) {}
	constexpr explicit array_ptr(Ptr dat) : array_ptr(dat, typename multi::array_ref<T, 0, Ptr>::extensions_type{}) {}

	constexpr explicit operator bool() const { return ref_.base(); }  // cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr explicit operator Ptr() const { return ref_.base(); }

	auto operator=(array_ptr const&) -> array_ptr&     = default;
	auto operator=(array_ptr&&) noexcept -> array_ptr& = default;

	friend constexpr auto operator==(array_ptr const& self, array_ptr const& other) -> bool { return self.ref_.base() == other.ref_.base(); }
	friend constexpr auto operator!=(array_ptr const& self, array_ptr const& other) -> bool { return self.ref_.base() != other.ref_.base(); }

	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto operator*() const -> multi::array_ref<T, 0, Ptr>& { return ref_; }  // moLINT(cppcoreguidelines-pro-type-const-cast) : TODO(correaa) make ref base class a mutable member

	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr auto operator->() const -> multi::array_ref<T, 0, Ptr>* { return &ref_; }  // moLINT(cppcoreguidelines-pro-type-const-cast) : TODO(correaa) make ref base class a mutable member
};

template<class TT, std::size_t N>
constexpr auto addressof(TT (&array)[N]) {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
	return array_ptr<
		// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
		std::decay_t<std::remove_all_extents_t<TT[N]>>, static_cast<dimensionality_type>(std::rank<TT[N]>{}), std::remove_all_extents_t<TT[N]>*>{&array};
}

template<class T, dimensionality_type D, typename Ptr = T*>
using array_cptr = array_ptr<T, D, typename std::pointer_traits<Ptr>::template rebind<T const>>;

template<dimensionality_type D, class P>
constexpr auto make_array_ref(P data, multi::extensions_t<D> extensions) {
	return array_ref<typename std::iterator_traits<P>::value_type, D, P>(data, extensions);
}

template<class P> auto make_array_ref(P data, extensions_t<0> exts) { return make_array_ref<0>(data, exts); }
template<class P> auto make_array_ref(P data, extensions_t<1> exts) { return make_array_ref<1>(data, exts); }
template<class P> auto make_array_ref(P data, extensions_t<2> exts) { return make_array_ref<2>(data, exts); }
template<class P> auto make_array_ref(P data, extensions_t<3> exts) { return make_array_ref<3>(data, exts); }
template<class P> auto make_array_ref(P data, extensions_t<4> exts) { return make_array_ref<4>(data, exts); }
template<class P> auto make_array_ref(P data, extensions_t<5> exts) { return make_array_ref<5>(data, exts); }

#ifdef __cpp_deduction_guides

template<class It, typename V = typename std::iterator_traits<It>::value_type>  // pointer_traits doesn't have ::value_type
array_ptr(It) -> array_ptr<V, 0, It>;
template<class It, typename V = typename std::iterator_traits<It>::value_type>  // pointer_traits doesn't have ::value_type
array_ptr(It, index_extensions<0>) -> array_ptr<V, 0, It>;
template<class It, typename V = typename std::iterator_traits<It>::value_type>
array_ptr(It, index_extensions<1>) -> array_ptr<V, 1, It>;
template<class It, typename V = typename std::iterator_traits<It>::value_type>
array_ptr(It, index_extensions<2>) -> array_ptr<V, 2, It>;
template<class It, typename V = typename std::iterator_traits<It>::value_type>
array_ptr(It, index_extensions<3>) -> array_ptr<V, 3, It>;

template<
	class T,
	std::size_t N,
	typename V = std::remove_all_extents_t<T[N]>, std::size_t D = std::rank_v<T[N]>  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
	>
array_ptr(T (*)[N]) -> array_ptr<V, static_cast<multi::dimensionality_type>(D)>;  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility

template<class Ptr> array_ref(Ptr, index_extensions<0>) -> array_ref<typename std::iterator_traits<Ptr>::value_type, 0, Ptr>;
template<class Ptr> array_ref(Ptr, index_extensions<1>) -> array_ref<typename std::iterator_traits<Ptr>::value_type, 1, Ptr>;
template<class Ptr> array_ref(Ptr, index_extensions<2>) -> array_ref<typename std::iterator_traits<Ptr>::value_type, 2, Ptr>;
template<class Ptr> array_ref(Ptr, index_extensions<3>) -> array_ref<typename std::iterator_traits<Ptr>::value_type, 3, Ptr>;
template<class Ptr> array_ref(Ptr, index_extensions<4>) -> array_ref<typename std::iterator_traits<Ptr>::value_type, 4, Ptr>;
template<class Ptr> array_ref(Ptr, index_extensions<5>) -> array_ref<typename std::iterator_traits<Ptr>::value_type, 5, Ptr>;

template<class It, class Tuple> array_ref(It, Tuple) -> array_ref<typename std::iterator_traits<It>::value_type, std::tuple_size_v<Tuple>, It>;

template<class It> const_subarray(It, It) -> const_subarray<typename It::element_type, It::dimensionality + 1, typename It::element_ptr, layout_t<It::dimensionality + 1>>;

template<class T> const_subarray(std::initializer_list<T>) -> const_subarray<T, 1>;

#endif

// TODO(correaa) move to utility
template<class T, std::size_t N>
constexpr auto rotated(T const (&array)[N]) noexcept {                                                 // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
	return multi::array_ref<std::remove_all_extents<T[N]>, std::rank<T[N]>{}, decltype(base(array))>(  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
			   base(array), extensions(array)
	)
		.rotated();
}

template<class T, std::size_t N>
constexpr auto rotated(T (&array)[N]) noexcept {                                                       // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
	return multi::array_ref<std::remove_all_extents<T[N]>, std::rank<T[N]>{}, decltype(base(array))>(  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : backwards compatibility
			   base(array), extensions(array)
	)
		.rotated();
}

template<class RandomAccessIterator, dimensionality_type D>
constexpr auto operator/(RandomAccessIterator data, multi::extensions_t<D> extensions)
	-> multi::array_ptr<typename std::iterator_traits<RandomAccessIterator>::value_type, D, RandomAccessIterator> {
	return {data, extensions};
}

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

template<class In, class T, dimensionality_type N, class TP, class = std::enable_if_t<(N > 1)>, class = decltype((void)adl_begin(*In{}), adl_end(*In{}))>
constexpr auto uninitialized_copy
	// require N>1 (this is important because it forces calling placement new on the pointer
	(In first, In last, multi::array_iterator<T, N, TP> dest) {  // NOLINT(performance-unnecessary-value-param) TODO(correaa) inverstigate why I can't make this In const& last
	while(first != last) {                                       // NOLINT(altera-unroll-loops) TODO(correaa) consider using an algorithm
		adl_uninitialized_copy(adl_begin(*first), adl_end(*first), adl_begin(*dest));
		++first;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		++dest;
	}
	return dest;
}

#if defined(__clang__) && (__clang_major__ >= 16) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif

// begin and end for forwarding reference are needed in this namespace
// to overwrite the behavior of std::begin and std::end
// which take rvalue-references as const-references.

template<class T> auto begin(T&& rng) -> decltype(std::forward<T>(rng).begin()) { return std::forward<T>(rng).begin(); }
template<class T> auto end(T&& rng) -> decltype(std::forward<T>(rng).end()) { return std::forward<T>(rng).end(); }

template<class T, std::size_t N, std::size_t M>
auto transposed(T (&array)[N][M]) -> decltype(auto) { return ~multi::array_ref<T, 2>(array); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

template<class T, dimensionality_type D, class TPtr = T const*>
using array_const_view = array_ref<T, D, TPtr> const&;

template<class T, dimensionality_type D, class TPtr = T*>
using array_view = array_ref<T, D, TPtr>&;

}  // end namespace boost::multi

#ifndef BOOST_MULTI_SERIALIZATION_ARRAY_VERSION
#define BOOST_MULTI_SERIALIZATION_ARRAY_VERSION 0  // NOLINT(cppcoreguidelines-macro-usage) gives user opportunity to select serialization version //NOSONAR
// #define BOOST_MULTI_SERIALIZATION_ARRAY_VERSION  0 // save data as flat array
// #define BOOST_MULTI_SERIALIZATION_ARRAY_VERSION -1 // save data as structured nested labels array
// #define BOOST_MULTI_SERIALIZATION_ARRAY_VERSION 16 // any other value, structure for N <= 16, flat otherwise N > 16

namespace boost::multi {
constexpr inline int serialization_array_version = BOOST_MULTI_SERIALIZATION_ARRAY_VERSION;
}  // end namespace boost::multi
#endif

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L) && !defined(_MSC_VER)
namespace std::ranges {  // NOLINT(cert-dcl58-cpp) to enable borrowed, nvcc needs namespace
template<typename Element, ::boost::multi::dimensionality_type D, class... Rest>
[[maybe_unused]] constexpr bool enable_borrowed_range<::boost::multi::subarray<Element, D, Rest...>> = true;  // NOLINT(misc-definitions-in-headers)

template<typename Element, ::boost::multi::dimensionality_type D, class... Rest>
[[maybe_unused]] constexpr bool enable_borrowed_range<::boost::multi::const_subarray<Element, D, Rest...>> = true;  // NOLINT(misc-definitions-in-headers)
}  // end namespace std::ranges
#endif

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#undef BOOST_MULTI_HD

// Copyright 2022-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <type_traits>

namespace boost {  // NOLINT(modernize-concat-nested-namespaces)
namespace multi {

// template<class T> struct is_trivially_default_constructible : std::is_trivially_default_constructible<T> {};
// template<class T> struct is_trivial : std::is_trivial<T> {};

}  // end namespace multi
}  // end namespace boost

// Copyright 2023-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace boost::multi::detail {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, std::size_t N>
class static_allocator {  // NOSONAR(cpp:S4963) this allocator has special semantics
#ifdef _MSC_VER
	#pragma warning(push)
	#pragma warning(disable : 4324)  // Warning that the structure is padded due to the below
#endif

// #if defined(__clang__)
// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wpadded"
// #endif

	BOOST_MULTI_NO_UNIQUE_ADDRESS alignas(T) std::array<std::byte, sizeof(T) * N> buffer_;

// #if defined(__clang__)
// #pragma clang diagnostic pop
// #endif

#ifdef _MSC_VER
	#pragma warning(pop)
#endif

#ifdef _MSC_VER
	#pragma warning(push)
	#pragma warning(disable : 4820)  // warning C4820: 'boost::multi::detail::static_allocator<main::T,32>': '3' bytes padding added after data member 'boost::multi::detail::static_allocator<main::T,32>::dirty_' [C:\Gitlab-Runner\builds\t3_1sV2uA\0\correaa\boost-multi\build\test\allocator.cpp.x.vcxproj]
#endif
	bool dirty_ = false;
#ifdef _MSC_VER
	#pragma warning(pop)
#endif

 public:
	using value_type = T;
	using pointer    = T*;

	template<class TT> struct rebind {
		using other = static_allocator<TT, N>;
	};

	static constexpr auto max_size() noexcept -> std::size_t { return N; }

	static_allocator() = default;  // NOLINT(cppcoreguidelines-pro-type-member-init,hicpp-member-init) buffer_ is not initialized

	template<class TT, std::size_t NN>
	explicit static_allocator(static_allocator<TT, NN> const& /*other*/) {  // NOLINT(hicpp-explicit-conversions,google-explicit-constructor) follow std::allocator  // NOSONAR
		// static_assert(sizeof(T) == sizeof(TT));
		static_assert(NN == N);
	}

	static_allocator(static_allocator const& /*other*/)  // std::vector makes a copy right away
	// = default;  // this copies the internal buffer
	{}

	// [[deprecated("don't move dynamic container with static_allocator")]]
	static_allocator(static_allocator&& /*other*/)  // this is called *by the elements* during move construction of a vector
		// = delete;
		// {throw std::runtime_error("don't move dynamic container with static_allocator");}  // this is called *by the elements* during move construction of a vector
		noexcept {}
	// noexcept {std::memmove(buffer_.data(), other.buffer_.data(), sizeof(T)*N);}
	// noexcept : buffer_{std::move(other.buffer_)} {}
	// noexcept = default;

	[[deprecated("don't move dynamic container with static_allocator")]]
	auto operator=(static_allocator const& /*other*/) -> static_allocator& = delete;

	[[deprecated("don't move dynamic container with static_allocator")]] auto operator=(static_allocator&& other) -> static_allocator& = delete;

	~static_allocator() = default;

	auto select_on_container_copy_construction() noexcept -> static_allocator = delete;
	// {return static_allocator{};}

	using propagate_on_container_move_assignment = std::false_type;  // this forces to call move assignment of the allocator by std::vector
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_swap            = std::false_type;

	static constexpr auto capacity() { return N; }

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4068)  // bug in MSVC 14.2/14.3
#endif
	BOOST_MULTI_NODISCARD("because otherwise it will generate a memory leak")
	auto allocate([[maybe_unused]] std::size_t n) -> pointer {
		assert(n <= N);
		assert(!dirty_);  // do not attempt to resize a vector with static_allocator
		// dirty_ = true;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"        // buffer_ is aligned as T
		return reinterpret_cast<pointer>(&buffer_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#pragma GCC diagnostic pop
	}
#ifdef _MSC_VER
#pragma warning( pop ) 
#endif

	void deallocate(pointer /*ptr*/, [[maybe_unused]] std::size_t n) {
		assert(n <= N);
	}

	using is_always_equal = std::true_type;
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class T, std::size_t N, class U>
constexpr auto operator==(static_allocator<T, N> const& /*a1*/, static_allocator<U, N> const& /*a2*/) noexcept { return true; }  // &a1 == &a2; }
// = delete;

template<class T, std::size_t N, class U>
auto operator!=(static_allocator<T, N> const& /*a1*/, static_allocator<U, N> const& /*a2*/) noexcept  // this is used *by the elements* when resizing a vector
{ return false; }                                                                                     // &a1 != &a2;}
// = delete

template<class T, std::size_t N, class U>
[[deprecated("don't swap dynamic container with static_allocator")]]
void swap(static_allocator<T, N>& a1, static_allocator<U, N>& a2) noexcept = delete;

}  // end namespace boost::multi::detail

#include <iterator>  // for std::sentinel_for
#include <memory>    // for std::allocator_traits
#include <stdexcept>
#include <type_traits>  // for std::common_reference
#include <utility>      // for std::move

#if __has_include(<memory_resource>)
#include <memory_resource>
// Apple clang provides the header but not the compiled library prior to version 16
#if (defined(__cpp_lib_memory_resource) && (__cpp_lib_memory_resource >= 201603)) && !(defined(__APPLE__) && defined(__clang_major__) && __clang_major__ <= 15) && (!defined(_LIBCPP_VERSION) || !(_LIBCPP_VERSION <= 160001))
#define BOOST_MULTI_HAS_MEMORY_RESOURCE
#endif
#endif

#if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<concepts>) && __has_include(<ranges>)
#include <concepts>  // for constructible_from  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <ranges>    // IWYU pragma: keep
#endif

// TODO(correaa) or should be (__CUDA__) or CUDA__ || HIP__
#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4626)  // assignment operator was implicitly defined as deleted
#endif

namespace boost::multi {

namespace detail {

template<class Allocator>
struct array_allocator {
	using allocator_type = Allocator;
	array_allocator()    = default;

 private:
	BOOST_MULTI_NO_UNIQUE_ADDRESS allocator_type alloc_;

	using allocator_traits = multi::allocator_traits<allocator_type>;

	using size_type_ = typename allocator_traits::size_type;  // NOLINT(readability-redundant-typename) typename needed in C++17
	using pointer_   = typename allocator_traits::pointer;    // NOLINT(readability-redundant-typename) typename needed in C++17

 protected:
	constexpr auto alloc() & -> auto& { return alloc_; }
	constexpr auto alloc() const& -> allocator_type const& { return alloc_; }

	constexpr explicit array_allocator(allocator_type const& alloc) : alloc_{alloc} {}  // NOLINT(modernize-pass-by-value)

	constexpr auto allocate(size_type_ n) -> pointer_ {
		return n ? allocator_traits::allocate(alloc_, n) : pointer_{nullptr};
	}
	constexpr auto allocate(size_type_ n, typename allocator_traits::const_void_pointer hint) -> pointer_ {  // NOLINT(readability-redundant-typename) typename needed in C++17
		return n ? allocator_traits::allocate(alloc_, n, hint) : pointer_{nullptr};
	}

	constexpr auto uninitialized_fill_n(pointer_ first, size_type_ count, typename allocator_traits::value_type const& value) {  // NOLINT(readability-redundant-typename) typename needed in C++17
		return adl_alloc_uninitialized_fill_n(alloc_, first, count, value);
	}

	template<typename It>
	auto uninitialized_copy_n(It first, size_type count, pointer_ d_first) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename std::pointer_traits<pointer_>::element_type> && !multi::force_element_trivial_default_construction<typename std::pointer_traits<pointer_>::element_type>) {
			adl_alloc_uninitialized_default_construct_n(alloc_, d_first, count);
		}
		return adl_copy_n(first, count, d_first);
#else
		return adl_alloc_uninitialized_copy_n(alloc_, first, count, d_first);
#endif
	}

	template<typename It>
	auto uninitialized_move_n(It first, size_type count, pointer_ d_first) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename std::pointer_traits<pointer_>::element_type> && !multi::force_element_trivial_default_construction<typename std::pointer_traits<pointer_>::element_type>) {
			adl_alloc_uninitialized_default_construct_n(alloc_, d_first, count);
		}
		return adl_copy_n(std::make_move_iterator(first), count, d_first);
#else
		return adl_alloc_uninitialized_move_n(alloc_, first, count, d_first);
#endif
	}

	template<class EP, typename It>
	auto uninitialized_copy_n(EP&& ep, It first, size_type count, pointer_ d_first) {
		return adl_uninitialized_copy_n(std::forward<EP>(ep), first, count, d_first);
	}

	template<typename It>
	auto destroy_n(It first, size_type n) { return adl_alloc_destroy_n(this->alloc(), first, n); }

 public:
	BOOST_MULTI_HD constexpr auto get_allocator() const noexcept -> allocator_type { return alloc_; }
};

}  // end namespace detail

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, dimensionality_type D, class DummyAlloc = std::allocator<T>>  // DummyAlloc mechanism allows using the convention array<T, an_allocator<>>, is an_allocator supports void template argument
struct dynamic_array                                                            // NOLINT(fuchsia-multiple-inheritance) : multiple inheritance used for composition
: protected detail::array_allocator<
	  typename allocator_traits<DummyAlloc>::template rebind_alloc<T>>
, public array_ref<T, D, typename multi::allocator_traits<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::pointer>
, boost::multi::random_iterable<dynamic_array<T, D, typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>> {
	static_assert(
		std::is_same_v<
			std::remove_const_t<typename multi::allocator_traits<DummyAlloc>::value_type>,
			typename dynamic_array::element_type> ||
			std::is_same_v<
				std::remove_const_t<typename multi::allocator_traits<DummyAlloc>::value_type>,
				void>,  // allocator template can be redundant or void (which can be a default for the allocator)
		"allocator value type must match array value type"
	);

 protected:
	using array_alloc = detail::array_allocator<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>;

 public:
	using detail::array_allocator<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::get_allocator;

	using allocator_type = typename detail::array_allocator<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::allocator_type;  // NOLINT(readability-redundant-typename) needed for C++17
	using decay_type     = array<T, D, allocator_type>;
	using layout_type    = typename array_ref<T, D, typename multi::allocator_traits<allocator_type>::pointer>::layout_type;  // NOLINT(readability-redundant-typename) needed for C++17

	using ref = array_ref<
		T, D,
		typename multi::allocator_traits<typename multi::allocator_traits<allocator_type>::template rebind_alloc<T>>::pointer>;

	auto operator new(std::size_t count) -> void* { return ::operator new(count); }
	auto operator new(std::size_t count, void* ptr) -> void* { return ::operator new(count, ptr); }

	void operator delete(void* ptr) noexcept { ::operator delete(ptr); }  // this overrides the deleted delete operator in reference (base) class subarray

 protected:
	using alloc_traits = /*typename*/ multi::allocator_traits<allocator_type>;

	auto uninitialized_value_construct() {
		return adl_alloc_uninitialized_value_construct_n(dynamic_array::alloc(), this->base_, this->num_elements());
	}

	constexpr auto uninitialized_default_construct() {
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			return adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->base_, this->num_elements());
		}
	}

	template<typename It> auto uninitialized_copy_elements(It first) {
		return array_alloc::uninitialized_copy_n(first, this->num_elements(), this->data_elements());
	}

	// template<typename It> auto uninitialized_move_elements(It first) {
	//  return array_alloc::uninitialized_move_n(first, this->num_elements(), this->data_elements());
	// }

	template<class EP, typename It> auto uninitialized_copy_elements(EP&& ep, It first) {
		return array_alloc::uninitialized_copy_n(std::forward<EP>(ep), first, this->num_elements(), this->data_elements());
	}

	constexpr void destroy() {
		if constexpr(!(std::is_trivially_destructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_destruction<typename dynamic_array::element_type>)) {
			array_alloc::destroy_n(this->data_elements(), this->num_elements());
		}
	}

	void allocate() {
		this->base_ = array_alloc::allocate(static_cast<typename multi::allocator_traits<typename dynamic_array::allocator_type>::size_type>(this->dynamic_array::num_elements()));  // NOLINT(readability-redundant-typename) needed for C++17
	}

 public:
	using value_type = /*typename*/ std::conditional_t<
		(D > 1),  // this parenthesis is needed
		array<typename dynamic_array::element_type, D - 1, allocator_type>,
		typename dynamic_array::element_type>;

	using typename ref::difference_type;
	using typename ref::size_type;

	explicit dynamic_array(allocator_type const& alloc) : array_alloc{alloc}, ref(nullptr, {}) {}

	using ref::operator();
	BOOST_MULTI_HD constexpr auto operator()() && -> decltype(auto) { return ref::element_moved(); }

	using ref::taked;

	constexpr auto taked(difference_type n) && -> decltype(auto) { return ref::taked(n).element_moved(); }

	using ref::dropped;

	constexpr auto dropped(difference_type n) && -> decltype(auto) { return ref::dropped(n).element_moved(); }

	// dynamic_array(dynamic_array&&) = delete;
	constexpr dynamic_array(dynamic_array&& other) noexcept(false)  // NOLINT(cppcoreguidelines-noexcept-move-operations,hicpp-noexcept-move,performance-noexcept-move-constructor)
	: array_alloc{other.alloc()},
	  ref{
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())),  // NOLINT(readability-redundant-typename) needed for C++17
		  other.extensions()
	  } {
		adl_alloc_uninitialized_move_n(
			this->alloc(),
			other.data_elements(),
			other.num_elements(),
			this->data_elements()
		);
		(void)std::move(other);
	}

	constexpr dynamic_array(decay_type&& other, allocator_type const& alloc) noexcept
	: array_alloc{alloc}, ref(std::exchange(other.base_, nullptr), other.extensions()) {
		std::move(other).layout_mutable() = typename dynamic_array::layout_type(typename dynamic_array::extensions_type{});  // = {};  careful! this is the place where layout can become invalid
	}

	explicit constexpr dynamic_array(decay_type&& other) noexcept
	: array_alloc{std::move(other.alloc())}, ref(std::exchange(other.base_, nullptr), other.extensions()) {
		std::move(other).layout_mutable() = typename dynamic_array::layout_type(typename dynamic_array::extensions_type{});  // = {};  careful! this is the place where layout can become invalid
	}

	// constexpr explicit dynamic_array(decay_type&& other) noexcept
	// : dynamic_array(std::move(other), allocator_type{}) {}  // 6b

#if __cplusplus >= 202002L && (!defined(__clang_major__) || (__clang_major__ != 10))
	template<class It, std::sentinel_for<It> Sentinel = It, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>  // NOLINT(readability-redundant-typename) needed for C++17
	constexpr explicit dynamic_array(It const& first, Sentinel const& last, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(layout_type{index_extension(adl_distance(first, last)) * multi::extensions(*first)}.num_elements())),  // NOLINT(readability-redundant-typename) needed for C++17
		  index_extension(adl_distance(first, last)) * multi::extensions(*first)
	  ) {
#if defined(__clang__) && defined(__CUDACC__)
		// TODO(correaa) add workaround for non-default constructible type and use adl_alloc_uninitialized_default_construct_n
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), ref::data_elements(), ref::num_elements());
		}
		adl_copy_n(first, last - first, ref::begin());
#else
		adl_alloc_uninitialized_copy(dynamic_array::alloc(), first, last, ref::begin());
#endif
	}
#else
	template<class It, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>
	constexpr explicit dynamic_array(It const& first, It const& last, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(layout_type{index_extension(adl_distance(first, last)) * multi::extensions(*first)}.num_elements())),
		  index_extension(adl_distance(first, last)) * multi::extensions(*first)
	  ) {
#if defined(__clang__) && defined(__CUDACC__)
		// TODO(correaa) add workaround for non-default constructible type and use adl_alloc_uninitialized_default_construct_n
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), ref::data_elements(), ref::num_elements());
		}
		adl_copy_n(first, last - first, ref::begin());
#else
		adl_alloc_uninitialized_copy(dynamic_array::alloc(), first, last, ref::begin());
#endif
	}
#endif

#if __cplusplus >= 202002L && (!defined(__clang_major__) || (__clang_major__ != 10))
	template<class It, std::sentinel_for<It> Sentinel, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>  // NOLINT(readability-redundant-typename) needed for C++17
	constexpr explicit dynamic_array(It const& first, Sentinel const& last)
	: dynamic_array(first, last, allocator_type{}) {}
#else
	template<class It, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>
	constexpr explicit dynamic_array(It const& first, It const& last) : dynamic_array(first, last, allocator_type{}) {}
#endif

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L)  //  && !defined(_MSC_VER)
 private:
	void extent_(typename dynamic_array::extensions_type const& extensions) {  // NOLINT(readability-redundant-typename) needed for C++17
		auto new_layout = typename dynamic_array::layout_t{extensions};
		if(new_layout.num_elements() == 0) {
			return;
		}
		this->layout_mutable() = new_layout;  // typename array::layout_t{extensions};
		this->base_            = this->dynamic_array::array_alloc::allocate(
            static_cast<typename multi::allocator_traits<typename dynamic_array::allocator_type>::size_type>(  // NOLINT(readability-redundant-typename) needed for C++17
                new_layout.num_elements()
            ),
            this->data_elements()  // used as hint
        );
	}

 public:
	template<
		class Range, class = std::enable_if_t<!std::is_base_of<dynamic_array, std::decay_t<Range>>{}>,
		class = decltype(std::declval<Range const&>().begin()),
		class = decltype(std::declval<Range const&>().end()),
		// class = decltype(/*dynamic_array*/ (std::declval<Range const&>().begin() - std::declval<Range const&>().end())),  // instantiation of dynamic_array here gives a compiler error in 11.0, partially defined type?
		class = std::enable_if_t<!is_subarray<Range const&>::value>>                                                                                                 // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
	requires std::is_convertible_v<std::ranges::range_reference_t<std::decay_t<std::ranges::range_reference_t<Range>>>, T> explicit dynamic_array(Range const& rng)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax  // NOSONAR
	: dynamic_array() {
		if(rng.size() == 0) {
			return;
		}
		auto outer_it = std::ranges::begin(rng);
		static_assert(D == 2);

		auto const& outer_ref   = *outer_it;
		auto        common_size = static_cast<size_type>(outer_ref.size());
		extent_({static_cast<size_type>(rng.size()), common_size});

		auto [is, js] = this->extensions();
		{
			index const i        = 0;
			auto        inner_it = std::ranges::begin(outer_ref);
			for(auto j : js) {              // NOLINT(altera-unroll-loops) TODO(correa) change to algorithm applied on elements
				(*this)[i][j] = *inner_it;  // rng[i][j];
				++inner_it;
			}
			++outer_it;
		}

		for(index i = 1; i != is.last(); ++i) {
			auto const& outer_ref2 = *outer_it;
			assert(static_cast<multi::size_t>(outer_ref2.size()) == common_size);

			auto inner_it = std::ranges::begin(outer_ref2);
			for(auto j : js) {              // NOLINT(altera-unroll-loops) TODO(correa) change to algorithm applied on elements
				(*this)[i][j] = *inner_it;  // rng[i][j];
				++inner_it;
			}
			++outer_it;
		}
	}
#endif

	template<
		class Range, class = std::enable_if_t<!std::is_base_of<dynamic_array, std::decay_t<Range>>{}>,
		class = decltype(std::declval<Range const&>().begin()),
		class = decltype(std::declval<Range const&>().end()),
		// class = decltype(/*dynamic_array*/ (std::declval<Range const&>().begin() - std::declval<Range const&>().end())),  // instantiation of dynamic_array here gives a compiler error in 11.0, partially defined type?
		class = std::enable_if_t<!is_subarray<Range const&>::value>>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
	// cppcheck-suppress noExplicitConstructor ; because I want to use equal for lazy assigments form range-expressions // NOLINTNEXTLINE(runtime/explicit)
	dynamic_array(Range const& rng)                     // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax  // NOSONAR
	: dynamic_array(std::begin(rng), std::end(rng)) {}  // Sonar: Prefer free functions over member functions when handling objects of generic type "Range".

	template<class TT>
	auto uninitialized_fill_elements(TT const& value) {
		return array_alloc::uninitialized_fill_n(this->data_elements(), this->num_elements(), value);
	}

	template<class TT, class... As>
	dynamic_array(array_ref<TT, D, As...> const& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref{
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())),  // NOLINT(readability-redundant-typename)
		  other.extensions()
	  } {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
		}
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
#else
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.data_elements(), other.num_elements(), this->data_elements());
#endif
	}

	dynamic_array(typename dynamic_array::extensions_type extensions, typename dynamic_array::element_type const& elem, allocator_type const& alloc)  // (2)  // NOLINT(readability-redundant-typename)
	: array_alloc{alloc},
	  ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{extensions}.num_elements()), nullptr), extensions} {  // NOLINT(readability-redundant-typename)
		array_alloc::uninitialized_fill_n(this->data_elements(), static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()), elem);                         // NOLINT(readability-redundant-typename)
	}

	template<class Element>
	explicit dynamic_array(
		Element const& elem, allocator_type const& alloc,
		std::enable_if_t<std::is_convertible_v<Element, typename dynamic_array::element_type> && (D == 0), int> /*dummy*/ = 0  // NOLINT(fuchsia-default-arguments-declarations) for classic sfinae, needed by MSVC?
	)
	: dynamic_array(typename dynamic_array::extensions_type{}, elem, alloc) {}

	constexpr dynamic_array(typename dynamic_array::extensions_type exts, typename dynamic_array::element_type const& elem)  // NOLINT(readability-redundant-typename)
	: array_alloc{},
	  array_ref<T, D, typename multi::allocator_traits<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::pointer>(  // NOLINT(readability-redundant-typename)
		  exts,
		  array_alloc::allocate(
			  static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t(exts).num_elements()),  // NOLINT(readability-redundant-typename)
			  nullptr
		  )
	  ) {
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type>) {
			array_alloc::uninitialized_fill_n(this->base(), static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()), elem);
		} else {                                                                                                                             // this workaround allows constexpr arrays for simple types
			adl_fill_n(this->base(), static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()), elem);  // NOLINT(readability-redundant-typename)
		}
	}

	template<class ValueType, class = decltype(std::declval<ValueType>().extensions()), std::enable_if_t<std::is_convertible_v<ValueType, typename dynamic_array::value_type>, int> = 0>                                                           // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value, allocator_type const& alloc)                                                                                                          // fill constructor
	: array_alloc{alloc}, ref(array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t(extension * value.extensions()).num_elements())), extension * value.extensions()) {  // NOLINT(readability-redundant-typename)
		static_assert(std::is_trivially_default_constructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_default_construction<typename dynamic_array::element_type>);                                                 // TODO(correaa) not implemented for non-trivial types,
		adl_fill_n(this->begin(), this->size(), value);                                                                                                                                                                                            // TODO(correaa) implement via .elements()? substitute with uninitialized version of fill, uninitialized_fill_n?
	}

	template<class ValueType, class = decltype(std::declval<ValueType>().extensions()), std::enable_if_t<std::is_convertible_v<ValueType, typename dynamic_array::value_type>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value)                                                                              // fill constructor
	: dynamic_array(extension, value, allocator_type{}) {}

	explicit dynamic_array(::boost::multi::extensions_t<D> const& extensions, allocator_type const& alloc)
	: array_alloc{alloc}, ref(array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{extensions}.num_elements())), extensions) {
		uninitialized_default_construct();
	}

	explicit dynamic_array(::boost::multi::extensions_t<D> const& exts)
	: dynamic_array(exts, allocator_type{}) {}

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                            // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_trivially_default_constructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unused*/, allocator_type const& alloc)
	: dynamic_array(extensions, alloc) {}

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                              // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	[[deprecated("****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****")]]
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unusued*/) = delete /*[["****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****")]]*/;

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                              // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	[[deprecated("****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****")]]
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unused*/, allocator_type const& /*alloc*/) = delete /*[["****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****"]]*/;

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                            // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_trivially_default_constructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unusued*/) : dynamic_array(extensions) {}

	template<class OtherT, class OtherEP, class OtherLayout, class = std::enable_if_t<std::is_assignable<typename ref::element_ref, typename multi::subarray<OtherT, D, OtherEP, OtherLayout>::element_type>{}>, class = decltype(adl_copy(std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().begin(), std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	constexpr dynamic_array(multi::const_subarray<OtherT, D, OtherEP, OtherLayout> const& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{other.extensions()}.num_elements())),
		  other.extensions()
	  ) {
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.elements().begin(), this->num_elements(), this->data_elements());
	}

	template<class F>  // TODO(correaa) make more generic, e.g.: take ArrayWithElementsLike
	constexpr dynamic_array(multi::restriction<D, F> const& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{other.extensions()}.num_elements())),
		  other.extensions()
	  ) {
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.elements().begin(), this->num_elements(), this->data_elements());
	}

	template<class F>  // ArrayElementsLike, class = typename ArrayElementsLike::elements_t>
	// cppcheck-suppress noExplicitConstructor  // NOLINTNEXTLINE(runtime/explicit)
	constexpr dynamic_array(multi::restriction<D, F> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) to allow terse syntax
	: dynamic_array(other, allocator_type{}) {}

	template<class OtherT, class OtherEP, class OtherLayout, class = std::enable_if_t<std::is_assignable<typename ref::element_ref, typename multi::subarray<OtherT, D, OtherEP, OtherLayout>::element_type>{}>, class = decltype(adl_copy(std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().begin(), std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	constexpr dynamic_array(multi::subarray<OtherT, D, OtherEP, OtherLayout>&& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{other.extensions()}.num_elements())),
		  other.extensions()
	  ) {
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), std::move(other).elements().begin(), this->num_elements(), this->data_elements());
	}

	template<
		class TT, class EElementPtr, class LLayout, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout>&>().base()), T>, int> = 0,
		class = decltype(adl_copy(std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().begin(), std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	explicit dynamic_array(multi::const_subarray<TT, D, EElementPtr, LLayout> const& other)
	: dynamic_array(other, allocator_type{}) {}

	template<
		class TT, class EElementPtr, class LLayout, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<multi::subarray<TT, D, EElementPtr, LLayout> const&>().base()), T>, int> = 0,
		class = decltype(adl_copy(std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().begin(), std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	// cppcheck-suppress noExplicitConstructor  // NOLINTNEXTLINE(runtime/explicit)
	constexpr /*implicit*/ dynamic_array(multi::const_subarray<TT, D, EElementPtr, LLayout> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: dynamic_array(other, allocator_type{}) {}

	template<
		class TT, class EElementPtr, class LLayout, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<multi::subarray<TT, D, EElementPtr, LLayout> const&>().base()), T>, int> = 0,
		class = decltype(adl_copy(std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().begin(), std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	// cppcheck-suppress noExplicitConstructor  // NOLINTNEXTLINE(runtime/explicit)
	constexpr /*implicit*/ dynamic_array(multi::subarray<TT, D, EElementPtr, LLayout>&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: dynamic_array(std::move(other), allocator_type{}) {}

	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr dynamic_array(multi::subarray<T, D, typename dynamic_array::element_ptr, typename dynamic_array::layout_type> const&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: dynamic_array(other, allocator_type{}) {}

	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr dynamic_array(multi::const_subarray<T, D, typename dynamic_array::element_ptr, typename dynamic_array::layout_type> const&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: dynamic_array(other, allocator_type{}) {}

	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr dynamic_array(multi::subarray<T, D, typename dynamic_array::element_ptr, typename dynamic_array::layout_type>&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: dynamic_array(std::move(other), allocator_type{}) {}

	template<class TT, class... Args, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
																																											   // cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	/*mplct*/ dynamic_array(array_ref<TT, D, Args...>& other)                                                                                                                  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		dynamic_array::uninitialized_copy_elements(other.data_elements());
	}

	template<class TT, class... Args, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(array_ref<TT, D, Args...>& other)                                                                                                                    // moLINT(fuchsia-default-arguments-declarations)
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		dynamic_array::uninitialized_copy_elements(other.data_elements());
	}

	// NOLINTNEXTLINE(modernize-use-constraints) TODO(correaa) for C++20
	template<class TT, class... Args, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&&>().base()), T>, int> = 0>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	/*mplct*/ dynamic_array(array_ref<TT, D, Args...>&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(std::move(other).data_elements());
	}

	template<class TT, class... Args, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(array_ref<TT, D, Args...>&& other)                                                                                                                    // NOLINT(fuchsia-default-arguments-declarations)
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(std::move(other).data_elements());
	}

	// NOLINTNEXTLINE(modernize-use-constraints) TODO(correaa) for C++20
	template<class TT, class... Args, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...> const&>().base()), T>, int> = 0>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	/*mplct*/ dynamic_array(array_ref<TT, D, Args...> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(other.data_elements());
	}

	template<class TT, class... Args, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...> const&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(array_ref<TT, D, Args...> const& other)                                                                                                                    // NOLINT(fuchsia-default-arguments-declarations)
	: array_alloc{},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())),
		  other.extensions()
	  ) {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(std::move(other).data_elements());
	}

	constexpr dynamic_array(dynamic_array const& other)  // 5b
	: array_alloc{
		  multi::allocator_traits<allocator_type>::select_on_container_copy_construction(other.alloc())
	  },
	  ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())  //,
		  ),
		  other.extensions()} {
		assert(this->stride() != 0);
		uninitialized_copy_elements(other.data_elements());
	}

	template<class ExecutionPolicy, std::enable_if_t<!std::is_convertible_v<ExecutionPolicy, typename dynamic_array::extensions_type>, int> = 0>  // NOLINT(modernize-use-constraints,modernize-type-traits) TODO(correaa) for C++20
	dynamic_array(ExecutionPolicy&& policy, dynamic_array const& other)
	: array_alloc{multi::allocator_traits<allocator_type>::select_on_container_copy_construction(other.alloc())}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements()), other.data_elements()), extensions(other)} {
		assert(this->stride() != 0);
		uninitialized_copy_elements(std::forward<ExecutionPolicy>(policy), other.data_elements());
	}

	// cppcheck-suppress noExplicitConstructor ; to allow assignment-like construction of nested arrays
	constexpr dynamic_array(std::initializer_list<typename dynamic_array<T, D>::value_type> values)
	: dynamic_array{(values.size() == 0) ? array<T, D>() : array<T, D>(values.begin(), values.end())} {}  // construct all with default constructor and copy to special memory at the end

	// template<class TT, std::enable_if_t<std::is_same_v<T, TT> || (D == 2), int> =0>
	// constexpr dynamic_array(std::initializer_list<std::initializer_list<TT>> il)
	// : dynamic_array{(il.size() == 0) ? array<T, D>() : array<T, D>(il.begin(), il.end())} {}  // construct all with default constructor and copy to special memory at the end

	dynamic_array(
		std::initializer_list<typename dynamic_array<T, D>::value_type> values,
		allocator_type const&                                           alloc
	)
	: dynamic_array{(values.size() == 0) ? dynamic_array<T, D>() : dynamic_array<T, D>(values.begin(), values.end()), alloc} {}

	template<class TT, std::size_t N>
	constexpr explicit dynamic_array(TT (&array)[N])  // @SuppressWarnings(cpp:S5945) NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backward compatibility // NOSONAR
	: dynamic_array(std::begin(array), std::end(array)) {
		assert(this->stride() != 0);
	}

	constexpr auto begin() const& noexcept -> typename dynamic_array::const_iterator { return ref::begin(); }
	constexpr auto end() const& noexcept -> typename dynamic_array::const_iterator { return ref::end(); }

	constexpr auto begin() && noexcept -> typename dynamic_array::move_iterator { return ref::begin(); }
	constexpr auto end() && noexcept -> typename dynamic_array::move_iterator { return ref::end(); }

	constexpr auto begin() & noexcept -> typename dynamic_array::iterator { return ref::begin(); }
	constexpr auto end() & noexcept -> typename dynamic_array::iterator { return ref::end(); }

	using ref::operator[];

	BOOST_MULTI_HD constexpr auto operator[](index idx) && -> decltype(auto) {
		return multi::move(ref::operator[](idx));
	}

	constexpr auto max_size() const noexcept { return static_cast<typename dynamic_array::size_type>(multi::allocator_traits<allocator_type>::max_size(this->alloc())); }  // TODO(correaa)  divide by nelements in under-dimensions?

 protected:
	constexpr void deallocate() {
		assert(this->stride() != 0);
		if(this->num_elements()) {
			multi::allocator_traits<allocator_type>::deallocate(this->alloc(), this->base_, static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()));
		}
	}

	void clear() noexcept {
		this->destroy();
		deallocate();
		this->layout_mutable() = typename dynamic_array::layout_type(typename dynamic_array::extensions_type{});
		assert(this->stride() != 0);
	}

 public:
	constexpr dynamic_array() noexcept
	: array_alloc{}, ref(nullptr, typename dynamic_array::extensions_type{}) {
		assert(this->stride() != 0);
		assert(this->size() == 0);
	}

#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
	constexpr ~dynamic_array()
#else
	~dynamic_array()
#endif
	{
		assert(this->stride() != 0);
		destroy();
		assert(this->stride() != 0);
		deallocate();
	}

	using element_const_ptr = typename std::pointer_traits<typename dynamic_array::element_ptr>::template rebind<typename dynamic_array::element_type const>;
	using element_move_ptr  = multi::move_ptr<typename dynamic_array::element_ptr>;

	using reference = std::conditional_t<
		(D > 1),
		subarray<typename dynamic_array::element_type, D - 1, typename dynamic_array::element_ptr>,
		std::conditional_t<
			D == 1,
			typename std::iterator_traits<typename dynamic_array::element_ptr>::reference,
			void>>;

	using const_reference = std::conditional_t<
		(D > 1),
		const_subarray<typename dynamic_array::element_type, D - 1, typename dynamic_array::element_ptr>,  // TODO(correaa) should be const_reference, but doesn't work witn rangev3?
		std::conditional_t<
			D == 1,
			decltype(*std::declval<typename dynamic_array::element_const_ptr>()),
			void>>;

	using iterator       = multi::array_iterator<T, D, typename dynamic_array::element_ptr>;
	using const_iterator = multi::array_iterator<T, D, typename dynamic_array::element_ptr, true>;

	friend auto get_allocator(dynamic_array const& self) -> allocator_type { return self.get_allocator(); }

	// cppcheck-suppress duplInheritedMember ; to override
	BOOST_MULTI_HD constexpr auto data_elements() const& -> element_const_ptr { return this->base_; }

	// cppcheck-suppress duplInheritedMember ; to override
	BOOST_MULTI_HD constexpr auto data_elements() & -> typename dynamic_array::element_ptr { return this->base_; }

	// cppcheck-suppress duplInheritedMember ; to override
	BOOST_MULTI_HD constexpr auto data_elements() && -> typename dynamic_array::element_move_ptr { return std::make_move_iterator(this->base_); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(dynamic_array const& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(dynamic_array& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(dynamic_array&& self) { return std::move(self).data_elements(); }

	constexpr auto base() & -> typename dynamic_array::element_ptr { return ref::base(); }
	constexpr auto base() const& -> typename dynamic_array::element_const_ptr { return typename dynamic_array::element_const_ptr{ref::base()}; }

	constexpr auto origin() & -> typename dynamic_array::element_ptr { return ref::origin(); }
	constexpr auto origin() const& -> typename dynamic_array::element_const_ptr { return ref::origin(); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array& self) -> typename dynamic_array::element_ptr { return self.origin(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array const& self) -> typename dynamic_array::element_const_ptr { return self.origin(); }

	template<class TT, typename EElementPtr, class LLayout>
	auto operator=(multi::const_subarray<TT, D, EElementPtr, LLayout> const& other) -> dynamic_array& {
		ref::operator=(other);  // TODO(correaa) : protect for self assigment
		assert(this->stride() != 0);
		return *this;
	}
	auto operator=(dynamic_array const& other) & -> dynamic_array& {
		if(std::addressof(other) == this) {
			return *this;
		}  // cert-oop54-cpp
		assert(other.extensions() == this->extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		assert(this->stride() != 0);
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	constexpr auto operator=(dynamic_array&& other) noexcept -> dynamic_array& {                               // lints  (cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
		assert(extensions(other) == dynamic_array::extensions());                                              // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : allow a constexpr-friendly assert
		adl_move(other.data_elements(), other.data_elements() + other.num_elements(), this->data_elements());  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) there is no std::move_n algorithm
		assert(this->stride() != 0);
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	template<class TT, class... As>
	auto operator=(dynamic_array<TT, D, As...> const& other) & -> dynamic_array& {
		assert(extensions(other) == dynamic_array::extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

	template<class Archive>
	// cppcheck-suppress duplInheritedMember ; to override
	void serialize(Archive& arxiv, unsigned int const version) { ref::serialize(arxiv, version); }

 private:
	void swap_(dynamic_array& other) noexcept { operator()().swap(other()); }

 public:
	friend void swap(dynamic_array& lhs, dynamic_array& rhs) noexcept { lhs.swap_(rhs); }
};

template<typename T, dimensionality_type D, class Alloc = std::allocator<T>>
using static_array [[deprecated("static_array has been renamed to dynamics_array (uses dynamic memory)")]] = dynamic_array<T, D, Alloc>;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<typename T, class Alloc>
struct dynamic_array<T, ::boost::multi::dimensionality_type{0}, Alloc>  // NOLINT(fuchsia-multiple-inheritance) : design
: protected detail::array_allocator<Alloc>
, public array_ref<T, 0, typename multi::allocator_traits<typename detail::array_allocator<Alloc>::allocator_type>::pointer> {
	static_assert(std::is_same_v<typename multi::allocator_traits<Alloc>::value_type, typename dynamic_array::element_type>, "allocator value type must match array value type");

 private:
	using array_alloc = detail::array_allocator<Alloc>;

 public:
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	// NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() && -> dynamic_array* = delete;  // NOSONAR(cpp:S877) NOLINT(google-runtime-operator) : delete to avoid taking address of temporary
	// NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() & -> dynamic_array* { return this; }  // NOSONAR(cpp:S877) NOLINT(google-runtime-operator) : override from base
	// NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() const& -> dynamic_array const* { return this; }  // NOSONAR(cpp:S877) NOLINT(google-runtime-operator) : override from base
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using array_alloc::get_allocator;
	using allocator_type = typename dynamic_array::allocator_type;
	using decay_type     = array<T, 0, Alloc>;

	template<class Ptr>
	void assign(Ptr data) & {
		if(data) {
			assert(this->num_elements() == 1);
			adl_copy_n(data, this->num_elements(), this->base());
		}
	}

	template<
		class Singleton, std::enable_if_t<!std::is_base_of_v<dynamic_array, Singleton> && !std::is_same_v<Singleton, typename dynamic_array::element_type>, int> = 0,
		class = decltype(adl_copy_n(&std::declval<Singleton>(), 1, typename dynamic_array::element_ptr{}))>
	auto operator=(Singleton const& single) -> dynamic_array& {
		assign(&single);
		return *this;
	}

 protected:
	using alloc_traits = typename multi::allocator_traits<allocator_type>;
	using ref          = array_ref<T, 0, typename multi::allocator_traits<typename multi::allocator_traits<Alloc>::template rebind_alloc<T>>::pointer>;

	auto uninitialized_value_construct() {
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			return adl_alloc_uninitialized_value_construct_n(dynamic_array::alloc(), this->base_, this->num_elements());
		}
	}

	template<typename It> auto uninitialized_copy(It first) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(this->alloc(), this->data_elements(), this->num_elements());
		}
		return adl_copy(first, this->num_elements(), this->data_elements());
#else
		return adl_alloc_uninitialized_copy_n(this->alloc(), first, this->num_elements(), this->data_elements());
#endif
	}

	template<typename It>
	auto uninitialized_move(It first) {
		return adl_alloc_uninitialized_move_n(this->alloc(), first, this->num_elements(), this->data_elements());
	}

	constexpr void destroy() {
		if constexpr(!(std::is_trivially_destructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_destruction<typename dynamic_array::element_type>)) {
			array_alloc::destroy_n(this->data_elements(), this->num_elements());
		}
	}

 public:
	using typename ref::difference_type;
	using typename ref::size_type;
	using typename ref::value_type;

	constexpr explicit dynamic_array(allocator_type const& alloc) : array_alloc{alloc} {}

	BOOST_MULTI_HD constexpr dynamic_array(decay_type&& other, allocator_type const& alloc) noexcept  // 6b
	: array_alloc{alloc}, ref{other.base_, other.extensions()} {
		std::move(other).ref::layout_t::operator=({});
	}

	using ref::operator==;
	using ref::operator!=;

	dynamic_array(
		typename dynamic_array::extensions_type const& extensions,
		typename dynamic_array::element const& elem, allocator_type const& alloc
	)
	: array_alloc{alloc},
	  ref(dynamic_array::allocate(
			  static_cast<typename multi::allocator_traits<allocator_type>::size_type>(
				  typename dynamic_array::layout_t{extensions}.num_elements()
			  )
		  ),
		  extensions) {
		uninitialized_fill(elem);
	}

	dynamic_array(typename dynamic_array::element_type const& elem, allocator_type const& alloc)
	: dynamic_array(typename dynamic_array::extensions_type{}, elem, alloc) {}

	template<typename OtherT, typename OtherEPtr, class OtherLayout>
	explicit dynamic_array(multi::const_subarray<OtherT, 0, OtherEPtr, OtherLayout> const& other, allocator_type const& alloc)
	: array_alloc{alloc}, ref(dynamic_array::allocate(other.num_elements()), extensions(other)) {
		assert(other.num_elements() <= 1);
		if(other.num_elements()) {
#if defined(__clang__) && defined(__CUDACC__)
			if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
				adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
			}
			adl_copy(other.base(), other.base() + other.num_elements(), this->base());
#else
			adl_alloc_uninitialized_copy(dynamic_array::alloc(), other.base(), other.base() + other.num_elements(), this->base());
#endif
		}
	}

	template<class TT, class... Args>
	explicit dynamic_array(multi::dynamic_array<TT, 0, Args...> const& other, allocator_type const& alloc)  // TODO(correaa) : call other constructor (above)
	: array_alloc{alloc}, ref(dynamic_array::allocate(static_cast<typename std::allocator_traits<Alloc>::size_type>(other.num_elements())), extensions(other)) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
		}
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
#else
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.data_elements(), other.num_elements(), this->data_elements());
#endif
	}

	template<class TT, class... Args>
	explicit dynamic_array(multi::dynamic_array<TT, 0, Args...> const& other)
	: dynamic_array(other, allocator_type{}) {}

	auto uninitialized_fill(typename dynamic_array::element_type const& elem) {
		array_alloc::uninitialized_fill_n(
			this->base_,
			static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()),
			elem
		);
	}

	template<class TT, class... Args>
	auto operator=(multi::const_subarray<TT, 0, Args...> const& other) -> dynamic_array& {
		adl_copy_n(other.base(), 1, this->base());
		return *this;
	}

	dynamic_array(
		typename dynamic_array::extensions_type const& extensions,
		typename dynamic_array::element_type const&    elem
	)  // 2
	: array_alloc{}, ref(dynamic_array::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{extensions}.num_elements()), nullptr), extensions) {
		uninitialized_fill(elem);
	}

	dynamic_array() : dynamic_array(multi::iextensions<0>{}) {}  // TODO(correaa) a noexcept will force a partially formed state for zero dimensional arrays

	explicit dynamic_array(typename dynamic_array::element_type const& elem)
	: dynamic_array(multi::iextensions<0>{}, elem) {}

	template<
		class Singleton, std::enable_if_t<!std::is_base_of_v<dynamic_array, Singleton> && !std::is_same_v<Singleton, typename dynamic_array::element_type>, int> = 0,  // NOLINT(modernize-type-traits) for C++20
		class = decltype(adl_copy_n(&std::declval<Singleton>(), 1, typename dynamic_array::element_ptr{}))>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax  // NOLINTNEXTLINE(runtime/explicit)
	/*implict*/ dynamic_array(Singleton const& single)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) this is used by the
	: ref(dynamic_array::allocate(1), typename dynamic_array::extensions_type{}) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
		}
		adl_copy_n(&single, typename multi::allocator_traits<Alloc>::size_type{1}, this->data_elements());
#else
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), &single, typename multi::allocator_traits<Alloc>::size_type{1}, this->data_elements());
#endif
	}

	template<class ValueType, typename = std::enable_if_t<std::is_same_v<ValueType, value_type>>>                                          // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value, allocator_type const& alloc)  // 3
	: dynamic_array(extension * extensions(value), alloc) {
		assert(this->stride() != 0);
		using std::fill;
		fill(this->begin(), this->end(), value);
	}

	template<class ValueType, typename = std::enable_if_t<std::is_same_v<ValueType, value_type>>>             // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value)  // 3
	: dynamic_array(extension, value, allocator_type{}) {}
	// : dynamic_array(extension * extensions(value)) {  // TODO(correaa) : call other constructor (above)
	//  assert(this->stride() != 0);
	//  using std::fill;
	//  fill(this->begin(), this->end(), value);
	// }

	explicit dynamic_array(typename dynamic_array::extensions_type const& extensions, allocator_type const& alloc)  // 3
	: array_alloc{alloc}, ref(dynamic_array::allocate(typename dynamic_array::layout_t{extensions}.num_elements()), extensions) {
		// assert(this->stride() != 0);
		uninitialized_value_construct();
	}
	explicit dynamic_array(typename dynamic_array::extensions_type const& extensions)  // 3
	: dynamic_array(extensions, allocator_type{}) {
		// assert(this->stride() != 0);
	}

	dynamic_array(dynamic_array const& other, allocator_type const& alloc)  // 5b
	: array_alloc{alloc}, ref(dynamic_array::allocate(other.num_elements()), extensions(other)) {
		assert(this->stride() != 0);
		uninitialized_copy_(other.data_elements());
	}

	dynamic_array(dynamic_array const& other)  // 5b
	: array_alloc{other.get_allocator()}, ref{dynamic_array::allocate(other.num_elements(), other.data_elements()), {}} {
		assert(this->stride() != 0);
		uninitialized_copy(other.data_elements());
	}

	dynamic_array(dynamic_array&& other) noexcept
	: array_alloc{other.get_allocator()}, ref(std::exchange(other.base_, nullptr), other.extensions()) {
		other.layout_mutable() = {};
		// other.layout_t<0>::operator=({});
		// , ref(dynamic_array::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements()), other.data_elements()), other.extensions()) {
		//  adl_alloc_uninitialized_move_n(
		//      this->alloc(),
		//      other.data_elements(),
		//      other.num_elements(),
		//      this->data_elements()
		//  );
		(void)std::move(other);
	}

 protected:
	void deallocate() {  // TODO(correaa) : move this to detail::array_allocator
		if(this->num_elements() && this->base_) {
			multi::allocator_traits<allocator_type>::deallocate(this->alloc(), this->base_, static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()));
		}
	}
	void clear() noexcept {
		this->destroy();
		deallocate();
		layout_t<0>::operator=({});
	}

 public:
	~dynamic_array() noexcept {
		this->destroy();
		deallocate();
	}
	using element_const_ptr = typename std::pointer_traits<typename dynamic_array::element_ptr>::template rebind<typename dynamic_array::element_type const>;

	BOOST_MULTI_FRIEND_CONSTEXPR auto get_allocator(dynamic_array const& self) -> allocator_type { return self.get_allocator(); }

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto base() & -> typename dynamic_array::element_ptr { return ref::base(); }
	constexpr auto base() const& -> typename dynamic_array::element_const_ptr { return ref::base(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	BOOST_MULTI_FRIEND_CONSTEXPR auto base(dynamic_array& self) -> typename dynamic_array::element_ptr { return self.base(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto base(dynamic_array const& self) -> typename dynamic_array::element_const_ptr { return self.base(); }

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto origin() & -> typename dynamic_array::element_ptr { return ref::origin(); }
	constexpr auto origin() const& -> typename dynamic_array::element_const_ptr { return ref::origin(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array& self) -> typename dynamic_array::element_ptr { return self.origin(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array const& self) -> typename dynamic_array::element_const_ptr { return self.origin(); }

	// NOSONAR
	constexpr operator typename std::iterator_traits<typename dynamic_array::element_const_ptr>::reference() const& {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return *(this->base_);
	}

	// NOSONAR
	constexpr operator std::add_rvalue_reference_t<typename std::iterator_traits<typename dynamic_array::element_ptr>::reference>() && {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return std::move(*(this->base_));
	}

	// NOSONAR
	constexpr operator typename std::iterator_traits<typename dynamic_array::element_ptr>::reference() & {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return *(this->base_);
	}

	template<class OtherElement, std::enable_if_t<std::is_convertible_v<typename dynamic_array::element_type, OtherElement>, int> = 0>  // NOLINT(modernize-use-constraints)
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr explicit operator OtherElement() const {
		return static_cast<OtherElement>(*(this->base_));
	}

	constexpr auto rotated() const& {  // cppcheck-suppress duplInheritedMember ; to overwrite
		typename dynamic_array::layout_t new_layout = this->layout();
		new_layout.rotate();
		return subarray<T, 0, typename dynamic_array::element_const_ptr>{new_layout, this->base_};
	}

	constexpr auto rotated() & {  // cppcheck-suppress duplInheritedMember ; to overwrite
		typename dynamic_array::layout_t new_layout = this->layout();
		new_layout.rotate();
		return subarray<T, 0, typename dynamic_array::element_ptr>{new_layout, this->base_};
	}

	constexpr auto rotated() && {  // cppcheck-suppress duplInheritedMember ; to overwrite
		typename dynamic_array::layout_t new_layout = this->layout();
		new_layout.rotate();
		return subarray<T, 0, typename dynamic_array::element_ptr>{new_layout, this->base_};
	}

	friend constexpr auto rotated(dynamic_array& self) -> decltype(auto) { return self.rotated(); }
	friend constexpr auto rotated(dynamic_array const& self) -> decltype(auto) { return self.rotated(); }

 private:
	constexpr auto unrotated_aux_() {
		typename dynamic_array::layout_t new_layout = *this;
		new_layout.unrotate();
		return subarray<T, 0, typename dynamic_array::element_const_ptr>{new_layout, this->base_};
	}

 public:
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto unrotated() & { return unrotated_aux_(); }
	constexpr auto unrotated() const& { return unrotated_aux_().as_const(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	friend constexpr auto unrotated(dynamic_array& self) -> decltype(auto) { return self.unrotated(); }
	friend constexpr auto unrotated(dynamic_array const& self) -> decltype(auto) { return self.unrotated(); }

	constexpr auto operator=(dynamic_array const& other) -> dynamic_array& {
		assert(extensions(other) == dynamic_array::extensions());  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : allow a constexpr-friendly assert
		if(this == &other) {
			return *this;
		}  // lints (cert-oop54-cpp) : handle self-assignment properly
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

 private:
	constexpr auto equal_extensions_if_(std::true_type /*true */, dynamic_array const& other) { return this->extensions() == extensions(other); }
	constexpr auto equal_extensions_if_(std::false_type /*false*/, dynamic_array const& /*other*/) { return true; }

 public:
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

	constexpr auto operator=(dynamic_array&& other) noexcept -> dynamic_array& {
		assert(equal_extensions_if_(std::integral_constant<bool, (dynamic_array::rank_v != 0)>{}, other));     // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : allow a constexpr-friendly assert
		adl_move(other.data_elements(), other.data_elements() + other.num_elements(), this->data_elements());  // there is no std::move_n algorithm  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	template<class TT, class... As, class = std::enable_if_t<std::is_assignable<typename dynamic_array::element_ref, TT>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	auto operator=(dynamic_array<TT, 0, As...> const& other) & -> dynamic_array& {
		assert(extensions(other) == dynamic_array::extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

	constexpr explicit operator subarray<value_type, 0, typename dynamic_array::element_const_ptr, typename dynamic_array::layout_t>() & {  // cppcheck-suppress duplInheritedMember ; to overwrite
		// cppcheck-suppress duplInheritedMember ; to overwrite
		return this->template dynamic_array_cast<value_type, typename dynamic_array::element_const_ptr>();  // cppcheck-suppress duplInheritedMember ; to overwrite
																											// return dynamic_array_cast<typename dynamic_array::value_type, typename dynamic_array::element_const_ptr>(*this);
	}

	template<class Archive>
	void serialize(Archive& arxiv, unsigned int const version) {  // cppcheck-suppress duplInheritedMember ; to overwrite
		ref::serialize(arxiv, version);
	}
};

template<class T, multi::dimensionality_type D, std::size_t Capacity = 4UL * 4UL>
using inplace_array = multi::dynamic_array<T, D, multi::detail::static_allocator<T, Capacity>>;

template<typename T, class Alloc>
struct array<T, 0, Alloc> : dynamic_array<T, 0, Alloc> {
	using dynamic_array<T, 0, Alloc>::dynamic_array;

	using dynamic_array<T, 0, Alloc>::operator=;

#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	BOOST_MULTI_HD constexpr auto operator[]() const& { return *(this->base()); }
#endif

#if !defined(__NVCOMPILER) || (__NVCOMPILER_MAJOR__ > 22 || (__NVCOMPILER_MAJOR__ == 22 && __NVCOMPILER_MINOR__ > 5))  // bug in nvcc 22.5: error: "operator=" has already been declared in the current scope
	template<class TT, class... Args>
	auto operator=(multi::array<TT, 0, Args...> const& other) & -> array& {
		if(other.base()) {
			adl_copy_n(other.base(), other.num_elements(), this->base());
		}
		return *this;
	}

	template<class TT, class... Args>
	auto operator=(multi::array<TT, 0, Args...> const& other) && -> array&& {  // NOLINT(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator) should assigment return auto& ?
		if(other.base()) {
			adl_copy_n(other.base(), other.num_elements(), this->base());
		}
		return std::move(*this);
	}
#endif

	auto reextent(typename array::extensions_type const& /*empty_extensions*/) -> array& {  // NOLINT(readability-redundant-typename)
		return *this;
	}

	// cppcheck-suppress duplInheritedMember ; to overwrite  // NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() && -> array* = delete;  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, ::boost::multi::dimensionality_type D, class Alloc>
struct array : dynamic_array<T, D, Alloc> {
	using static_ = dynamic_array<T, D, Alloc>;

	static_assert(
		std::is_same_v<typename multi::allocator_traits<Alloc>::value_type, T> || std::is_same_v<typename multi::allocator_traits<Alloc>::value_type, void>,
		"only exact type of array element or void (default) is allowed as allocator value type"
	);

	// cppcheck-suppress duplInheritedMember ; to override  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() && -> array* = delete;  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary
	// cppcheck-suppress duplInheritedMember ; to override  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() & -> array* { return this; }  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary
	// cppcheck-suppress duplInheritedMember ; to override  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() const& -> array const* { return this; }  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary

	template<class Archive, class ArTraits = multi::archive_traits<Archive>>
	void serialize(Archive& arxiv, unsigned int const version) {  // cppcheck-suppress duplInheritedMember ; to override
		auto extensions_ = this->extensions();

		arxiv& ArTraits::make_nvp("extensions", extensions_);  // don't try `using ArTraits::make_nvp`, make_nvp is a static member
		if(this->extensions() != extensions_) {
			clear();
			this->reextent(extensions_);
		}
		static_::serialize(arxiv, version);
	}

	// vvv workaround for MSVC 14.3 and ranges, TODO(correaa) good solution would be to inherit from const_subarray
	BOOST_MULTI_HD operator subarray<T, D, typename array::element_const_ptr, typename array::layout_type> const&() const {     // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return reinterpret_cast<subarray<T, D, typename array::element_const_ptr, typename array::layout_type> const&>(*this);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
	}

	// move this to dynamic_array
	template<
		class Range, std::enable_if_t<!has_extensions<std::decay_t<Range>>::value, int> = 0,
		class = decltype(Range{std::declval<typename array::const_iterator>(), std::declval<typename array::const_iterator>()})>
	constexpr explicit operator Range() const {  // cppcheck-suppress duplInheritedMember ; to overwrite
		// vvv Range{...} needed by Windows GCC?
		return Range{this->begin(), this->end()};  // NOLINT(fuchsia-default-arguments-calls) e.g. std::vector(it, it, alloc = {})
	}

	// move this to dynamic_array
	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>                          // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN const&() const& { return this->template to_carray_<TTN>(); }  // cppcheck-suppress duplInheritedMember ; to override

	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>                // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN&() && { return this->template to_carray_<TTN>(); }  // cppcheck-suppress duplInheritedMember ; to override

	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>               // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN&() & { return this->template to_carray_<TTN>(); }  // cppcheck-suppress duplInheritedMember ; to override

	// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved) false positive in clang-tidy 17-20 ?
	using dynamic_array<T, D, Alloc>::dynamic_array;  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) passing c-arrays to base
	using typename dynamic_array<T, D, Alloc>::value_type;

	// #if defined(_MSC_VER)
	//  explicit array(typename array::extensions_type const& exts)  // NOTE(correaa) if you think you need to implement this for MSVC is because MSVC needs to be compiled in permissive- mode for C++17
	//  : dynamic_array<T, D, Alloc>(exts) { }
	// #endif

	// cppcheck-suppress noExplicitConstructor ; to allow assignment-like construction of nested arrays
	constexpr array(std::initializer_list<typename dynamic_array<T, D>::value_type> ilv)
	: static_{
		  (ilv.size() == 0) ? array<T, D>()
							: array<T, D>(ilv.begin(), ilv.end())
	  } {
	}

	template<
		class OtherT,
		class = std::enable_if_t<std::is_constructible_v<typename dynamic_array<T, D>::value_type, OtherT> && !std::is_convertible_v<OtherT, typename dynamic_array<T, D>::value_type> && (D == 1)>>  // NOLINT(modernize-use-constraints,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) TODO(correaa) for C++20
	constexpr explicit array(std::initializer_list<OtherT> ilv)                                                                                                                                       // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) inherit explicitness of conversion from the elements
	: static_{
		  (ilv.size() == 0) ? array<T, D>()()
							: array<T, D>(ilv.begin(), ilv.end()).element_transformed([](auto const& elem) noexcept { return static_cast<T>(elem); })
	  } {}

	array()             = default;
	array(array const&) = default;

	~array() = default;

	auto reshape(typename array::extensions_type extensions) & -> array& {
		typename array::layout_t const new_layout{extensions};  // TODO(correaa) implement move-reextent in terms of reshape
		assert(new_layout.num_elements() == this->num_elements());
		this->layout_mutable() = new_layout;
		assert(this->stride() != 0);
		return *this;
	}

	auto clear() noexcept -> array& {  // cppcheck-suppress duplInheritedMember ; to override
		static_::clear();
		assert(this->stride() != 0);
		return *this;
	}
	friend auto clear(array& self) noexcept -> array& { return self.clear(); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(array const& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(array& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(array&& self) { return std::move(self).data_elements(); }

	friend BOOST_MULTI_HD constexpr auto move(array& self) -> decltype(auto) { return std::move(self); }
	friend BOOST_MULTI_HD constexpr auto move(array&& self) -> decltype(auto) { return std::move(self); }

	BOOST_MULTI_HD constexpr array(array&& other, Alloc const& alloc) noexcept
	: dynamic_array<T, D, Alloc>{std::move(other), alloc} {
		assert(this->stride() != 0);
	}

	// BOOST_MULTI_HD constexpr array(array&& other) noexcept : array{std::move(other), other.get_allocator()} {
	// 	assert(this->stride() != 0);
	// }

	BOOST_MULTI_HD constexpr array(array&& other) noexcept : dynamic_array<T, D, Alloc>{std::move(other)} {
		assert(this->stride() != 0);
	}

	// friend auto get_allocator(array const& self) -> typename array::allocator_type { return self.get_allocator(); }

	void swap(array& other) noexcept {
		using std::swap;
		if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_swap::value) {
			swap(this->alloc(), other.alloc());
		}
		swap(this->base_, other.base_);
		swap(
			this->layout_mutable(),
			other.layout_mutable()
		);
		assert(this->stride() != 0);
	}

#ifndef NOEXCEPT_ASSIGNMENT
	auto operator=(array&& other) noexcept -> array& {
		if(this == std::addressof(other)) {
			return *this;
		}
		clear();
		this->base_ = other.base_;
		if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_move_assignment::value) {
			this->alloc() = std::move(other.alloc());
		}
		this->layout_mutable() = std::exchange(other.layout_mutable(), typename array::layout_type(typename array::extensions_type{}));
		assert(this->stride() != 0);
		assert(other.stride() != 0);
		return *this;
	}

	auto operator=(array const& other) -> array& {
		if(array::extensions() == other.extensions()) {
			if(this == &other) {
				return *this;
			}  // required by cert-oop54-cpp
			if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_copy_assignment::value) {
				this->alloc() = other.alloc();
			}
			static_::operator=(other);
		} else {
			clear();
			if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_copy_assignment::value) {
				this->alloc() = other.alloc();
			}
			this->layout_mutable() = other.layout();
			array::allocate();
			array::uninitialized_copy_elements(other.data_elements());
		}
		return *this;
	}
#else
	auto operator=(array o) noexcept -> array& { return swap(o), *this; }
#endif

	template<typename OtherT, typename OtherEP, class OtherLayout>
	auto operator=(multi::const_subarray<OtherT, D, OtherEP, OtherLayout> const& other) -> array& {
		if(array::extensions() == other.extensions()) {
			static_::operator=(other);  // TODO(correaa) : protect for self assigment
		} else {
			operator=(array{other});
		}
		return *this;
	}

	template<class TT, class AAlloc>
	auto operator=(multi::array<TT, D, AAlloc> const& other) -> array& {
		if(array::extensions() == other.extensions()) {
			static_::operator=(other);
		} else if(this->num_elements() == other.extensions().num_elements()) {
			reshape(other.extensions());
			static_::operator=(other);
		} else {
			operator=(static_cast<array>(other));
		}
		return *this;
	}

	template<
		class Range, class = decltype(std::declval<static_&>().operator=(std::declval<Range&&>())),
		std::enable_if_t<!has_data_elements<std::decay_t<Range>>::value, int> = 0,
		std::enable_if_t<has_extensions<std::decay_t<Range>>::value, int>     = 0,
		std::enable_if_t<!std::is_base_of_v<array, std::decay_t<Range>>, int> = 0>  // NOLINT(modernize-use-constraints,modernize-type-traits) for C++20
	auto operator=(Range&& other) -> array& {
		if(array::extensions() == other.extensions()) {
			this->operator()() = std::forward<Range>(other);
		} else if(this->num_elements() == other.extensions().num_elements()) {
			reshape(other.extensions());
			this->operator()() = std::forward<Range>(other);
		} else {
			operator=(static_cast<array>(std::forward<Range>(other)));
		}
		return *this;
	}

	template<
		class Range, class = decltype(std::declval<static_&>().operator=(std::declval<Range&&>())),
		std::enable_if_t<!std::is_base_of_v<array, std::decay_t<Range>>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	auto from(Range&& other) -> array& {                                            // TODO(correaa) : check that LHS is not read-only?
		if(array::extensions() == other.extensions()) {
			this->operator()() = other;
		} else if(this->num_elements() == other.extensions().num_elements()) {
			reshape(other.extensions());
			this->operator()() = other;
		} else {
			operator=(static_cast<array>(std::forward<Range>(other)));
		}
		return *this;
	}

	friend void swap(array& self, array& other) noexcept(true /*noexcept(self.swap(other))*/) { self.swap(other); }

	void assign(typename array::extensions_type extensions, typename array::element_type const& elem) {
		if(array::extensions() == extensions) {
			adl_fill_n(this->base_, this->num_elements(), elem);
		} else {
			this->clear();
			(*this).array::layout_t::operator=(layout_t<D>{extensions});
			this->base_ = this->static_::array_alloc::allocate(this->num_elements(), nullptr);
			adl_alloc_uninitialized_fill_n(this->alloc(), this->base_, this->num_elements(), elem);
		}
	}

	template<class It>
	auto assign(It first, It last) -> array& {  // cppcheck-suppress duplInheritedMember ; to overwrite
		using std::all_of;
		using std::next;
		if(adl_distance(first, last) == this->size()) {
			static_::ref::assign(first);
		} else {
			this->operator=(array(first, last));
		}
		return *this;
	}
	void assign(std::initializer_list<value_type> values) {
		if(values.size() != 0) {
			assign(values.begin(), values.end());
		}
	}

	template<class Range> auto assign(Range&& other) & -> decltype(assign(adl_begin(std::forward<Range>(other)), adl_end(std::forward<Range>(other)))) {
		return assign(adl_begin(std::forward<Range>(other)), adl_end(std::forward<Range>(other)));  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved)
	}

	auto operator=(std::initializer_list<value_type> values) -> array& {
		if(values.size() == 0) {
			this->clear();
		} else {
			assign(values.begin(), values.end());
		}
		return *this;
	}

	auto reextent(typename array::extensions_type const& extensions) && -> array&& {  // NOLINT(readability-redundant-typename)
		if(extensions == this->extensions()) {
			return std::move(*this);
		}

		auto new_layout = typename array::layout_t{extensions};

		if(new_layout.num_elements() != this->layout().num_elements()) {
			this->destroy();
			this->deallocate();

			this->layout_mutable() = new_layout;  // typename array::layout_t{extensions};
			this->base_            = this->static_::array_alloc::allocate(
                static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(
                    new_layout.num_elements()
                ),
                this->data_elements()  // used as hint
            );

			if constexpr(!(std::is_trivially_default_constructible_v<typename array::element_type> || multi::force_element_trivial_default_construction<typename array::element_type>)) {
				adl_alloc_uninitialized_value_construct_n(this->alloc(), this->base_, this->num_elements());
			}
		} else {
			this->layout_mutable() = new_layout;
		}

		return std::move(*this);
	}

	auto reextent(typename array::extensions_type const& extensions) & -> array& {  // NOLINT(readability-redundant-typename)
		if(extensions == this->extensions()) {
			return *this;
		}
		auto&& tmp = typename array::ref(
			this->static_::array_alloc::allocate(
				static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(
					typename array::layout_t{extensions}.num_elements()
				),
				this->data_elements()  // used as hint
			),
			extensions
		);
		if constexpr(!(std::is_trivially_default_constructible_v<typename array::element_type> || multi::force_element_trivial_default_construction<typename array::element_type>)) {
			adl_alloc_uninitialized_value_construct_n(this->alloc(), tmp.data_elements(), tmp.num_elements());
		}
		auto const is = intersection(this->extensions(), extensions);
		tmp.apply(is) = this->apply(is);  // TODO(correaa) : use (and implement) `.move();`
		this->destroy();
		this->deallocate();
		this->base_            = tmp.base();
		this->layout_mutable() = tmp.layout();
		return *this;
	}

	[[nodiscard]] constexpr auto operator+() const& { return array{*this}; }  // cppcheck-suppress duplInheritedMember ; to overwrite
	[[nodiscard]] constexpr auto operator+() && { return array{*this}; }      // cppcheck-suppress duplInheritedMember ; to overwrite

	auto reextent(typename array::extensions_type const& exs, typename array::element_type const& elem) & -> array& {  // NOLINT(readability-redundant-typename)
		if(exs == this->extensions()) {
			return *this;
		}

		// array tmp(x, e, this->get_allocator());  // TODO(correaa) opportunity missed to use hint allocation
		// auto const is = intersection(this->extensions(), x);
		// tmp.apply(is) = this->apply(is);
		// swap(tmp);

		// implementation with hint
		auto&& tmp = typename array::ref(
			this->static_::array_alloc::allocate(
				static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(typename array::layout_t{exs}.num_elements()),
				this->data_elements()  // use as hint
			),
			exs
		);
		this->uninitialized_fill_n(tmp.data_elements(), static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(tmp.num_elements()), elem);
		auto const is = intersection(this->extensions(), exs);
		tmp.apply(is) = this->apply(is);
		this->destroy();
		this->deallocate();
		this->base_            = tmp.base();  // TODO(correaa) : use (and implement) `.move();`
		this->layout_mutable() = tmp.layout();
		//  (*this).array::layout_t::operator=(tmp.layout());

		return *this;
	}
	// template<class... Indices> constexpr auto reindex(Indices... idxs) && -> array&& {
	//  this->layout_mutable() = this->layout_mutable().creindex(idxs...);
	//  return std::move(*this);
	// }
	// template<class... Indices> constexpr auto reindex(Indices... idxs) & -> array& {
	//  this->layout_mutable() = this->layout_mutable().creindex(idxs...);
	//  // this->layout_mutable().reindex(idxs...);
	//  return *this;
	// }

	// ~array() {
	//  assert(this->stride() != 0);
	// }
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __cpp_deduction_guides

#define BOOST_MULTI_IL std::initializer_list  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing, TODO(correaa) use template typedef instead of macro

// vvv MSVC 14.3 in c++17 mode needs paranthesis in dimensionality_type(d)
template<class T> dynamic_array(BOOST_MULTI_IL<T>) -> dynamic_array<T, static_cast<dimensionality_type>(1U), std::allocator<T>>;  // MSVC needs the allocator argument error C2955: 'boost::multi::dynamic_array': use of class template requires template argument list
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<T>>) -> dynamic_array<T, static_cast<dimensionality_type>(2U), std::allocator<T>>;
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>) -> dynamic_array<T, static_cast<dimensionality_type>(3U), std::allocator<T>>;
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>) -> dynamic_array<T, static_cast<dimensionality_type>(4U), std::allocator<T>>;
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>>) -> dynamic_array<T, static_cast<dimensionality_type>(5U), std::allocator<T>>;

// TODO(correaa) add zero dimensional case?
template<class T> array(BOOST_MULTI_IL<T>) -> array<T, static_cast<dimensionality_type>(1U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<T>>) -> array<T, static_cast<dimensionality_type>(2U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>) -> array<T, static_cast<dimensionality_type>(3U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>) -> array<T, static_cast<dimensionality_type>(4U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>>) -> array<T, static_cast<dimensionality_type>(5U)>;

#undef BOOST_MULTI_IL

template<class T> array(T[]) -> array<T, static_cast<dimensionality_type>(1U)>;  // NOSONAR(cpp:S5945) NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

//  vvv these are necessary to catch {n, m, ...} notation (or single integer notation)
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<0>, T) -> array<T, static_cast<dimensionality_type>(0U)>;  // TODO(correaa) use some std::allocator_traits instead of is_allocator
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<1>, T) -> array<T, static_cast<dimensionality_type>(1U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<2>, T) -> array<T, static_cast<dimensionality_type>(2U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<3>, T) -> array<T, static_cast<dimensionality_type>(3U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<4>, T) -> array<T, static_cast<dimensionality_type>(4U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<5>, T) -> array<T, static_cast<dimensionality_type>(5U)>;

// generalization, will not work with naked {n, m, ...} notation (or single integer notation)
template<dimensionality_type D, class T, class = std::enable_if_t<!boost::multi::is_allocator_v<T>>>
array(iextensions<D>, T) -> array<T, D>;

template<class MatrixRef, class DT = typename MatrixRef::decay_type, class T = typename DT::element_type, dimensionality_type D = DT::rank_v, class Alloc = typename DT::allocator_type>
array(MatrixRef) -> array<T, D, Alloc>;

template<class MatValues, class T = typename MatValues::element, dimensionality_type D = MatValues::rank_v>
array(MatValues) -> array<T, D>;

template<class MatValues, class T = typename MatValues::element, dimensionality_type D = MatValues::rank_v, class Alloc = std::allocator<T>, class = std::enable_if_t<multi::is_allocator_v<Alloc>>>  /// , class Alloc = typename DT::allocator_type>
array(MatValues, Alloc) -> array<T, D, Alloc>;

template<typename T, dimensionality_type D, typename P> array(subarray<T, D, P>) -> array<T, D>;

template<
	class Range, std::enable_if_t<!has_extensions<Range>::value, int> = 0,
	typename V = decltype(*::std::begin(std::declval<Range const&>()))
	// typename V = typename std::iterator_traits<decltype(::std::begin(std::declval<Range const&>()))>::value_type
	>
array(Range) -> array<V, 1>;

template<class Reference>
auto operator+(Reference&& ref) -> decltype(array(std::forward<Reference>(ref))) {
	return array(std::forward<Reference>(ref));
}

#endif  // ends defined(__cpp_deduction_guides)

template<class T, std::size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
auto decay(T const (&arr)[N]) noexcept -> multi::array<std::remove_all_extents_t<T[N]>, std::rank_v<T[N]>> {
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	return multi::array_cref<std::remove_all_extents_t<T[N]>, std::rank_v<T[N]>>(data_elements(arr), extensions(arr));
}

template<class T, std::size_t N>
struct array_traits<T[N], void, void> {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	using reference  = T&;
	using element    = std::remove_all_extents_t<T[N]>;  // NOSONAR(cpp:S5945) NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	using decay_type = multi::array<T, 1>;
};

}  // end namespace boost::multi

namespace boost::multi::pmr {

#ifdef BOOST_MULTI_HAS_MEMORY_RESOURCE
template<class T, boost::multi::dimensionality_type D>
using array = boost::multi::array<T, D, std::pmr::polymorphic_allocator<T>>;
#else
template<class T, boost::multi::dimensionality_type D>
struct [[deprecated("no PMR allocator")]] array;  // your version of C++ doesn't provide polymorphic_allocators
#endif

}  // end namespace boost::multi::pmr

// common_reference for compatibility with ranges
#if defined(__cpp_lib_common_reference) || defined(__cpp_lib_ranges)
// TODO(correaa) achieve this by normal inheritance
// NOLINTBEGIN(cert-dcl58-cpp)
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array     &&,          ::boost::multi::array<T, D, A...>                         &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array     &&,          ::boost::multi::array<T, D, A...>                    const&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                         &, typename ::boost::multi::array<T, D, A...>::basic_const_array     &&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                    const&, typename ::boost::multi::array<T, D, A...>::basic_const_array     &&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array       ,          ::boost::multi::array<T, D, A...>                         &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array       ; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                    const&, typename ::boost::multi::array<T, D, A...>::basic_const_array const&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array const&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array const&,          ::boost::multi::array<T, D, A...>                    const&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array const&; };

// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array      &,          ::boost::multi::array<T, D, A...>                         &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array      &; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                         &, typename ::boost::multi::array<T, D, A...>::basic_const_array      &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array      &; };
// NOLINTEND(cert-dcl58-cpp)
#endif

namespace boost::serialization {

template<typename T> struct version;  // in case serialization was not included before

template<typename T, boost::multi::dimensionality_type D, class A>
struct version<boost::multi::array<T, D, A>> {
	using type = std::integral_constant<int, BOOST_MULTI_SERIALIZATION_ARRAY_VERSION>;  // TODO(correaa) use constexpr variable here, not a macro
	// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class) for backward compatibility with Boost Serialization
	enum /*class value_t*/ { value = type::value };  // NOSONAR(cpp:S3642)  // https://community.sonarsource.com/t/suppress-issue-in-c-source-file/43154/24
};

}  // end namespace boost::serialization

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#undef BOOST_MULTI_HD

// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <type_traits>

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
	assert(((xs == arrs.extensions()) && ...));
	return apply_bind_t<F, std::decay_t<A>, std::decay_t<As>...>{std::forward<F>(fun), std::forward<A>(arr), std::forward<As>(arrs)...} ^ xs;
	//	return [fun = std::forward<F>(fun), &arr, &arrs...](auto... is) { return fun(arr[is...], arrs[is...]...); } ^ arr.extensions();
}

template<class F, class A, class B>
constexpr auto map(F&& fun, A&& alpha, B&& omega) {
	if constexpr(!multi::has_dimensionality<std::decay_t<A>>::value) {
		return map(std::forward<F>(fun), [alpha = std::forward<A>(alpha)]() { return alpha; } ^ multi::extensions_t<0>{}, std::forward<B>(omega));
	} else if constexpr(!multi::has_dimensionality<std::decay_t<B>>::value) {
		return map(std::forward<F>(fun), std::forward<A>(alpha), [omega = std::forward<B>(omega)]() { return omega; } ^ multi::extensions_t<0>{});
	} else {
		if constexpr(std::decay_t<A>::dimensionality < std::decay_t<B>::dimensionality) {
			return map(std::forward<F>(fun), alpha.repeated(omega.size()), omega);
		} else if constexpr(std::decay_t<B>::dimensionality < std::decay_t<A>::dimensionality) {
			return map(std::forward<F>(fun), alpha, omega.repeated(alpha.size()));
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
	template<class AA>                                                 // , std::enable_if_t<!std::is_base_v<exp_bind_t<A>, std::decay_t<AA> >, int> =0>
	explicit exp_bind_t(AA&& a) noexcept : a_{std::forward<AA>(a)} {}  // NOLINT(bugprone-forwarding-reference-overload)

	template<class... Is>
	constexpr auto operator()(Is... is) const {
		using ::std::exp;
		return exp(multi::detail::invoke_square(a_, is...));  // a_[is...] in C++23
	}
};

template<class A> exp_bind_t(A) -> exp_bind_t<A>;

template<class A>
constexpr auto exp(A&& alpha) {
	auto xs = alpha.extensions();
	// auto hm = alpha.home();
	// return exp_bind_t(hm) ^ xs;
	return exp_bind_t<A>(std::forward<A>(alpha)) ^ xs;
	// return exp_bind_t<typename bind_category<A>::type>{std::forward<A>(alpha)} ^ xs;
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

#endif  //  BOOST_MULTI_HPP
