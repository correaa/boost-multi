// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_IO_HPP
#define BOOST_MULTI_IO_HPP
// #pragma once

#include "boost/multi/utility.hpp"

#include <array>   // for std::array
#include <cctype>  // for std::isdigit
#include <iostream>
#include <utility>  // for std::index_sequence, std::make_index_sequence

// #if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
// #if __has_include(<format>)
// #include <boost/multi/io/format.hpp>
// #endif
// #endif

namespace boost::multi {

namespace detail {

template<class T, typename = decltype(std::declval<T&>().reextent(std::declval<typename T::extents_type>()))>
auto        has_reextent_aux(T const&) -> std::true_type;
inline auto has_reextent_aux(...) -> std::false_type;

template<class T> struct has_reextent : decltype(has_reextent_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg) match style of has_dimensionality/has_member_move in utility.hpp

template<class Array, std::enable_if_t<!has_dimensionality<Array>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
void print(std::ostream& os, Array const& arr, std::string_view /*open*/, std::string_view /*sep*/, std::string_view /*close*/, std::string_view /*tag*/, int /*indent*/) {
	os << arr;
}

template<class Array, std::enable_if_t<has_dimensionality<Array>::value && Array::dimensionality == 0, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
void print(std::ostream& os, Array const& arr, std::string_view /*open*/, std::string_view /*sep*/, std::string_view /*close*/, std::string_view /*tag*/, int /*indent*/) {
	assert(!arr.empty());
	os << static_cast<typename Array::element_cref>(arr);
}

template<class Array, std::enable_if_t<has_dimensionality<Array>::value && (Array::dimensionality > 0), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
void print(std::ostream& os, Array const& arr, std::string_view open, std::string_view sep, std::string_view close, std::string_view tab, int indent) {
	if(has_reextent<Array>::value) {
		os << arr.extents() << ' ';
	}
	for(auto count = 0; count != indent; ++count) {  // NOLINT(altera-unroll-loops)
		os << tab;
	}
	os << open[0];
	if constexpr(Array::dimensionality > 1) {
		os << '\n';
	}
	for(auto idx : arr.extent()) {  // NOLINT(altera-unroll-loops) TODO(correaa) use an algorithm
		multi::detail::print(os, arr[idx], open.size() == 1 ? open : open.substr(1), sep.size() == 1 ? sep : sep.substr(1), close.size() == 1 ? close : close.substr(1), tab.size() == 1 ? tab : tab.substr(1), indent + 1);
		if(idx != arr.extent().back()) {
			os << sep[0];
			if constexpr(Array::dimensionality > 1) {
				os << '\n';
			} else {
				os << ' ';
			}
		}
	}

	if constexpr(Array::dimensionality > 1) {
		os << sep[0] << ' ' << '\n';
		for(auto count = 0; count != indent; ++count) {  // NOLINT(altera-unroll-loops) TODO(correaa) use an algorithm
			os << tab;
		}
	}

	os << close[0];
}

template<class Array, std::enable_if_t<!has_dimensionality<Array>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
void parse(std::istream& is, Array& arr, std::string_view /*open*/, std::string_view /*sep*/, std::string_view /*close*/) {
	is >> arr;
}

template<class Array, std::enable_if_t<has_dimensionality<Array>::value && Array::dimensionality == 0, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
void parse(std::istream& is, Array& arr, std::string_view /*open*/, std::string_view /*sep*/, std::string_view /*close*/) {
	assert(!arr.empty());
	is >> static_cast<typename Array::element_ref>(arr);
}

// reads the format print() writes (same open/sep/close grammar), independently of print():
// if Array can reextent (an owning multi::array), read the extents header first and resize
// to match. if it can't (a subarray/array_ref view, or any recursive sub-level of an owning
// array, since only the top-level owning array has reextent), write directly into the
// existing shape instead. either way, a shape mismatch between the stream and the target
// surfaces as an ordinary stream failure (missing/extra separator or closing bracket, or a
// failed element extraction) rather than as a special-cased size check.
template<class Array, std::enable_if_t<has_dimensionality<Array>::value && (Array::dimensionality > 0), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
void parse(std::istream& is, Array& arr, std::string_view open, std::string_view sep, std::string_view close) {
	if constexpr(has_reextent<Array>::value) {
		typename Array::extents_type exts;
		is >> exts;
		if(!is) {
			return;
		}
		arr.reextent(exts);
	}

	is >> std::ws;
	if(is.peek() != open[0]) {
		is.setstate(std::ios::failbit);
		return;
	}
	is.get();

	auto const ext   = arr.extent();
	bool       first = true;
	for(auto idx : ext) {  // NOLINT(altera-unroll-loops) TODO(correaa) use an algorithm
		if(!first) {
			is >> std::ws;
			if(is.peek() != sep[0]) {
				is.setstate(std::ios::failbit);
				return;
			}
			is.get();
		}
		first      = false;
		auto&& sub = arr[idx];  // arr[idx] is a prvalue "reference" proxy; name it so it's an lvalue parse() can bind to
		multi::detail::parse(is, sub, open.size() == 1 ? open : open.substr(1), sep.size() == 1 ? sep : sep.substr(1), close.size() == 1 ? close : close.substr(1));
		if(!is) {
			return;
		}
	}

	is >> std::ws;
	if(is.peek() == sep[0]) {  // tolerate the trailing separator print() emits for dimensionality > 1
		is.get();
		is >> std::ws;
	}
	if(is.peek() != close[0]) {
		is.setstate(std::ios::failbit);
		return;
	}
	is.get();
}

}  // namespace detail

template<class Array, std::enable_if_t<has_dimensionality<Array>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
auto operator<<(std::ostream& os, Array const& arr) -> std::ostream& {
	multi::detail::print(os, arr, "{", ",", "}", "\t", 0);
	return os;
}

template<class Array, std::enable_if_t<has_dimensionality<Array>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
auto operator>>(std::istream& is, Array& arr) -> std::istream& {
	multi::detail::parse(is, arr, "{", ",", "}");
	return is;
}

template<typename Integer>
auto operator<<(std::ostream& os, extent_t<Integer> const& ext) -> std::ostream& {
	if(ext.empty()) {
		return os << "[)";
	}
	if(ext.front() != 0) {
		return os << "[" << ext.front() << ", " << ext.back() + 1 << ")";
	}
	return os << ext.size();
}

// parses the mirror image of the extent_t operator<< above.
// subtleties:
//  - a bare number `n` (no brackets) means the extent [0, n), matching the print side
//    omitting brackets exactly when front() == 0; it does NOT mean "a single index n".
//  - "[)" is the empty extent, but printing collapses any empty range to that spelling
//    regardless of its original `first`; parsing it back therefore can't recover a
//    nonzero original first_ and always yields the canonical empty extent [0, 0).
//  - "[a, b)" gives first_ = a, last_ = b directly (b is already exclusive, same as what
//    operator<< writes via ext.back() + 1), no off-by-one adjustment needed on read.
template<typename Integer>
auto operator>>(std::istream& is, extent_t<Integer>& ext) -> std::istream& {
	is >> std::ws;
	if(is.peek() == '[') {
		is.get();  // consume '['
		is >> std::ws;
		if(is.peek() == ')') {
			is.get();  // consume ')'
			ext = extent_t<Integer>{};
			return is;
		}
		Integer first{};
		Integer last{};
		is >> first >> std::ws;
		if(is.peek() == ',') {
			is.get();  // consume ','
		}
		is >> last >> std::ws;
		if(is.peek() == ')') {
			is.get();  // consume ')'
		}
		ext = extent_t<Integer>{first, last};
		return is;
	}
	Integer size{};
	is >> size;
	ext = extent_t<Integer>{size};  // bare number n -> [0, n)
	return is;
}

namespace detail {

template<class Exts, std::size_t... Is>
auto print_extents(std::ostream& os, Exts const& exts, std::string_view open, std::string_view sep, std::string_view close, std::index_sequence<Is...> /*seq*/) -> std::ostream& {
	os << open;
	((os << (Is == 0 ? "" : sep) << exts.template get<Is>()), ...);
	return os << close;
}

// separator between extents_t elements is " x "; rather than matching that exact token,
// treat any run of characters that isn't the start of a number or a nested '[' as separator.
// Keeps parsing robust to the exact spacing/spelling used.
inline void skip_extents_separator(std::istream& is) {
	is >> std::ws;
	while(is.good()) {
		auto const chr = is.peek();
		if(chr == std::char_traits<char>::eof()) {
			break;
		}
		if((std::isdigit(static_cast<unsigned char>(chr)) != 0) || chr == '-' || chr == '[') {
			break;
		}
		is.get();
	}
}

template<dimensionality_type D, std::size_t... Is>
auto make_extents(std::array<index_extension, sizeof...(Is)> const& exts, std::index_sequence<Is...> /*seq*/) -> extents_t<D> {
	return extents_t<D>{exts[Is]...};
}

template<dimensionality_type D>
auto parse_extents(std::istream& is) -> extents_t<D> {
	is >> std::ws;
	if(is.peek() == '[') {
		is.get();  // consume '['
	}

	std::array<index_extension, static_cast<std::size_t>(D)> exts{};
	for(std::size_t i = 0; i != static_cast<std::size_t>(D); ++i) {  // NOLINT(altera-unroll-loops)
		if(i != 0) {
			skip_extents_separator(is);
		}
		is >> exts[i];
	}

	is >> std::ws;
	if(is.peek() == ']') {
		is.get();  // consume ']'
	}

	return make_extents<D>(exts, std::make_index_sequence<static_cast<std::size_t>(D)>{});
}

}  // namespace detail

// e.g. print_extents(os, exts) -> "[2 x 2 x 3]" (default, same as operator<<)
//      print_extents(os, exts, "(", ", ", ")") -> "(2, 2, 3)" (Python tuple literal)
template<dimensionality_type D>
auto print_extents(std::ostream& os, extents_t<D> const& exts, std::string_view open = "[", std::string_view sep = " x ", std::string_view close = "]") -> std::ostream& {
	return multi::detail::print_extents(os, exts, open, sep, close, std::make_index_sequence<static_cast<std::size_t>(D)>{});
}

template<dimensionality_type D>
auto operator<<(std::ostream& os, extents_t<D> const& exts) -> std::ostream& {
	return print_extents(os, exts);
}

template<dimensionality_type D>
auto operator>>(std::istream& is, extents_t<D>& exts) -> std::istream& {
	exts = multi::detail::parse_extents<D>(is);
	return is;
}

}  // namespace boost::multi

#endif  // BOOST_MULTI_IO_HPP
