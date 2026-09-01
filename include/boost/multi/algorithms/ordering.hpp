// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// `ordering` computes the permutation of indices that would sort a (multi)dimensional
// range *without moving the data* (a.k.a. argsort).  The result is written into a
// caller-provided output range (no internal allocation).  Because it orders by index,
// it handles non-zero-based arrays naturally and never copies/moves the (possibly
// proxy-referenced) elements.
//
// The sort backend is not hard-coded to `std::sort`: it dispatches to an ADL-found
// `sort` (e.g. `thrust::sort` when the output iterator is a thrust iterator) and falls
// back to `std::sort` otherwise.  Overloads taking an execution policy as the first
// argument forward it (e.g. `std::execution::par`, or a thrust execution policy).
// NOTE: because dispatch is by ADL, `boost::multi` must not declare an iterator-triple
// `sort(first, last, comp)`, which would otherwise be selected for Multi iterators.

// #pragma once
#ifndef BOOST_MULTI_ALGORITHMS_ORDERING_HPP
#define BOOST_MULTI_ALGORITHMS_ORDERING_HPP

#include <algorithm>    // for std::sort, std::copy
#include <functional>   // for std::less
#include <type_traits>  // for std::enable_if_t, std::void_t
#include <utility>      // for std::forward, std::declval

namespace boost::multi {

namespace detail {
// detects a Multi array/subarray (has `.extension()`), used to tell an array argument
// apart from an execution-policy argument in the policy overloads below.
template<class T, class = void> struct has_extension : std::false_type {};
template<class T>
struct has_extension<T, std::void_t<decltype(std::declval<T const&>().extension())>> : std::true_type {};

// overload-priority tag: priority_tag<1> is preferred over priority_tag<0>.
template<int N> struct priority_tag : priority_tag<N - 1> {};
template<> struct priority_tag<0> {};

// Prefer an ADL-found `sort` (e.g. `thrust::sort` for thrust iterators) and fall back to
// `std::sort`.  Crucially, `std::sort` is NOT brought into scope in the ADL overload, so a
// same-signature `thrust::sort` wins unambiguously instead of tying with `std::sort`.
template<class It, class Compare>
auto sort_dispatch(priority_tag<1> /*prefer ADL*/, It first, It last, Compare comp)
	-> decltype(sort(first, last, comp)) {
	return sort(first, last, comp);  // ADL only (no `using std::sort`)
}
template<class It, class Compare>
auto sort_dispatch(priority_tag<0> /*fallback*/, It first, It last, Compare comp) -> void {
	std::sort(first, last, comp);
}

template<class Policy, class It, class Compare>
auto sort_dispatch(priority_tag<1> /*prefer ADL*/, Policy&& policy, It first, It last, Compare comp)
	-> decltype(sort(std::forward<Policy>(policy), first, last, comp)) {
	return sort(std::forward<Policy>(policy), first, last, comp);  // ADL only (e.g. thrust policies)
}
template<class Policy, class It, class Compare>
auto sort_dispatch(priority_tag<0> /*fallback*/, Policy&& policy, It first, It last, Compare comp) -> void {
	std::sort(std::forward<Policy>(policy), first, last, comp);  // std execution policies
}
}  // namespace detail

// Writes into [first, ...) the permutation of `arr`'s indices such that
// `arr[result[0]], arr[result[1]], ...` is non-decreasing according to `comp`.
// `first` must point to a mutable random-access range of at least `arr.size()` elements.
// `arr` is not modified.  Returns the end of the written range.
template<class Array1D, class RandomAccessIt, class Compare, std::enable_if_t<detail::has_extension<Array1D>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++17
auto ordering(Array1D const& arr, RandomAccessIt first, Compare comp) -> RandomAccessIt {
	auto const last = std::copy(arr.extension().begin(), arr.extension().end(), first);  // seed output with the index values of `arr`

	detail::sort_dispatch(
		detail::priority_tag<1>{}, first, last,
		[&arr, comp](auto idx1, auto idx2) { return comp(arr[idx1], arr[idx2]); }
	);

	return last;
}

template<class Array1D, class RandomAccessIt>
auto ordering(Array1D const& arr, RandomAccessIt first) -> RandomAccessIt {
	return ordering(arr, first, std::less<>{});
}

// Policy-aware overloads: `policy` (e.g. `std::execution::par`, or a thrust policy) is
// forwarded to the sort.  Disambiguated from the overloads above by requiring the second
// argument to be a Multi array (an execution policy is not).
template<class Policy, class Array1D, class RandomAccessIt, class Compare, std::enable_if_t<detail::has_extension<Array1D>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++17
auto ordering(Policy&& policy, Array1D const& arr, RandomAccessIt first, Compare comp) -> RandomAccessIt {
	auto const last = std::copy(arr.extension().begin(), arr.extension().end(), first);

	detail::sort_dispatch(
		detail::priority_tag<1>{}, std::forward<Policy>(policy), first, last,
		[&arr, comp](auto idx1, auto idx2) { return comp(arr[idx1], arr[idx2]); }
	);

	return last;
}

template<class Policy, class Array1D, class RandomAccessIt, std::enable_if_t<detail::has_extension<Array1D>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++17
auto ordering(Policy&& policy, Array1D const& arr, RandomAccessIt first) -> RandomAccessIt {
	return ordering(std::forward<Policy>(policy), arr, first, std::less<>{});
}

// To *apply* the resulting `order` (or any index permutation) there is no need for a
// dedicated algorithm here:
//   - out of place (gather):  dst[k] = src[order[k]], i.e.
//       std::transform(order.begin(), order.end(), dst.begin(), [&src](auto idx) { return src[idx]; });
//     (or, lazily, a view:  order | std::views::transform([&src](auto idx) { return src[idx]; }))
//   - in place:  boost::algorithm::apply_permutation(src.begin(), src.end(), order.begin(), order.end());
// Both work for N-dimensional `src`, where `src[idx]` is a whole slice.

}  // end namespace boost::multi

#endif  // BOOST_MULTI_ALGORITHMS_ORDERING_HPP
