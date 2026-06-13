// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// `ordering` computes the permutation of indices that would sort a (multi)dimensional
// range *without moving the data* (a.k.a. argsort).  The result is written into a
// caller-provided output range (no internal allocation; the sort is in-place over the
// indices via `std::sort`).  Because it orders by index, it handles non-zero-based
// arrays naturally and never copies/moves the (possibly proxy-referenced) elements.

// #pragma once
#ifndef BOOST_MULTI_ALGORITHMS_ORDERING_HPP
#define BOOST_MULTI_ALGORITHMS_ORDERING_HPP

#include <algorithm>   // for std::sort, std::copy
#include <functional>  // for std::less

namespace boost::multi {

// Writes into [first, ...) the permutation of `arr`'s indices such that
// `arr[result[0]], arr[result[1]], ...` is non-decreasing according to `comp`.
// `first` must point to a mutable random-access range of at least `arr.size()` elements.
// `arr` is not modified.  Returns the end of the written range.
template<class Array1D, class RandomAccessIt, class Compare>
auto ordering(Array1D const& arr, RandomAccessIt first, Compare comp) -> RandomAccessIt {
	auto const           ext  = arr.extension();
	RandomAccessIt const last = std::copy(ext.begin(), ext.end(), first);  // seed output with the index values of `arr`

	std::sort(
		first, last,
		[&arr, comp](auto idx1, auto idx2) { return comp(arr[idx1], arr[idx2]); }
	);

	return last;
}

template<class Array1D, class RandomAccessIt>
auto ordering(Array1D const& arr, RandomAccessIt first) -> RandomAccessIt {
	return ordering(arr, first, std::less<>{});
}

}  // end namespace boost::multi

#endif  // BOOST_MULTI_ALGORITHMS_ORDERING_HPP
