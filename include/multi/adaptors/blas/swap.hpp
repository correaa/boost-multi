// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2019-2023 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_BLAS_SWAP_HPP
#define MULTI_ADAPTORS_BLAS_SWAP_HPP
#pragma once

#include "../blas/core.hpp"

namespace boost::multi::blas {

template<class It1, class It2>
auto swap(It1 first, It2 last, It2 first2) -> It2 {
	assert(stride(first) == stride(last));
	using std::distance;
	auto const dist = distance(first, last);
	swap(dist, base(first), stride(first), base(first2), stride(first2));
	return first2 + dist;
}

template<class X1D, class Y1D>
auto swap(X1D&& x, Y1D&& y) -> Y1D&& {  // NOLINT(readability-identifier-length) x, y conventional blas names
	assert( size(x) == size(y) );
	assert( offset(x) == 0 and offset(y) == 0 );
	swap( begin(x), end(x), begin(y) );
	return std::forward<Y1D>(y);
}

} // end namespace boost::multi::blas
#endif
