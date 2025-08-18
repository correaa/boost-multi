// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ADAPTORS_THRUST_REDUCE_BY_INDEX_HPP_
#define BOOST_MULTI_ADAPTORS_THRUST_REDUCE_BY_INDEX_HPP_
#pragma once

#include <thrust/reduce.h>
#include <thrust/iterator/discard_iterator.h>

namespace boost::multi::thrust{
    
namespace detail {
template<class SizeType>
struct divide_by {
	SizeType divsr;
	__host__ __device__ constexpr auto operator()(SizeType divdn) const -> SizeType { return divdn/divsr; }
};
}

template<class ExecutionPolicy, class T, class S>
auto reduce_by_index(ExecutionPolicy&& ep, T const& M, S&& sums) -> S&& {
	assert(M.extension() == sums.extension());

	auto const row_ids_begin =
	    ::thrust::make_transform_iterator(
			::thrust::make_counting_iterator(std::ptrdiff_t{0}),
			detail::divide_by<decltype(M.elements().size())>{M.elements().size()/M.size()}
	    )
	;
	auto const row_ids_end = row_ids_begin + M.elements().size();

	// auto const row_ids_begin =
	//     thrust::make_transform_iterator(
	// 		M.extensions().elements().begin(),
	//         [] __host__ __device__ (typename T::indexes e) -> std::ptrdiff_t { using std::get; return get<0>(e); }
	//     )
	// ;
	// auto const row_ids_end = row_ids_begin + M.num_elements();

	::thrust::reduce_by_key(
        std::forward<ExecutionPolicy>(ep),
		row_ids_begin, row_ids_end,
		M.elements().begin(),
		::thrust::make_discard_iterator(),
		sums.begin()
	);

	return std::forward<S>(sums);
}

template<class T, class S>
auto reduce_by_index(T const& M, S&& sums) -> S&& {
    return reduce_by_index(::thrust::cuda::par, M, std::forward<S>(sums));
}

template<class T>
auto reduce_by_index(T const& M) {
    multi::array<typename T::element, T::dimensionality - 1, typename T::allocator_type> ret(M[0].extensions(), M.get_allocator());
    return reduce_by_index(M, std::move(ret));
}

}

#endif  // BOOST_MULTI_ADAPTORS_THRUST_REDUCE_BY_INDEX_HPP_
