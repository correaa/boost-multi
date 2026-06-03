// Copyright 2025-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ADAPTORS_THRUST_REDUCE_BY_INDEX_HPP
#define BOOST_MULTI_ADAPTORS_THRUST_REDUCE_BY_INDEX_HPP
#include <type_traits>
#pragma once

#include <thrust/functional.h>
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

template<class ExecutionPolicy, class T, class SIt>
auto reduce_by_index(ExecutionPolicy&& ep, T const& M, SIt sums_first) 
-> std::enable_if_t<
	std::is_base_of_v<std::input_iterator_tag, typename std::iterator_traits<SIt>::iterator_category>,
	SIt
> {
	auto const row_index = [] __host__ __device__ (typename T::index i, typename T::index j) -> typename T::index { return i; } ^ M.extents();

	// // row-index keys via a named functor (an extended __host__ __device__ lambda here
	// // trips nvcc's closure-placeholder substitution inside a function template)
	// auto const row_ids_begin =
	//     ::thrust::make_transform_iterator(
	// 		::thrust::make_counting_iterator(std::ptrdiff_t{0}),
	// 		detail::divide_by<decltype(M.num_elements())>{M.num_elements()/M.size()}
	//     )
	// ;
	// auto const row_ids_end = row_ids_begin + M.num_elements();

	auto const row_ids_begin = row_index.elements().begin();
	auto const row_ids_end   = row_index.elements().end();

	return ::thrust::reduce_by_key(
        std::forward<ExecutionPolicy>(ep),
		row_ids_begin, row_ids_end,
		M.elements().begin(),
		::thrust::make_discard_iterator(),
		sums_first
	).second;
}

template<class ExecutionPolicy, class T, class S>
auto reduce_by_index(ExecutionPolicy&& ep, T const& M, S&& sums)
-> std::enable_if_t<sizeof(std::declval<S&&>().extension()) != 0, S&&> {
	assert(M.extension() == sums.extension());

	auto sums_end = reduce_by_index(std::forward<ExecutionPolicy>(ep), M, sums.begin());
	assert(sums_end == sums.end());

	return std::forward<S>(sums);
}

template<class ExecutionPolicy, class T, class S, class BinaryOp>
auto reduce_by_index(ExecutionPolicy&& ep, T const& M, S&& sums, BinaryOp&& op) -> S&& {
	assert(M.extension() == sums.extension());

	auto const row_ids_begin =
	    ::thrust::make_transform_iterator(
			::thrust::make_counting_iterator(std::ptrdiff_t{0}),
			detail::divide_by<decltype(M.num_elements())>{M.num_elements()/M.size()}
	    )
	;
	auto const row_ids_end = row_ids_begin + M.num_elements();

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
		sums.begin(),
		::thrust::equal_to<typename T::index>{},
		std::forward<BinaryOp>(op)
	);

	return std::forward<S>(sums);
}

template<class T, class S>
auto reduce_by_index(T const& M, S&& sums) -> S&& {
    return reduce_by_index(::thrust::cuda::par, M, std::forward<S>(sums));
}

template<class T, class S, class BinOp>
auto reduce_by_index(T const& M, S&& sums, BinOp&& op) -> S&& {
    return reduce_by_index(::thrust::cuda::par, M, std::forward<S>(sums), std::forward<BinOp>(op));
}

template<class T>
auto reduce_by_index(T const& M) {
    multi::array<typename T::element, T::dimensionality - 1, typename T::allocator_type> ret(M[0].extents(), M.get_allocator());
    return reduce_by_index(M, std::move(ret));
}

template<
	class T, class BinOp, class TE = typename T::element,
	std::enable_if_t<! multi::has_extents<std::decay_t<BinOp>>::value> =0,
	class = decltype(std::declval<BinOp>()(std::declval<TE>(), std::declval<TE>()))
>
auto reduce_by_index(T const& M, BinOp&& op) {
    multi::array<TE, T::dimensionality - 1, typename T::allocator_type> ret(M.layout().sub().extents(), M.get_allocator());
    return reduce_by_index(M, std::move(ret), std::forward<BinOp>(op));
}

}

#endif  // BOOST_MULTI_ADAPTORS_THRUST_REDUCE_BY_INDEX_HPP_
