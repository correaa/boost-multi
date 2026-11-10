// Copyright 2019-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ADAPTORS_LAPACK_GEQRF_HPP
#define BOOST_MULTI_ADAPTORS_LAPACK_GEQRF_HPP

#include "boost/multi/adaptors/blas/filling.hpp"
#include "boost/multi/utility.hpp"  // for size

#include <algorithm>  // for min
#include <cassert>
#include <memory>     // for allocator_traits
#include <stdexcept>  // for runtime_error
#include <string>     // for operator+, to_string, allocator
#include <utility>    // for forward

extern "C" {
using integer = int const&;
using integer_out = int&;

void dgeqrf_(  // NOLINT(readability-identifier-naming) externally linked
	integer,   // M,
	integer,   // N,
	double*,   // A  double precision, dimension( lda, * )   A,
	integer,   // LDA,
	double*,   // TAU,  // double precision, dimension( * )    TAU,
	double*,   // WORK,  // double precision, dimension( * )    WORK,
	integer,   // LWORK,
	integer_out  // INFO
);
}

// namespace boost{namespace multi{namespace lapack{
namespace boost::multi::lapack {

using blas::filling;

template<class Array2D, class TAU, class Allocator>
auto geqrf(Array2D&& aa, TAU& tau, Allocator alloc) -> Array2D&& {
//  assert( stride(~a) == 1);
	assert( size(tau) == std::min(size(~aa), size(aa)) );

	double dwork;  // NOLINT(cppcoreguidelines-init-variables) delayed initialization
	int    info;   // NOLINT(cppcoreguidelines-init-variables) delayed initialization
	dgeqrf_(
		static_cast<int>((~aa).size()), static_cast<int>(aa.size()), aa.base(), static_cast<int>(aa.stride()),
		tau.base(),
		&dwork, -1,
		info  // cppcheck-suppress uninitvar ;
	);
	if(info != 0) {  // cppcheck-suppress identicalConditionAfterEarlyExit ; TODO(correaa) investigate
		throw std::runtime_error("Error in DGEQRF work estimation, info: " + std::to_string(info));
	}

	auto const  lwork = static_cast<int>(dwork);
	auto* const work  = alloc.allocate(static_cast<typename std::allocator_traits<Allocator>::size_type>(lwork));

	dgeqrf_(
		static_cast<int>(size(~aa)), static_cast<int>(size(aa)), aa.base(), static_cast<int>(aa.stride()),
		tau.base(),
		work, lwork,
		info
	);
	alloc.deallocate(work, static_cast<typename std::allocator_traits<Allocator>::size_type>(lwork));

	if(info != 0) {  // cppcheck-suppress identicalConditionAfterEarlyExit ; TODO(correaa) investigate
		throw std::runtime_error("Error in DGESVD computation, info: " + std::to_string(info));
	}

	return std::forward<Array2D>(aa);
}

template<class Array2D, class TAU, class Allocator = std::allocator<double>>
auto geqrf(Array2D&& aa, TAU&& tau) -> Array2D&& {
	return geqrf(std::forward<Array2D>(aa), std::forward<TAU>(tau), Allocator{});
}

}  // end namespace boost::multi::lapack

#endif
