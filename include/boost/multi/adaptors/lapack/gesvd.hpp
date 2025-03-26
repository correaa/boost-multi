// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ADAPTORS_LAPACK_GESVD_HPP
#define BOOST_MULTI_ADAPTORS_LAPACK_GESVD_HPP

#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <string>

extern "C" {
void dgesvd_(char const& jobu, char const& jobvt, int const& mm, int const& nn, double* aa, int const& lda, double* ss, double* uu, int const& ldu, double* vt, int const& ldvt, double* work, int const& lwork, int& info);  // NOLINT(readability-identifier-naming)
}

namespace boost::multi::lapack {

template<class Alloc, class AArray2D, class UArray2D, class SArray1D, class VTArray2D>
void gesvd(AArray2D&& AA, UArray2D&& UU, SArray1D&& ss, VTArray2D&& VV, Alloc alloc) {
	assert( AA.size() == UU.size() );
	assert( (~AA).size() == VV.size() );
	assert( ss.size() == std::min(UU.size(), VV.size()) );

	assert((~AA).stride() == 1);
	assert(ss.stride() == 1);
	assert((~UU).stride() == 1);
	assert((~VV).stride() == 1);

	int    info;   // NOLINT(cppcoreguidelines-init-variables) init by function
	double dwork;  // NOLINT(cppcoreguidelines-init-variables) init by function

	dgesvd_(
		'A' /*all left*/, 'A' /*all right*/, 
		static_cast<int>(VV.size()), static_cast<int>(UU.size()),
		AA.base(), static_cast<int>(AA.stride()),
		ss.base(),
		VV.base(), static_cast<int>(VV.stride()),
		UU.base(), static_cast<int>(UU.stride()),
		&dwork, -1, info
	);
	if(info != 0) { throw std::runtime_error("Error in DGESVD work estimation, info: " + std::to_string(info)); }

	int const     lwork = static_cast<int>(dwork);
	double* const work  = alloc.allocate(lwork);

	dgesvd_(
		'A' /*all left*/, 'A' /*all right*/,
		static_cast<int>(VV.size()), static_cast<int>(UU.size()),
		AA.base(), static_cast<int>(AA.stride()),
		ss.base(),
		VV.base(), static_cast<int>(VV.stride()),
		UU.base(), static_cast<int>(UU.stride()),
		work, lwork, info
	);
	alloc.deallocate(work, lwork);

	if(info != 0) { throw std::runtime_error("Error in DGESVD computation, info: " + std::to_string(info)); }

	(void)std::forward<AArray2D>(AA);
}

template<
	template<typename> class AllocT = std::allocator,
	class AArray2D, class UArray2D, class SArray1D, class VTArray2D,
	class Alloc = AllocT<typename std::decay_t<AArray2D>::element_type>
>
void gesvd(AArray2D&& AA, UArray2D&& UU, SArray1D&& ss, VTArray2D&& VV) {
	return gesvd(std::forward<AArray2D>(AA), std::forward<UArray2D>(UU), std::forward<SArray1D>(ss), std::forward<VTArray2D>(VV), Alloc{});
}

}  // end namespace boost::multi::lapack

#endif  // BOOST_MULTI_ADAPTORS_LAPACK_GESVD_HPP
