// Copyright 2019-2024 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_BLAS_SIDE_HPP
#define MULTI_ADAPTORS_BLAS_SIDE_HPP

namespace boost::multi::blas {

enum class side : char {
	left  = 'L',
	right = 'R'//,
//  pre_multiply = 'R',
//  post_multiply = 'L'
};

inline auto swap(side sid) -> side {
	switch(sid) {
		case side::left : return side::right;
		case side::right: return side::left ;
	} __builtin_unreachable();  // LCOV_EXCL_LINE
}

} // end namespace boost::multi::blas

#endif
