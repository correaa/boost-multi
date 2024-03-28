// Copyright 2019-2024 Alfredo A. Correa

#pragma once

namespace boost::multi::blas {

enum class side : char {
	left  = 'L',
	right = 'R'
};

inline auto swap(side sid) -> side {
	switch(sid) {
		case side::left : return side::right;
		case side::right: return side::left ;
	} __builtin_unreachable();  // LCOV_EXCL_LINE
}

} // end namespace boost::multi::blas
