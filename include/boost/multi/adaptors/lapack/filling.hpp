// Copyright 2024 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_LAPACK_FILLING_HPP
#define MULTI_ADAPTORS_LAPACK_FILLING_HPP
#pragma once

// TODO(correaa)  #include "multi/blas/filling.hpp"

namespace boost::multi::lapack {

enum class filling : char {
	lower = 'U',
	upper = 'L',
};

inline auto flip(filling side) -> filling {
	switch(side) {
	case filling::lower: return filling::upper;
	case filling::upper: return filling::lower;
	}
	__builtin_unreachable();  // LCOV_EXCL_LINE
}

inline auto operator-(filling side) -> filling { return flip(side); }
inline auto operator+(filling side) -> filling { return side; }

}  // namespace boost::multi::lapack

#endif
