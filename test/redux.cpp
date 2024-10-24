// Copyright 2018-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>

#include <boost/multi/array.hpp>  // for array, implicit_cast, explicit_cast

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	std::int64_t const maxsize = 39062;  // 390625;
	std::int64_t const nmax = 1000;  // 10000;


	// auto pp = [] /*__host__ __device__*/ (long ix, long iy) -> double { return double(ix) * double(iy); };

	auto const nx = nmax;  // for(long nx = 1; nx <= nmax; nx *= 10)
	auto const ny = maxsize;  // for(long ny = 1; ny <= maxsize; ny *= 5) 

	multi::array<double, 2> K2D({nmax, maxsize});

	for(std::int64_t ix = 0; ix < nx; ix++) {
		for(std::int64_t iy = 0; iy < ny; iy++) {
			K2D[ix][iy] = static_cast<double>(ix)*static_cast<double>(iy);
		}
	}

	// for(long iy = 0; iy < sizey; iy++) {
	// 	for(long ix = 0; ix < sizex; ix++) {
	// 		accumulator[ix] += kernel(ix, iy);
	// 	}
	// }

	return boost::report_errors();
}
