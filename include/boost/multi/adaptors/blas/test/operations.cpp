// Copyright 2019-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include "boost/multi/adaptors/blas/operations.hpp"  // for H, H_t

#include "boost/multi/array.hpp"

#include <boost/core/lightweight_test.hpp>

#include <cmath>
#include <complex>

namespace multi = boost::multi;
namespace blas  = multi::blas;

auto main() -> int {  // NOLINT(bugprone-exception-escape)
	{
		using complex = std::complex<double>;
		auto const I  = complex{0.0, 1.0};  // NOLINT(readability-identifier-length)

		multi::array<complex, 2> B = {  // NOLINT(readability-identifier-length)
			{1.0 - 3.0 * I, 6.0 + 2.0 * I},
			{8.0 + 2.0 * I, 2.0 + 4.0 * I},
			{2.0 - 1.0 * I, 1.0 + 1.0 * I},
		};

		blas::H(B)[1][1] = 10.0 + 50.0 * I;

		BOOST_TEST( std::abs( B[1][1].real() -  10.0 ) < 1e-12 );
		BOOST_TEST( std::abs( B[1][1].imag() - -50.0 ) < 1e-12 );
	}

	return boost::report_errors();
}
