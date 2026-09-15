// Copyright 2019-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// #include "../../../adaptors/cuda.hpp"
#include "boost/multi/array.hpp"
#include "boost/multi/io.hpp"

#include "boost/multi/adaptors/blas/dot.hpp"

#include <boost/core/lightweight_test.hpp>

#include <cassert>
#include <complex>
#include <numeric>

namespace multi = boost::multi;
namespace blas  = multi::blas;

auto main() -> int {  // NOLINT(bugprone-exception-escape)
	// BOOST_AUTO_TEST_CASE(blas_conjugated_gpu)
	// {
	//		using complex = std::complex<double>;
	//		constexpr complex I{0.0, 1.0};

	// 	cuda::array<complex, 1> const acu = {1.0 +     I, 2.0 + 3.0*I, 3.0 + 2.0*I, 4.0 - 9.0*I};
	// 	cuda::array<complex, 1> const bcu = {5.0 + 2.0*I, 6.0 + 6.0*I, 7.0 + 2.0*I, 8.0 - 3.0*I};

	// 	{
	// 		cuda::array<complex, 0> ccu;
	// 		blas::dot(acu, bcu, ccu);
	// 		BOOST_TEST( ccu() == 19.0 - 27.0*I );
	// 	}
	// 	BOOST_TEST( blas::C(bcu)[1] == 2.0 - 3.0*I );
	// }

	{
		using complex = std::complex<double>;
		auto const I  = complex{0.0, 1.0};

		multi::array<complex, 2> B = {
			{1.0 - 3.0 * I, 6.0 + 2.0 * I},
			{8.0 + 2.0 * I, 2.0 + 4.0 * I},
			{2.0 - 1.0 * I, 1.0 + 1.0 * I}
		};

		std::cout << B << '\n';

		namespace blas = multi::blas;
		// multi::array<complex, 2> conjB = blas::H(B);

		std::cout << blas::H(B) << '\n';

		blas::H(B)[1][1] = 10.0 + 50.0 * I;

		std::cout << B << '\n';
	}

	return boost::report_errors();
}
