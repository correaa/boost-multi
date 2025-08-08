// Copyright 2021-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <tuple>  // IWYU pragma: keep
// IWYU pragma: no_include <type_traits>                      // for add_const<>::type

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	multi::array<int, 2> A2D({5, 7}, 1);
	auto const           A2Dx = A2D.extension();

	BOOST_TEST( &A2D() == &A2D(A2Dx) );

	auto const A2Dxs = A2D.extensions();

	using std::get;

	BOOST_TEST( get<0>(A2Dxs[1][2]) == 1 );
	BOOST_TEST( get<1>(A2Dxs[1][2]) == 2 );

	BOOST_TEST( get<0>(A2Dxs) == A2Dx );
	BOOST_TEST( get<1>(A2Dxs) == A2D[0].extension() );

	BOOST_TEST( &A2D() == &A2D(get<0>(A2D.extensions()), get<1>(A2D.extensions())) );
	BOOST_TEST( &A2D() == &std::apply(A2D, A2Dxs) );

	BOOST_TEST( A2Dxs.size() == A2D.size() );
	BOOST_TEST( A2Dxs.sizes() == A2D.sizes() );

	auto const [ni, nj] = A2Dxs.sizes();
	for(int i = 0; i != ni; ++i) {      // NOLINT(altera-unroll-loops)
		for(int j = 0; j != nj; ++j) {  // NOLINT(altera-unroll-loops)
			auto const [first, second] = A2Dxs[i][j];
			BOOST_TEST( first == i );
			BOOST_TEST( second == j );
		}
	}

	auto const [is, js] = A2Dxs;
	for(auto const i : is) {      // NOLINT(altera-unroll-loops)
		for(auto const j : js) {  // NOLINT(altera-unroll-loops)
			auto const [first, second] = A2Dxs[i][j];
			BOOST_TEST( first == i );
			BOOST_TEST( second == j );
		}
	}

	BOOST_TEST( get<0>(A2Dxs).size() == 5 );

	multi::array<int, 1> const A1D({37}, 1);
	BOOST_TEST( A1D.size() == 37 );
	BOOST_TEST( A1D.num_elements() == 37 );
	BOOST_TEST( A1D.extensions().num_elements() == 37 );

	BOOST_TEST( A1D.extensions().elements().size() == A1D.extensions().num_elements() );

	return boost::report_errors();
}
