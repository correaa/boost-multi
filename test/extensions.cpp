// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <iostream>
#include <tuple>  // IWYU pragma: keep
// IWYU pragma: no_include <type_traits>                      // for add_const<>::type

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	auto       A2D  = multi::array<int, 2>({5, 7}, 1);
	auto const A2Dx = A2D.extension();

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

	// auto it2d = A2Dxs.elements().begin(); (void)it2d;

	multi::array<int, 1> const A1D({37}, 1);
	BOOST_TEST( A1D.size() == 37 );
	BOOST_TEST( A1D.num_elements() == 37 );
	BOOST_TEST( A1D.extensions().num_elements() == 37 );

	BOOST_TEST( A1D.extensions().elements().size() == A1D.extensions().num_elements() );
	{
		auto it = A1D.extensions().elements().begin();
		BOOST_TEST( get<0>(*it) == 0 );
		++it;
		BOOST_TEST( get<0>(*it) == 1 );

		it = A1D.extensions().elements().end();
		--it;
		BOOST_TEST( get<0>(*it) == 36 );
	}
	{
		auto x1d = multi::extensions_t<1>(3);

		auto it = x1d.elements().begin();
		BOOST_TEST( get<0>(*it) == 0 );

		++it;
		BOOST_TEST( get<0>(*it) == 1 );

		++it;
		BOOST_TEST( get<0>(*it) == 2 );

		++it;
		BOOST_TEST( it == x1d.elements().end() );

		--it;
		BOOST_TEST( get<0>(*it) == 2 );

		--it;
		BOOST_TEST( get<0>(*it) == 1 );

		--it;
		BOOST_TEST( get<0>(*it) == 0 );
		BOOST_TEST( it == x1d.elements().begin() );
	}
	{
		auto x1d = multi::extensions_t<1>(3);

		auto it = x1d.elements().begin();
		BOOST_TEST( get<0>(*it) == 0 );

		++it;
		BOOST_TEST( get<0>(*it) == 1 );

		it += 2;

		BOOST_TEST( it == x1d.elements().end() );

		it -= 3;
		BOOST_TEST( get<0>(*it) == 0 );
		BOOST_TEST( it == x1d.elements().begin() );

		BOOST_TEST( x1d.elements().end() - x1d.elements().begin() == 3 );
	}
	{
		multi::extensions_t<2> const x2d({4, 3});

		auto ll = [](auto x, auto y) { return x + y; };
		multi::f_extensions_t<2, decltype(ll)> x2df({4, 2}, ll);  (void)x2df;
		auto val = x2df[3][1];
		BOOST_TEST(val == 4);
		// multi::detail::what(x2df[1]);
		// std::cout << x2df[1][2] << std::endl;

		// auto x2d_trd = x2d.element_transformed([](auto is) { using std::get; return get<0>(is) + get<1>(is); });

		BOOST_TEST( x2d.elements().end() - x2d.elements().begin() == 12 );

		auto it = x2d.elements().begin();

		BOOST_TEST( it == x2d.elements().begin() );

		using std::get;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		++it;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		++it;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		BOOST_TEST( it - x2d.elements().begin() == 2 );

		++it;
		BOOST_TEST( 1 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		++it;
		BOOST_TEST( 1 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		++it;
		BOOST_TEST( 1 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		BOOST_TEST( it - x2d.elements().begin() ==  5 );
		BOOST_TEST( x2d.elements().begin() - it == -5 );
		BOOST_TEST( x2d.elements().end() - it == 7 );
		BOOST_TEST( x2d.elements().end() - x2d.elements().begin() == 12 );

		++it;
		BOOST_TEST( 2 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		++it;
		BOOST_TEST( 2 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		++it;
		BOOST_TEST( 2 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		++it;
		BOOST_TEST( 3 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		++it;
		BOOST_TEST( 3 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		++it;
		BOOST_TEST( 3 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		++it;
		BOOST_TEST( it ==  x2d.elements().end() );

		--it;
		BOOST_TEST( 3 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		--it;
		BOOST_TEST( 3 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		--it;
		BOOST_TEST( 3 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		--it;
		BOOST_TEST( 2 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		--it;
		BOOST_TEST( 2 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		--it;
		BOOST_TEST( 2 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		--it;
		BOOST_TEST( 1 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		--it;
		BOOST_TEST( 1 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		--it;
		BOOST_TEST( 1 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		--it;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 2 == get<1>(*it) );

		--it;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 1 == get<1>(*it) );

		--it;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		BOOST_TEST( it ==  x2d.elements().begin() );
	}
	{
		multi::extensions_t<2> const x2d({4, 3});

		auto it = x2d.elements().begin();

		BOOST_TEST( it == x2d.elements().begin() );

		using std::get;
		BOOST_TEST( 0 == get<0>(*it) );
		BOOST_TEST( 0 == get<1>(*it) );

		BOOST_TEST( 0 == get<0>(*(it + 2)) );
		BOOST_TEST( 2 == get<1>(*(it + 2)) );

		BOOST_TEST( 1 == get<0>(*(it + 5)) );
		BOOST_TEST( 2 == get<1>(*(it + 5)) );

		auto const it2  = it + 5;
		auto const it22 = it - (-5);
		BOOST_TEST( it2 == it22 );

		BOOST_TEST( 1 == get<0>(*(it2)) );
		BOOST_TEST( 2 == get<1>(*(it2)) );

		std::cout << "x y " << get<0>(*(it2 - 1)) << ' ' << get<1>(*(it2 - 1)) << '\n';

		BOOST_TEST( 1 == get<0>(*(it2-1)) );
		BOOST_TEST( 1 == get<1>(*(it2-1)) );

		std::cout << "x y " << get<0>(*(it2 - 2)) << ' ' << get<1>(*(it2 - 2)) << '\n';
		BOOST_TEST( 1 == get<0>(*(it2-2)) );
		BOOST_TEST( 0 == get<1>(*(it2-2)) );

		auto const it3  = it2 - 5;
		auto const it33 = it2 + (-5);
		BOOST_TEST( it3 == it33 );

		BOOST_TEST( it3 == it );

		// it -= 2;

		// std::cout << "x y " << get<0>(*it) << ' ' << get<1>(*it) << '\n';
		// BOOST_TEST( 0 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// it += 2;

		// BOOST_TEST( 0 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// it += 3;

		// BOOST_TEST( 1 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// it -= 3;

		// std::cout << get<0>(*it) << ' ' << get<1>(*it) << std::endl;

		// BOOST_TEST( 0 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 1 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 2 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 2 == get<0>(*it) );
		// BOOST_TEST( 1 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 2 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 3 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 3 == get<0>(*it) );
		// BOOST_TEST( 1 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( 3 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// ++it;
		// BOOST_TEST( it ==  x2d.elements().end() );

		// --it;
		// BOOST_TEST( 3 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 3 == get<0>(*it) );
		// BOOST_TEST( 1 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 3 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 2 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 2 == get<0>(*it) );
		// BOOST_TEST( 1 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 2 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 1 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 1 == get<0>(*it) );
		// BOOST_TEST( 1 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 1 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 0 == get<0>(*it) );
		// BOOST_TEST( 2 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 0 == get<0>(*it) );
		// BOOST_TEST( 1 == get<1>(*it) );

		// --it;
		// BOOST_TEST( 0 == get<0>(*it) );
		// BOOST_TEST( 0 == get<1>(*it) );

		// BOOST_TEST( it ==  x2d.elements().begin() );
	}

	return boost::report_errors();
}
