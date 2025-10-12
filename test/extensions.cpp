// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/detail/extensions.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <algorithm>  // IWYU pragma: keep  // for std::equal
#include <tuple>      // IWYU pragma: keep
// IWYU pragma: no_include <type_traits>    // for add_const<>::type
// IWYU pragma: no_include <variant>        // for get, iwyu bug

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

		auto ll = [](auto xx, auto yy) {
			return xx + yy;
		};
		multi::f_extensions_t<2, decltype(ll)> const x2df({4, 2}, ll);
		(void)x2df;
		auto val = x2df[3][1];
		BOOST_TEST(val == 4);

		auto elems = x2df.elements();
		BOOST_TEST( elems[7] == 4 );
		BOOST_TEST( *(x2df.elements().begin() + 1) == 1 + 0 );

		// BOOST_TEST( *(*(x2df.begin()).begin()) == 0 )

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

		// auto it2d = x2d.begin();

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

		BOOST_TEST( 1 == get<0>(*(it2-1)) );
		BOOST_TEST( 1 == get<1>(*(it2-1)) );

		BOOST_TEST( 1 == get<0>(*(it2-2)) );
		BOOST_TEST( 0 == get<1>(*(it2-2)) );

		auto const it3  = it2 - 5;
		auto const it33 = it2 + (-5);
		BOOST_TEST( it3 == it33 );

		BOOST_TEST( it3 == it );
	}
	{
		multi::array<int, 1> const arr(10);

		auto xn = decltype(arr.extension())(10);
		BOOST_TEST( xn. size() == 10 );

		multi::extension_t const xn2(10);
		BOOST_TEST( xn2.size() == 10 );

		xn = xn2;

		multi::detail::extensions const xns2{xn2};
		using std::get;
		BOOST_TEST( get<0>(xns2) == xn2 );

		multi::detail::extensions const xns2d{xn2, xn2};
		auto [xns2d_a, xns2d_b] = xns2d;

		BOOST_TEST( xns2d_a == xn2 );
		BOOST_TEST( xns2d_b == xn2 );

		multi::extensions_t<2> const met2{xns2d};

		multi::layout_t<2> const lyt(met2);
		multi::layout_t<2> const lyt_2(xns2d);

		BOOST_TEST( lyt == lyt_2 );

		// multi::array<int, 1> const arr2({xn2});
	}
	{
		auto const x2df = [](auto x, auto y) { return x + y; } ^ multi::extensions_t<2>(3, 4);

		// boost::multi::f_extensions_t<2, decltype(ll)> x2df(multi::extensions_t<2>(3, 4), ll);
		BOOST_TEST( x2df.elements()[0] == 0 );
		BOOST_TEST( x2df.elements()[1] == 1 );
		BOOST_TEST( x2df.elements()[2] == 2 );
		BOOST_TEST( x2df.elements()[3] == 3 );
		BOOST_TEST( x2df.elements()[4] == 1 );
		BOOST_TEST( x2df.elements()[5] == 2 );

		BOOST_TEST( x2df[2][1] == 2 + 1 );

		multi::array<multi::index, 2> const arr2df = [](auto x, auto y) { return x + y; } ^ multi::extensions_t<2>(3, 4);

		BOOST_TEST( arr2df(2, 1) == 2 + 1 );
		BOOST_TEST( arr2df[2][1] == 2 + 1 );

		BOOST_TEST(std::equal(
			arr2df.elements().begin(), arr2df.elements().end(),
			([](auto x, auto y) { return x + y; } ^ multi::extensions_t<2>(3, 4)).elements().begin()
		));

		BOOST_TEST(std::equal(
			arr2df.elements().begin(), arr2df.elements().end(),
			(multi::extensions_t<2>(3, 4)->*[](auto x, auto y) { return x + y; }).elements().begin()
		));

		BOOST_TEST(   arr2df.elements().begin() != arr2df.elements().end()  );
		BOOST_TEST( !(arr2df.elements().begin() == arr2df.elements().end()) );

		BOOST_TEST( arr2df[2][1] == ([](auto x, auto y) { return x + y; } ^ multi::extensions_t<2>(3, 4))[2][1] );

		BOOST_TEST( arr2df[2][1] == ([](auto x, auto y) { return x + y; } ^ multi::extensions_t(3, 4))[2][1] );
		BOOST_TEST(
			arr2df[2][1]
			== multi::extensions_t<2>(3, 4).element_transformed( [](auto const& idxs) { using std::get; return get<0>(idxs) + get<1>(idxs); })[2][1]
		);
		BOOST_TEST(
			arr2df[2][1]
			== multi::extensions_t<2>(3, 4).element_transformed( [](auto idxs) {auto [xx, yy] = idxs; return xx + yy; })[2][1]
		);
	}
	{
		multi::extensions_t<3> const xs{3, 4, 5};

		BOOST_TEST( xs.sub() == multi::extensions_t<2>(4, 5) );
	}
	{
		multi::array<int, 2> const arr({3, 4});

		auto const& xs = arr.extensions();

		using std::get;
		BOOST_TEST( get<0>(xs[0][0]) == 0 );
		BOOST_TEST( get<1>(xs[0][0]) == 0 );

		BOOST_TEST(   xs.begin() != xs.end()  );
		BOOST_TEST( !(xs.begin() == xs.end()) );

		BOOST_TEST( xs[0] == xs[0] );
		BOOST_TEST( xs[0] != xs[1] );

		BOOST_TEST( xs[0] == *xs.begin() );
		BOOST_TEST( xs[1] == *(xs.begin() + 1) );

		auto it = xs.begin();
		++it;
		BOOST_TEST( *it == xs[1] );

		auto const& values = [](auto ii, auto jj) { return ii + jj; } ^ arr.extensions();

		BOOST_TEST( values.extensions() == arr.extensions() );
		BOOST_TEST( *values.elements().begin() == 0 );
		BOOST_TEST( values.elements().begin() != values.elements().end() );
		BOOST_TEST( values[0][0] == 0 );
		BOOST_TEST( values.begin() != values.end() );

		// auto arr2 = multi::array<boost::multi::index, 2>(arr.extensions());

		// arr2.elements() = values.elements();

		// arr2() = values;

		// arr2 = values;
	}

	return boost::report_errors();
}
