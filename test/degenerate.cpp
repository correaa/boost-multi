// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, dynamic_array, num_elements

#include <boost/core/lightweight_test.hpp>

// IWYU pragma: no_include <utility>  // for forward, declval, move

namespace multi = boost::multi;

auto main() -> int {
	{
		multi::array<int, 2> const A({0, 0}, 0);
		BOOST_TEST( A.size() == 0 );

		multi::array<int, 2> const B({6, 0}, 0);
		BOOST_TEST( B.size() == 0 );

		BOOST_TEST( A == B );
	}
	{
		multi::extents_t<2> const e1(6, 0);
		BOOST_TEST( e1.size() == 6 );

		multi::extents_t<2> const e2(0, 0);
		BOOST_TEST( e2.size() == 0 );

		BOOST_TEST( e1 != e2 );

		multi::layout_t<2> const l1(e1);
		multi::layout_t<2> const l2(e2);

		BOOST_TEST( l1 == l2 );

		multi::extents_t<2> const a1 = l1.extents();
		multi::extents_t<2> const a2 = l2.extents();

		BOOST_TEST( a1.size() == 0 );
		BOOST_TEST( a2.size() == 0 );
	}
	{
		using std::get;

		multi::extents_t<2> const e1(0, 6);
		BOOST_TEST( e1.size() == 0 );
		BOOST_TEST( get<1>(e1).size() == 6 );

		multi::extents_t<2> const e2(0, 0);
		BOOST_TEST( e2.size() == 0 );
		BOOST_TEST( get<1>(e2).size() == 0 );

		BOOST_TEST( e1 != e2 );

		multi::layout_t<2> const l1(e1);
		multi::layout_t<2> const l2(e2);

		// BOOST_TEST( l1 == l2 );

		multi::extents_t<2> const a1 = l1.extents();
		multi::extents_t<2> const a2 = l2.extents();

		BOOST_TEST( a1.size() == 0 );
		BOOST_TEST( get<1>(a1).size() == 6 );
		BOOST_TEST( a2.size() == 0 );
		BOOST_TEST( get<1>(a2).size() == 0 );
	}

	return boost::report_errors();
}
