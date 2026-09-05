// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, dynamic_array, num_elements

#include <boost/core/lightweight_test.hpp>

// IWYU pragma: no_include <utility>  // for forward, declval, move

namespace multi = boost::multi;

auto main() -> int {
	{
		multi::array<int, 2> A({0, 0}, 0);
		BOOST_TEST( A.size() == 0 );

		multi::array<int, 2> B({6, 0}, 0);
		BOOST_TEST( B.size() == 0 );

		BOOST_TEST( A == B );
	}
	{
		multi::extents_t<2> e1(6, 0);
		BOOST_TEST( e1.size() == 6 );

		multi::extents_t<2> e2(0, 0);
		BOOST_TEST( e2.size() == 0 );

		BOOST_TEST( e1 != e2 );

		multi::layout_t<2> l1(e1);
		multi::layout_t<2> l2(e2);

		BOOST_TEST( l1 == l2 );

		multi::extents_t<2> a1 = l1.extents();
		multi::extents_t<2> a2 = l2.extents();

		BOOST_TEST( a1.size() == 0 );
		BOOST_TEST( a2.size() == 0 );
	}
	{
	}
	// auto C = A[1];

	// using std::get;

	// std::cout << get<0>(A.sizes()) << '\t' << get<1>(A.sizes()) << std::endl;
	// std::cout << get<0>(B.sizes()) << std::endl;

	return boost::report_errors();
}
