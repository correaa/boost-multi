// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, dynamic_array, num_elements

#include <tuple>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {
	{
		multi::array<int, 1> const A = {1, 2, 3, 4};
		BOOST_TEST( A.size() == 4 );
	}
	{
		auto const tup = std::make_tuple(1, 2, 3);
		multi::array<int, 1> const A(tup);
		BOOST_TEST( A.size() == 3 );
	}
	{
		multi::array<int, 2> const B({3, 2}, 0);
		multi::array<int, 1> const Bsizes(B.sizes());
		BOOST_TEST( Bsizes.size() == 2 );
	}

	return boost::report_errors();
}
