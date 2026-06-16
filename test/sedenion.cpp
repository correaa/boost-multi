// Copyright 2026 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for implicit_cast, explicit_cast

#include <boost/core/lightweight_test.hpp>

#include <numeric>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	{
		multi::array<int, 2> arr(multi::extents_t<2>{2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 2*2 );
	}
	{
		multi::array<int, 4> arr(multi::extents_t<4>{2, 2, 2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 2*2*2*2 );
	}
	{
		multi::array<int, 8> arr(multi::extents_t<8>{2, 2, 2, 2, 2, 2, 2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 256 );
	}
	{
		multi::array<int, 10> arr(multi::extents_t<10>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 256*2*2 );
	}
#ifdef __clang__
	{
		multi::array<int, 12> arr( multi::extents_t<12>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 256*2*2*2*2 );
	}
	{
		multi::array<int, 14> arr( multi::extents_t<14>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 256*2*2*2*2*2*2 );
	}
	{
		multi::array<int, 16> arr( multi::extents_t<16>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, 0);
		BOOST_TEST( arr.num_elements() == 65536 );
	}
#endif

	return boost::report_errors();
}
