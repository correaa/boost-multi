// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/algorithms/ordering.hpp>  // for ordering
#include <boost/multi/array.hpp>                // for array

#include <boost/core/lightweight_test.hpp>

#include <array>       // for array
#include <functional>  // for greater
#include <iterator>    // for next

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// basic ascending order into a caller-provided std::array
	{
		multi::array<int, 1> const arr = {3, 1, 2};

		std::array<multi::index, 3> order{};

		BOOST_TEST( multi::ordering(arr, order.begin()) == order.end() );

		BOOST_TEST( order[0] == 1 );
		BOOST_TEST( order[1] == 2 );
		BOOST_TEST( order[2] == 0 );

		// the data must be untouched
		BOOST_TEST( arr[0] == 3 );
		BOOST_TEST( arr[1] == 1 );
		BOOST_TEST( arr[2] == 2 );

		// reading through the order gives a sorted view
		BOOST_TEST( arr[order[0]] == 1 );
		BOOST_TEST( arr[order[1]] == 2 );
		BOOST_TEST( arr[order[2]] == 3 );
	}

	// descending order via custom comparator
	{
		multi::array<int, 1> const arr = {3, 1, 2};

		std::array<multi::index, 3> order{};

		multi::ordering(arr, order.begin(), std::greater<>{});

		BOOST_TEST( arr[order[0]] == 3 );
		BOOST_TEST( arr[order[1]] == 2 );
		BOOST_TEST( arr[order[2]] == 1 );
	}

	// already sorted -> identity permutation
	{
		multi::array<int, 1> const arr = {1, 2, 3, 4};

		std::array<multi::index, 4> order{};

		multi::ordering(arr, order.begin());

		BOOST_TEST( order[0] == 0 );
		BOOST_TEST( order[1] == 1 );
		BOOST_TEST( order[2] == 2 );
		BOOST_TEST( order[3] == 3 );
	}

	// output into a caller-provided multi::array, floating-point elements
	{
		multi::array<double, 1> const arr = {2.5, -1.0, 0.0, 9.9, 3.3};

		multi::array<multi::index, 1> order(arr.extents());

		BOOST_TEST( multi::ordering(arr, order.begin()) == order.end() );

		for(multi::index k = 0; k + 1 != order.size(); ++k) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			BOOST_TEST( arr[order[k]] <= arr[order[k + 1]] );
		}
	}

	// ties: std::sort is not stable, so only require the result to be sorted and a valid permutation
	{
		multi::array<int, 1> const arr = {2, 2, 1, 3, 1};

		multi::array<multi::index, 1> order(arr.extents());

		multi::ordering(arr, order.begin());

		for(multi::index k = 0; k + 1 != order.size(); ++k) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			BOOST_TEST( arr[order[k]] <= arr[order[k + 1]] );
		}

		multi::index sum = 0;
		for(auto idx : order) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			sum += idx;
		}
		BOOST_TEST( sum == (0 + 1 + 2 + 3 + 4) );  // it is a permutation of {0..4}
	}

	// caller-provided raw pointer (via std::array::data) as output
	{
		multi::array<int, 1> const arr = {5, 4, 6};

		std::array<multi::index, 3> buf{};

		// compare raw pointers on both sides: MSVC's std::array::end() is a wrapped iterator, not a pointer
		BOOST_TEST( multi::ordering(arr, buf.data()) == std::next(buf.data(), 3) );

		BOOST_TEST( arr[buf[0]] == 4 );
		BOOST_TEST( arr[buf[1]] == 5 );
		BOOST_TEST( arr[buf[2]] == 6 );
	}

	return boost::report_errors();
}
