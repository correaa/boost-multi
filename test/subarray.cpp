// Copyright 2019-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for implicit_cast, explicit_cast

#include <boost/core/lightweight_test.hpp>

// IWYU pragma: no_include <type_traits>                      // for add_const_t
#include <utility>      // for as_const

namespace multi = boost::multi;

int main() {
	// BOOST_AUTO_TEST_CASE(subarray_assignment)
	{
		multi::array<int, 3> A({3, 4, 5}, 99);

		auto constA2 = std::as_const(A)[2];
		BOOST_TEST( constA2[1][1] == 99 );

		auto A2 = A[2];
		BOOST_TEST( A2[1][1] == 99 );

		//  what(constA2, A2);
		//  A2[1][1] = 88;
	}

	// BOOST_AUTO_TEST_CASE(subarray_base)
	{
		multi::array<int, 3> A({3, 4, 5}, 99);

		auto&& Asub  = A();
		*Asub.base() = 88;

		BOOST_TEST( A[0][0][0] == 88 );

		*A().base() = 77;

		BOOST_TEST( A[0][0][0] == 77 );

		// *std::as_const(Asub).base() = 66;  // should not compile, read-only
	}

	return boost::report_errors();
}
