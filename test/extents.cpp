// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/detail/extents.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <tuple>  // IWYU pragma: keep
// IWYU pragma: no_include <type_traits>  // for integral_constant

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	{
		multi::array<int, 1> const arr1d(3);

		auto const x1d = multi::extents(arr1d.extension());

		BOOST_TEST( x1d.size() == 3 );

		auto const y1d = multi::extents(3);
		BOOST_TEST( y1d.size() == 3 );
	}
	{
		multi::extents const x2d(4, 3);
		BOOST_TEST( x2d.size() == 4 );
		auto [x0, x1] = x2d;

		BOOST_TEST( x0.size() == 4 );
		BOOST_TEST( x1.size() == 3 );

		using std::get;
		BOOST_TEST( x0 == get<0>(x2d) );
		BOOST_TEST( x1 == get<1>(x2d) );

		auto it = x2d.begin();
		++it;

#if defined(__cpp_concepts) && (__cpp_concepts >= 201907L)
		static_assert(std::incrementable<decltype(it)>);
#endif
		// BOOST_TEST( *it == xs[1] );
	}

	return boost::report_errors();
}
