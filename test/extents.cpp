// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/detail/extents.hpp>

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <iterator>  // IWYU pragma: keep  // for incrementable
#include <tuple>     // IWYU pragma: keep
// IWYU pragma: no_include <type_traits>  // for integral_constant

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape,readability-function-cognitive-complexity)
	{
		multi::array<int, 1> const arr1d(3);

		auto const x1d = multi::extents_t(arr1d.extension());

		BOOST_TEST( x1d.size() == 3 );

		auto const y1d = multi::extents_t(3);
		BOOST_TEST( y1d.size() == 3 );
	}
	{
		multi::extents_t const x2d(4, 3);
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
		*it;
		x2d.begin()[1];
		// BOOST_TEST( *it == xs[1] );

		auto x2d_it  = x2d.begin();
		auto x2d_it2 = x2d_it + 2;
		auto x2d_it3 = x2d_it2 + 1;
		BOOST_TEST( x2d_it3 == x2d.begin() + 3 );
	}
	{
		multi::array<int, 3> const arr({2, 3, 5});
		auto [is, js, ks] = arr.extents();
		multi::array<int, 3> const brr({ks, js, is}, multi::uninitialized_elements);
	}
	{
		multi::array<int, 1> const arr({5});
		using std::get;
		auto is = get<0>(arr.extents());

		multi::array<int, 1> const brr(multi::extensions_t<1>{is});
	}
	{
		multi::array<int, 1> const arr({5});
		auto [is] = arr.extents();

		multi::array<int, 1> const brr(multi::extensions_t<1>{is});

		BOOST_TEST( arr.extents() == brr.extents() );
	}
	{
		multi::array<int, 2> const arr({2, 3});

		BOOST_TEST( arr.size() == 2 );
		BOOST_TEST( arr.transposed().size() == 3 );

		auto [is, js] = arr.extents();

		BOOST_TEST( is.size() == 2 );
		BOOST_TEST( js.size() == 3 );

		multi::array<int, 2> const brr(multi::extensions_t<2>{js, is});

		BOOST_TEST( brr.size() == 3 );

		BOOST_TEST( brr.size() == arr.transposed().size() );
		BOOST_TEST( brr.extents() == arr.transposed().extents() );
	}
	{
		multi::array<int, 2> const arr({2, 3});

		BOOST_TEST( arr.size() == 2 );
		BOOST_TEST( arr.transposed().size() == 3 );

		auto [is, js] = arr.extents();

		BOOST_TEST( is.size() == 2 );
		BOOST_TEST( js.size() == 3 );

		multi::array<int, 2> const brr(multi::extensions_t<2>{js, is});  // braced `{js, is}` would read as iota-rows (extension_t -> array<int,1>); use extensions_t<2> for extents

		BOOST_TEST( brr.size() == 3 );

		BOOST_TEST( brr.size() == arr.transposed().size() );
		BOOST_TEST( brr.extents() == arr.transposed().extents() );
	}
	{
		multi::array<int, 3> const arr({2, 3, 5});
		auto [is, js, ks] = arr.extents();
		multi::array<int, 3> const brr(multi::extensions_t<3>{ks, js, is});

		BOOST_TEST( brr.extents() == arr.rotated().transposed().extents() );
	}
	{
		multi::array<int, 3> const arr({2, 3, 5});
		auto [is, js, ks] = arr.extents();
		multi::array<int, 3> const brr({ks, js, is});

		BOOST_TEST( brr.extents() == arr.rotated().transposed().extents() );
	}

	return boost::report_errors();
}
