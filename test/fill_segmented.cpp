// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, apply, operator==
// #include <boost/multi/detail/what.hpp>

#include <boost/core/lightweight_test.hpp>

#include <algorithm>

namespace {

template<class SegIt, class T>
void fill_segmented(SegIt first, SegIt last, T x) {
	// typedef segmented_iterator_traits<SegIt> traits;
	// typename traits::segment_iterator sf = traits::segment(first);
	// typename traits::segment_iterator sl = traits::segment(last);
	// typename traits::local_iterator   lf = traits::local(first);

	auto sf = first.segment();
	auto sl = last.segment();

	auto lf = first.local();

	while(true) {  // NOLINT(altera-unroll-loops)
		// typename traits::local_iterator le =
		//     (sf == sl) ? traits::local(last) : traits::end(sf);

		auto le = (sf == sl) ? last.local() : sf.end();

		std::fill(lf, le, x);
		if(sf == sl) {
			break;
		}
		// lf = traits::begin(++sf);
		lf = lf = (++sf).begin();
	}
}

}  // namespace

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity)
	multi::array<int, 2> arr1({3, 5});
	std::fill(arr1().flatted().begin(), arr1().flatted().end(), 7);

	BOOST_TEST( arr1[1][1] == 7 );

	multi::array<int, 2> arr2 = {
		{1, 2, 3},
		{4, 5, 6}
	};
	BOOST_TEST( arr2.flattened().begin().segment().size() == 3 );

	BOOST_TEST( arr2.flattened().begin().segment()[0] == 1 );
	BOOST_TEST( arr2.flattened().begin().segment()[1] == 2 );
	BOOST_TEST( arr2.flattened().begin().segment()[2] == 3 );

	BOOST_TEST( arr2.flattened().begin().segment().num_elements() == 3 );

	BOOST_TEST( arr2[0].size() == 3 );

	BOOST_TEST( arr2[0][0] == 1 );
	BOOST_TEST( arr2[0][1] == 2 );
	BOOST_TEST( arr2[0][2] == 3 );

	BOOST_TEST( arr2[0] == arr2.flattened().begin().segment() );

	BOOST_TEST( (arr2.flattened().begin() + 0).segment() == arr2[0] );
	BOOST_TEST( (arr2.flattened().begin() + 1).segment() == arr2[0] );
	BOOST_TEST( (arr2.flattened().begin() + 2).segment() == arr2[0] );

	BOOST_TEST( (arr2.flattened().begin() + 3).segment() == arr2[1] );
	BOOST_TEST( (arr2.flattened().begin() + 4).segment() == arr2[1] );
	BOOST_TEST( (arr2.flattened().begin() + 5).segment() == arr2[1] );

	BOOST_TEST( (arr2.flattened().begin() + 0).local() == arr2[0].begin() );
	BOOST_TEST( (arr2.flattened().begin() + 1).local() == arr2[0].begin() + 1 );
	BOOST_TEST( (arr2.flattened().begin() + 2).local() == arr2[0].begin() + 2 );

	BOOST_TEST( (arr2.flattened().begin() + 3).local() == arr2[1].begin() );
	BOOST_TEST( (arr2.flattened().begin() + 4).local() == arr2[1].begin() + 1 );
	BOOST_TEST( (arr2.flattened().begin() + 5).local() == arr2[1].begin() + 2 );

	// segment()/local() must be const-correct on a const array (regression for the bug
	// where they hardcoded the non-const pointer type and failed to compile)
	{
		multi::array<int, 2> const carr = {
			{1, 2, 3},
			{4, 5, 6}
		};

		BOOST_TEST( carr.flattened().begin().segment().size() == 3 );
		BOOST_TEST( carr.flattened().begin().segment().num_elements() == 3 );
		BOOST_TEST( carr.flattened().begin().segment()[0] == 1 );

		BOOST_TEST( carr.flattened().begin().segment() == carr[0] );
		BOOST_TEST( (carr.flattened().begin() + 4).segment() == carr[1] );

		BOOST_TEST( (carr.flattened().begin() + 0).local() == carr[0].begin() );
		BOOST_TEST( (carr.flattened().begin() + 4).local() == carr[1].begin() + 1 );
	}

	return boost::report_errors();
}
