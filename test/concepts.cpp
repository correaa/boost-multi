// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2022 Alfredo A. Correa

// #define BOOST_TEST_MODULE "C++ Unit Tests for Multi concepts"  // NOLINT(cppcoreguidelines-macro-usage) title
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include <multi/array.hpp>

namespace multi = boost::multi;

using NDArrays = boost::mpl::list<
	multi::array<double, 1>,
	multi::array<double, 2>,
	multi::array<double, 3>
>;

BOOST_AUTO_TEST_CASE_TEMPLATE(convertibles, NDArray, NDArrays)
{
	static_assert( std::is_convertible_v<typename NDArray::      reference, typename NDArray::value_type> );
	static_assert( std::is_convertible_v<typename NDArray::const_reference, typename NDArray::value_type> );

	static_assert( std::is_same_v<typename NDArray::element_type, typename multi::array<double, 1>::value_type>);
	static_assert( std::is_same_v<typename NDArray::element_ref , typename multi::array<double, 1>::reference >);

	using NDRef = typename NDArray::ref;

	static_assert( std::is_convertible_v<NDRef, NDArray> );

	static_assert( std::is_convertible_v<typename NDRef::      reference, typename NDRef::value_type> );
	static_assert( std::is_convertible_v<typename NDRef::const_reference, typename NDRef::value_type> );

	static_assert( std::is_same_v<typename NDRef::element_type, typename multi::array<double, 1>::value_type> );
	static_assert( std::is_same_v<typename NDRef::element_ref , typename multi::array<double, 1>::reference > );
}
