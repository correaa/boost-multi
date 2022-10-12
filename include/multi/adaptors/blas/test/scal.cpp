// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
// © Alfredo A. Correa 2019-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS scal"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../blas/scal.hpp"

#include "../../../array.hpp"

namespace multi = boost::multi;
namespace blas = multi::blas;

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_n) {
	multi::array<double, 2> arr = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	BOOST_REQUIRE( (arr[0][2] == 3.) and (arr[2][2] == 11.) );

	blas::scal_n(2., arr[2].begin(), arr[2].size());
	BOOST_REQUIRE( arr[0][2] == 3. and arr[2][2] == 11.*2. );
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_it) {
	multi::array<double, 2> arr = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	BOOST_REQUIRE( arr[0][2] == 3. );
	BOOST_REQUIRE( arr[2][2] == 11.);

	blas::scal(2., arr[2].begin(), arr[2].end());
	BOOST_REQUIRE( arr[0][2] == 3. );
	BOOST_REQUIRE(arr[2][2] == 11.*2. );
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_real) {
	multi::array<double, 2> arr = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	BOOST_REQUIRE( arr[0][2] ==  3. );
	BOOST_REQUIRE( arr[2][2] == 11. );

	BOOST_REQUIRE(  blas::scal(1., arr[2]) ==  arr[2] );
	BOOST_REQUIRE( &blas::scal(1., arr[2]) == &arr[2] );
	BOOST_REQUIRE( +blas::scal(1., arr[2]) ==  arr[2] );

	blas::scal(2., arr[2]);
	BOOST_REQUIRE( arr[0][2] == 3. and arr[2][2] == 11.*2. );

	BOOST_REQUIRE( &blas::scal(1., arr[2]) == &arr[2] );
}
