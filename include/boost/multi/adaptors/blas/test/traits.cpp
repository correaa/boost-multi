// Copyright 2019-2024 Alfredo A. Correa

#include<boost/test/included/unit_test.hpp>

#include <boost/multi/array.hpp>
#include <boost/multi/adaptors/blas/traits.hpp>

#include<complex>

namespace multi = boost::multi;
namespace blas = multi::blas;

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_traits_simple_array) {
	multi::array<double, 2> arr;
	BOOST_REQUIRE( arr.empty() );
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_traits) {
	static_assert( blas::is_d<double>{} );
	static_assert( blas::is_s<float >{} );

	static_assert( blas::is_c<std::complex<float>>{} );
	static_assert( blas::is_z<std::complex<double>>{} );
}
