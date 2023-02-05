// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi complex"
#include <boost/test/unit_test.hpp>

#include <boost/mpl/list.hpp>

#include "../../complex.hpp"

namespace multi = boost::multi;

using float_types = boost::mpl::list<float, double>;

BOOST_AUTO_TEST_CASE_TEMPLATE(complex_ctors, T, float_types) {
	{
		multi::complex<T> a = T{1.0} + multi::imaginary<T>{T{2.0}};
		BOOST_REQUIRE( real(a) == T{1.0});
		BOOST_REQUIRE( imag(a) == T{2.0});
	}
	{
		multi::complex<T> a = T{1.0} + T{2.0} * multi::imaginary<T>::i;
		BOOST_REQUIRE( real(a) == T{1.0});
		BOOST_REQUIRE( imag(a) == T{2.0});
	}
	// 	{
	//		multi::complex<T> a = T{1.0} + multi::imaginary{T{2.0}};
	// 		BOOST_REQUIRE( real(a) == T{1.0});
	// 		BOOST_REQUIRE( imag(a) == T{2.0});
	// 	}
}

BOOST_AUTO_TEST_CASE(double_complex_literals) {
	using namespace multi::literals;
	multi::complex<double> a = 1.0 + 2.0_i;
	//	multi::complex<double> a = 1.0 + 2.0i;  // literal i is not standard

	BOOST_REQUIRE( real(a) == 1.0 );
	BOOST_REQUIRE( imag(a) == 2.0 );
}

BOOST_AUTO_TEST_CASE(float_complex_literals) {
	using namespace multi::literals;
	//  multi::complex<float> a = 1.0f + 2.0  _i;  // may induced an undesired or forbidden conversion
	//  multi::complex<float> a = 1.0f + 2.0 f_i;  // literal f_i is not standard
	//	multi::complex<float> a = 1.0f + 2.0_f_i;
	multi::complex<float> a = 1.0f + 2.0_if;

	BOOST_REQUIRE( real(a) == 1.0f );
	BOOST_REQUIRE( imag(a) == 2.0f );
}
