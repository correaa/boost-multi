// Copyright 2018-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wold-style-cast"
	#pragma clang diagnostic ignored "-Wundef"
	#pragma clang diagnostic ignored "-Wconversion"
	#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wundef"
	#pragma GCC diagnostic ignored "-Wconversion"
	#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#ifndef BOOST_TEST_MODULE
	#define BOOST_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>

#if defined(__clang__)
	#pragma clang diagnostic pop
#elif defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

#include <boost/multi/array.hpp>  // for array

#include <complex>  // for complex
#include <string>   // for operator==, string

namespace multi = boost::multi;

inline auto what_is(multi::array<double, 2> const& /*arr*/) { return std::string{"real"}; }                   // NOLINT(fuchsia-default-arguments-calls)
inline auto what_is(multi::array<std::complex<double>, 2> const& /*arr*/) { return std::string{"complex"}; }  // NOLINT(fuchsia-default-arguments-calls)

BOOST_AUTO_TEST_CASE(multi_array_overload) {
	multi::array<double, 2> const               real_A({10, 20});
	multi::array<std::complex<double>, 2> const cplx_A({10, 20});

	std::string const real_str    = what_is(real_A);
	std::string const complex_str = what_is(cplx_A);

	BOOST_REQUIRE( real_str    == "real"    );
	BOOST_REQUIRE( complex_str == "complex" );
}
