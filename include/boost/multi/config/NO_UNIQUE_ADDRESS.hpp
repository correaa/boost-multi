// Copyright 2019-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_CONFIG_NO_UNIQUE_ADDRESS_HPP_
#define BOOST_MULTI_CONFIG_NO_UNIQUE_ADDRESS_HPP_

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(name) 0
#endif

#if __has_cpp_attribute(no_unique_address) >=201803 and not defined(__NVCC__) and not defined(__PGI)
	// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) this macro will be needed until C++20
	#define BOOST_MULTI_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
	// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) this macro will be needed until C++20
	#define BOOST_MULTI_NO_UNIQUE_ADDRESS
#endif

#endif  // BOOST_MULTI_CONFIG_NO_UNIQUE_ADDRESS_HPP_
