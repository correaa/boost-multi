// Copyright 2019-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_CONFIG_NODISCARD_HPP_
#define BOOST_MULTI_CONFIG_NODISCARD_HPP_

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(name) 0
#endif

#ifndef BOOST_MULTI_NODISCARD
#if defined(__NVCC__)
	#define BOOST_MULTI_NODISCARD(MsG)
#elif (__has_cpp_attribute(nodiscard) and (__cplusplus>=201703L))
	#if (__has_cpp_attribute(nodiscard)>=201907) and (__cplusplus>201703L)
		#define BOOST_MULTI_NODISCARD(MsG) [[nodiscard]]  // [[nodiscard(MsG)]] in c++20 empty message is not allowed with paren
	#else
		#define BOOST_MULTI_NODISCARD(MsG) [[nodiscard]]  // NOLINT(cppcoreguidelines-macro-usage) TODO(correaa) check if this is needed in C++17
	#endif
#elif __has_cpp_attribute(gnu::warn_unused_result)
	#define BOOST_MULTI_NODISCARD(MsG) [[gnu::warn_unused_result]]
#else
	#define BOOST_MULTI_NODISCARD(MsG)
#endif
#endif

#ifndef BOOST_MULTI_NODISCARD_CLASS
	#if(__has_cpp_attribute(nodiscard) and not defined(__NVCC__) and (not defined(__clang__) or (defined(__clang__) and (__cplusplus >= 202002L))))
		#if (__has_cpp_attribute(nodiscard)>=201907)
			#define BOOST_MULTI_NODISCARD_CLASS(MsG) [[nodiscard_(MsG)]]
		#else
			#define BOOST_MULTI_NODISCARD_CLASS(MsG) [[nodiscard]]
		#endif
	#else
		#define BOOST_MULTI_NODISCARD_CLASS(MsG)
	#endif
#endif

#endif  // BOOST_MULTI_CONFIG_NODISCARD_HPP_
