// Copyright 2019-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_CONFIG_ASSERT_HPP
#define BOOST_MULTI_DETAIL_CONFIG_ASSERT_HPP

#include<cassert>

#ifdef BOOST_MULTI_ASSERT_DISABLE  // to activate bounds check compile in debug mode (default) with -DBOOST_MULTI_ACCESS_DEBUG
	#define BOOST_MULTI_ASSERT(Expr) /*empty*/ // NOLINT(cppcoreguidelines-macro-usage
#else
	#define BOOST_MULTI_ASSERT(Expr) assert(Expr)  // NOLINT(cppcoreguidelines-macro-usage)
#endif

#endif  // BOOST_MULTI_DETAIL_CONFIG_ASSERT_HPP
