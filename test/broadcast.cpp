// Copyright 2023-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// #if defined(__clang__)
//  #pragma clang diagnostic push
//  #pragma clang diagnostic ignored "-Wconversion"
//  #pragma clang diagnostic ignored "-Wold-style-cast"
//  #pragma clang diagnostic ignored "-Wsign-conversion"
//  #pragma clang diagnostic ignored "-Wundef"
// #elif defined(__GNUC__)
//  #pragma GCC diagnostic push
//  #pragma GCC diagnostic ignored "-Warray-bounds="
//  #if (__GNUC__ > 7)
//      #pragma GCC diagnostic ignored "-Wcast-function-type"
//  #endif
//  #pragma GCC diagnostic ignored "-Wconversion"
//  #pragma GCC diagnostic ignored "-Wold-style-cast"
//  #pragma GCC diagnostic ignored "-Wsign-conversion"
//  #pragma GCC diagnostic ignored "-Wstringop-overflow="
//  #pragma GCC diagnostic ignored "-Wundef"
// #endif

// #ifndef BOOST_TEST_MODULE
//  #define BOOST_TEST_MAIN
// #endif

// #include <boost/test/included/unit_test.hpp>

// #if defined(__clang__)
//  #pragma clang diagnostic pop
// #elif defined(__GNUC__)
//  #pragma GCC diagnostic pop
// #endif

#include <boost/multi/array.hpp>

#include <algorithm>  // for std::ranges::fold_left

#include <boost/core/lightweight_test.hpp>
#define BOOST_AUTO_TEST_CASE(CasenamE) [[maybe_unused]] void* CasenamE;

namespace multi = boost::multi;

int main() {
BOOST_AUTO_TEST_CASE(broadcast_as_fill) {
	multi::array<int, 1> b = { 10, 11 };

	multi::array<int, 2> B({ 10, 2 });

	// std::fill  (B.begin(), B.end(), b);                                       // canonical way
	std::fill_n(B.begin(), B.size(), b);  // canonical way

	// std::copy_n(b.broadcasted().begin(), B.size(), B.begin());                // doesn't work because faulty implementation of copy_n
	// thrust::copy_n(b.broadcasted().begin(), B.size(), B.begin());                // equivalent, using broadcast

	// std::copy_n(b.broadcasted().begin(), b.broadcasted().size(), B.begin());  // incorrect, undefined behavior, no useful size()
	// std::copy  (b.broadcasted().begin(), b.broadcasted().end(), B.begin());   // incorrect, undefined behavior, non-terminating loop (end is not reacheable)
	// B = b.broadcasted();

	BOOST_TEST( B[0] == b );
	BOOST_TEST( B[1] == b );

	BOOST_TEST( std::all_of(B.begin(), B.end(), [b](auto const& row) { return row == b; }) );
}

return boost::report_errors();}
