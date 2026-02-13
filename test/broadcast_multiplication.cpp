// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/broadcast.hpp>
#include <boost/multi/io.hpp>
#include <boost/multi/restriction.hpp>  // for restriction, operator!=

#include <boost/core/lightweight_test.hpp>  // IWYU pragma: keep

#include <iostream>
#include <numeric>
// IWYU pragma: no_include <utility>  // for forward

namespace multi = boost::multi;

// NOLINTBEGIN(readability-identifier-length)
int main() {  // NOLINT(readability-function-cognitive-complexity)
	multi::array<int, 1> a(13);
	std::iota(a.begin(), a.end(), 0);

	std::cout << "a = " << a << '\n';

	multi::array<int, 1> b(11);
	std::iota(b.begin(), b.end(), 0);

	std::cout << "b = " << b << '\n';

	// this method uses a restriction directly
	auto const& M1 = multi::restricted(
		[&a, &b](auto i, auto j) { return a[i] * b[j]; },
		multi::extensions_t<2>(a.extension(), b.extensions())
	);

	std::cout << "M1 = " << M1 << '\n';
	BOOST_TEST( M1[5][7] == 5*7 );

#if (!defined(__GNUC__) || (__GNUC__ > 9)) || defined(__clang__)
#if !defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ > 11)
	// this method brings both arrays to the right (final) dimensionality and then does elementwise mult
	using multi::broadcast::operator*;  // cppcheck-suppress constStatement
	auto const& M2 = (~a.repeated(b.size())) * b.repeated(a.size());

	std::cout << "M2 = " << M2 << '\n';
	BOOST_TEST( M2[5][7] == 5*7 );
#endif
#endif

	// it is actually only necessary to bring one to the final dimension
	using multi::broadcast::operator*;  // cppcheck-suppress constStatement
	auto const& M3 = ~a.repeated(b.size()) * b;

	std::cout << "M3 = " << M3 << '\n';
	BOOST_TEST( M3[5][7] == 5*7 );

	// if one insists in using row and column 2D arrays (assignments are for brevity)
	multi::array<int, 2> A({a.size(), 1});
	(~A)[0] = a;
	multi::array<int, 2> B({1, b.size()});
	B[0] = b;

	using multi::broadcast::operator*;  // cppcheck-suppress constStatement
	auto const& M4 = (~((~A)[0].repeated(B[0].size()))) * B[0];

	std::cout << "M4 = " << M4 << '\n';
	BOOST_TEST( M4[5][7] == 5*7 );

	return boost::report_errors();
}
// NOLINTEND(readability-identifier-length)
