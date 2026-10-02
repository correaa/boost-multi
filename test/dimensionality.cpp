// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>
#include <boost/multi/restriction.hpp>  // IWYU pragma: keep  // for bind_front_t, restriction

#include <boost/core/lightweight_test.hpp>

#include <utility>  // IWYU pragma: keep

namespace multi = boost::multi;

namespace {
// NOLINTNEXTLINE(readability-identifier-naming) nvcc (<12.4) cannot use a function-local lambda/type as a template argument within the same function, so this must live at namespace scope
struct proj_fn {
	constexpr auto operator()(multi::index /*row*/, multi::index /*col*/) const { return 1; }
};
}  // namespace

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	multi::array<int, 2> const arr;

	BOOST_TEST( arr.dimensionality == 2 );
	static_assert(decltype(arr)::dimensionality == 2);

	auto restr = proj_fn{} ^ multi::extents_t<2>(2, 2);

	BOOST_TEST( restr[1][1] == 1 );
	BOOST_TEST( restr.dimensionality == 2 );
	static_assert(decltype(restr)::dimensionality == 2);

	return boost::report_errors();
}
