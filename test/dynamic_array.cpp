// Copyright 2019-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for implicit_cast, explicit_cast

#include <boost/core/lightweight_test.hpp>

class copyable {
};

struct non_copyable {
	non_copyable() = default;

	non_copyable(non_copyable const&) = delete;
	non_copyable(non_copyable&&)      = default;

	auto operator=(non_copyable const&) -> non_copyable& = default;
	auto operator=(non_copyable&&) -> non_copyable&      = default;

	~non_copyable() = default;
};

struct non_default_constructible {
	explicit non_default_constructible(int /*unused*/) {}
	non_default_constructible() = delete;
};

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	{
		multi::dynamic_array<copyable, 2> const arr({2, 2});

		auto barr = arr;

		BOOST_TEST( barr.size() == 2 );
	}
	{
		multi::dynamic_array<non_copyable, 2> const arr({2, 2});
		// auto barr = arr;  // doesn't work (ok) because arr is not copy constructible
		// auto barr = std::move(arr);  // doesn't work (ok) because arr is not copy constructible

		BOOST_TEST( arr.size() == 2 );
	}
	{
		// multi::dynamic_array<non_default_constructible, 2> arr({2, 2});  // ok, doesn't work, because it is not copy-constructible
	}
	// {
	// 	multi::dynamic_array<non_default_constructible, 2> arr({2, 2}, non_default_constructible{1});

	// 	auto barr = arr;
	// }

	return boost::report_errors();
}
