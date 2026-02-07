// Copyright 2016 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

struct dummy {
    dummy(int) {}
};


namespace {
template<class T>
class indirect_initializer_list {
    std::initializer_list<T> impl_;

  public:
    indirect_initializer_list(std::initializer_list<T> impl) : impl_{impl} {}
};

// [[maybe_unused]] int fun(std::initializer_list<int>) { return 33; }
[[maybe_unused]] int fun(indirect_initializer_list<int>) { return 33; }
[[maybe_unused]] int fun(dummy const&) { return 44; }
}

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
    auto res = fun({1, 2});

    BOOST_TEST( res == 44 );

    // clang-format off
	{
		{ multi::array<int, 1> const arr({9, 9, 9}); BOOST_TEST(arr.num_elements() == 3); }
		{ multi::array<int, 1> const arr({9, 9});    BOOST_TEST(arr.num_elements() == 2); }
		{ multi::array<int, 1> const arr({9});       BOOST_TEST(arr.num_elements() == 1); }

		{ multi::array<int, 3> const arr({9, 9, 9}); BOOST_TEST(arr.num_elements() == 9L*9L*9L ); }
		{ multi::array<int, 2> const arr({9, 9});    BOOST_TEST(arr.num_elements() == 9L*9L); }
		{ multi::array<int, 1> const arr({9});       BOOST_TEST(arr.num_elements() != 9L); } // PATTERN BREAKS
	}
	// clang-format on

	return boost::report_errors();
}
