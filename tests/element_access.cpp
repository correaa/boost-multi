#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x; exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi element access"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#if __cpp_lib_apply>=201603
#include<tuple> // apply
#else
#include<experimental/tuple>
#endif

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_tests_element_access){
	multi::array<double, 2> m({3, 3}, {});
	std::array<int, 2> p = {1, 2};
	BOOST_REQUIRE( &m[p[0]][p[1]] == &m(p[0], p[1]) );
	using std::
#if not(__cpp_lib_apply>=201603)
		experimental::
#endif
		apply;
	BOOST_REQUIRE( &m[p[0]][p[1]] == &apply(m, p) );
}

BOOST_AUTO_TEST_CASE(multi_test_constness_reference){
	multi::array<double, 2> m({3, 3}, 99.);
	BOOST_REQUIRE( m(1, {0, 3})[1] == 99. );
	BOOST_REQUIRE( m({0, 3}, 1)[1] == 99. );
	BOOST_REQUIRE( m({0, 3}, {0, 3})[1][1] == 99. );

	m(1, {0, 3})[1] = 88.; BOOST_REQUIRE( m(1, {0, 3})[1] == 88. );
	m({0, 3}, 1)[1] = 77.; BOOST_REQUIRE( m({0, 3}, 1)[1] == 77. );
	m({0, 3}, {0, 3})[1][1] = 66.; BOOST_REQUIRE( m({0, 3}, 1)[1] == 66. );
}

BOOST_AUTO_TEST_CASE(multi_test_constness){
	multi::array<double, 2> const m({3, 3}, 99.);
	BOOST_REQUIRE( m(1, {0, 3})[1] == 99. );
	BOOST_REQUIRE( m({0, 3}, 1)[1] == 99. );
	BOOST_REQUIRE( m({0, 3}, {0, 3})[1][1] == 99. );

// none of these lines should compile because m is read-only
//	m(1, {0, 3})[1] = 88.; BOOST_REQUIRE( m(1, {0, 3})[1] == 88. );
//	m({0, 3}, 1)[1] = 77.; BOOST_REQUIRE( m({0, 3}, 1)[1] == 77. );
//	m({0, 3}, {0, 3})[1][1] = 66.; BOOST_REQUIRE( m({0, 3}, 1)[1] == 66. );
}

