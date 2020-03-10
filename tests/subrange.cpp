#ifdef COMPILATION_INSTRUCTIONS
$CXX $0 -o $0x -lboost_unit_test_framework &&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi range selection"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_array_range_section){

	multi::array<double, 4> A({10, 20, 30, 40});

	auto&& all = A({0, 10}, {0, 20}, {0, 30}, {0, 40});
	BOOST_REQUIRE( &A[1][2][3][4] == &all[1][2][3][4] );

	auto&& sub = A({0, 5}, {0, 10}, {0, 15}, {0, 20});
	BOOST_REQUIRE( &sub[1][2][3][4] == &A[1][2][3][4] );

}

