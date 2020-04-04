#ifdef COMPILATION_INSTRUCTIONS
nvcc -x cu $0 -o $0x -lboost_unit_test_framework &&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi range selection"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<experimental/tuple>

#include "../array.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_array_range_section){

	multi::array<double, 4> A({10, 20, 30, 40});

	auto&& all = A({0, 10}, {0, 20}, {0, 30}, {0, 40});
	BOOST_REQUIRE( &A[1][2][3][4] == &all[1][2][3][4] );

	auto&& sub = A({0, 5}, {0, 10}, {0, 15}, {0, 20});
	BOOST_REQUIRE( &sub[1][2][3][4] == &A[1][2][3][4] );

	using std::experimental::apply;
	auto&& all_apply = apply(A, extensions(A));
	BOOST_REQUIRE( &A[1][2][3][4] == &all_apply[1][2][3][4] );

	auto&& element_apply = apply(A, std::array<int, 4>{1, 2, 3, 4});
	BOOST_REQUIRE( &A[1][2][3][4] == &element_apply );

	auto&& element_apply2 = apply(A, std::tuple<int, int, int, int>{1, 2, 3, 4});
	BOOST_REQUIRE( &A[1][2][3][4] == &element_apply );


}

