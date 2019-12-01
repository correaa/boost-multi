#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o$0x -lboost_unit_test_framework &&$0x&& rm $0x; exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi rotate"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_rotate){
	double a[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	double b[4][5];
	multi::array_ref<double, 2> A(&a[0][0], {4, 5});
	multi::array_ref<double, 2> B(&b[0][0], {4, 5});
	rotated(B) = rotated(A);
	BOOST_REQUIRE( B[1][1] == 6  );
	BOOST_REQUIRE( B[2][1] == 11 );
	BOOST_REQUIRE( B[1][2] == 7  );
	BOOST_REQUIRE( (B <<1) == (A <<1) );
	BOOST_REQUIRE( (B<<1)[2][1] == 7 );

	{
		multi::array<double, 2> a = {
			{00, 01},
			{10, 11}
		};
		BOOST_REQUIRE( a[1][0] == 10 );
		BOOST_REQUIRE( (a <<1)[0][1] == 10 );
		BOOST_REQUIRE( &a[1][0] == &(a << 1)[0][1] );
		(a<<1)[0][1] = 100;
		BOOST_REQUIRE( a[1][0] == 100 );
	}
	{
		multi::array<double, 2> const a = {
			{00, 01},
			{10, 11}
		};
		BOOST_REQUIRE( (a<<1)[0][1] == 10 );
	}

}

