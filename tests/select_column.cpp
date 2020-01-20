#ifdef COMPILATION_INSTRUCTIONS
$CXX $0 -o$0x -DBOOST_TEST_DYN_LINK -lboost_unit_test_framework &&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi range selection"
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_array_range_section){
	multi::array<double, 2> A = {
		{00., 01., 02.},
		{10., 11., 12.},
		{20., 21., 22.},
		{30., 31., 32.},
	};
	BOOST_REQUIRE( size(A) == 4 );

	auto&& col2( A(A.extension(0), 2) ); // select column #2 
	// same as A(extesion(A), 2)
	// same as A(A.extension(0), 2);
	// same as rotated(A)[2];
	BOOST_REQUIRE( col2.size(0) == size(A) );

	BOOST_REQUIRE( dimensionality(col2) == 1 );
	BOOST_REQUIRE( size(col2) == size(A) );
	BOOST_REQUIRE( col2.size() == size(A) );
	BOOST_REQUIRE(( col2 == multi::array<double, 1>{02., 12., 22., 32.} ));
	BOOST_REQUIRE(( col2 == multi::array<double, 1>(rotated(A)[2]) ));
	BOOST_REQUIRE(( col2 == rotated(A)[2] ));
	BOOST_REQUIRE(( col2 == A(A.extension(0), 2) ));
}

