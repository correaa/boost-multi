#ifdef COMPILATION_INSTRUCTIONS
$CXX catch_main.o -std=c++17 -Wall -Wextra -Wpedantic $0 -o $0x && $0x && rm $0x; exit
#endif

#include<catch2/catch.hpp>

#include "../array.hpp"

namespace multi = boost::multi;

TEST_CASE( "Array partitioned 1D", "[array]"){
	multi::array<double, 1>	A1 = {0, 1, 2, 3, 4, 5};
	auto&& A2_ref = A1.partitioned(2);
	REQUIRE(dimensionality(A2_ref)==dimensionality(A1)+1);
	REQUIRE(size(A2_ref)==2);
	REQUIRE(size(A2_ref[0])==3);
	REQUIRE( &A2_ref[1][0] == &A1[3] );
}

TEST_CASE( "Array partitioned 2D", "[array]"){
	multi::array<double, 2>	A2 = 
		{
			{  0,  1,  2,  3,  4,  5}, 
			{  6,  7,  8,  9, 10, 11}, 

			{ 12, 13, 14, 15, 16, 17}, 
			{ 18, 19, 20, 21, 22, 23}, 
		}
	;
	auto&& A3_ref = A2.partitioned(2);
	REQUIRE( dimensionality(A3_ref) == dimensionality(A2)+1 );
	REQUIRE( num_elements(A3_ref) == num_elements(A2) );
	REQUIRE( size(A3_ref)==2 );
	REQUIRE( size(A3_ref[0])==2 );
	REQUIRE( size(A3_ref[0][0])==6 );
	REQUIRE( &A3_ref[1][1][0] == &A2[3][0] );
}

