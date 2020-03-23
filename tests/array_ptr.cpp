#ifdef COMPILATION_INSTRUCTIONS
$CXX $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_array_ptr_test){
	double a[4][5] = {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	double b[4][5];
#ifdef __cpp_deduction_guides
	auto&& A = *multi::array_ptr(&a[0][0], {4, 5});
	multi::array_ref B(&b[0][0], {4, 5});
#else
	auto&& A = *multi::array_ptr<double, 2>(&a[0][0], {4, 5});
	multi::array_ref<double, 2, double*> B(&b[0][0], {4, 5});
#endif
	BOOST_REQUIRE( size(A) == 4 );
	BOOST_REQUIRE( size(A) == size(B) );

	B = A;
	BOOST_REQUIRE( B == A );

	BOOST_REQUIRE( size(rotated(A)) == 5 );
	BOOST_REQUIRE( size(rotated(B)) == 5 );
	BOOST_REQUIRE( size(rotated(A)) == size(rotated(B)) );
	
	BOOST_REQUIRE( std::distance(begin(rotated(B)), end(rotated(B))) == 5 );		
	rotated(A) = rotated(B);
	BOOST_REQUIRE( b[1][2] == a[1][2] );
}

