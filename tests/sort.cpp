#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra $0 -o$0x -lboost_unit_test_framework&&$0x&&rm $0x; exit
#endif

#define BOOST_TEST_MODULE "Unit Tests for Multi sort"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include<algorithm> // for sort
#include<vector>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_array_ref_stable_sort){

// begin-snippet: multi_array_ref_stable_sort
	std::vector<double> v = {1.,2.,3.};
	double d2D[4][5] = {
		{150, 16, 17, 18, 19},
		{ 30,  1,  2,  3,  4}, 
		{100, 11, 12, 13, 14}, 
		{ 50,  6,  7,  8,  9} 
	};
	multi::array_ref<double, 2> d2D_ref(&d2D[0][0], {4, 5});
	
	BOOST_REQUIRE( not std::is_sorted(begin(d2D_ref), end(d2D_ref) ) );
	std::stable_sort( begin(d2D_ref), end(d2D_ref) );
	BOOST_REQUIRE( std::is_sorted( begin(d2D_ref), end(d2D_ref) ) );

//	BOOST_REQUIRE( not std::is_sorted( begin(rotated(d2D_ref)), end(rotated(d2D_ref))) );
//	std::stable_sort( begin(rotated(d2D_ref)), end(rotated(d2D_ref)) );
//	BOOST_REQUIRE( std::is_sorted( begin(rotated(d2D_ref)), end(rotated(d2D_ref))) );

	BOOST_REQUIRE( not std::is_sorted( d2D_ref.begin(1), d2D_ref.end(1) ) );
	std::stable_sort( d2D_ref.begin(1), d2D_ref.end(1) );
	BOOST_REQUIRE( std::is_sorted( d2D_ref.begin(1), d2D_ref.end(1) ) );
// end-snippet

}

