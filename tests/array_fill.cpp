#ifdef COMPILATION_INSTRUCTIONS
$CXX -std=c++14 -Wall -Wextra -Wpedantic $0 -DBOOST_TEST_DYN_LINK -lboost_unit_test_framework -o $0x &&$0x $@&&rm $0x;exit
#endif

#include "../array.hpp"

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi fill"
#include<boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(fill){
	namespace multi = boost::multi;

	multi::array<double, 2> d2D = {
		{150., 16., 17., 18., 19.},
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	using std::all_of;
	BOOST_REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto& e){return e==5.;}) );

	using std::fill;

	fill(begin(d2D[1]), end(d2D[1]), 8.);
	BOOST_REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto& e){return e==8.;}) );

	fill(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), 8.);
	BOOST_REQUIRE( all_of(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), [](auto&& e){return e==8.;}) );

//	fill( d2D({0, 4}).begin(1), d2D({0, 4}).end(1), 9.);
//	fill(begin(d2D({0, 4}, 1)), end(d2D({0, 4}, 1)), 9.);
}

