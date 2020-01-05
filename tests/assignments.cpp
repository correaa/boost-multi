#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi assignments"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../multi/array.hpp"

#include<iostream>
#include<vector>

namespace multi = boost::multi;
using std::cout;

multi::array_ref<double, 2> make_ref(double* p){return {p, {5, 7}};}

BOOST_AUTO_TEST_CASE(assignments){
	{
		std::vector<double> v(5*7, 99.);

		multi::array<double, 2> A{{5, 7}, 33.};
		multi::array_ref<double, 2>(v.data(), {5, 7}) = A;
		BOOST_REQUIRE( v[9] == 33. );
		BOOST_REQUIRE( not v.empty() );
		BOOST_REQUIRE( not empty(A) );

		multi::array<double, 1> V;
		BOOST_REQUIRE( V.empty() );
	}
	{
		std::vector<double> v(5*7, 99.), w(5*7, 33.);

		multi::array_ref<double, 2> B{w.data(), {5, 7}};
		make_ref(v.data()) = B;
		make_ref(v.data()) = B.sliced(0,5);

		BOOST_REQUIRE( v[9] == 33. );
	}
	{
		std::vector<double> v(5*7, 99.), w(5*7, 33.);

		make_ref(v.data()) = make_ref(w.data());

		BOOST_REQUIRE( v[9] == 33. );
	}
}

