#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4-*-
$CXX $0 -o $0x -lboost_unit_test_framework&&$0x --report_level=detailed&&rm $0x;exit
#endif
// © Alfredo Correa 2019-2020

#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi one based"
#include<boost/test/unit_test.hpp>

#include<iostream>

#include "../array.hpp"
//#include "../adaptors/cuda.hpp"

#include<complex>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(one_based_1D){

	multi::array<double, 1> Af({{1, 1 + 10}}, 0.);
	Af[1] = 1.;
	Af[2] = 2.;
	Af[3] = 3.;

	BOOST_REQUIRE( Af[1] = 1. );
	BOOST_REQUIRE( *Af.data_elements() == 1. );
	BOOST_REQUIRE( size(Af) == 10 );
	BOOST_REQUIRE( extension(Af).start() == 1 );
	BOOST_REQUIRE( extension(Af).finish() == 11 );

	auto Af1 = multi::array<double, 1>(10, 0.).reindex(1);

	multi::array<double, 1> B({{0, 10}}, 0.);
	B[0] = 1.;
	B[1] = 2.;
	B[2] = 3.;

	BOOST_REQUIRE( size(B) == 10 );
	BOOST_REQUIRE( B != Af );
	BOOST_REQUIRE( std::equal(begin(Af), end(Af), begin(B)) );

	BOOST_REQUIRE( Af.reindexed(0) == B );
}

BOOST_AUTO_TEST_CASE(one_based_2D){

	multi::array<double, 2> Af({{1, 1 + 10}, {1, 1 + 20}}, 0.);
	Af[1][1] = 1.;
	Af[2][2] = 2.;
	Af[3][3] = 3.;

	BOOST_REQUIRE( Af[1][1] = 1. );
	BOOST_REQUIRE( *Af.data_elements() == 1. );
	BOOST_REQUIRE( size(Af) == 10 );
	BOOST_REQUIRE( extension(Af).start()  ==  1 );
	BOOST_REQUIRE( extension(Af).finish() == 11 );

	auto Af1 = multi::array<double, 2>({10, 10}, 0.).reindex(1, 1);

	multi::array<double, 2> B({{0, 10}, {0, 20}}, 0.);
	B[0][0] = 1.;
	B[1][1] = 2.;
	B[2][2] = 3.;

	BOOST_REQUIRE( size(B) == 10 );
	BOOST_REQUIRE( B != Af );
	BOOST_REQUIRE( std::equal(begin(Af.reindexed(0, 0)), end(Af.reindexed(0, 0)), begin(B)) );
	BOOST_REQUIRE( std::equal(begin(Af), end(Af), begin(B.reindexed(1, 1))) );
	BOOST_REQUIRE( std::equal(begin(Af), end(Af), begin(B.reindexed(0, 1))) );
	
	BOOST_REQUIRE( Af.reindexed(0, 0) == B );

}

