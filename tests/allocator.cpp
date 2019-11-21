#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi allocators"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include "../../multi/memory/stack.hpp"

#include<vector>
#include<complex>
#include<iostream>
#include<scoped_allocator>

namespace multi = boost::multi;
using std::cout;


BOOST_AUTO_TEST_CASE(array_allocators){
{
	multi::array<double, 3> AA({10, 20, 30}, 99.);
	BOOST_REQUIRE( AA[5][10][15] == 99. );
}
{
	std::vector<multi::array<double, 2>> VA;
	for(int i = 0; i != 10; ++i)
		VA.emplace_back(multi::index_extensions<2>{i, i}, i);
	BOOST_REQUIRE( size(VA[8]) == 8 );
	BOOST_REQUIRE( VA[2][0][0] == 2 );
}
{
	std::vector<multi::array<double, 2>> VA(33);
	for(int i = 0; i != 33; ++i)
		VA[i] = multi::array<double, 2>({i + 1, i + 1}, 99.);
	BOOST_REQUIRE( size(VA[4]) == 5 );
	BOOST_REQUIRE( VA[4][1][1] == 99. );
}
{
	std::vector<multi::array<double, 2>> VA(33);
	for(int i = 0; i != 33; ++i)
		VA[i] = multi::array<double, 2>({i, i}, 99.);
	BOOST_REQUIRE( size(VA[4]) == 4 );
	BOOST_REQUIRE( VA[4][1][1] == 99. );
}
{
	multi::array<double, 3> A{};
	BOOST_REQUIRE( size(A) == 0 );
	multi::array<double, 3> B{};
	BOOST_REQUIRE( size(B) == 0 );
}
{
	multi::array<multi::array<double, 3>, 2> AA({10, 20}, multi::array<double, 3>{});
	for(int i = 0; i != 10; ++i)
		for(int j = 0; j != 20; ++j)
			AA[i][j] = multi::array<double, 3>({i+j, i+j, i+j}, 99.);
	BOOST_REQUIRE( size(AA[9][19]) == 9 + 19 );
	BOOST_REQUIRE( AA[9][19][1][1][1] == 99. );
}
{
	multi::array<multi::array<double, 3>, 2> AA({10, 20}, multi::array<double, 3>{});
	BOOST_REQUIRE( size(AA[9][19]) == 0 );
	for(int i = 0; i != 10; ++i)
		for(int j = 0; j != 20; ++j)
			AA[i][j] = multi::array<double, 3>({i+j, i+j, i+j}, 99.);
	BOOST_REQUIRE( size(AA[9][19]) == 9 + 19 );
	BOOST_REQUIRE( AA[9][19][1][1][1] == 99. );
}
{
	multi::array<multi::array<double, 3>, 2> AA({100, 200});
	AA[9][19] = multi::array<double, 3>{};
//	return;
	BOOST_REQUIRE( size(AA[9][19]) == 0 );
	BOOST_REQUIRE( size(AA[8][18]) == 0 ); // no, cannot check because multi::array is only partially formed

	for(int i = 0; i != 10; ++i)
		for(int j = 0; j != 20; ++j)
			AA[i][j] = multi::array<double, 3>({i+j, i+j, i+j}, 99.);
	BOOST_REQUIRE( size(AA[9][19]) == 9 + 19 );
	BOOST_REQUIRE( AA[9][19][1][1][1] == 99. );
}
{
	multi::array<multi::array<double, 3>, 2> AA({10, 20});
	for(int i = 0; i != 10; ++i)
		for(int j = 0; j != 20; ++j)
			AA[i][j] = multi::array<double, 3>({i+j, i+j, i+j}, 99.);
	BOOST_REQUIRE( size(AA[9][19]) == 9+19 );
	BOOST_REQUIRE( AA[9][19][1][1][1] == 99. );
}
}

