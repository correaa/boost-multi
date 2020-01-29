#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wfatal-errors $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2018-2020
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi allocators"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<iostream>
#include<vector>
#include "../array.hpp"

namespace multi = boost::multi;

template<class MA>
decltype(auto) take(MA&& ma){return ma[0];}

BOOST_AUTO_TEST_CASE(iterator_1d){
{
	multi::array<double, 1> A(100, 99.); 
	BOOST_REQUIRE( size(A) == 100 );
	BOOST_REQUIRE( begin(A) < end(A) );
	BOOST_REQUIRE( end(A) - begin(A) == size(A) );
}
{
	multi::array<double, 1> A({100}, 99.); 
	BOOST_REQUIRE( size(A) == 100 );
	BOOST_REQUIRE( begin(A) < end(A) );
}
}
BOOST_AUTO_TEST_CASE(iterator_2d){
{
	multi::array<double, 2> A({120, 140}, 99.); 
	BOOST_REQUIRE( size(A) == 120 );
	BOOST_REQUIRE( cbegin(A) < cend(A) );
	BOOST_REQUIRE( cend(A) - cbegin(A) == size(A) );
}
{
	multi::array<double, 2> A(120, multi::array<double, 1>(140, 123.)); 
	BOOST_REQUIRE( size(A) == 120 );
	BOOST_REQUIRE( std::get<1>(sizes(A)) == 140 );
	BOOST_REQUIRE( A[119][139] == 123. );
}
{
	std::vector<double> v(10000);
	multi::array_ref<double, 2> A(v.data(), {100, 100}); 
	BOOST_REQUIRE(size(A) == 100);
	begin(A)[4][3] = 2.; // ok 
	using multi::static_array_cast;
//	auto const& A_const = static_array_cast<double const>(A);
//	begin(A_const)[4][3] = 2.; // error, read only
}
{
	std::vector<double> V(10000);
	multi::array_ref<double, 2, std::vector<double>::iterator> A(begin(V), {100, 100}); 
	BOOST_REQUIRE(size(A) == 100);
//	begin(arr)[4][3] = 2.;
	A[1][0] = 99.;
	BOOST_REQUIRE( cbegin(A) < cend(A) );
	BOOST_REQUIRE( V[100] == 99. );
}
{
	multi::array<double, 3>::reverse_iterator rit;
	BOOST_REQUIRE(( rit.base() == multi::array<double, 3>::reverse_iterator{}.base() ));
	BOOST_REQUIRE(( multi::array<double, 3>::reverse_iterator{}.base() == multi::array<double, 3>::reverse_iterator{}.base() ));
	BOOST_REQUIRE(( multi::array<double, 3>::reverse_iterator{} == multi::array<double, 3>::reverse_iterator{} ));
	BOOST_REQUIRE(( multi::array<double, 3>::reverse_iterator{} == multi::array<double, 3>::reverse_iterator{} ));
}
{
}
}
#if 0
	return 0;

	multi::array<double, 3> A =
//	#if defined(__INTEL_COMPILER)
//		(double[3][2][2])
//	#endif
		{
			{
				{ 1.2,  1.1}, { 2.4, 1.}
			},
			{
				{11.2,  3.0}, {34.4, 4.}
			},
			{
				{ 1.2,  1.1}, { 2.4, 1.}
			}
		}
	;
	assert( begin(A) < end(A) );
	assert( cbegin(A) < cend(A) );
	assert( begin(A[0]) < end(A[0]) );

	multi::array<double, 1>::const_iterator i;
	assert( begin(A[0]) < end(A[0]) );

	assert( size(A) == 3 and size(A[0]) == 2 and size(A[0][0]) == 2 and A[0][0][1] == 1.1 );
	assert(( multi::array<double, 3>::reverse_iterator{A.begin()} == rend(A) ));

	assert( begin(A) < end(A) );
	assert( cbegin(A) < cend(A) );
//	assert( crbegin(A) < crend(A) );
//	assert( crend(A) > crbegin(A) );
	assert( end(A) - begin(A) == size(A) );
	assert( rend(A) - rbegin(A) == size(A) );

	assert( size(*begin(A)) == 2 );
	assert( size(begin(A)[1]) == 2 );
	assert( &(A[1][1].begin()[0]) == &A[1][1][0] );
	assert( &A[0][1][0] == &A[0][1][0] );
	assert( &((*A.begin())[1][0]) == &A[0][1][0] );
	assert( &((*A.begin()).operator[](1)[0]) == &A[0][1][0] );
	assert( &(A.begin()->operator[](1)[0]) == &A[0][1][0] );
	assert( &(A.begin()->operator[](1).begin()[0]) == &A[0][1][0] );
	assert( &((A.begin()+1)->operator[](1).begin()[0]) == &A[1][1][0] );
	assert( &((begin(A)+1)->operator[](1).begin()[0]) == &A[1][1][0] );
	assert( &((cbegin(A)+1)->operator[](1).begin()[0]) == &A[1][1][0] );

	multi::array<double, 3>::iterator it; assert(( it == multi::array<double, 3>::iterator{} ));
	--it;
	it = begin(A);                                    assert( it == begin(A) );
	multi::array<double, 3>::iterator it2 = begin(A); assert(it == it2);
	it = end(A);                                      assert(it != it2);
	assert(it > it2);
	multi::array<double, 3>::iterator it3{it};        assert( it3 == it );
	multi::array<double, 3>::const_iterator cit;
	cit = it3;                                        assert( cit == it3 );
	assert((begin(A) == multi::array<double, 3>::iterator{rend(A)}));
	{
		std::vector<double> vv = {1.,2.,3.};
		auto it = vv.begin();
		auto rit = vv.rend();
		assert(std::vector<double>::reverse_iterator{it} == rit);
	}
}
#endif

