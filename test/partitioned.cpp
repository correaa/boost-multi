#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo Correa 2018-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi partitioned operation"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include<experimental/tuple>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(array_partitioned_1d){
	multi::array<double, 1>	A1 = {0, 1, 2, 3, 4, 5};
	auto&& A2_ref = A1.partitioned(2);
	static_assert( std::decay<decltype(A2_ref)>::type::dimensionality == decltype(A1)::dimensionality+1 , "!");
	BOOST_REQUIRE( dimensionality(A2_ref)==dimensionality(A1)+1 );
	BOOST_REQUIRE( size(A2_ref)==2 );
	BOOST_REQUIRE( size(A2_ref[0])==3 );
	BOOST_REQUIRE( &A2_ref[1][0] == &A1[3] );
}

BOOST_AUTO_TEST_CASE(array_partitioned_2d){
	multi::array<double, 2>	A2 = 
		{
			{  0,  1,  2,  3,  4,  5}, 
			{  6,  7,  8,  9, 10, 11}, 

			{ 12, 13, 14, 15, 16, 17}, 
			{ 18, 19, 20, 21, 22, 23}, 
		}
	;
	auto&& A3_ref = A2.partitioned(2);
	BOOST_REQUIRE( dimensionality(A3_ref) == dimensionality(A2)+1 );
	BOOST_REQUIRE( num_elements(A3_ref) == num_elements(A2) );
	BOOST_REQUIRE( size(A3_ref)==2 );
	BOOST_REQUIRE( size(A3_ref[0])==2 );
	BOOST_REQUIRE( size(A3_ref[0][0])==6 );
	BOOST_REQUIRE( &A3_ref[1][1][0] == &A2[3][0] );
}

BOOST_AUTO_TEST_CASE(array_partitioned){
	multi::array<std::string, 2> A2 = 
		{
			{  "s0P0",  "s1P0"},
			{  "s0P1",  "s1P1"},
			{  "s0P2",  "s1P2"},
			{  "s0P3",  "s1P3"},
			{  "s0P4",  "s1P4"},
			{  "s0P5",  "s1P5"},
		}; 

	BOOST_REQUIRE(  size(A2) == 6 );
	BOOST_REQUIRE(( sizes(A2) == decltype(sizes(A2)){6, 2} ));

	BOOST_REQUIRE( size(A2.partitioned(3)) == 3 );
	BOOST_REQUIRE( dimensionality(A2.partitioned(3)) == 3 );

	BOOST_REQUIRE(( sizes(A2.partitioned(3)) == decltype(sizes(A2.partitioned(3))){3, 2, 2} ));
	
	BOOST_REQUIRE( size(A2.partitioned(1)) == 1 );
	BOOST_REQUIRE( dimensionality(A2.partitioned(1)) == 3 );
	BOOST_REQUIRE( &A2.partitioned(1).rotated()[3][1][0] == &A2[3][1] );
}

BOOST_AUTO_TEST_CASE(array_encoded_subarray){

	multi::array<double, 2> A = { // A[walker][encoded_property]
		{99, 99, 0.00, 0.01, 0.10, 0.11, 99},
		{99, 99, 1.00, 1.01, 1.10, 1.11, 99},
		{99, 99, 2.00, 2.01, 2.10, 2.11, 99},
		{99, 99, 3.00, 3.01, 3.10, 3.11, 99},
		{99, 99, 4.00, 4.01, 4.10, 4.11, 99},
		{99, 99, 5.00, 5.01, 5.10, 5.11, 99},
	};

	multi::iextension const encoded_2x2_range = {2, 6};
//	auto&& B = A(A.extension(), encoded_2x2_range).rotated().partitioned(2).unrotated();
	auto&& B = A.rotated()(encoded_2x2_range).partitioned(2).unrotated();

	BOOST_REQUIRE( dimensionality(B) == 3 );
	BOOST_TEST_REQUIRE( std::get<0>(sizes(B)) == 6 );
	BOOST_REQUIRE( std::get<1>(sizes(B)) == 2 );
	BOOST_REQUIRE( std::get<2>(sizes(B)) == 2 );
	BOOST_REQUIRE( &B[4][1][0] == &A[4][4] );
	BOOST_REQUIRE( B[4][1][0] == 4.10 );

	BOOST_REQUIRE((
		B[4] == multi::array<double, 2>{
			{4.00, 4.01},
			{4.10, 4.11}
		}
	));

	B[4][1][0] = 1111.;
	BOOST_REQUIRE( A[4][4] == 1111. );
}

BOOST_AUTO_TEST_CASE(array_partitioned_add_to_last){

	multi::array<double, 3>	A3 = {
		{
			{  0,  1,  2,  3,  4,  5}, 
			{  6,  7,  8,  9, 10, 11}, 
			{ 12, 13, 14, 15, 16, 17}, 
			{ 18, 19, 20, 21, 22, 23}, 
		},
		{
			{  0,  1,  2,  3,  4,  5}, 
			{  6,  7,  8,  9, 10, 11}, 
			{ 12, 13, 14, 15, 16, 17}, 
			{ 18, 19, 20, 21, 22, 23}, 
		}
	};

	auto strides = std::experimental::apply([](auto... e){return std::array<long, sizeof...(e)>{long{e}...};}, A3.strides());
//	auto const strides = std::apply([](auto... e){return std::array{long{e}...};}, A3.strides());

	BOOST_REQUIRE( std::is_sorted(strides.rbegin(), strides.rend()) and A3.num_elements() == A3.nelems() ); // contiguous c-ordering

	auto&& A4 = A3.reinterpret_array_cast<double>(1);
//	auto&& A4 = A3.partitioned(1).rotated();
//	auto&& A4 = A3.rotated().partitioned(size(A3<<1)).unrotated();

	BOOST_REQUIRE(( A3.extensions() == decltype(A3.extensions()){2, 4, 6} ));
	BOOST_REQUIRE(( A4.extensions() == decltype(A4.extensions()){2, 4, 6, 1} ));

	BOOST_REQUIRE( A4.is_flattable() );
	BOOST_REQUIRE( A4.flatted().is_flattable() );

	BOOST_REQUIRE( &A4[1][2][3][0] == &A3[1][2][3] );
}

