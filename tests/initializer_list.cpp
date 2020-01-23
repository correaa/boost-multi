#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wfatal-errors $0 -o$0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi initializer_list"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include<complex>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_tests_initializer_list){
{
	auto il = {1.2, 3.4, 5.6};
	multi::static_array<double, 1> const A(begin(il), end(il));
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( A[2] == il.begin()[2] );
}
{
	multi::static_array<double, 1> const A = {1.2, 3.4, 5.6};
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( A[2] == 5.6 );
	BOOST_REQUIRE(( A == multi::static_array<double, 1>{1.2, 3.4, 5.6} ));
	BOOST_REQUIRE(( A == decltype(A){1.2, 3.4, 5.6} ));
}
{
#if __cpp_deduction_guides
	multi::static_array const A = {1.2, 3.4, 5.6};
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( A[2] == 5.6 );
	BOOST_REQUIRE(( A == multi::static_array{1.2, 3.4, 5.6} ));
#endif
}
{
	auto il = {1.2, 3.4, 5.6};
	multi::array<double, 1> const A(il.begin(), il.end());
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( A[2] == 5.6 );
}
{
	multi::array<double, 1> const A = {1.2, 3.4, 5.6};
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( A[2] == 5.6 );
	BOOST_REQUIRE(( A == multi::array<double, 1>{1.2, 3.4, 5.6} ));
	BOOST_REQUIRE(( A == decltype(A){1.2, 3.4, 5.6} ));
}
{
#if __cpp_deduction_guides
	multi::array const A = {1.2, 3.4, 5.6};
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( A[2] == 5.6 );
	BOOST_REQUIRE(( A == multi::array{1.2, 3.4, 5.6} ));
#endif
}
{
	double const a[3] = {1.1, 2.2, 3.3};
	using multi::num_elements;
	BOOST_REQUIRE( num_elements(a) == 3 );

	using std::begin; using std::end;
	multi::static_array<double, 1> const A(begin(a), end(a));
	BOOST_REQUIRE( size(A) == 3 );
}
#if 0
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
	multi::static_array<double, 1> const A = (double[3])// warning: ISO C++ forbids compound-literals [-Wpedantic]
		{1.1, 2.2, 3.3}
	;
#pragma GCC diagnostic pop
	assert( size(A)==3 and A[1] == 2.2 );
}
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
	multi::static_array<double, 1> const A = (double[])// warning: ISO C++ forbids compound-literals [-Wpedantic]
		{1.1, 2.2, 3.3}
	;
#pragma GCC diagnostic pop
	assert( size(A)==3 and A[1] == 2.2 );
}
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
	multi::array 
#if not __cpp_deduction_guides
		<double, 1>
#endif
const A = 
(double[])// warning: ISO C++ forbids compound-literals [-Wpedantic]
		{1.1, 2.2, 3.3}
	;
#pragma GCC diagnostic pop
	assert( size(A)==3 and A[1] == 2.2 );
}
#endif
{
	std::array<double, 3> a = {1.1, 2.2, 3.3};
	multi::array<double, 1> const A(begin(a), end(a));
	BOOST_REQUIRE(( A == decltype(A){1.1, 2.2, 3.3} ));
}
{
#if __cpp_deduction_guides
	std::array a = {1.1, 2.2, 3.3};
	multi::array<double, 1> const A(begin(a), end(a));
	assert(( A == decltype(A){1.1, 2.2, 3.3} ));
#endif
}
{
	multi::static_array<double, 2> const A = {
		{ 1.2,  2.4, 3.6, 8.9},
		{11.2, 34.4, 5.6, 1.1},
		{15.2, 32.4, 5.6, 3.4}
	};
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( size(A[0]) == 4 );
	BOOST_REQUIRE(( A == decltype(A)
		{
			{ 1.2,  2.4, 3.6, 8.9},
			{11.2, 34.4, 5.6, 1.1},
			{15.2, 32.4, 5.6, 3.4}
		}
	));
}
{
	multi::array<double, 2> const A = {
		{ 1.2,  2.4, 3.6},
		{11.2, 34.4, 5.6},
		{15.2, 32.4, 5.6}
	};
	BOOST_REQUIRE( size(A) == 3 );
	BOOST_REQUIRE( size(A) == 3 and size(A[0]) == 3 );
	BOOST_REQUIRE( A[1][1] == 34.4 );
}
{
	double const a[3][2] = {
		{ 1.2,  2.4},
		{11.2, 34.4},
		{15.2, 32.4}
	};
	using std::begin; using std::end;
	multi::static_array<double, 2> A(begin(a), end(a));
}
{
	double const a[3][2] = {
		{ 1.2,  2.4},
		{11.2, 34.4},
		{15.2, 32.4}
	};
	using std::begin; using std::end;
	multi::static_array<double, 2> A(begin(a), end(a));
}
{
	double const staticA[3][2] = 
		{
			{ 1.2,  2.4},
			{11.2, 34.4},
			{15.2, 32.4}
		}
	;
	multi::static_array<double, 2> const A(std::begin(staticA), std::end(staticA));
	BOOST_REQUIRE(( A == multi::static_array<double, 2>{
			{ 1.2,  2.4},
			{11.2, 34.4},
			{15.2, 32.4}
		}
	));
}
#if 0
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
	multi::array<double, 2> A = 
		(double const[][2]) // warns with -Wpedantic
		{
			{ 1.2,  2.4},
			{11.2, 34.4},
			{15.2, 32.4}
		}
	;
#pragma GCC diagnostic pop
	assert( size(A) == 3 );
}
#endif
{
	multi::array<double, 2> A = 
#if defined(__INTEL_COMPILER)
		(double const[3][4])
#endif
		{
			{ 1.2,  2.4},
			{11.2, 34.4},
			{15.2, 32.4}
		}
	;
}
{
	std::array<std::array<double, 2>, 3> a = {{
		{{1.,2.}},
		{{2.,4.}},
		{{3.,6.}}
	}};
	multi::array<double, 2> A(begin(a), end(a));
	BOOST_REQUIRE( num_elements(A) == 6 and A[2][1] == 6. );
}
{
	multi::array<double, 3> const A = 
		{
			{
				{ 1.2, 0.}, 
				{ 2.4, 1.}
			},
			{
				{11.2,  3.}, 
				{34.4,  4.}
			},
			{
				{15.2, 99.}, 
				{32.4,  2.}
			}
		}
	;
	BOOST_REQUIRE( A[1][1][0] == 34.4 and A[1][1][1] == 4.   );
}
{
	using complex = std::complex<double>;
	constexpr complex I(0.,1.);
	multi::array<complex, 2> b = {
		{2. + 1.*I, 1. + 3.*I, 1. + 7.*I},
		{3. + 4.*I, 4. + 2.*I, 0. + 0.*I}
	};
}
{
	using std::string;
	multi::array<string, 3> B3 = {
		{ {"000", "001", "002"}, 
		  {"010", "011", "012"} },
		{ {"100", "101", "102"}, 
		  {"110", "111", "112"} }
	};
	BOOST_REQUIRE( num_elements(B3)==12 and B3[1][0][1] == "101" );
}
#if __cpp_deduction_guides
 {	multi::array A = {1., 2., 3.}; BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 and num_elements(A)==3 and A[1]==2. ); static_assert( typename decltype(A)::rank{}==1 );
}{	multi::array A = {1., 2.};     BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 and num_elements(A)==2 and A[1]==2. ); BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 );
}{	multi::array A = {0, 2};       BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 and num_elements(A)==2 and A[1]==2. ); BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 );
}{	multi::array A = {9.};         BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 and num_elements(A)==1 and A[0]==9. ); BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 );
}{	multi::array A = {9};          BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 and num_elements(A)==1 and A[0]==9. ); BOOST_REQUIRE( multi::rank<decltype(A)>{}==1 );
}{	multi::array A = {
		{1., 2., 3.}, 
		{4., 5., 6.}
	}; BOOST_REQUIRE( multi::rank<decltype(A)>{}==2 and num_elements(A)==6 );
}
#endif

}

