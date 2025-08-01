#define BOOST_TEST_MODULE "C++ Unit Tests for Multi HIP thrust"
// #include<boost/test/unit_test.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(vector){
	// H has storage for 4 integers
	thrust::host_vector<int> H(4);

	// initialize individual elements
	H[0] = 14;
	H[1] = 20;
	H[2] = 38;
	H[3] = 46;

	// H.size() returns the size of vector H
	BOOST_TEST_REQUIRE( H.size() == 4 );

	// print contents of H
	BOOST_TEST_REQUIRE( H[2] == 38 );

	// resize H
	H.resize(2);

	BOOST_REQUIRE( H.size() == 2 );

	// Copy host_vector H to device_vector D
	thrust::device_vector<int> D = H;

//  f(D.data());

	// elements of D can be modified
	D[0] = 99;
	D[1] = 88;

	thrust::device_ptr<int> p = D.data();  // this works with rocm hip 5.6
//  thrust::pointer<int, thrust::detail::seq_t> p = D.data();  // this works with rocm hip 5.6

	BOOST_REQUIRE( p[0] == 99 );

	BOOST_TEST_REQUIRE( D[1] == 88 );
}
