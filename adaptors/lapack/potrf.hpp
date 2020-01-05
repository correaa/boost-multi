#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -std=c++17 -Wall -Wextra -Wpedantic  -D_TEST_MULTI_ADAPTORS_LAPACK_POTRF -DADD_ $0.cpp -o $0x `pkg-config --libs blas lapack` -lboost_unit_test_framework &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_LAPACK_POTRF_HPP
#define MULTI_ADAPTORS_LAPACK_POTRF_HPP

#include "../../../multi/array.hpp"
#include "../lapack/core.hpp"
#include "../blas/numeric.hpp"

#include "../lapack/core.hpp"
#include "../blas/side.hpp"
#include "../blas/filling.hpp"

#include<cassert>

namespace boost{
namespace multi{
namespace lapack{

using blas::filling;

template<class Iterator>
Iterator potrf(filling t, Iterator first, Iterator last){
	assert( stride(first) == stride(last) );
	assert( first->stride() == 1 );
	auto n = std::distance(first, last);
	auto lda = stride(first);
	int info;
	::potrf(static_cast<char>(t), n, base(first), lda, info);
	assert( info >= 0 );
	return info==0?last:first + info;
}

using blas::flip;

template<class A2D>
decltype(auto) potrf(filling t, A2D&& A){
	if(stride(A)==1){
		auto last = potrf(flip(t), begin(rotated(A)), end(rotated(A)));
		return A({0, distance(begin(rotated(A)), last)}, {0, distance(begin(rotated(A)), last)});
	}
	auto last = potrf(t, begin(A), end(A));
	using std::distance;
	return A({0, distance(begin(A), last)}, {0, distance(begin(A), last)});
}

#if __cplusplus>=201703L and __has_cpp_attribute(nodiscard)>=201603
#define NODISCARD(MsG) [[nodiscard]]
#elif __has_cpp_attribute(gnu::warn_unused_result)
#define NODISCARD(MsG) [[gnu::warn_unused_result]]
#else
#define NODISCARD(MsG)
#endif

template<class A2D> NODISCARD("result is returned because third argument is const")
decltype(auto) potrf(filling t, A2D const& A){
	auto ret = decay(A);
	auto last = potrf(t, ret); assert( size(last) == size(ret) );
	return ret;
}

template<class A2D>
decltype(auto) potrf(A2D&& A){
	return potrf(blas::detect_triangular(A), A);
}

}}}

#if _TEST_MULTI_ADAPTORS_LAPACK_POTRF

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi lapack adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<iostream>

namespace multi = boost::multi;
namespace lapack = multi::lapack;

using complex = std::complex<double>;

template<class M> decltype(auto) print(M const& C){
	using std::cout;
	using multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j) cout << C[i][j] << ' ';
		cout << std::endl;
	}
	return cout << std::endl;
}

BOOST_AUTO_TEST_CASE(lapack_potrf, *boost::unit_test::tolerance(0.00001) ){
	auto const I = complex(0.,1.);
{
	multi::array<complex, 2> A = {
		 {167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN    , 167.381               , 126.746 + 0.0327519*I},
		 {NAN    , NAN                   , 167.231              }
	};
	using lapack::filling;
	using lapack::potrf;
	potrf(filling::upper, begin(A), end(A));//A is hermitic upper triangular (implicit below)
	BOOST_TEST( real(A[1][2]) == 3.78646 );
	BOOST_TEST( imag(A[1][2]) == 0.0170734 );
	BOOST_TEST( A[2][1] != A[2][1] );
}
{
	multi::array<complex, 2> A = {
		{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		{NAN    , 167.381               , 126.746 + 0.0327519*I},
		{NAN    , NAN                   , 167.231              }
	};
	using multi::lapack::filling;
	using multi::lapack::potrf;
	potrf(filling::upper, A); // A is hermitic in upper triangular (implicit below)
	BOOST_TEST( real(A[1][2]) == 3.78646 );
	BOOST_TEST( imag(A[1][2]) == 0.0170734 );
	BOOST_TEST( A[2][1] != A[2][1] );
}
{
	multi::array<complex, 2> A =
		{{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN, 167.381, 126.746 + 0.0327519*I},
		 {NAN, NAN , 167.231}}
	;
	multi::array<complex, 2> At = rotated(A);
	auto&& Att = rotated(At);
	using lapack::filling;
	using lapack::potrf;
	print(Att);
	potrf(filling::upper, Att); // A is hermitic in the upper triangular (implicit hermitic below)
	print(Att);
	BOOST_TEST( real(Att[1][2]) == 3.78646 );
	BOOST_TEST( imag(Att[1][2]) == 0.0170734 );
	BOOST_TEST( Att[2][1] != Att[2][1] );
}
{
	multi::array<complex, 2> A =
		{{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN, 167.381, 126.746 + 0.0327519*I},
		 {NAN, NAN , 167.231}}
	;
	using lapack::potrf;
	potrf(A); // A is hermitic in the upper triangular (implicit hermitic below)
	BOOST_TEST( real(A[1][2]) == 3.78646 );
	BOOST_TEST( imag(A[1][2]) == 0.0170734 );
	BOOST_TEST( A[2][1] != A[2][1] );
}
{
	multi::array<complex, 2> A =
		{{190., 126., 125.},
		 {NAN , 111., 122.},
		 {NAN , NAN , 135.}}
	;
	using lapack::filling;
	using lapack::potrf;
	potrf(A); // A is hermitic in the upper triangular (implicit hermitic below)
	BOOST_TEST( A[2][1] != A[2][1] );
}

}

#endif
#endif

