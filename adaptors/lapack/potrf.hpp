#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -std=c++17 -Wall -Wextra -Wpedantic  -D_TEST_MULTI_ADAPTORS_LAPACK_POTRF -DADD_ $0.cpp -o $0x `pkg-config --libs blas lapack` -lboost_unit_test_framework &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_LAPACK_POTRF_HPP
#define MULTI_ADAPTORS_LAPACK_POTRF_HPP

#include "../../../multi/array.hpp"
#include "../lapack/core.hpp"
#include "../blas/numeric.hpp"

namespace boost{
namespace multi{
namespace lapack{

/*
template<class T>
struct uhermitian : public multi::array<T, 2>{
	static auto uplo(){return 'U';}
//	using multi::array<T, 2>::array;
	uhermitian(multi::array<T, 2>&& ma) : multi::array<T, 2>{std::move(ma)}{}
	decltype(auto) operator()(index i, index j) const{
		return multi::array<T, 2>::operator[](std::min(i, j))[std::max(i, j)];
	}
};*/

/*
template<class T>
struct utriangular : public multi::array<T, 2>{
	utriangular(multi::array<T, 2>&& ma) : multi::array<T, 2>{std::move(ma)}{}
	decltype(auto) operator()(index i, index j) const{
		if(i > j) return 0;
		return multi::array<T, 2>::operator[](i)[j];
	}	
};*/

}}}

#if 1
#include "../lapack/core.hpp"
#include "../blas/operations.hpp"
#include<cassert>

namespace boost{
namespace multi{

namespace lapack{

using triangular = blas::triangular;

template<class Iterator>
Iterator potrf(triangular t, Iterator first, Iterator last){
	assert( stride(first) == stride(last) );
	assert( first->stride() == 1 );
	auto n = std::distance(first, last);
	auto lda = stride(first);
	int info;
	::potrf(static_cast<char>(t), n, base(first), lda, info);
	assert( info >= 0 );
	return info==0?last:first + info;
}

template<class A2D>
decltype(auto) potrf(triangular t, A2D&& A){
	if(stride(A)==1){
		auto last = potrf(flip(t), begin(rotated(A)), end(rotated(A)));
		return A({0, distance(begin(rotated(A)), last)}, {0, distance(begin(rotated(A)), last)});
	}
	auto last = potrf(t, begin(A), end(A));
	using std::distance;
	return A({0, distance(begin(A), last)}, {0, distance(begin(A), last)});
}

template<class A2D>
#if __cplusplus>=201703L
#if __has_cpp_attribute(nodiscard)>=201603
[[nodiscard
#if __has_cpp_attribute(nodiscard)>=201907
("result is returned because third argument is const")
#endif
]]
#endif
#endif 
decltype(auto) potrf(triangular t, A2D const& A){
	auto ret = decay(A);
	auto last = potrf(t, ret); assert( size(last) == size(ret) );
	return ret;
}

template<class A2D>
decltype(auto) potrf(A2D&& A){
	return potrf(blas::detect_triangular(A), A);
}

}}}
#endif

#if _TEST_MULTI_ADAPTORS_LAPACK_POTRF

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi lapack adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<iostream>

namespace multi = boost::multi;
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
	multi::array<complex, 2> A =
		{{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN, 167.381, 126.746 + 0.0327519*I},
		 {NAN, NAN , 167.231}}
	;
	using multi::lapack::triangular;
	using multi::lapack::potrf;
	potrf(triangular::upper, begin(A), end(A)); // A is hermitic in the upper triangular (implicit hermitic below)
	print(A);
	BOOST_TEST( real(A[1][2]) == 3.78646 );
	BOOST_TEST( imag(A[1][2]) == 0.0170734 );
}
{
	multi::array<complex, 2> A =
		{{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN, 167.381, 126.746 + 0.0327519*I},
		 {NAN, NAN , 167.231}}
	;
	using multi::lapack::triangular;
	using multi::lapack::potrf;
	potrf(triangular::upper, A); // A is hermitic in the upper triangular (implicit hermitic below)
	print(A);
	
}
{
	multi::array<complex, 2> A =
		{{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN, 167.381, 126.746 + 0.0327519*I},
		 {NAN, NAN , 167.231}}
	;
	multi::array<complex, 2> At = rotated(A);
	auto&& Att = rotated(At);
	using multi::lapack::triangular;
	using multi::lapack::potrf;
	print(Att);
	potrf(triangular::upper, Att); // A is hermitic in the upper triangular (implicit hermitic below)
	print(Att);
}
{
	multi::array<complex, 2> A =
		{{167.413, 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {NAN, 167.381, 126.746 + 0.0327519*I},
		 {NAN, NAN , 167.231}}
	;
	using multi::lapack::triangular;
	using multi::lapack::potrf;
	potrf(A); // A is hermitic in the upper triangular (implicit hermitic below)
	print(A);
}

//	multi::lapack::uhermitian<complex> H{std::move(A)};
//	multi::lapack::potrf(H);
//	multi::lapack::utriangular<complex> 
//	assert( &H(1, 2) == &H(2, 1) );

#if 0
	multi::array<complex, 2> A =
		{{167.413               , 126.804 - 0.00143505*I, 125.114 - 0.1485590*I},
		 {126.804 + 0.00143505*I, 167.381               , 126.746 + 0.0327519*I},
		 {125.114 + 0.14855900*I, 126.746 - 0.0327519*I , 167.231              }}
	;
	multi::lapack::uhermitian<complex> H{std::move(A)};
	assert( &H(1, 2) == &H(2, 1) );
#endif
/*
	multi::array<complex, 2> A = 
	{
		{3.23+0.00*I, 1.51-1.92*I, 1.90+0.84*I,   0.42+2.50*I},
		{1.51+1.92*I, 3.58+0.00*I, -0.23+1.11*I, -1.18+1.37*I},
		{1.90-0.84*I, -0.23-1.11*I, 4.09+0.00*I,  2.33-0.14*I},
		{0.42-2.50*I, -1.18-1.37*I, 2.33+0.14*I,  4.29+0.00*I}
	};
*/
//	multi::lapack::uhermitian<complex> H = std::move(A);
}

#endif
#endif

