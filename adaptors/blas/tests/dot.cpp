#ifdef COMPILATION_INSTRUCTIONS
clang++ -Wall -Wextra -Wpedantic -Wfatal-errors $0 -o $0x `pkg-config --libs blas` -lcudart -lcublas -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS dot"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

//#include "../../blas.hpp"
#include "../../blas/dot.hpp"

#include "../../../array.hpp"
#include "../../blas/cuda.hpp"

#include "../../../adaptors/cuda.hpp"

#include<complex>
#include<cassert>
#include<numeric>

using std::cout;
namespace multi = boost::multi;
namespace blas = multi::blas;

BOOST_AUTO_TEST_CASE(blas_dot){
	{
		multi::array<double, 2> CA = {
			{1.,  2.,  3.,  4.},
			{5.,  6.,  7.,  8.},
			{9., 10., 11., 12.}
		};
		using blas::dot;
		auto d = dot(CA[1], CA[2]);
		assert( d() == std::inner_product(begin(CA[1]), begin(CA[2]), end(CA[1]), 0.) );
	}
	{
		multi::array<float, 1> const A = {1.,2.,3.};
		multi::array<float, 1> const B = {1.,2.,3.};
		using blas::dot;
		auto f = dot(A, B); // sdot
		assert( f() == std::inner_product(begin(A), end(A), begin(B), float{0}) );
	}
//	{
//		multi::array<float, 1> const A = {1.,2.,3.};
//		multi::array<float, 1> const B = {1.,2.,3.};
//		using multi::blas::dot;
//		auto d = dot<double>(A, B); // dsdot
//		assert( d == std::inner_product(begin(A), end(A), begin(B), double{0}) );
//	}
//	{
//		multi::array<float, 1> const A = {1.,2.,3.};
//		multi::array<float, 1> const B = {1.,2.,3.};
//		using multi::blas::dot;
//		auto d = dot<double>(begin(A) + 1, end(A), begin(B) + 1); // dsdot, mixed precision can be called explicitly
//		static_assert( std::is_same<decltype(d), double>{}, "!");
//		assert( d == std::inner_product(begin(A) + 1, end(A), begin(B) + 1, double{0}) );
//	}
	using complex = std::complex<double>;
	{
		using Z = std::complex<double>; Z const I{0, 1};
		multi::array<Z, 1> const A = {I, 2.*I, 3.*I};
		using blas::dot;
		assert( dot(A, A)() == std::inner_product(begin(A), end(A), begin(A), std::complex<double>(0)) );
	}
	{
		using Z = std::complex<double>; Z const I{0, 1};
		multi::array<Z, 1> const A = {I, 2.*I, 3.*I};
		using blas::dot;
		using blas::conjugated;
	//	std::cout << dot(A, conjugated(A)) << std::endl; // zdotc
		assert( dot(A, conjugated(A))() == std::inner_product(begin(A), end(A), begin(A), std::complex<double>(0), std::plus<>{}, [](auto&& a, auto&& b){return a*conj(b);}) );
	}
	{
		using Z = std::complex<double>; Z const I{0, 1};
		multi::array<Z, 1> const A = {I, 2.*I, 3.*I};
		using blas::dot; 
	//	std::cout << dot(A, A) << std::endl; // zdotc (dot defaults to dotc for complex)
		using blas::conjugated;
		assert( dot(A, conjugated(A))() == std::inner_product(begin(A), end(A), begin(A), std::complex<double>(0), std::plus<>{}, [](auto&& a, auto&& b){return a*conj(b);}) );
	}
	{
		multi::array<float, 1> const A = {1.,2.,3.};
		multi::array<float, 1> const B = {1.,2.,3.};
		using multi::blas::dot;
	//	auto f = dot(1.2, A, B); // sdsdot, 1.2 is demoted to 1.2f
	//	assert( f == 1.2f + std::inner_product(begin(A), end(A), begin(B), float{0}) );
	}
	{
	//	multi::array<double, 1> const A = {1.,2.,3.};
	//	multi::array<double, 1> const B = {1.,2.,3.};
	//	using multi::blas::dot;
	//	auto f = dot(1.2, A, B); // this strange function is only implement for floats
	}
	{
		complex const I{0, 1};
		namespace cuda = multi::cuda;
		cuda::array<complex, 1> const acu = {5. + 2.*I, 6. + 6.*I, 7. + 2.*I, 8. - 3.*I};
		cuda::array<complex, 1> const bcu = {5. + 2.*I, 6. + 6.*I, 7. + 2.*I, 8. - 3.*I};

		using blas::conjugated;
		using blas::dot;
		{
			cuda::array<complex, 0> ccu;
			dot(acu, bcu, ccu);
			BOOST_REQUIRE( ccu() == complex(121, 72) );
		}
		{
			cuda::array<complex, 0> ccu;
			dot(acu, conjugated(bcu), ccu);
			BOOST_REQUIRE( ccu() == complex(227, 0) );
		}
		{
			cuda::array<complex, 1> ccu = {1., 2.};
			dot(acu, conjugated(bcu), ccu[0]);
			BOOST_REQUIRE( ccu[0] == complex(227, 0) );
		}
	}

}

