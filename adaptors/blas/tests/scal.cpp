#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x `pkg-config --cflags --libs blas` -Wno-deprecated-declarations -lboost_unit_test_framework -lcudart&&$0x&&rm $0x;exit
#endif

#include "../../blas.hpp"
#include "../../blas/numeric.hpp"
//#include "../../../array.hpp"
#include "../../../adaptors/cuda.hpp"

#include<complex>
#include<cassert>

using std::cout;
namespace multi = boost::multi;

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS copy"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_real){
	multi::array<double, 2> A = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	BOOST_REQUIRE( A[0][2] == 3. and A[2][2] == 11. );
	multi::blas::scal(2., A[2]); // dscal
	BOOST_REQUIRE( A[0][2] == 3. and A[2][2] == 11.*2. );
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_complex_real_case){
	using complex = std::complex<double>;
	multi::array<complex, 2> A = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	BOOST_TEST( A[0][2] == 3. );
	BOOST_TEST( A[2][2] == 11. );

	multi::blas::scal(2., A[2]); // zscal (2. is promoted to complex later)
	BOOST_TEST( A[0][2] == 3. );
	BOOST_REQUIRE( A[2][2] == 11.*2. );

	multi::blas::scal(1./2, A[2]); // zdscal
	BOOST_TEST( A[0][2] == 3. );
	BOOST_TEST( A[2][1] == 10. );
	BOOST_TEST( A[2][2] == 11. );

	multi::blas::scal(2., begin(A[2]), begin(A[2]) + 2);
	BOOST_TEST( A[0][2] == 3. );
	BOOST_TEST( A[2][1] == 20. );
	BOOST_TEST( A[2][2] == 11. );
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_complex){
	using complex = std::complex<double>;
	constexpr complex I(0, 1);
	multi::array<complex, 2> A = {
		{1. + 2.*I, 2. + 3.*I, 3. + 4.*I, 4. + 5.*I},
		{5. + 2.*I, 6. + 3.*I, 7. + 4.*I, 8. + 5.*I},
		{1. + 1.*I, 2. + 2.*I, 3. + 3.*I, 4. + 4.*I}
	};
	using multi::blas::scal;
	scal(2., A[1]); // zscal (2. is promoted to complex later)
	BOOST_TEST( A[1][2] == 14. + 8.*I );

	scal(3.*I, A[0]);
	BOOST_TEST( A[0][1] == (2. + 3.*I)*3.*I );

	using multi::blas::imag;
	scal(2., imag(A[2]));
	assert( A[2][1] == 2. + 4.*I );
}

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_scal_cuda){
	namespace cuda = multi::cuda;
	using complex = std::complex<double>;
	constexpr complex I(0, 1);
	cuda::managed::array<complex, 2> A = {
		{1. + 2.*I, 2. + 3.*I, 3. + 4.*I, 4. + 5.*I},
		{5. + 2.*I, 6. + 3.*I, 7. + 4.*I, 8. + 5.*I},
		{1. + 1.*I, 2. + 2.*I, 3. + 3.*I, 4. + 4.*I}
	};
	using multi::blas::scal;
	scal(2., A[1]); // zscal (2. is promoted to complex later)
	BOOST_TEST( A[1][2] == 14. + 8.*I );

//	cuda::managed::array<double, 1> a = {1., 2., 3.};
//	scal(2., a);
	using multi::blas::imag;
//	imag(A);
	{
		namespace cuda = multi::cuda;
		cuda::managed::array<complex, 2> Bgpu = {
			{1.+1.*I, 2.+2.*I}, 
			{3.+3.*I, 4.+4.*I}
		};
		using multi::blas::imag;
		assert( imag(Bgpu)[1][1] == imag(Bgpu[1][1]) );
		scal(2., imag(Bgpu[1]));
		assert( imag(Bgpu[1][1])==8. );
	}
//	what(imag(A[2]));
}

