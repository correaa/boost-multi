#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXXX $CXXFLAGS $0 -o $0.$X `pkg-config --libs blas` -lboost_unit_test_framework&&$0.$X&&rm $0.$X;exit
#endif

#include "../../blas.hpp"
#include "../../../array.hpp"

#include<complex>
#include<cassert>

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS copy"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

namespace multi = boost::multi;
namespace blas = multi::blas;

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_copy_real){
	namespace blas = multi::blas;
	multi::array<double, 2> A = {
		{1.,  2.,  3.,  4.},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	BOOST_REQUIRE( A[0][2] == 3. );
	BOOST_REQUIRE( A[2][2] == 11. );

	blas::copy(A[0], A[2]);
	BOOST_REQUIRE( A[0][2] == 3. );
	BOOST_REQUIRE( A[2][2] == 3. );

//	multi::blas::copy(begin(A[1]), end(A[1]), begin(A[2])); // dcopy
	blas::copy( A[1]({0, size(A[1])}), A[2]({0, size(A[1])}) );
	BOOST_REQUIRE( A[1][3] == 8. );
	BOOST_REQUIRE( A[2][3] == 8. );

	multi::array<double, 1> AR3 = blas::copy(rotated(A)[3]); // dcopy
	BOOST_REQUIRE( AR3[1] == A[1][3] );
}

using complex = std::complex<double>; constexpr complex I{0, 1};

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_copy_complex){
	namespace blas = multi::blas;
	multi::array<complex, 2> A = {
		{1. + 3.*I,  2. + 4.*I,  3. + 5.*I,  4. + 6.*I},
		{5.,  6.,  7.,  8.},
		{9., 10., 11., 12.}
	};
	blas::copy(A[0], A[2]);
	BOOST_REQUIRE( A[0][2] == 3. + 5.*I );
}
//BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_copy_cuda_complex){
//	namespace cuda = multi::cuda;
//	namespace blas = multi::blas;
//	cuda::array<complex, 2> A = {
//		{1. + 3.*I,  2. + 4.*I,  3. + 5.*I,  4. + 6.*I},
//		{5.,  6.,  7.,  8.},
//		{9., 10., 11., 12.}
//	};

//	blas::copy(A[0], A[2]);
//	BOOST_REQUIRE( A[0][2] == 3. + 5.*I );
//	BOOST_REQUIRE( A[2][2] == 3. + 5.*I );
//}

//BOOST_AUTO_TEST_CASE(multi_adaptors_blas_test_copy_cuda_managed_complex){
//	namespace cuda = multi::cuda;
//	namespace blas = multi::blas;

//	cuda::managed::array<complex, 2> A = {
//		{1. + 3.*I,  2. + 4.*I,  3. + 5.*I,  4. + 6.*I},
//		{5.,  6.,  7.,  8.},
//		{9., 10., 11., 12.}
//	};
//	blas::copy(A[0], A[2]);
//	BOOST_REQUIRE( A[0][2] == 3. + 5.*I );
//	BOOST_REQUIRE( A[2][2] == 3. + 5.*I );
//}

BOOST_AUTO_TEST_CASE(multi_blas_copy){
	multi::array<double, 1> const A = {1., 2., 3., 4.};
	multi::array<double, 1> B = {5., 6., 7., 8.};
	blas::copy(A, B);
	BOOST_REQUIRE( B == A );
	
	B = blas::copy(A);
	BOOST_REQUIRE( B == A );
}

BOOST_AUTO_TEST_CASE(multi_blas_copy_complex){
	multi::array<std::complex<double>, 1> const A = {1., 2., 3., 4.};
	multi::array<std::complex<double>, 1> B = {5., 6., 7., 8.};
	blas::copy(A, B);
	BOOST_REQUIRE( B == A );
	
	B = blas::copy(A);
	BOOST_REQUIRE( B == A );
}

BOOST_AUTO_TEST_CASE(multi_blas_copy_context){
	multi::array<double, 1> const A = {1., 2., 3., 4.};
	multi::array<double, 1> B = {5., 6., 7., 8.};
	blas::context ctx;
	blas::copy(ctx, A, B);
	BOOST_REQUIRE( B == A );

	B = blas::copy(ctx, A);
}

BOOST_AUTO_TEST_CASE(multi_blas_copy_row){
	multi::array<double, 2> const A = {
		{1., 2., 3.},
		{4., 5., 6.},
		{7., 8., 9.}
	};
	multi::array<double, 1> B(3);
	blas::copy(rotated(A)[0], B);
	BOOST_REQUIRE( B == rotated(A)[0] );
}

