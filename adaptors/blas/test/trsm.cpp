#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 -o $0x -lcudart -lcublas `pkg-config --libs blas` -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS trsm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

//#include "../../../memory/adaptors/cuda/managed/ptr.hpp"

#include "../../../adaptors/blas.hpp"
//#include "../../../adaptors/blas/cuda.hpp"

//#include "../../../adaptors/cuda.hpp"
#include "../../../array.hpp"

namespace multi = boost::multi;

template<class M> decltype(auto) print(M const& C){
	using multi::size; using std::cout;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j) cout<< C[i][j] <<' ';
		cout<<std::endl;
	}
	return cout<<std::endl;
}

namespace utf = boost::unit_test;
using complex = std::complex<double>; constexpr complex I{0, 1};

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_0x0, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<double, 2> const A;
	{
		multi::array<double, 2> B;
		blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_1x1, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<double, 2> const A = {
		{10.,},
	};
	{
		multi::array<double, 2> B = {
			{3.,},
		};
		blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[0][0] == 3./10. );
	}
	{
		multi::array<double, 2> B = {
			{3.,},
		};
		blas::trsm(blas::U, blas::diagonal::general, 2., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[0][0] == 2.*3./10. );
	}
	{
		multi::array<double, 2> B = {
			{3., 4., 5.},
		};
		blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[0][1] == 4./10. );
	}
	{
		multi::array<double, 2> B = {
			{3., 4., 5.},
		};
		multi::array<double, 2> Bcpy = blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( Bcpy[0][1] == 4./10. );
	}
	{
		multi::array<double, 2> B = {
			{3., 4., 5.},
		};
		BOOST_TEST( blas::trsm(blas::U, blas::diagonal::general, 1., A, B)[0][1] == 4./10. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex, *utf::tolerance(0.00001)){
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	namespace blas = multi::blas;
	using blas::filling;
	using blas::hermitized;
	trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST_REQUIRE( real(B[1][2]) == 2.33846 );
	BOOST_TEST_REQUIRE( imag(B[1][2]) == -0.0923077 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_rectangular, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I, 3. + 2.*I},
		{2. - 2.*I, 7. - 2.*I},
		{3. + 1.*I, 4. + 8.*I}
	};
	blas::trsm(blas::filling::lower, 2.+1.*I, blas::H(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST_REQUIRE( real(B[2][0]) == -4.16471 );
	BOOST_TEST_REQUIRE( imag(B[2][0]) ==  8.25882 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_column, *utf::tolerance(0.00001)){
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I},
		{2. - 2.*I},
		{3. + 1.*I}
	};
	namespace blas = multi::blas;
	blas::trsm(blas::filling::lower, 2.+1.*I, blas::H(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST_REQUIRE( real(B[2][0]) == -4.16471 );
	BOOST_TEST_REQUIRE( imag(B[2][0]) ==  8.25882 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_column_cpu, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. - 9.*I},
		{2. - 2.*I},
		{3. + 1.*I}
	};
	blas::trsm(blas::filling::lower, 2.+1.*I, blas::H(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST_REQUIRE( real(B[2][0]) == -4.16471 );
	BOOST_TEST_REQUIRE( imag(B[2][0]) ==  8.25882 );
}


BOOST_AUTO_TEST_CASE(multi_blas_trsm_real_square, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{999999.,  7.,  1.},
		{999999., 999999.,  8.}
	};
	{
		multi::array<double, 2> B = {
			{1., 3., 4.},
			{2., 7., 1.},
			{3., 4., 2.}
		};
		blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[1][2] == 0.107143 );
	}
	{
		multi::array<double, 2> AT = rotated(A);
		multi::array<double, 2> B = {
			{1., 3., 4.},
			{2., 7., 1.},
			{3., 4., 2.}
		};
		blas::trsm(blas::U, blas::diagonal::general, 1., blas::T(AT), B);
		BOOST_TEST( B[1][2] == 0.107143 );
	}
	{
		multi::array<double, 2> AT = rotated(A);
		multi::array<double, 2> B = {
			{1., 3., 4.},
			{2., 7., 1.},
			{3., 4., 2.}
		};
		multi::array<double, 2> BT = rotated(B);
		blas::trsm(blas::U, blas::diagonal::general, 1., blas::T(AT), blas::T(BT));
		BOOST_TEST( rotated(BT)[1][2] == 0.107143 );
	}
	{
		multi::array<double, 2> AT = rotated(A);
		multi::array<double, 2> B = {
			{1., 3., 4.},
			{2., 7., 1.},
			{3., 4., 2.}
		};
		multi::array<double, 2> BT = rotated(B);
		blas::trsm(blas::U, blas::diagonal::general, 1., A, blas::T(BT));
		BOOST_TEST( rotated(BT)[1][2] == 0.107143 );
	}
}


BOOST_AUTO_TEST_CASE(multi_blas_trsm_real_nonsquare, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{ 0.,  7.,  1.},
		{ 0.,  0.,  8.}
	};
	{
		multi::array<double, 2> B = {
			{1., 3., 4., 8.},
			{2., 7., 1., 9.},
			{3., 4., 2., 1.},
		};
		multi::array<double, 2> BT = rotated(B);
		blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[1][2] == 0.107143 );

		blas::trsm(blas::U, blas::diagonal::general, 1., A, blas::T(BT));
		BOOST_TEST( rotated(BT)[1][2] == 0.107143 );
	}
	{
		multi::array<double, 2> B = {
			{1., 3., 4., 8.},
			{2., 7., 1., 9.},
			{3., 4., 2., 1.},
		};
		multi::array<double, 2> AT = rotated(A);
		multi::array<double, 2> BT = rotated(B);
		blas::trsm(blas::U, blas::diagonal::general, 1., blas::T(AT), B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[1][2] == 0.107143 );

		blas::trsm(blas::U, blas::diagonal::general, 1., blas::T(AT), blas::T(BT));
		BOOST_TEST( rotated(BT)[1][2] == 0.107143 );
	}
	{
		multi::array<double, 2> B = {
			{1.},
			{2.},
			{3.},
		};
		blas::trsm(blas::U, blas::diagonal::general, 1., A, B); // B=Solve(A.X=alpha*B, X) B=A⁻¹B, B⊤=B⊤.(A⊤)⁻¹, A upper triangular (implicit zeros below)
		BOOST_TEST( B[2][0] == 0.375 );
	}
	{
		multi::array<double, 2> B = {
			{1.},
			{2.},
			{3.},
		};
		multi::array<double, 2> BT = rotated(B);
		blas::trsm(blas::U, blas::diagonal::general, 1., A, blas::T(BT));
		BOOST_TEST( rotated(BT)[2][0] == 0.375 );
	}
}


BOOST_AUTO_TEST_CASE(multi_blas_trsm_real_nonsquare_default_diagonal_gemm_check, *utf::tolerance(0.00001)){
	namespace blas = multi::blas;
	multi::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{ 0.,  7.,  1.},
		{ 0.,  0.,  8.}
	};
	multi::array<double, 2> B = {
		{1.},
		{2.},
		{3.}
	};
	{
		auto S = blas::trsm(blas::U, blas::diagonal::general, 1., A, B);
	//	BOOST_REQUIRE( S[2][0] == 0.375 );

	//	auto Bck = blas::gemm(1., A, S);
	//	BOOST_REQUIRE( Bck[2][0] == 3. );

	//	for(int i{};i<3;++i)
	//		for(int j{};j<size(rotated(B));++j) 
	//			BOOST_CHECK_SMALL( Bck[i][j]-B[i][j] , 0.00001);
	}
	{
	//	multi::array<double, 2> const BT = rotated(B);
	//	auto Bck=blas::gemm(1., A, blas::trsm(blas::U, blas::diagonal::general, 1., A, blas::T(BT)));
	//	for(int i{};i<3;++i)
	//		for(int j{};j<size(rotated(B));++j) BOOST_CHECK_SMALL(Bck[i][j]-B[i][j], 0.00001);
	}
	{
	//	auto const AT = rotated(A);
	//	auto Bck = blas::gemm(1., blas::T(AT), trsm(blas::U, blas::diagonal::general, 1., blas::T(AT), B));
	//	for(int i{};i<3;++i)for(int j{};j<size(rotated(B));++j) BOOST_CHECK_SMALL(Bck[i][j]-B[i][j], 0.00001);
	}
	{
		auto const AT = rotated(A).decay();
		auto const BT = rotated(B).decay();
	//	auto const Bck = blas::gemm(1., A, blas::trsm(blas::U, blas::diagonal::general, 1., blas::T(AT), blas::T(BT)));
	//	for(int i{};i<3;++i)for(int j{};j<size(rotated(B));++j) BOOST_REQUIRE_SMALL(Bck[i][j]-B[i][j], 0.00001);
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_nonsquare_default_diagonal_hermitized_gemm_check_no_const, *utf::tolerance(0.00001)){
	using complex = std::complex<double>; complex const I{0, 1};
	multi::array<complex, 2> const A = {
		{ 1. + 4.*I,  3.,  4.- 10.*I},
		{ 0.,  7.- 3.*I,  1.},
		{ 0.,  0.,  8.- 2.*I}
	};
	multi::array<complex, 2> B = {
		{1. + 1.*I, 2. + 1.*I, 3. + 1.*I},
		{5. + 3.*I, 9. + 3.*I, 1. - 1.*I}
	};
	using multi::blas::trsm;
	using multi::blas::filling;
	using multi::blas::hermitized;
	trsm(filling::upper, A, hermitized(B)); // B†←A⁻¹.B†, B←B.A⁻¹†, B←(A⁻¹.B†)†
	BOOST_TEST( imag(B[1][2]) == -0.147059 );
}


BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_nonsquare_default_diagonal_hermitized_gemm_check, *utf::tolerance(0.00001)){
	using complex = std::complex<double>; complex const I{0, 1};
	multi::array<complex, 2> const A = {
		{ 1. + 4.*I,  3.,  4.- 10.*I},
		{ 0.,  7.- 3.*I,  1.},
		{ 0.,  0.,  8.- 2.*I}
	};
	namespace blas = multi::blas;
	{
		{
			multi::array<complex, 2> B = {
				{1. + 1.*I, 5. + 3.*I},
				{2. + 1.*I, 9. + 3.*I},
				{3. + 1.*I, 1. - 1.*I},
			};
			auto S = blas::trsm(blas::filling::lower, blas::diagonal::general, 1., blas::H(A), B); // S = A⁻¹†.B, S† = B†.A⁻¹
			BOOST_TEST( real(S[2][1]) == 1.71608  );
		}
		{
			multi::array<complex, 2> B = {
				{1. + 1.*I, 2. + 1.*I, 3. + 1.*I},
				{5. + 3.*I, 9. + 3.*I, 1. - 1.*I}
			};
			auto S =+ blas::trsm(blas::filling::upper, 1., A, blas::H(B)); // S = A⁻¹B†, S†=B.A⁻¹†, S=(B.A⁻¹)†, B <- S†, B <- B.A⁻¹†
			BOOST_TEST( imag(S[2][1]) == +0.147059 );
			BOOST_TEST( imag(B[1][2]) == -0.147059 );
		}
		{
			multi::array<complex, 2> B = {
				{1. + 1.*I, 2. + 1.*I, 3. + 1.*I},
				{5. + 3.*I, 9. + 3.*I, 1. - 1.*I}
			};
			auto S =+ blas::trsm(blas::filling::upper, 2., A, blas::H(B)); // S = A⁻¹B†, S†=B.A⁻¹†, S=(B.A⁻¹)†, B <- S†, B <- B.A⁻¹†
			BOOST_TEST( imag(S[2][1]) == +0.147059*2. );
			BOOST_TEST( imag(B[1][2]) == -0.147059*2. );
		}
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_real_1x1_check, *utf::tolerance(0.00001)){
	multi::array<double, 2> const A = {
		{ 4.},
	};
	namespace blas = multi::blas;
	{
		{
			multi::array<double, 2> B = {
				{5.},
			};
			auto S =+ blas::trsm(blas::filling::upper, blas::diagonal::general, 3., A, B);
			BOOST_REQUIRE( S[0][0] == 3.*5./4. );
		}
		{
			multi::array<double, 2> B = {
				{5.},
			};
			auto S =+ blas::trsm(blas::filling::upper, 1., A, B);
			BOOST_REQUIRE( S[0][0] == 1.*5./4. );
		}
		{
			multi::array<double, 2> B = {
				{5.},
			};
			auto S =+ blas::trsm(blas::filling::upper, A, B);
			BOOST_REQUIRE( S[0][0] == 1.*5./4. );
		}
	}
}

//BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_1x1_check, *utf::tolerance(0.00001)){
//	using complex = std::complex<double>; complex const I = complex{0, 1};
//	multi::array<complex, 2> const A = {
//		{ 4. + 2.*I},
//	};
//	namespace blas = multi::blas;
//	{
//		multi::array<complex, 2> B = {
//			{5. + 1.*I},
//		};
//		{
//			auto S = blas::trsm(blas::filling::upper, blas::diagonal::general, 3.+5.*I, A, B);
//			BOOST_TEST( real(S[0][0]) == real((3.+5.*I)*B[0][0]/A[0][0]) );
//			BOOST_TEST( imag(S[0][0]) == imag((3.+5.*I)*B[0][0]/A[0][0]) );
//		}
//	}
//}


//BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_column_cuda, *utf::tolerance(0.00001)){
//	namespace cuda = multi::cuda;
//	cuda::array<complex, 2> A = {
//		{ 1.,  3.,  4.},
//		{NAN,  7.,  1.},
//		{NAN, NAN,  8.}
//	};
////	multi::cuda::array<complex, 2> const B = {
////		{1.},
////		{2.},
////		{3.}
////	};
//	namespace blas = multi::blas;
////	auto Bcpy = blas::trsm(blas::filling::upper, 1., A, B); // B ⬅ α Inv[A].B, B† ⬅ B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
////	multi::array<complex, 2> Bcpu = Bcpy;
////	BOOST_TEST_REQUIRE( std::real(Bcpu[2][0]) == 0.375 );
////	BOOST_TEST_REQUIRE( std::imag(Bcpu[2][0]) == 0.    );
//}

#if 0



//template<class T> void what(T&&) = delete;



BOOST_AUTO_TEST_CASE(multi_blas_trsm_double_column_cuda, *utf::tolerance(0.00001)){
	multi::cuda::array<double, 2> const A = {
		{ 1.,  3.,  4.},
		{NAN,  7.,  1.},
		{NAN, NAN,  8.}
	};
	multi::cuda::array<double, 2> B = {
		{1.},
		{2.},
		{3.}
	};
	namespace blas = multi::blas;
	using blas::filling;	
	using blas::hermitized;
	trsm(filling::upper, 1., A, B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( B[2][0] == 0.375 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_column_cuda2, *utf::tolerance(0.00001)){
	multi::cuda::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::cuda::array<complex, 2> B = {
		{1. - 9.*I},
		{2. - 2.*I},
		{3. + 1.*I}
	};
	namespace blas = multi::blas;
	using blas::filling;	
	using blas::hermitized;
	trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	multi::array<complex, 2> Bcpu = B;
	BOOST_TEST( real(Bcpu[2][0]) == -4.16471 );
	BOOST_TEST( imag(Bcpu[2][0]) ==  8.25882 );
}

BOOST_AUTO_TEST_CASE(multi_blas_cuda_trsm_complex, *utf::tolerance(0.00001)){
	multi::cuda::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::cuda::array<complex, 2> const B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};

	namespace blas = multi::blas;
	using blas::filling;	
	using blas::hermitized;
//	auto C = trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	auto C = trsm(filling::lower, 1., hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit 
}

BOOST_AUTO_TEST_CASE(multi_blas_cuda_managed_trsm_complex, *utf::tolerance(0.00001)){
	multi::cuda::managed::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::cuda::managed::array<complex, 2> const B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};

	namespace blas = multi::blas;
	using blas::filling;
	using blas::hermitized;
	auto C = trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
}
#endif

