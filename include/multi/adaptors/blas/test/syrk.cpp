// Copyright 2019-2024 Alfredo A. Correa

// #define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS scal"
#include <boost/test/unit_test.hpp>

#include <multi/adaptors/blas/syrk.hpp>

#include <multi/array.hpp>

BOOST_AUTO_TEST_CASE(dummy_test) {
}

namespace multi = boost::multi;
// namespace blas  = multi::blas;

// template<class M> decltype(auto) print(M const& C){
//   using boost::multi::size;
//   for(int i = 0; i != size(C); ++i){
//     for(int j = 0; j != size(C[i]); ++j)
//       std::cout << C[i][j] << ' ';
//     std::cout << std::endl;
//   }
//   return std::cout << std::endl;
// }

BOOST_AUTO_TEST_CASE(multi_blas_syrk_real) {
	// NOLINTNEXTLINE(readability-identifier-length)
	multi::array<double, 2> const a = {
		{1.0, 3.0, 4.0},
		{9.0, 7.0, 1.0},
	};
	{
		multi::array<double, 2> c({3, 3}, 9999.0);  // NOLINT(readability-identifier-length)
		namespace blas = multi::blas;

		using blas::filling;
		using blas::transposed;

		syrk(filling::lower, 1.0, transposed(a), 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_REQUIRE( c[2][1] ==   19.0 );
		BOOST_REQUIRE( c[1][2] == 9999.0 );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.0);  // NOLINT(readability-identifier-length)
		namespace blas = multi::blas;

		using blas::filling;
		using blas::transposed;

		syrk(filling::upper, 1.0, transposed(a), 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_REQUIRE( c[1][2] ==   19.0 );
		BOOST_REQUIRE( c[2][1] == 9999.0 );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.0);  // NOLINT(readability-identifier-length)
		namespace blas = multi::blas;

		using blas::filling;
		using blas::syrk;

		syrk(filling::lower, 1.0, a, 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_REQUIRE( c[1][0] ==   34.0 );
		BOOST_REQUIRE( c[0][1] == 9999.0 );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.0);  // NOLINT(readability-identifier-length)

		namespace blas = multi::blas;

		using blas::filling;

		syrk(filling::upper, 1.0, a, 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, a⸆a, `c` in lower triangular

		BOOST_REQUIRE( c[0][1] ==   34.0 );
		BOOST_REQUIRE( c[1][0] == 9999.0 );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.0);  // NOLINT(readability-identifier-length)

		namespace blas = multi::blas;

		using blas::filling;

		syrk(filling::upper, 1.0, a, 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, a⸆a, `c` in lower triangular

		BOOST_REQUIRE( c[0][1] ==   34.0 );
		BOOST_REQUIRE( c[1][0] == 9999.0 );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_real_special_case) {
	// NOLINTNEXTLINE(readability-identifier-length)
	multi::array<double, 2> const a = {
		{1.0, 3.0, 4.0},
	};
	{
		multi::array<double, 2> c({1, 1}, 9999.0);  // NOLINT(readability-identifier-length)

		namespace blas = multi::blas;
		using blas::filling;

		syrk(filling::lower, 1.0, a, 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_TEST( c[0][0] == 1.0*1.0 + 3.0*3.0 + 4.0*4.0 );
	}
	{
		multi::array<double, 2> c({1, 1}, 9999.0);  // NOLINT(readability-identifier-length)

		namespace blas = multi::blas;
		using blas::filling;

		syrk(filling::upper, 1.0, a, 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_TEST( c[0][0] == 1.0*1.0 + 3.0*3.0 + 4.0*4.0 );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_complex_real_case) {
	using complex = std::complex<double>;
	auto const I  = complex{0.0, 1.0};  // NOLINT(readability-identifier-length) imag unit

	// NOLINTNEXTLINE(readability-identifier-length)
	multi::array<complex, 2> const a = {
		{1.0 + I * 0.0, 3.0 + I * 0.0, 4.0 + I * 0.0},
		{9.0 + I * 0.0, 7.0 + I * 0.0, 1.0 + I * 0.0},
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.0 + I * 0.0);  // NOLINT(readability-identifier-length)

		namespace blas = multi::blas;

		using blas::filling;
		using blas::transposed;

		syrk(filling::lower, 1.0, transposed(a), 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_REQUIRE( real(c[2][1]) ==   19.0 );
		BOOST_REQUIRE( real(c[1][2]) == 9999.0 );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_complex) {
	using complex = std::complex<double>;

	constexpr auto const I = complex{0.0, 1.0};

	multi::array<complex, 2> const a = {
		{1.0 + 3.0 * I, 3.0 - 2.0 * I, 4.0 + 1.0 * I},
		{9.0 + 1.0 * I, 7.0 - 8.0 * I, 1.0 - 3.0 * I},
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.0 + I * 0.0);
		namespace blas = multi::blas;

		syrk(blas::filling::lower, 1.0, blas::T(a), 0.0, c);  // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular

		BOOST_TEST( real(c[2][1]) == - 3.0 );
		BOOST_TEST( imag(c[2][1]) == -34.0 );
	}
	// {
	//  multi::array<complex, 2> c({2, 2}, 9999.);
	//  namespace blas = multi::blas;
	//  using blas::filling;
	//  syrk(filling::lower, 1., a, 0., c);  // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
	//  BOOST_REQUIRE( c[1][0] == complex(18., -21.) );
	//  BOOST_REQUIRE( c[0][1] == 9999. );
	// }
	// {
	//  multi::array<complex, 2> c({2, 2}, 9999.);
	//  namespace blas = multi::blas;
	//  using blas::filling;
	//  syrk(filling::upper, 1., a, 0., c);  // c⸆=c=aa⸆=(aa⸆)⸆, `c` in upper triangular
	//  BOOST_REQUIRE( c[0][1] == complex(18., -21.) );
	//  BOOST_REQUIRE( c[1][0] == 9999. );
	// }
}

// BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_operation_complex){
//   using complex = std::complex<double>;
//   constexpr auto const I = complex{0., 1.};
//   multi::array<complex, 2> const a = {
//     { 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
//     { 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
//   };
//   {
//     multi::array<complex, 2> c({2, 2}, 9999.);
//     using multi::blas::filling;
//     syrk(filling::lower, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[1][0]==complex(18., -21.) );
//     BOOST_REQUIRE( c[0][1]==9999. );
//   }
//   {
//     multi::array<complex, 2> c({3, 3}, 9999.);
//     namespace blas = multi::blas;
//     using blas::filling;
//     using blas::transposed;
//     syrk(filling::lower, 1., transposed(a), 0., c); // c⸆=c=a⸆a=(aa⸆)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[2][1]==complex(-3.,-34.) );
//     BOOST_REQUIRE( c[1][2]==9999. );
//   }
//   {
//     multi::array<complex, 2> c({3, 3}, 9999.);
//     namespace blas = multi::blas;
//     using blas::filling;
//     using blas::transposed;
//     syrk(filling::lower, 1., rotated(a), 0., c); // c⸆=c=a⸆a=(aa⸆)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[2][1]==complex(-3.,-34.) );
//     BOOST_REQUIRE( c[1][2]==9999. );
//   }
// }

// BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_operation_real){
//   multi::array<double, 2> const a = {
//     { 1., 3., 4.},
//     { 9., 7., 1.}
//   };
//   {
//     multi::array<double, 2> c({2, 2}, 9999.);
//     using multi::blas::filling;
//     syrk(filling::lower, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[1][0] == 34. );
//     BOOST_REQUIRE( c[0][1] == 9999. );
//   }
//   {
//     multi::array<double, 2> c({2, 2}, 9999.);
//     using multi::blas::filling;
//     syrk(filling::upper, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in upper triangular
//     BOOST_REQUIRE( c[0][1] == 34. );
//     BOOST_REQUIRE( c[1][0] == 9999. );
//   }
//   {
//     multi::array<double, 2> c({3, 3}, 9999.);
//     using multi::blas::filling;
//     syrk(filling::lower, 1., rotated(a), 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[2][1] == 19. );
//     BOOST_REQUIRE( c[1][2] == 9999. );
//   }
//   {
//     multi::array<double, 2> c({3, 3}, 9999.);
//     namespace blas = multi::blas;
//     using blas::transposed;
//     using blas::filling;
//     syrk(filling::lower, 1., transposed(a), 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[2][1] == 19. );
//     BOOST_REQUIRE( c[1][2] == 9999. );
//   }
//   {
//     multi::array<double, 2> c({3, 3}, 9999.);
//     namespace blas = multi::blas;
//     using blas::transposed;
//     using blas::filling;
//     syrk(filling::upper, 1., transposed(a), 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in upper triangular
//     BOOST_REQUIRE( c[1][2] == 19. );
//     BOOST_REQUIRE( c[2][1] == 9999. );
//   }
//   {
//     multi::array<double, 2> c({2, 2}, 9999.);
//     using multi::blas::filling;
//     using multi::blas::transposed;
//     syrk(filling::upper, 1., a, 0., transposed(c)); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in upper triangular
//     BOOST_REQUIRE( c[0][1] == 9999. );
//     BOOST_REQUIRE( c[1][0] == 34. );
//   }
// }

// BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_implicit_zero){
//   multi::array<double, 2> const a = {
//     { 1., 3., 4.},
//     { 9., 7., 1.}
//   };
//   {
//     multi::array<double, 2> c({2, 2}, 9999.);
//     using multi::blas::filling;
//     syrk(filling::lower, 1., a, c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[1][0] == 34. );
//     BOOST_REQUIRE( c[0][1] == 9999. );
//   }
// }

// BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_symmetrization){
//   multi::array<double, 2> const a = {
//     { 1., 3., 4.},
//     { 9., 7., 1.}
//   };
//   {
//     multi::array<double, 2> c({2, 2}, 9999.);
//     using multi::blas::syrk;
//     using multi::blas::gemm;
//     using multi::blas::T;
//     syrk(1., a, c); // c⸆=c=aa⸆=(aa⸆)⸆
//     BOOST_REQUIRE( c[1][0] == 34. );
//     BOOST_REQUIRE( c[0][1] == 34. );
//     BOOST_REQUIRE( syrk(a) == gemm(a, T(a)) );
//   }
//   {
//     using multi::blas::syrk;
//     multi::array<double, 2> c = syrk(1., a); // c⸆=c=aa⸆=(aa⸆)⸆
//     BOOST_REQUIRE( c[1][0] == 34. );
//     BOOST_REQUIRE( c[0][1] == 34. );
//   }
//   {
//     using multi::blas::syrk;
//     multi::array<double, 2> c = syrk(a); // c⸆=c=aa⸆=(aa⸆)⸆
//     BOOST_REQUIRE( c[1][0] == 34. );
//     BOOST_REQUIRE( c[0][1] == 34. );
//   }
//   {
//     using multi::blas::transposed;
//     using multi::blas::syrk;
//     multi::array<double, 2> c = syrk(transposed(a)); // c⸆=c=a⸆a=(a⸆a)⸆
//     BOOST_REQUIRE( c[2][1] == 19. );
//     BOOST_REQUIRE( c[1][2] == 19. );
//   }
// }

// #if 0

//}

//}

// #if 0
//   {

//    {
//      multi::array<complex, 2> C({2, 2}, 9999.);
//      syrk(1., rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
//      assert( C[1][0] == complex(18., -21.) );
//    }
//    {
//      multi::array<complex, 2> C({2, 2}, 9999.);
//      syrk(rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
//      assert( C[1][0] == complex(18., -21.) );
//    }
//    {
//      complex C[2][2];
//      using multi::rotated;
//      syrk(rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
//      assert( C[1][0] == complex(18., -21.) );
//    }
//    {
//      auto C = syrk(1., A); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is everywhere)
//      assert( C[1][2]==complex(-3.,-34.) );
//    }
//    {
////      what(rotated(syrk(A)));
//      multi::array C = rotated(syrk(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
//      print(C) <<"---\n";
//    }
//
//  }
// #if 0
//  {
//    multi::array<complex, 2> const A = {
//      { 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
//      { 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
//    };
//    auto C = rotated(syrk(A)).decay(); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
//    print(C) <<"---\n";
////    print(C) <<"---\n";
//  }
//  return 0;
//  {
//    multi::array<complex, 2> const A = {
//      { 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
//      { 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
//    };
//    auto C = syrk(rotated(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering)
//    print(C) <<"---\n";
//  }
// #endif
// #endif
//}

// BOOST_AUTO_TEST_CASE(multi_blas_syrk_herk_fallback){
//   multi::array<double, 2> const a = {
//     { 1., 3., 4.},
//     { 9., 7., 1.}
//   };
//   {
//     multi::array<double, 2> c({2, 2}, 9999.);
//     namespace blas = multi::blas;
//     using blas::filling;
//     syrk(filling::lower, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
//     BOOST_REQUIRE( c[1][0] == 34. );
//     BOOST_REQUIRE( c[0][1] == 9999. );
//   }
// }
// #endif

// #endif
