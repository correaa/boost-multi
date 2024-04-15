// Copyright 2019-2024 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuSolver potrf"
#include <boost/test/unit_test.hpp>

#include "../../blas/gemm.hpp"
#include "../../blas/herk.hpp"

#include "../../lapack/potrf.hpp"

#include <iostream>
#include <random>

namespace multi = boost::multi;
// namespace lapack = multi::lapack;
namespace blas = multi::blas;

using complex = std::complex<double>;

std::ostream& operator<<(std::ostream& os, std::complex<double> const& c) {
	return os << real(c) << " + I*" << imag(c);
}

template<class M> decltype(auto) print(M const& C, std::string const msg = "") {
	using multi::size;
	using std::cout;
	cout << msg << "\n"
	     << '{';
	for(int i = 0; i != size(C); ++i) {
		cout << '{';
		for(int j = 0; j != size(C[i]); ++j) {
			cout << C[i][j];
			if(j + 1 != size(C[i]))
				cout << ", ";
		}
		cout << '}' << std::endl;
		if(i + 1 != size(C))
			cout << ", ";
	}
	return cout << '}' << std::endl;
}

template<class M>
M&& randomize(M&& A) {
	std::mt19937 eng{123};
	auto         gen = [&]() {
                                                                                                                                                                                                      auto unif = std::uniform_real_distribution<>{-1, 1};
                                                                                                                                                                                                      return std::complex<double>(unif(eng), unif(eng));
	};
	std::for_each(begin(A), end(A), [&](auto&& r) { std::generate(begin(r), end(r), gen); });
	return std::forward<M>(A);
}

/*
BOOST_AUTO_TEST_CASE(orthogonalization_over_rows, *boost::unit_test::tolerance(0.00001)){
	auto A = randomize(multi::array<complex, 2>({3, 10}));
	lapack::onrm(A);

	using blas::herk;
	using blas::hermitized;
	using blas::filling;
	auto id = herk(filling::upper, A);
	BOOST_TEST( real(id[1][1]) == 1. ); BOOST_TEST( imag(id[1][1]) == 0. );
	BOOST_TEST( real(id[1][2]) == 0. ); BOOST_TEST( imag(id[1][2]) == 0. );
}
*/

// BOOST_AUTO_TEST_CASE(orthogonalization_over_rows_cuda, *boost::unit_test::tolerance(0.00001)) {
//  auto Acpu = randomize(multi::array<complex, 2>({3, 10}));

//  multi::cuda::array<complex, 2> A = Acpu;

//  using namespace blas;
//  using namespace lapack;

//  trsm(filling::lower, hermitized(potrf(filling::upper, herk(filling::upper, A))), A);

//  Acpu    = A;
//  auto id = herk(filling::upper, Acpu);
//  BOOST_TEST( real(id[1][1]) == 1. );
//  BOOST_TEST( imag(id[1][1]) == 0. );
//  BOOST_TEST( real(id[1][2]) == 0. );
//  BOOST_TEST( imag(id[1][2]) == 0. );
// }

/*
BOOST_AUTO_TEST_CASE(orthogonalization_over_columns, *boost::unit_test::tolerance(0.00001)){

	auto A = randomize( multi::array<complex, 2>({10, 3}) );
	using blas::hermitized;
	lapack::onrm(hermitized(A));

	using blas::filling;
	auto id = herk(filling::upper, hermitized(A));
	BOOST_TEST( real(id[1][1]) == 1. ); BOOST_TEST( imag(id[1][1]) == 0. );
	BOOST_TEST( real(id[1][2]) == 0. ); BOOST_TEST( imag(id[1][2]) == 0. );
}*/

BOOST_AUTO_TEST_CASE(numericalalgorithmsgroup, *boost::unit_test::tolerance(0.0000001)) {

	double const nan = std::numeric_limits<double>::quiet_NaN();
	auto const   I   = complex{0.0, 1.0};

	multi::array<complex, 2> const A_gold = {
		{3.23 + 0.00 * I,  1.51 - 1.92 * I,  1.90 + 0.84 * I,  0.42 + 2.50 * I},
		{1.51 + 1.92 * I,  3.58 + 0.00 * I, -0.23 + 1.11 * I, -1.18 + 1.37 * I},
		{1.90 - 0.84 * I, -0.23 - 1.11 * I,  4.09 + 0.00 * I,  2.33 - 0.14 * I},
		{0.42 - 2.50 * I, -1.18 - 1.37 * I,  2.33 + 0.14 * I,  4.29 + 0.00 * I},
	};

	auto A = A_gold;

	multi::lapack::potrf(multi::lapack::filling::upper, A);

	auto AA = A;

	for(auto i = 0; i != 4; ++i) {
		for(auto j = 0; j != i; ++j) {
			AA[i][j] = 0.0;
		}
	}

	auto const C = +blas::herk(1.0, blas::H(AA));  // +blas::gemm(1.0, blas::H(AA), AA);

	print(A_gold, "A gold");
	print(C, "recover");

	for(auto i = 0; i != 4; ++i) {
		for(auto j = 0; j != 4; ++j) {
			BOOST_TEST( real(A_gold[i][j]) == real(C[i][j]) );
			BOOST_TEST( imag(A_gold[i][j]) == imag(C[i][j]) );
		}
	}
}

BOOST_AUTO_TEST_CASE(lapack_potrf, *boost::unit_test::tolerance(0.00001)) {

	double const nan = std::numeric_limits<double>::quiet_NaN();
	auto const   I   = complex{0.0, 1.0};

	{
		multi::array<complex, 2> A = {
			{      167.413, 126.804 - 0.00143505 * I, 125.114 - 0.1485590 * I},
			{nan + nan * I,                  167.381, 126.746 + 0.0327519 * I},
			{nan + nan * I,            nan + nan * I,                 167.231},
		};

		print(A, "original A");
		using boost::multi::lapack::filling;
		using boost::multi::lapack::potrf;

		potrf(filling::upper, A);  // A is hermitic in upper triangular (implicit below)
		BOOST_TEST( real(A[1][2]) == 3.78646 );
		BOOST_TEST( imag(A[1][2]) == 0.0170734 );
		//  BOOST_TEST( A[2][1] != A[2][1] );
		print(A, "decomposition");

		multi::array<complex, 2> C(A.extensions(), complex{0.0, 0.0});

		multi::array<complex, 2> AA = A;
		auto const [is, js]         = AA.extensions();
		for(auto i : is) {
			for(auto j = 0; j != i; ++j) {
				AA[i][j] = std::conj(A[j][i]);
			}
		}

		blas::gemm(complex{1.0, 0.0}, blas::H(AA), AA, complex{0.0, 0.0}, C);

		print(C, "recovery");
	}
	// {
	//  multi::cuda::managed::array<complex, 2> A = {
	//      {167.413, 126.804 - 0.00143505 * I, 125.114 - 0.1485590 * I},
	//      {    NAN,                  167.381, 126.746 + 0.0327519 * I},
	//      {    NAN,                      NAN,                 167.231},
	//  };
	//  using lapack::filling;
	//  using lapack::potrf;
	//  potrf(filling::upper, A);  // A is hermitic in upper triangular (implicit below)
	//  BOOST_TEST( real(A[1][2]) == 3.78646 );
	//  BOOST_TEST( imag(A[1][2]) == 0.0170734 );
	//  //  BOOST_TEST( A[2][1] != A[2][1] );
	// }
	// {
	//  multi::cuda::array<complex, 2> A = {
	//      {167.413, 126.804 - 0.00143505 * I, 125.114 - 0.1485590 * I},
	//      {    NAN,                  167.381, 126.746 + 0.0327519 * I},
	//      {    NAN,                      NAN,                 167.231},
	//  };
	//  using lapack::filling;
	//  using lapack::potrf;
	//  potrf(filling::upper, A);  // A is hermitic in upper triangular (implicit below)
	//  multi::array<complex, 2> A_copy = A;
	//  print(A_copy);
	// }
}
