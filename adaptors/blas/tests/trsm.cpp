#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc -x cu --expt-relaxed-constexpr`#c++` $0 -o $0x -Wno-deprecated-declarations -lcudart -lcublas -lboost_unit_test_framework `pkg-config --libs blas` `#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` &&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../../memory/adaptors/cuda/managed/ptr.hpp"

#include "../../../adaptors/blas.hpp"
#include "../../../adaptors/blas/cuda.hpp"

#include "../../../adaptors/cuda.hpp"
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

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex, *utf::tolerance(0.00001)){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> const B = {
		{1. - 9.*I, 3. + 2.*I, 4. + 3.*I},
		{2. - 2.*I, 7. - 2.*I, 1. - 1.*I},
		{3. + 1.*I, 4. + 8.*I, 2. + 7.*I}
	};
	namespace blas = multi::blas;
	using blas::filling;	
	using blas::hermitized;
	auto C = trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( std::real(C[1][2]) == 2.33846 );
	BOOST_TEST( std::imag(C[1][2]) == -0.0923077 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_rectangular, *utf::tolerance(0.00001)){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> const B = {
		{1. - 9.*I, 3. + 2.*I},
		{2. - 2.*I, 7. - 2.*I},
		{3. + 1.*I, 4. + 8.*I}
	};
	namespace blas = multi::blas;
	using blas::filling;
	using blas::hermitized;
	auto C = trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(C[2][0]) == -4.16471 );
	BOOST_TEST( imag(C[2][0]) ==  8.25882 );
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_column, *utf::tolerance(0.00001)){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
	multi::array<complex, 2> const A = {
		{ 1. + 2.*I,  3. - 1.*I,  4. + 9.*I},
		{NAN       ,  7. + 4.*I,  1. + 8.*I},
		{NAN       , NAN       ,  8. + 2.*I}
	};
	multi::array<complex, 2> const B = {
		{1. - 9.*I},
		{2. - 2.*I},
		{3. + 1.*I}
	};
	namespace blas = multi::blas;
	using blas::filling;	
	using blas::hermitized;
	auto C = trsm(filling::lower, 2.+1.*I, hermitized(A), B); // B=alpha Inv[A†].B, B†=B†.Inv[A], Solve(A†.X=B, X), Solve(X†.A=B†, X), A is upper triangular (with implicit zeros below)
	BOOST_TEST( real(C[2][0]) == -4.16471 );
	BOOST_TEST( imag(C[2][0]) ==  8.25882 );
}

BOOST_AUTO_TEST_CASE(multi_blas_cuda_trsm_complex, *utf::tolerance(0.00001)){
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
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
	using complex = std::complex<double>;
	constexpr complex I{0., 1.};
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


