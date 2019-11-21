#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX `#--compiler-options` -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_SYRK $0.cpp -o $0x -lboost_unit_test_framework \
`pkg-config --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_SYRK_HPP
#define MULTI_ADAPTORS_BLAS_SYRK_HPP

#include "../blas/core.hpp"
#include "../blas/copy.hpp"

#include "../blas/numeric.hpp"
#include "../blas/operations.hpp"
#include "../blas/side.hpp"

namespace boost{
namespace multi{namespace blas{

template<typename AA, typename BB, class A2D, class C2D>
C2D&& syrk(triangular c_side, real_operation a_op, AA alpha, A2D const& a, BB beta, C2D&& c){
	if(stride(c)==1 and stride(c[0])!=1) syrk(flip(c_side), transpose(a_op), alpha, rotated(a), beta, rotated(c));
	else{
		assert( stride(c[0])==1 );
		assert( stride(a[0])==1 ); // sources and destination are incompatible layout
		assert( size(c) == size(c[0]) );
		assert( a_op==real_operation::transposition?size(a[0])==size(c):size(a)==size(c) ); 
		using boost::multi::blas::core::syrk;
		syrk(
			static_cast<char>(c_side), static_cast<char>(a_op), size(c), 
			a_op==real_operation::transposition?size(a):size(*begin(a)), 
			alpha, base(a), stride(a), beta, base(c), stride(c)
		);
	}
	return std::forward<C2D>(c);
}

template<typename AA, typename BB, class A2D, class C2D>
C2D&& syrk(triangular c_side, AA alpha, A2D const& a, BB beta, C2D&& c){
	if(stride(a)==1){
		if(stride(c)==1) syrk(flip(c_side), real_operation::transposition, alpha, rotated(a), beta, rotated(std::forward<C2D>(c)));
		else             syrk(c_side      , real_operation::transposition, alpha, rotated(a), beta,        (std::forward<C2D>(c)));
	}else{
		if(stride(c)==1) syrk(flip(c_side), real_operation::identity     , alpha,        (a), beta, rotated(std::forward<C2D>(c)) );
		else             syrk(c_side      , real_operation::identity     , alpha,        (a), beta,        (std::forward<C2D>(c)));
	}
	return std::forward<C2D>(c);
}

template<typename AA, class A2D, class C2D>
C2D&& syrk(triangular c_side, AA alpha, A2D const& a, C2D&& c){
	syrk(c_side, alpha, a, 0., c);
	return std::forward<C2D>(c);
}

template<typename AA, class A2D, class C2D>
C2D&& syrk(AA alpha, A2D const& a, C2D&& c){
	if(stride(c)==1) syrk(triangular::upper, alpha, a, rotated(c));
	else             syrk(triangular::lower, alpha, a,        (c));
	assert( size(c) == size(rotated(c)) );
	for(typename std::decay_t<C2D>::difference_type i = 0; i != size(c); ++i)
		blas::copy(rotated(c)[i]({i+1, size(c)}), c[i]({i+1, size(c)}) );
	//	blas::copy(begin(rotated(c)[i])+i+1, end(rotated(c)[i]), begin(c[i])+i+1);
	return std::forward<C2D>(c);
}

template<typename AA, class A2D, class Ret = typename A2D::decay_type>
auto syrk(AA alpha, A2D const& a){
	return syrk(alpha, a, Ret({size(a), size(a)}, get_allocator(a)));
}

template<class A2D> auto syrk(A2D const& A){return syrk(1., A);}

}}}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if _TEST_MULTI_ADAPTORS_BLAS_SYRK

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../blas/gemm.hpp"

#include "../../array.hpp"
#include "../../utility.hpp"

#include <boost/timer/timer.hpp>

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>

//#include<catch.hpp>

using std::cout;
using std::cerr;

namespace multi = boost::multi;

template<class M> decltype(auto) print(M const& C){
	using boost::multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j)
			std::cout << C[i][j] << ' ';
		std::cout << std::endl;
	}
	return std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_enums){
	BOOST_REQUIRE( multi::blas::complex_operation::hermitian == multi::blas::operation::hermitian );
	BOOST_REQUIRE( multi::blas::complex_operation::identity == multi::blas::operation::identity );
	multi::blas::operation op = multi::blas::complex_operation::identity;
	BOOST_REQUIRE( op == multi::blas::complex_operation::identity );
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_real){

	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::transposition, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1] == 19. ); 
		BOOST_REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::transposition, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][2] == 19. );
		BOOST_REQUIRE( c[2][1] == 9999. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][0] == 34. ); 
		BOOST_REQUIRE( c[0][1] == 9999. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::identity, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, a⸆a, `c` in lower triangular
		BOOST_REQUIRE( c[0][1] == 34. ); 
		BOOST_REQUIRE( c[1][0] == 9999. );
	}
	{
		multi::array<double, 2> const at = rotated(a);
		multi::array<double, 2> ct({2, 2}, 9999.);
		auto&& a_ = rotated(at);
		auto&& c_ = rotated(ct);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::identity, 1., a_, 0., c_); // c⸆=c=a⸆a=(a⸆a)⸆, a⸆a, `c` in lower triangular
		BOOST_REQUIRE( c_[0][1] == 34. ); 
		BOOST_REQUIRE( c_[1][0] == 9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_real_special_case){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
	};
	{
		multi::array<double, 2> c({1, 1}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		//BOOST_REQUIRE( c[1][0] == 34. ); 
		//BOOST_REQUIRE( c[0][1] == 9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_complex_real_case){
	using complex = std::complex<double>;
	multi::array<complex, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::operation;
		syrk(triangular::lower, operation::transposition, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1] == 19. );
		BOOST_REQUIRE( c[1][2] == 9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_complex){
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::transposition, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1] == complex(-3., -34.) );
		BOOST_REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][0] == complex(18., -21.) );
		BOOST_REQUIRE( c[0][1] == 9999. );
	}
	{
		multi::array<complex, 2> const at = rotated(a);
		multi::array<complex, 2> ct({2, 2}, 9999.);
		auto&& a_ = rotated(at);
		auto&& c_ = rotated(ct);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::upper, real_operation::identity, 1., a_, 0., c_); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in upper triangular
		BOOST_REQUIRE( c_[0][1] == complex(18., -21.) ); 
		BOOST_REQUIRE( c_[1][0] == 9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_operation_complex){
	using complex = std::complex<double>;
	constexpr auto const I = complex{0., 1.};
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][0]==complex(18., -21.) );
		BOOST_REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::transposed;
		syrk(triangular::lower, 1., transposed(a), 0., c); // c⸆=c=a⸆a=(aa⸆)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(-3.,-34.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::transposed;
		syrk(triangular::lower, 1., rotated(a), 0., c); // c⸆=c=a⸆a=(aa⸆)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(-3.,-34.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_operation_real){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 9999. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::upper, 1., a, 0., c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in upper triangular
		BOOST_REQUIRE( c[0][1] == 34. );
		BOOST_REQUIRE( c[1][0] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., rotated(a), 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1] == 19. );
		BOOST_REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::transposed;
		using multi::blas::triangular;
		syrk(triangular::lower, 1., transposed(a), 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[2][1] == 19. );
		BOOST_REQUIRE( c[1][2] == 9999. );
	}
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::transposed;
		using multi::blas::triangular;
		syrk(triangular::upper, 1., transposed(a), 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in upper triangular
		BOOST_REQUIRE( c[1][2] == 19. );
		BOOST_REQUIRE( c[2][1] == 9999. );
	}
	{
		multi::array<double, 2> const at = rotated(a);
		multi::array<double, 2> ct({2, 2}, 9999.);
		auto&& a_ = rotated(at);
		auto&& c_ = rotated(ct);
		using multi::blas::triangular;
		syrk(triangular::upper, 1., a_, 0., c_); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in upper triangular
		print(c_);
		BOOST_REQUIRE( c_[0][1] == 34. ); 
		BOOST_REQUIRE( c_[1][0] == 9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_implicit_zero){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		syrk(triangular::lower, 1., a, c); // c⸆=c=aa⸆=(aa⸆)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 9999. );
	}
}


BOOST_AUTO_TEST_CASE(multi_blas_syrk_automatic_symmetrization){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::syrk;
		syrk(1., a, c); // c⸆=c=aa⸆=(aa⸆)⸆
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::syrk;
		multi::array<double, 2> c = syrk(1., a); // c⸆=c=aa⸆=(aa⸆)⸆
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::syrk;
		multi::array<double, 2> c = syrk(a); // c⸆=c=aa⸆=(aa⸆)⸆
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::transposed;
		using multi::blas::syrk;
		multi::array<double, 2> c = syrk(transposed(a)); // c⸆=c=a⸆a=(a⸆a)⸆
		BOOST_REQUIRE( c[2][1] == 19. );
		BOOST_REQUIRE( c[1][2] == 19. );
	}
#if 0
	{
		{
			multi::array<complex, 2> C({2, 2}, 9999.);
			syrk(1., rotated(A), C); // C^T = C =  A*A^T = (A*A^T)^T, A*A^T, C are C-ordering, information in C lower triangular
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			multi::array<complex, 2> C({2, 2}, 9999.);
			syrk(1., rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			multi::array<complex, 2> C({2, 2}, 9999.);
			syrk(rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			complex C[2][2];
			using multi::rotated;
			syrk(rotated(A), rotated(C)); // C^T=C=A*A^T=(A*A^T)^T
			assert( C[1][0] == complex(18., -21.) );
		}
		{
			auto C = syrk(1., A); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is everywhere)
			assert( C[1][2]==complex(-3.,-34.) );
		}
		{
//			what(rotated(syrk(A)));
			multi::array C = rotated(syrk(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
			print(C) <<"---\n";
		}
		
	}
#if 0
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		auto C = rotated(syrk(A)).decay(); // C = C^T = A^T*A, C is a value type matrix (with C-ordering, information is in upper triangular part)
		print(C) <<"---\n";
//		print(C) <<"---\n";
	}
	return 0;
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		auto C = syrk(rotated(A)); // C = C^T = A^T*A, C is a value type matrix (with C-ordering)
		print(C) <<"---\n";
	}
#endif
#endif
}

BOOST_AUTO_TEST_CASE(multi_blas_syrk_herk_fallback){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	using multi::blas::triangular;
	using multi::blas::real_operation;
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		using multi::blas::real_operation;
		syrk(triangular::lower, real_operation::identity, 1., a, 0., c); // c⸆=c=a⸆a=(a⸆a)⸆, `c` in lower triangular
		BOOST_REQUIRE( c[1][0] == 34. ); 
		BOOST_REQUIRE( c[0][1] == 9999. );
	}
}


#endif
#endif

