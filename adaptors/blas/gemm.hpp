#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_GEMM $0.cpp -o $0x -lboost_unit_test_framework \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_GEMM_HPP
#define MULTI_ADAPTORS_BLAS_GEMM_HPP

#include "../blas/core.hpp"
#include "../blas/operations.hpp"

namespace boost{
namespace multi{
namespace blas{

using multi::blas::core::gemm;

template<class AA, class BB, class A2D, class B2D, class C2D>//, typename = std::enable_if_t<not is_complex_array<std::decay_t<C2D>>{}> >
auto gemm(real_operation a_op, real_operation b_op, AA alpha, A2D const& A, B2D const& B, BB beta, C2D&& C)
->decltype(gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C)), std::forward<C2D&&>(C))
{
	assert( stride(rotated(A))== 1 );
	assert( stride(rotated(B))== 1 );
	assert( stride(rotated(C))== 1 );
	[&](){switch(a_op){
		case real_operation::identity: switch(b_op){
			case real_operation::identity: 
				assert( size(C) == size(A) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case real_operation::transposition: 
				assert( size(C) == size(A) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('T', 'N', size(C[0]), size(A), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));				
		}
		case real_operation::transposition: switch(b_op){
			case real_operation::identity: 
				assert( size(C) == size(rotated(A)) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'T', size(C[0]), size(A[0]), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case real_operation::transposition: 
				assert( size(C) == size(rotated(A)) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('T', 'T', size(C[0]), size(A[0]), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
		}
	};}();
	return std::forward<C2D>(C);
}

template<class AA, class BB, class A2D, class B2D, class C2D, typename = std::enable_if_t<is_complex_array<std::decay_t<C2D>>{}> >
auto gemm(complex_operation a_op, complex_operation b_op, AA alpha, A2D const& A, B2D const& B, BB beta, C2D&& C)
->decltype(gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C)), std::forward<C2D&&>(C))
{
	using multi::blas::core::gemm;
	assert( stride(rotated(A))== 1 );
	assert( stride(rotated(B))== 1 );
	assert( stride(rotated(C))== 1 );
	[&](){switch(a_op){
		case complex_operation::identity: switch(b_op){
			case complex_operation::identity: 
				assert( size(C) == size(A) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case complex_operation::hermitian: 
				assert( size(C) == size(A) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('C', 'N', size(C[0]), size(A), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));				
		}
		case complex_operation::hermitian: switch(b_op){
			case complex_operation::identity: 
				assert( size(C) == size(rotated(A)) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'C', size(C[0]), size(A[0]), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case complex_operation::hermitian: 
				assert( size(C) == size(rotated(A)) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('C', 'C', size(C[0]), size(A[0]), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
		}
	};}();
	return std::forward<C2D>(C);	
}

template<class AA, class BB, class A2D, class B2D, class C2D, typename = std::enable_if_t<is_complex_array<std::decay_t<C2D>>{}> >
auto gemm(real_operation a_op, complex_operation b_op, AA alpha, A2D const& A, B2D const& B, BB beta, C2D&& C)
->decltype(gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C)), std::forward<C2D&&>(C))
{
	using multi::blas::core::gemm;
	assert( stride(rotated(A))== 1 );
	assert( stride(rotated(B))== 1 );
	assert( stride(rotated(C))== 1 );
	[&](){switch(a_op){
		case real_operation::identity: switch(b_op){
			case complex_operation::identity: 
				assert( size(C) == size(A) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case complex_operation::hermitian: 
				assert( size(C) == size(A) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('C', 'N', size(C[0]), size(A), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));				
		}
		case real_operation::transposition: switch(b_op){
			case complex_operation::identity: 
				assert( size(C) == size(rotated(A)) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'T', size(C[0]), size(A[0]), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case complex_operation::hermitian: 
				assert( size(C) == size(rotated(A)) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('C', 'T', size(C[0]), size(A[0]), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
		}
	};}();
	return std::forward<C2D>(C);	
}

template<class AA, class BB, class A2D, class B2D, class C2D, typename = std::enable_if_t<is_complex_array<std::decay_t<C2D>>{}> >
auto gemm(complex_operation a_op, real_operation b_op, AA alpha, A2D const& A, B2D const& B, BB beta, C2D&& C)
->decltype(gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C)), std::forward<C2D&&>(C))
{
	using multi::blas::core::gemm;
	assert( stride(rotated(A))== 1 );
	assert( stride(rotated(B))== 1 );
	assert( stride(rotated(C))== 1 );
	[&](){switch(a_op){
		case complex_operation::identity: switch(b_op){
			case real_operation::identity: 
				assert( size(C) == size(A) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'N', size(C[0]), size(A), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case real_operation::transposition: 
				assert( size(C) == size(A) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('T', 'N', size(C[0]), size(A), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));				
		}
		case complex_operation::hermitian: switch(b_op){
			case real_operation::identity: 
				assert( size(C) == size(rotated(A)) );
				assert( std::get<1>(sizes(B)) == std::get<1>(sizes(C)) );
				return gemm('N', 'T', size(C[0]), size(A[0]), size(B), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
			case real_operation::transposition: 
				assert( size(C) == size(rotated(A)) );
				assert( size(B) == std::get<1>(sizes(C)) );
				return gemm('C', 'T', size(C[0]), size(A[0]), size(B[0]), alpha, base(B), stride(B), base(A), stride(A), beta, base(C), stride(C));
		}
	};}();
	return std::forward<C2D>(C);
}

template<class M> bool is_c_ordering(M const& m){return stride(rotated(m))==1;}

template<class AA, class BB, class A2D, class B2D, class C2D>
void gemm_aux(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::false_type, std::false_type){
	gemm(
		is_c_ordering(A)?real_operation::identity:real_operation::transposition, 
		is_c_ordering(B)?real_operation::identity:real_operation::transposition, 
		a, 
		is_c_ordering(A)?rotated(rotated(A)):rotated(A), 
		is_c_ordering(B)?rotated(rotated(B)):rotated(B), 
		b, C
	);
}

template<class AA, class BB, class A2D, class B2D, class C2D>
void gemm_aux(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::true_type, std::false_type){
	gemm(
		is_c_ordering(A)?complex_operation::identity:complex_operation::hermitian, 
		is_c_ordering(B)?complex_operation::identity:complex_operation::hermitian, 
		a, 
		is_c_ordering(A)?conjugated(A):rotated(conjugated(A)), 
		is_c_ordering(B)?rotated(rotated(B)):rotated(B), 
		b, C
	);
}

template<class AA, class BB, class A2D, class B2D, class C2D>
void gemm_aux(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::false_type, std::true_type){
	gemm(
		is_c_ordering(A)?complex_operation::identity:complex_operation::hermitian, 
		is_c_ordering(B)?complex_operation::identity:complex_operation::hermitian, 
		a, 
		is_c_ordering(A)?rotated(rotated(A)):rotated(A), 
		is_c_ordering(B)?conjugated(B):rotated(conjugated(B)), 
		b, C
	);
}

template<class AA, class BB, class A2D, class B2D, class C2D>
void gemm_aux(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::true_type, std::true_type){
	gemm(
		is_c_ordering(A)?complex_operation::identity:complex_operation::hermitian, 
		is_c_ordering(B)?complex_operation::identity:complex_operation::hermitian, 
		a, 
		is_c_ordering(A)?conjugated(A):rotated(conjugated(A)), 
		is_c_ordering(B)?conjugated(B):rotated(conjugated(B)), 
		b, C
	);
}

template<class AA, class BB, class A2D, class B2D, class C2D>
decltype(auto) gemm_aux2(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::false_type){
	assert( stride(rotated(C)) == 1 );
	gemm_aux(a, A, B, b, is_c_ordering(C)?rotated(rotated(C)):rotated(C), multi::blas::is_hermitized<A2D>{}, multi::blas::is_hermitized<B2D>{});
	return std::forward<C2D>(C);
}

template<class AA, class BB, class A2D, class B2D, class C2D>
decltype(auto) gemm_aux2(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::true_type){
	assert( stride(rotated(C)) == 1 );
	gemm_aux(a, conjugated(A), conjugated(B), b, is_c_ordering(C)?rotated(rotated(C)):rotated(C), multi::blas::is_hermitized<A2D>{}, multi::blas::is_hermitized<B2D>{});
	return std::forward<C2D>(C);
}

template<class AA, class BB, class A2D, class B2D, class C2D>
decltype(auto) gemm(AA a, A2D const& A, B2D const& B, BB b, C2D&& C){
	gemm_aux2(a, A, B, b, C, multi::blas::is_hermitized<C2D>{});
	return std::forward<C2D>(C);	
}

}}}

#if _TEST_MULTI_ADAPTORS_BLAS_GEMM

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../array.hpp"
#include "../../utility.hpp"

#include <boost/timer/timer.hpp>

#include<complex>
#include<cassert>
#include<iostream>
#include<numeric>
#include<algorithm>


namespace multi = boost::multi;

template<class M> decltype(auto) print(M const& C){
	using std::cout;
	using multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j)
			cout<< C[i][j] <<' ';
		cout<<std::endl;
	}
	return cout<<std::endl;
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_real_square_implementation){
	multi::array<double, 2> const a = {
		{ 1., 3.},
		{ 9., 7.},
	};
	multi::array<double, 2> const b = {	
		{ 11., 12.},
		{  7., 19.},
	};
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::operation;
		gemm(operation::identity, operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][1] == 241. );
		using multi::blas::real_operation;
		BOOST_REQUIRE( operation{real_operation::identity} == operation::identity );
	}
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::operation;
		gemm(operation::identity, operation::transposition, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][1] == 196. );
		using multi::blas::real_operation;
		BOOST_REQUIRE( operation{real_operation::identity} == operation::identity );
	}
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::operation;
		gemm(operation::transposition, operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE(( c[1][1] == 169. and c[1][0] == 82. ));
		using multi::blas::real_operation;
		BOOST_REQUIRE( operation{real_operation::identity} == operation::identity );
	}
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::operation;
		gemm(operation::transposition, operation::transposition, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][1] == 154. );
		using multi::blas::real_operation;
		BOOST_REQUIRE( operation{real_operation::identity} == operation::identity );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_real_nonsquare_implementation){
	multi::array<double, 2> const a = {
		{ 1., 3., 1.},
		{ 9., 7., 1.},
	};
	multi::array<double, 2> const b = {	
		{ 11., 12., 1.},
		{  7., 19., 1.},
		{  1.,  1., 1.}
	};
	{
		multi::array<double, 2> c({2, 3});
		using multi::blas::real_operation;
		gemm(real_operation::identity, real_operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][2] == 17. );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_real_nonsquare2_implementation){
	multi::array<double, 2> const a = {
		{ 1., 3.},
		{ 9., 7.},
		{ 1., 1.}
	};
	multi::array<double, 2> const b = {	
		{ 11., 12.},
		{  7., 19.}
	};
	{
		multi::array<double, 2> c({3, 2});
		using multi::blas::real_operation;
		gemm(real_operation::identity, real_operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[2][1] == 31. );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_real_nonsquare3_implementation){
	multi::array<double, 2> const a = {
		{ 1., 9., 1.},
		{ 3., 7., 1.}
	};
	multi::array<double, 2> const b = {	
		{ 11., 12.},
		{  7., 19.}
	};
	{
		multi::array<double, 2> c({3, 2});
		using multi::blas::real_operation;
		gemm(real_operation::transposition, real_operation::identity, 1., a, b, 0., c); // c=a⸆b, c⸆=b⸆a
		BOOST_REQUIRE( c[2][1] == 31. );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_square_implementation){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{ 1. + 2.*I, 3. - 3.*I},
		{ 9. + 1.*I, 7. + 4.*I},
	};
	multi::array<complex, 2> const b = {	
		{ 11. + 1.*I, 12. + 1.*I},
		{  7. + 8.*I, 19. - 2.*I},
	};
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::real_operation;
		gemm(real_operation::identity, real_operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][0] == complex(115, 104) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::real_operation;
		gemm(real_operation::identity, real_operation::transposition, 1., a, b, 0., c); // c=ab⸆, c⸆=ba⸆
		BOOST_REQUIRE( c[1][0] == complex(178, 75) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::real_operation;
		gemm(real_operation::transposition, real_operation::identity, 1., a, b, 0., c); // c=a⸆b, c⸆=b⸆a
		BOOST_REQUIRE(( c[1][1] == complex(180, 29) and c[1][0] == complex(53, 54) ));
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::real_operation;
		gemm(real_operation::transposition, real_operation::transposition, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE(( c[1][1] == complex(186, 65) and c[1][0] == complex(116, 25) ));
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		gemm(complex_operation::identity, complex_operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][0] == complex(115, 104) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		gemm(complex_operation::hermitian, complex_operation::identity, 1., a, b, 0., c); // c=a†b, c†=b†a
		BOOST_REQUIRE( c[1][0] == complex(111, 64) and c[1][1] == complex(158, -51) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		gemm(complex_operation::identity, complex_operation::hermitian, 1., a, b, 0., c); // c=ab†, c†=ba†
		BOOST_REQUIRE( c[1][0] == complex(188, 43) and c[1][1] == complex(196, 25) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		gemm(complex_operation::hermitian, complex_operation::hermitian, 1., a, b, 0., c); // c=a†b†, c†=ba
		BOOST_REQUIRE( c[1][0] == complex(116, -25) and c[1][1] == complex(186, -65) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		using multi::blas::real_operation;
		gemm(real_operation::transposition, complex_operation::hermitian, 1., a, b, 0., c); // c=a⸆b†, c†=ba⸆†
		BOOST_REQUIRE( c[1][0] == complex(118, 5) and c[1][1] == complex(122, 45) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		using multi::blas::real_operation;
		gemm(real_operation::transposition, complex_operation::hermitian, 1., a, b, 0., c); // c=a⸆b†, c†=ba⸆†
		BOOST_REQUIRE( c[1][0] == complex(118, 5) and c[1][1] == complex(122, 45) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::complex_operation;
		using multi::blas::real_operation;
		gemm(real_operation::transposition, real_operation::transposition, 1., a, b, 0., c); // c=a⸆b⸆, c⸆=ba
		BOOST_REQUIRE( c[1][0] == complex(116, 25) and c[1][1] == complex(186, 65) );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare_implementation){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{ 1. + 2.*I, 3. - 3.*I},
		{ 9. + 1.*I, 7. + 4.*I},
		{ 1.       , 1.       }
	};
	multi::array<complex, 2> const b = {	
		{ 11. + 1.*I, 12. + 1.*I},
		{  7. + 8.*I, 19. - 2.*I},
	};
	{
		multi::array<complex, 2> c({3, 2});
		using multi::blas::real_operation;
		gemm(real_operation::identity, real_operation::identity, 1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[2][1] == complex(31, -1) );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare2_implementation){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{ 1. + 2.*I, 9. + 1.*I, 1.},
		{ 3. - 3.*I, 7. + 4.*I, 1.}
	};
	multi::array<complex, 2> const b = {	
		{ 11. + 1.*I, 12. + 1.*I},
		{  7. + 8.*I, 19. - 2.*I},
	};
	{
		multi::array<complex, 2> c({3, 2});
		using multi::blas::real_operation;
		gemm(real_operation::transposition, real_operation::identity, 1., a, b, 0., c); // c=a⸆b, c⸆=b⸆a
		BOOST_REQUIRE( c[2][1] == complex(31, -1) );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare3_implementation){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{ 1. + 2.*I, 9. - 1.*I, 1.},
		{ 3. + 3.*I, 7. - 4.*I, 1.}
	};
	multi::array<complex, 2> const b = {	
		{ 11. + 1.*I, 12. + 1.*I},
		{  7. + 8.*I, 19. - 2.*I},
	};
	{
		multi::array<complex, 2> c({3, 2});
		using multi::blas::complex_operation;
		using multi::blas::real_operation;
		gemm(complex_operation::hermitian, real_operation::identity, 1., a, b, 0., c); // c=a⸆b, c⸆=b⸆a
		BOOST_REQUIRE( c[2][1] == complex(31, -1) );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_real_square_automatic){
	multi::array<double, 2> const a = {
		{ 1., 3.},
		{ 9., 7.},
	};
	multi::array<double, 2> const b = {	
		{ 11., 12.},
		{  7., 19.},
	};
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][0] == 148 and c[1][1] == 241 );
	}
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., a, rotated(b), 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][1] == 196. );
	}
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(a), b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE(( c[1][1] == 169. and c[1][0] == 82. ));
	}
	{
		multi::array<double, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(a), rotated(b), 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][1] == 154. );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_real_nonsquare_automatic){
	multi::array<double, 2> const a = {
		{ 1., 3., 1.},
		{ 9., 7., 1.},
	};
	multi::array<double, 2> const b = {	
		{ 11., 12., 1.},
		{  7., 19., 1.},
		{  1.,  1., 1.}
	};
	{
		multi::array<double, 2> c({2, 3});
		using multi::blas::gemm;
		gemm(1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][2] == 17. );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_square_automatic){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{ 1. + 2.*I, 3. - 3.*I},
		{ 9. + 1.*I, 7. + 4.*I},
	};
	multi::array<complex, 2> const b = {	
		{ 11. + 1.*I, 12. + 1.*I},
		{  7. + 8.*I, 19. - 2.*I},
	};
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][0] == complex(115, 104) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., a, rotated(b), 0., c); // c=ab⸆, c⸆=ba⸆
		BOOST_REQUIRE( c[1][0] == complex(178, 75) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(a), b, 0., c); // c=a⸆b, c⸆=b⸆a
		BOOST_REQUIRE(( c[1][1] == complex(180, 29) and c[1][0] == complex(53, 54) ));
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(a), rotated(b), 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE(( c[1][1] == complex(186, 65) and c[1][0] == complex(116, 25) ));
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		gemm(1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][0] == complex(115, 104) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(a), b, 0., c); // c=a†b, c†=b†a
		BOOST_REQUIRE( c[1][0] == complex(111, 64) and c[1][1] == complex(158, -51) );
	}
	{
		multi::array<complex, 2> c({2, 2});
	//	using multi::blas::complex_operation;
	//	gemm(complex_operation::identity, complex_operation::hermitian, 1., a, b, 0., c); // c=ab†, c†=ba†
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., a, hermitized(b), 0., c); // c=ab†, c†=ba†
		BOOST_REQUIRE( c[1][0] == complex(188, 43) and c[1][1] == complex(196, 25) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(a), hermitized(b), 0., c); // c=a†b†, c†=ba
		BOOST_REQUIRE( c[1][0] == complex(116, -25) and c[1][1] == complex(186, -65) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::transposed;
		using multi::blas::hermitized;
		using multi::blas::gemm;
	//	gemm(1., transposed(a), hermitized(b), 0., c); // c=a⸆b†, c†=ba⸆†
	//	print(c);
	//	BOOST_REQUIRE( c[1][0] == complex(118, 5) and c[1][1] == complex(122, 45) );
	}
	{
		multi::array<complex, 2> c({2, 2});
		using multi::blas::gemm;
		using multi::blas::transposed;
		gemm(1., transposed(a), transposed(b), 0., c); // c=a⸆b⸆, c⸆=ba
		BOOST_REQUIRE( c[1][0] == complex(116, 25) and c[1][1] == complex(186, 65) );
	}
}


BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_implementation, *boost::unit_test::disabled()){
	using multi::blas::gemm;
	multi::array<double, 2> Ccpu({4, 2});
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		using multi::blas::real_operation;
		gemm(real_operation::identity, real_operation::identity, 1., A, B, 0., Ccpu); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
//		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
#if 0
#if 0 // TODO: support c-arrays
	{
		double A[2][3] = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		double B[3][4] = {
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		double C[4][2];
		gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
	}
#endif
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm(1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<double, 2> const A = {
			{1., 9.}, 
			{3., 7.}, 
			{4., 1.}
		};
		multi::array<double, 2> const B = {
			{ 11., 12., 4., 0.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<double, 2> C({4, 2}, 0.);
		gemm('N', 'T', 1., A, B, 0., C); // C^T = A^T*B , C = (A^T*B)^T, C = B^T*A , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n";
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {
			{11., 7., 11.}, 
			{12., 19., 12.}, 
			{4., 1., 4.}, 
			{3., 2., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm('T', 'N', 1., A, B, 0., C); // C^T = A*B^T , C = (A*B^T)^T, C = B*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<double, 2> const A = {
			{1., 9.}, 
			{3., 7.}, 
			{4., 1.}
		};
		multi::array<double, 2> const B = {
			{11., 7., 11.}, 
			{12., 19., 12.}, 
			{4., 1., 4.}, 
			{3., 2., 1.}
		};
		multi::array<double, 2> C({4, 2});
		gemm('N', 'N', 1., A, B, 0., C); // C^T = A^T*B^T , C = (A^T*B^T)^T, C = B*A, if A, B, and C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> C({2, 2});
		gemm('T', 'N', 1., A, A, 0., C); // C^T = C = A*A^T , C = (A*A^T)^T, C = A*A^T = C^T, if A, B, C are c-ordering (e.g. array or array_ref)
		print(rotated(C)) << "---\n"; //{{76., 117., 23., 13.}, {159., 253., 47., 42.}}
	}
	constexpr auto const I = complex{0., 1.};
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({2, 2});
		gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
		print(C) << "---\n";
	//	multi::array<complex, 2> CC({2, 2});
	//	using multi::blas::herk;
	//	herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
	//	print(CC) << "---\n";
	}
#if 0 //TODO: support c-arrays
	{
		complex const A[2][3] = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		complex C[2][2];
		gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
	//	print(C) << "---\n";
	//	complex CC[2][2];
	//	using multi::blas::herk;
	//	herk('U', 'C', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
	//	print(CC) << "---\n";
	}
#endif
	{
		multi::array<complex, 2> const A = {
			{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
			{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
		};
		multi::array<complex, 2> C({3, 3});
		gemm('N', 'C', 1., A, A, 0., C); // C^H = C = A^H*A , C = (A^H*A)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
	//	print(C) << "---\n";
	//	multi::array<complex, 2> CC({3, 3});
	//	using multi::blas::herk;
	//	herk('U', 'N', 1., A, 0., CC); // CC^H = CC = A*A^H, CC = (A*A^H)^H, CC = A*A^H, C lower triangular
	//	print(CC) << "---\n";
	}
	return;
	{
#if 0
	{
		using std::complex;
		{
			multi::array<complex<double>, 2> const A({1000, 1000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.1);
			multi::array<std::complex<double>, 2> C({size(A), size(A)});
			{
			boost::timer::auto_cpu_timer t;
			using multi::blas::herk;
			herk('U', 'C', 1., A, 0., C); // C^H = C = A*A^H, C = (A*A^H)^H, C = A*A^H, C lower triangular. if A, B, C are c-ordering (e.g. array or array_ref)
			}
			cerr << C[10][1] << std::endl;
		}
		{
			multi::array<complex<double>, 2> const A({1000, 1000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.1);
			multi::array<std::complex<double>, 2> C({size(A), size(A)});
			{
			boost::timer::auto_cpu_timer t;
			gemm('C', 'N', 1., A, A, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
			}
			cerr << C[10][1] << std::endl;
		}
		{
			multi::array<complex<double>, 2> const A({2000, 2000}); std::iota(data_elements(A), data_elements(A) + num_elements(A), 0.1);
			multi::array<complex<double>, 2> const B({2000, 2000}); std::iota(data_elements(B), data_elements(B) + num_elements(B), 1.2);
			multi::array<std::complex<double>, 2> C({size(A), size(A)});
			{
			boost::timer::auto_cpu_timer t;
			gemm('C', 'N', 1., A, B, 0., C); // C^H = C = A*A^H , C = (A*A^H)^H, C = A*A^H = C^H, if A, B, C are c-ordering (e.g. array or array_ref)
			}
			cerr << C[1222][134] << std::endl;
		}
	//	print(C) << "---\n";
	//	print(CC) << "---\n";
	}
#endif

	}
	return; 
#if 0
	{
		multi::array<double, 2> const A = {
			{ 1., 1., 1.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 1., 0., 0.},
			{ 1., 0., 0.},
			{ 1., 0., 0.},
		};
		using multi::blas::gemm;

		multi::array<double, 2> C1({3, 3});
		gemm('N', 'N', 1., B, A, 0., C1); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C1);

		multi::array<double, 2> C2({3, 3});
		gemm('N', 'N', 1., A, B, 0., C2); // C = B*A , C^T = (A^T).(B^T) , if A, B, C are c-ordering
		print(C2);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 1., 1.},
			{ 0., 0., 0.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 1., 0., 0., 0., 0.},
			{ 1., 0., 0., 0., 0.},
			{ 1., 0., 0., 0., 0,}
		};
		multi::array<double, 2> C({4, 5});

		using multi::blas::gemm;
		gemm('N', 'N', 1., B, A, 0., C); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2., 3.},
			{ 1., 4., 5.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm('T', 'N', 1., A, A, 0., C); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0.},
			{ 1., 1., 1.}
		};
		multi::array<double, 2> const B = {
			{ 0., 1., 0.},
			{ 0., 1., 0.},
			{ 0., 1., 0.}
		};
		multi::array<double, 2> C({2, 3});

		using multi::blas::gemm;
		gemm('N', 'N', 1., B, A, 0., C); // C = A*B , C^T = (B^T).(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0.},
			{ 0., 0., 1.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 0., 1., 0.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> C({3, 3});

		using multi::blas::gemm; 
		gemm('T', 'T', 1., B, A, 0., C); // C = (B*A)^T , C^T = A*B, C=A^T*B^T , if A, B, C are c-ordering
		print(C);

	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0., 0.},
			{ 0., 1., 0., 0.},
			{ 0., 0., 0., 0.},
			{ 0., 0., 0., 0.},
			{ 0., 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 0., 0., 0., 0., 0.},
			{ 0., 1., 0., 0., 0.},
			{ 0., 0., 0., 0., 0.}
		};
		multi::array<double, 2> C({4, 3});

		using multi::blas::gemm;
		gemm('T', 'T', 1., B, A, 0., C); //C = (B*A)^T, C^T = A*B, C=A^T*B^T, if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 0., 0.},
			{ 0., 1., 0.},
			{ 0., 0., 0.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 0., 0., 0., 0.},
			{ 0., 1., 0., 0.},
		};
		multi::array<double, 2> C({3, 2});

		using multi::blas::gemm;
		gemm('T', 'T', 1., B, A, 0., C); //C = (B*A)^T, C^T = A*B, C=A^T*B^T, if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1.},
			{ 0., 1.}
		};
		multi::array<double, 2> const B = {
			{ 1., 1.},
			{ 0., 0.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm; 
		gemm('T', 'N', 1., B, A, 0., C); // C = A*(B^T) , C^T = B*(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1., 0., 0.},
			{ 0., 1., 0., 0.}
		};
		multi::array<double, 2> const B = {
			{ 1., 1., 1., 1.},
			{ 0., 0., 0., 0.},
			{ 0., 0., 0., 0.}
		};
		multi::array<double, 2> C({2, 3});
		using multi::blas::gemm;
		gemm('T', 'N', 1., B, A, 0., C); // C = A*(B^T) , C^T = B*(A^T) , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1.},
			{ 0., 1.}
		};
		multi::array<double, 2> const B = {
			{ 1., 1.},
			{ 0., 0.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm('N', 'T', 1., B, A, 0., C); // C = ((B^T)*A)^T , C^T = B^T*A, C = (A^T)*B , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 0., 1., 0., 0.},
			{ 0., 1., 0., 0.},
		};
		multi::array<double, 2> const B = {
			{ 1., 1., 1.},
			{ 0., 0., 0.}
		};
		multi::array<double, 2> C({4, 3});
		using multi::blas::gemm;
		gemm('N', 'T', 1., B, A, 0., C); // C = ((B^T)*A)^T , C^T = B^T*A, C = (A^T)*B , if A, B, C are c-ordering
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, B, 0., C); // C = A.B
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), B, 0., C); // C = A^T.B
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, rotated(B), 0., C); // C = A.B^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), rotated(B), 0., C); // C = A^T.B^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, B, 0., rotated(C)); // C^T = A.B, C = B^T.A^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), B, 0., rotated(C)); // C^T = A^T.B, C = B^T.A
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., A, rotated(B), 0., rotated(C)); // C^T = A.B^T, C = B.A^T
		print(C);
	}
	{
		multi::array<double, 2> const A = {
			{ 1., 2.},
			{ 3., 4.}
		};
		multi::array<double, 2> const B = {
			{ 5., 6.},
			{ 7., 8.}
		};
		multi::array<double, 2> C({2, 2});
		using multi::blas::gemm;
		gemm(1., rotated(A), rotated(B), 0., rotated(C)); // C^T = A^T.B^T, C = B.A
		print(C);
	}
#endif
#endif
}

#endif
#endif

