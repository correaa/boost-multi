#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -std=c++17 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_GEMM $0.cpp -o $0x -lboost_unit_test_framework \
`pkg-config --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_GEMM_HPP
#define MULTI_ADAPTORS_BLAS_GEMM_HPP

#include "../blas/core.hpp"
#include "../blas/operations.hpp"

#if __cplusplus>=201703L and __has_cpp_attribute(nodiscard)>=201603
#define NODISCARD(MsG) [[nodiscard]]
#elif __has_cpp_attribute(gnu::warn_unused_result)
#define NODISCARD(MsG) [[gnu::warn_unused_result]]
#else
#define NODISCARD(MsG)
#endif

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
C2D&& gemm_aux2(AA a, A2D const& A, B2D const& B, BB b, C2D&& C, std::false_type){
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

template<class AA, class A2D, class B2D, class C2D = typename A2D::decay_type>
NODISCARD("second argument is const")
auto gemm(AA a, A2D const& A, B2D const& B){
	assert(get_allocator(A) == get_allocator(B));
	C2D ret({size(A), size(rotated(B))});
	gemm(a, A, B, 0., ret);
	return ret;
}

template<class A2D, class B2D> auto gemm(A2D const& A, B2D const& B){return gemm(1., A, B);}

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

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare_automatic){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{ 1. + 2.*I, 3. - 3.*I, 1.-9.*I},
		{ 9. + 1.*I, 7. + 4.*I, 1.-8.*I},
	};
	multi::array<complex, 2> const b = {	
		{ 11.+1.*I, 12.+1.*I, 4.+1.*I, 8.-2.*I},
		{  7.+8.*I, 19.-2.*I, 2.+1.*I, 7.+1.*I},
		{  5.+1.*I,  3.-1.*I, 3.+8.*I, 1.+1.*I}
	};
	{
		multi::array<complex, 2> c({2, 4});
		using multi::blas::gemm;
		gemm(1., a, b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
	}
}

#endif
#endif

