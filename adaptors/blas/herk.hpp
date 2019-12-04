#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -std=c++17 -Wall -Wextra -Wpedantic -D_TEST_MULTI_ADAPTORS_BLAS_HERK $0.cpp -o $0x -lboost_unit_test_framework \
`pkg-config --cflags --libs blas` \
`#-Wl,-rpath,/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -L/usr/local/Wolfram/Mathematica/12.0/SystemFiles/Libraries/Linux-x86-64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5` \
-lboost_timer &&$0x&& rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#ifndef MULTI_ADAPTORS_BLAS_HERK_HPP
#define MULTI_ADAPTORS_BLAS_HERK_HPP

#include "../blas/core.hpp"
#include "../blas/copy.hpp" 
#include "../blas/scal.hpp" 
#include "../blas/syrk.hpp" // fallback to real case

#include "../blas/side.hpp"
#include "../blas/operations.hpp"

//#include<iostream> //debug
//#include<type_traits> // void_t

namespace boost{
namespace multi{namespace blas{

template<class A> auto base_aux(A&& a, std::false_type){return base(a);}
template<class A> auto base_aux(A&& a, std::true_type){return underlying(base(a));}

template<class AA, class BB, class A2D, class C2D, typename = std::enable_if_t<is_complex_array<std::decay_t<C2D>>{}>>
C2D&& herk(fill c_side, complex_operation a_op, AA alpha, A2D const& a, BB beta, C2D&& c){
	using multi::blas::core::herk;
	if(stride(c)==1 and stride(c[0])!=1) herk(flip(c_side), hermitize(a_op), alpha, rotated(a), beta, rotated(c));
	else{
		assert( stride(c[0])==1 ); // sources and destination are incompatible layout
		assert( stride(a[0])==1 ); // sources and destination are incompatible layout
		assert( size(c[0]) == size(c) );
		assert( a_op==complex_operation::hermitian?size(a[0])==size(c):size(a)==size(c) );
		auto base_x = base_aux(a, is_hermitized<A2D>{});
		herk(
			static_cast<char>(c_side), static_cast<char>(a_op), size(c), 
			a_op==complex_operation::hermitian?size(a):size(*begin(a)), 
			alpha, base_x, stride(a), beta, base(c), stride(c)
		);
	}
	return std::forward<C2D>(c);
}

template<class AA, class BB, class A2D, class C2D, typename = std::enable_if_t<not is_complex_array<std::decay_t<C2D>>{}>>
C2D&& herk(fill c_side, complex_operation a_op, AA alpha, A2D const& a, BB beta, C2D&& c, void* = 0){
	syrk(c_side, a_op==complex_operation::hermitian?real_operation::transposition:real_operation::identity, alpha, a, beta, c);
	return std::forward<C2D>(c);
}

template<class AA, class BB, class A2D, class C2D, class = typename A2D::element_ptr>
void herk_aux(fill c_side, AA alpha, A2D const& a, BB beta, C2D&& c, std::true_type){
	herk(c_side, complex_operation::hermitian, alpha, hermitized(a), beta, c);
}

template<class AA, class BB, class A2D, class C2D, class = typename A2D::element_ptr>
void herk_aux(fill c_side, AA alpha, A2D const& a, BB beta, C2D&& c, std::false_type){
	herk(c_side, complex_operation::identity, alpha, a, beta, std::forward<C2D>(c));
}

template<class AA, class BB, class A2D, class C2D, class = typename A2D::element_ptr>
C2D&& herk(fill c_side, AA alpha, A2D const& a, BB beta, C2D&& c){
#if __cpp_if_constexpr>=201606
	if constexpr(is_hermitized<A2D>{}) herk(c_side, complex_operation::hermitian, alpha, hermitized(a), beta, c);
	else                               herk(c_side, complex_operation::identity , alpha, a, beta, std::forward<C2D>(c));
#else
	herk_aux(c_side, alpha, a, beta, std::forward<C2D>(c), is_hermitized<A2D>{});
#endif
	return std::forward<C2D>(c);
}

template<class AA, class A2D, class C2D, class = typename A2D::element_ptr>
C2D&& herk(fill c_side, AA alpha, A2D const& a, C2D&& c){
	return herk(c_side, alpha, a, 0., std::forward<C2D>(c));
}

template<typename AA, class A2D, class C2D>
void herk_aux(AA alpha, A2D const& a, C2D&& c, std::true_type){
	{
		herk(fill::lower, alpha, a, c);
		using multi::rotated;
		using multi::size;
		assert( size(c) == size(rotated(c)) );
		for(typename std::decay_t<C2D>::difference_type i = 0; i != size(c); ++i){
			blas::copy( rotated(c)[i]({i + 1, size(c)}), c[i]({i+1, size(c)}) );
		//	blas::copy(rotated(c)[i]({i + 1, size(c)})
		//	blas::copy(begin(rotated(c)[i])+i+1, end(rotated(c)[i]), begin(c[i])+i+1);
			blas::scal(-1., begin(imag(c[i]))+i+1, end(imag(c[i])));
		}
	}
}

template<typename A, class A2D, class C2D>
void herk_aux(A alpha, A2D const& a, C2D&& c, std::false_type){syrk(alpha, a, c);}

template<typename AA, class A2D, class C2D>
C2D&& herk(AA alpha, A2D const& a, C2D&& c){
#if __cpp_if_constexpr>=201606
	if constexpr(is_complex_array<std::decay_t<C2D>>{}){
		herk(fill::lower, alpha, a, c);
		using multi::rotated;
		using multi::size;
		for(typename std::decay_t<C2D>::difference_type i = 0; i != size(c); ++i){
			blas::copy( rotated(c)[i]({i + 1, size(c)}), c[i]({i+1, size(c)}) );
		//	blas::copy(begin(rotated(c)[i])+i+1, end(rotated(c)[i]), begin(c[i])+i+1);
			blas::scal(-1., begin(imag(c[i]))+i+1, end(imag(c[i])));
		}
	}else syrk(alpha, a, c);
#else
	herk_aux(alpha, a, std::forward<C2D>(c), is_complex_array<std::decay_t<C2D>>{});
#endif
	return std::forward<C2D>(c);
}

template<class AA, class A2D, class Ret = typename A2D::decay_type>
NODISCARD("second argument is const")
auto herk(AA alpha, A2D const& a){
	auto s = size(a);
	Ret ret(typename Ret::extensions_type{s, s}, get_allocator(a));
	assert( size(ret)==size(*begin(ret)) );
	herk(alpha, a, ret);
	return ret;
}

template<class A2D> auto herk(A2D const& a){return herk(1., a);}

}}

}

#if _TEST_MULTI_ADAPTORS_BLAS_HERK

//#include "../blas/gemm.hpp" TODO: test herk againt gemm

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>


namespace multi = boost::multi;

#include<catch.hpp>
#include "../../array.hpp"

#include<iostream>
#include<numeric>

#include <boost/timer/timer.hpp>

template<class M> decltype(auto) print(M const& C){
	using std::cout;
	using boost::multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j) cout << C[i][j] << ' ';
		cout << std::endl;
	}
	return cout << std::endl;
}

using complex = std::complex<double>;
constexpr auto const I = complex{0., 1.};

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_real_case){
	multi::array<complex, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	using multi::blas::fill;
	using multi::blas::operation;
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		assert( size(c) == size(*begin(a)) );
		herk(fill::lower, operation::hermitian, 1., a, 0., c);//c†=c=a†a=(a†a)†, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(19.,0.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		herk(fill::upper, operation::hermitian, 1., a, 0., c);//c†=c=a†a=(a†a)†, `c` in lower triangular
		BOOST_REQUIRE( c[1][2]==complex(19.,0.) );
		BOOST_REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> const a_rot = rotated(a);
		auto&& a_ = rotated(a_rot);
		multi::array<complex, 2> c({3, 3}, 9999.);
		auto&& c_ = rotated(c);
		herk(fill::upper, operation::hermitian, 1., a_, 0., c_);//c_†=c_=a_†a_=(a_†a_)†, `c_` in lower triangular
		BOOST_REQUIRE( c_[1][2]==complex(19.,0.) );
		BOOST_REQUIRE( c_[2][1]==9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_basic_transparent_interface_special_case){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I}
	};
	using multi::blas::fill;
	using multi::blas::complex_operation;	
	{
		multi::array<complex, 2> c({1, 1}, 9999.);
		using multi::blas::herk;
		herk(fill::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `a` and `c` are c-ordering, information in c lower triangular
		BOOST_REQUIRE( c[0][0]==complex(40., 0.) );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_basic_transparent_interface){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	using multi::blas::fill;
	using multi::blas::complex_operation;
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::herk;
		herk(fill::lower, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, information in `c` lower triangular
		BOOST_REQUIRE( c[2][1]==complex(41.,2.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::herk;
		herk(fill::upper, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, `c` in upper triangular
		BOOST_REQUIRE( c[1][2]==complex(41., -2.) );
		BOOST_REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(fill::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `a` and `c` are c-ordering, information in c lower triangular
		BOOST_REQUIRE( c[1][0]==complex(50., -49.) );
		BOOST_REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(fill::upper, complex_operation::identity, 1., a, 0., c); //c†=c=aa†, `c` in upper triangular
		BOOST_REQUIRE( c[0][1]==complex(50., 49.) );
		BOOST_REQUIRE( c[1][0]==9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_basic_enum_interface){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	using multi::blas::fill;
	using multi::blas::complex_operation;
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		herk(fill::lower, complex_operation::hermitian, 1., a, 0., c); //c†=c=a†a=(a†a)†, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(41.,2.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using namespace multi::blas;
		herk(fill::upper, complex_operation::hermitian, 1., a, 0., c); //c†=c=a†a=(a†a)†, `c` in upper triangular
		BOOST_REQUIRE( c[1][2]==complex(41., -2.) );
		BOOST_REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(fill::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `c` in lower triangular
		BOOST_REQUIRE( c[1][0]==complex(50., -49.) );
		BOOST_REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(fill::upper, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `c` in upper triangular
		BOOST_REQUIRE( c[0][1]==complex(50., 49.) );
		BOOST_REQUIRE( c[1][0]==9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_basic_explicit_enum_interface){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::fill;
		using multi::blas::complex_operation;
		herk(fill::lower, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(41.,2.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::fill;
		using multi::blas::complex_operation;
		herk(fill::upper, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, `c` in upper triangular
		BOOST_REQUIRE( c[1][2]==complex(41., -2.) );
		BOOST_REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(fill::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†=(aa†)†, `c` in lower triangular
		BOOST_REQUIRE( c[1][0]==complex(50., -49.) );
		BOOST_REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(fill::upper, complex_operation::identity, 1., a, 0., c); // c†=c=aa†=(aa†)†, `c` in upper triangular
		BOOST_REQUIRE( c[0][1]==complex(50., 49.) );
		BOOST_REQUIRE( c[1][0]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(fill::upper, complex_operation::identity, 1., a, 0., c); // c†=c=aa†=(aa†)†, `c` in upper triangular
		BOOST_REQUIRE( c[0][1]==complex(50., 49.) );
		BOOST_REQUIRE( c[1][0]==9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_automatic_operator_interface){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::fill;
		using multi::blas::hermitized;
		herk(fill::lower, 1., hermitized(a), 0., c); // c=c†=a†a, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(41., 2.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::fill;
		herk(fill::lower, 1., a, 0., c); // c=c†=aa†, `c` in lower triangular
		BOOST_REQUIRE( c[1][0]==complex(50., -49.) );
		BOOST_REQUIRE( c[0][1]==9999. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_automatic_operator_interface_implicit_no_sum){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::fill;
		using multi::blas::hermitized;
		herk(fill::lower, 1., hermitized(a), c); // c=c†=a†a, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(41., 2.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::fill;
		herk(fill::lower, 1., a, c); // c=c†=aa†, `c` in lower triangular
		BOOST_REQUIRE( c[1][0]==complex(50., -49.) );
		BOOST_REQUIRE( c[0][1]==9999. );
	}
}

template<class T> void what(T&&);

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_automatic_ordering_and_symmetrization){

	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::hermitized;
		herk(1., hermitized(a), c); // c†=c=a†a
		BOOST_REQUIRE( c[2][1]==complex(41.,  2.) );
		BOOST_REQUIRE( c[1][2]==complex(41., -2.) );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(1., a, c); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == complex(50., -49.) );
		BOOST_REQUIRE( c[0][1] == complex(50., +49.) );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(1., a); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == complex(50., -49.) );
		BOOST_REQUIRE( c[0][1] == complex(50., +49.) );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(1., hermitized(a)); // c†=c=a†a

		BOOST_REQUIRE( size(hermitized(a))==3 );		
		BOOST_REQUIRE( c[2][1] == complex(41., +2.) );
		BOOST_REQUIRE( c[1][2] == complex(41., -2.) );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(a); // c†=c=a†a
//		what(multi::pointer_traits<decltype(base(a))>::default_allocator_of(base(a)));
		BOOST_REQUIRE( c[1][0] == complex(50., -49.) );
		BOOST_REQUIRE( c[0][1] == complex(50., +49.) );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(hermitized(a)); // c†=c=a†a
		BOOST_REQUIRE( c[2][1] == complex(41., +2.) );
		BOOST_REQUIRE( c[1][2] == complex(41., -2.) );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_automatic_ordering_and_symmetrization_real_case){

	multi::array<complex, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::hermitized;
		herk(1., hermitized(a), c); // c†=c=a†a
		BOOST_REQUIRE( c[2][1]==19. );
		BOOST_REQUIRE( c[1][2]==19. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(1., a, c); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(1., a); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(1., hermitized(a)); // c†=c=a†a

		BOOST_REQUIRE( size(hermitized(a))==3 );		
		BOOST_REQUIRE( c[2][1]==19. );
		BOOST_REQUIRE( c[1][2]==19. );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(a); // c†=c=a†a
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(hermitized(a)); // c†=c=a†a
		BOOST_REQUIRE( c[2][1]==19. );
		BOOST_REQUIRE( c[1][2]==19. );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_real_automatic_ordering_and_symmetrization_real_case){

	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	{
		multi::array<double, 2> c({3, 3}, 9999.);
		using multi::blas::hermitized;
		using multi::blas::herk;
		herk(1., hermitized(a), c); // c†=c=a†a
		BOOST_REQUIRE( c[2][1]==19. );
		BOOST_REQUIRE( c[1][2]==19. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(1., a, c); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		multi::array<double, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(1., a, c); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::herk;
		multi::array<double, 2> c = herk(1., a); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
#if 0
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(1., hermitized(a)); // c†=c=a†a

		BOOST_REQUIRE( size(hermitized(a))==3 );		
		REQUIRE( c[2][1]==19. );
		REQUIRE( c[1][2]==19. );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(a); // c†=c=a†a
		BOOST_REQUIRE( c[1][0] == 34. );
		BOOST_REQUIRE( c[0][1] == 34. );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(hermitized(a)); // c†=c=a†a
		BOOST_REQUIRE( c[2][1]==19. );
		BOOST_REQUIRE( c[1][2]==19. );
	}
#endif
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_real_case){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	using multi::blas::fill;
	using multi::blas::operation;
	{
		static_assert( not boost::multi::blas::is_complex_array<multi::array<double, 2>>{} , "!");
		multi::array<double, 2> c({2, 2}, 9999.);
		syrk(fill::lower, operation::identity, 1., a, 0., c);//c†=c=aa†=(aa†)†, `c` in lower triangular
	}
	{
		static_assert( not boost::multi::blas::is_complex_array<multi::array<double, 2>>{} , "!");
		multi::array<double, 2> c({2, 2}, 9999.);
		herk(fill::lower, operation::identity, 1., a, 0., c);//c†=c=aa†=(aa†)†, `c` in lower triangular
	}
	{
		static_assert( not boost::multi::blas::is_complex_array<multi::array<double, 2>>{} , "!");
		using multi::blas::herk;
		multi::array<double, 2> c = herk(a);//c†=c=aa†=(aa†)†, `c` in lower triangular
	}
	{
		static_assert( not boost::multi::blas::is_complex_array<multi::array<double, 2>>{} , "!");
		using multi::blas::herk;
		using multi::blas::hermitized;
		auto&& aa = hermitized(a);
		BOOST_REQUIRE( aa == multi::blas::transposed(a) );
		static_assert( not boost::multi::blas::is_complex_array<std::decay_t<decltype(aa)>>{} , "!");

		multi::array<double, 2> c = herk(hermitized(a));//c†=c=aa†=(aa†)†, `c` in lower triangular
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_herk_complex_timing){
	multi::array<complex, 2> const a({4000, 4000}); std::iota(data_elements(a), data_elements(a) + num_elements(a), 0.2);
	multi::array<complex, 2> c({4000, 4000}, 9999.);
	boost::timer::auto_cpu_timer t;
	using multi::blas::herk;
	using multi::blas::hermitized;
	herk(1., hermitized(a), c); // c†=c=a†a
}

#endif
#endif

