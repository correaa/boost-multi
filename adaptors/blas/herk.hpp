#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -std=c++17 -Wall -Wextra -Wpedantic `#-Wfatal-errors` -D_TEST_MULTI_ADAPTORS_BLAS_HERK .DCATCH_CONFIG_MAIN $0.cpp -o $0x \
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

#include "../blas/operations.hpp"

#include<type_traits> // void_t

namespace boost{
namespace multi{namespace blas{

//enum class trans : char{N='N', C='C'};

//enum class triangular : char{
//	lower = static_cast<char>(uplo::U),
//	upper = static_cast<char>(uplo::L),
//};

//triangular flip(triangular side){
//	switch(side){
//		case triangular::lower: return triangular::upper;
//		case triangular::upper: return triangular::lower;
//	}
//}

//enum class complex_operation : char{
//	hermitian = static_cast<char>(trans::N),
//	identity  = static_cast<char>(trans::C),
//};

complex_operation hermitize(complex_operation op){
	switch(op){
		case complex_operation::hermitian: return complex_operation::identity ;
		case complex_operation::identity : return complex_operation::hermitian;
	} __builtin_unreachable();
}

template<class AA, class BB, class A2D, class C2D, typename = std::enable_if_t< is_complex_array<std::decay_t<C2D>>{}>
>
C2D&& herk(triangular c_side, complex_operation a_op, AA alpha, A2D const& a, BB beta, C2D&& c){
	if(stride(c)!=1){
		assert( stride(a)!=1 ); // sources and destination are incompatible layout
		assert( size(c)==(a_op==complex_operation::hermitian?size(*begin(a)):size(a)) );
		herk(
			static_cast<char>(c_side), static_cast<char>(a_op), size(c), 
			a_op==complex_operation::hermitian?size(a):size(*begin(a)), 
			alpha, base(a), stride(a), beta, base(c), stride(c)
		);
	}else herk(flip(c_side), hermitize(a_op), alpha, rotated(a), beta, rotated(c));
	return std::forward<C2D>(c);
}

template<class AA, class BB, class A2D, class C2D, typename = std::enable_if_t<not is_complex_array<std::decay_t<C2D>>{}>>
C2D&& herk(triangular c_side, complex_operation a_op, AA alpha, A2D const& a, BB beta, C2D&& c, void* = 0){
	syrk(c_side, a_op==complex_operation::hermitian?real_operation::transposition:real_operation::identity, alpha, a, beta, c);
	return std::forward<C2D>(c);
}

template<class A>
constexpr bool is_hermitized(A&& a){
	using ptr = decltype(base(a));
	return
		   std::is_same<std::decay_t<ptr>, boost::multi::blas::detail::conjugater<const std::complex<double>*>>{}
		or std::is_same<std::decay_t<ptr>, boost::multi::blas::detail::conjugater<      std::complex<double>*>>{}
	;
}

template<class AA, class BB, class A2D, class C2D, class = typename A2D::element_ptr>
C2D&& herk(triangular c_side, AA alpha, A2D const& a, BB beta, C2D&& c){
	if(is_hermitized(a))
		herk(c_side, complex_operation::hermitian, alpha, hermitized(a), beta, c);
	else
		herk(c_side, complex_operation::identity, alpha, a, beta, std::forward<C2D>(c));
	return std::forward<C2D>(c);
}

template<class AA, class A2D, class C2D, class = typename A2D::element_ptr>
C2D&& herk(triangular c_side, AA alpha, A2D const& a, C2D&& c){
	return herk(c_side, alpha, a, 0., std::forward<C2D>(c));
}

template<typename AA, class A2D, class C2D>
C2D&& herk(AA alpha, A2D const& a, C2D&& c){
	herk(triangular::lower, alpha, a, c);
	using multi::rotated;
	using multi::size;
	for(typename std::decay_t<C2D>::difference_type i = 0; i != size(c); ++i){
		blas::copy(begin(rotated(c)[i])+i+1, end(rotated(c)[i]), begin(c[i])+i+1);
		blas::scal(-1., begin(imag(c[i]))+i+1, end(imag(c[i])));
	}
	return std::forward<C2D>(c);
}

template<class AA, class A2D>
auto herk(AA alpha, A2D const& A){
	multi::array<typename A2D::element, 2> ret({size(A), size(A)});
	herk(alpha, A, ret);
	return ret;
}

template<class A2D, class R = typename A2D::decay_type>
auto herk(A2D const& A){return herk(1., A);}

}}

}

#if _TEST_MULTI_ADAPTORS_BLAS_HERK

//#include "../blas/gemm.hpp" TODO: test herk againt gemm

namespace multi = boost::multi;

#include<catch.hpp>

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

TEST_CASE("multi::blas::herk complex (real case)", "[report]"){
	multi::array<complex, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	using multi::blas::triangular;
	using multi::blas::operation;
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		assert( size(c) == size(*begin(a)) );
		herk(triangular::lower, operation::hermitian, 1., a, 0., c);//c†=c=a†a=(a†a)†, `c` in lower triangular
		REQUIRE( c[2][1]==complex(19.,0.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		herk(triangular::upper, operation::hermitian, 1., a, 0., c);//c†=c=a†a=(a†a)†, `c` in lower triangular
		REQUIRE( c[1][2]==complex(19.,0.) );
		REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> const a_rot = rotated(a);
		auto&& a_ = rotated(a_rot);
		multi::array<complex, 2> c({3, 3}, 9999.);
		auto&& c_ = rotated(c);
		herk(triangular::upper, operation::hermitian, 1., a_, 0., c_);//c_†=c_=a_†a_=(a_†a_)†, `c_` in lower triangular
		REQUIRE( c_[1][2]==complex(19.,0.) );
		REQUIRE( c_[2][1]==9999. );
	}
}

TEST_CASE("multi::blas::herk complex basic transparent interface", "[report]"){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	using multi::blas::triangular;
	using multi::blas::complex_operation;
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::herk;
		herk(triangular::lower, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, information in `c` lower triangular
		REQUIRE( c[2][1]==complex(41.,2.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::herk;
		herk(triangular::upper, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, `c` in upper triangular
		REQUIRE( c[1][2]==complex(41., -2.) );
		REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(triangular::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `a` and `c` are c-ordering, information in c lower triangular
		REQUIRE( c[1][0]==complex(50., -49.) );
		REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(triangular::upper, complex_operation::identity, 1., a, 0., c); //c†=c=aa†, `c` in upper triangular
		REQUIRE( c[0][1]==complex(50., 49.) );
		REQUIRE( c[1][0]==9999. );
	}
}

TEST_CASE("multi::blas::herk complex basic enum interface", "[report]"){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	using multi::blas::triangular;
	using multi::blas::complex_operation;
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		herk(triangular::lower, complex_operation::hermitian, 1., a, 0., c); //c†=c=a†a=(a†a)†, `c` in lower triangular
		REQUIRE( c[2][1]==complex(41.,2.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using namespace multi::blas;
		herk(triangular::upper, complex_operation::hermitian, 1., a, 0., c); //c†=c=a†a=(a†a)†, `c` in upper triangular
		REQUIRE( c[1][2]==complex(41., -2.) );
		REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(triangular::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `c` in lower triangular
		REQUIRE( c[1][0]==complex(50., -49.) );
		REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(triangular::upper, complex_operation::identity, 1., a, 0., c); // c†=c=aa†, `c` in upper triangular
		REQUIRE( c[0][1]==complex(50., 49.) );
		REQUIRE( c[1][0]==9999. );
	}
}

TEST_CASE("multi::blas::herk complex basic explicit enum interface", "[report]"){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::complex_operation;
		herk(triangular::lower, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, `c` in lower triangular
		REQUIRE( c[2][1]==complex(41.,2.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::complex_operation;
		herk(triangular::upper, complex_operation::hermitian, 1., a, 0., c); // c†=c=a†a=(a†a)†, `c` in upper triangular
		REQUIRE( c[1][2]==complex(41., -2.) );
		REQUIRE( c[2][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(triangular::lower, complex_operation::identity, 1., a, 0., c); // c†=c=aa†=(aa†)†, `c` in lower triangular
		REQUIRE( c[1][0]==complex(50., -49.) );
		REQUIRE( c[0][1]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(triangular::upper, complex_operation::identity, 1., a, 0., c); // c†=c=aa†=(aa†)†, `c` in upper triangular
		REQUIRE( c[0][1]==complex(50., 49.) );
		REQUIRE( c[1][0]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using namespace multi::blas;
		herk(triangular::upper, complex_operation::identity, 1., a, 0., c); // c†=c=aa†=(aa†)†, `c` in upper triangular
		REQUIRE( c[0][1]==complex(50., 49.) );
		REQUIRE( c[1][0]==9999. );
	}
}

TEST_CASE("multi::blas::herk complex automatic operator interface", "[report]"){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::hermitized;
		herk(triangular::lower, 1., hermitized(a), 0., c); // c=c†=a†a, `c` in lower triangular
		REQUIRE( c[2][1]==complex(41., 2.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		herk(triangular::lower, 1., a, 0., c); // c=c†=aa†, `c` in lower triangular
		REQUIRE( c[1][0]==complex(50., -49.) );
		REQUIRE( c[0][1]==9999. );
	}
}

TEST_CASE("multi::blas::herk complex automatic operator interface implicit no-sum", "[report]"){
	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::hermitized;
		herk(triangular::lower, 1., hermitized(a), c); // c=c†=a†a, `c` in lower triangular
		REQUIRE( c[2][1]==complex(41., 2.) );
		REQUIRE( c[1][2]==9999. );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::triangular;
		herk(triangular::lower, 1., a, c); // c=c†=aa†, `c` in lower triangular
		REQUIRE( c[1][0]==complex(50., -49.) );
		REQUIRE( c[0][1]==9999. );
	}
}

TEST_CASE("multi::blas::herk complex automatic ordering and symmetrization", "[report]"){

	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::hermitized;
		herk(1., hermitized(a), c); // c†=c=a†a
		REQUIRE( c[2][1]==complex(41.,  2.) );
		REQUIRE( c[1][2]==complex(41., -2.) );
	}
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		herk(1., a, c); // c†=c=aa†
		REQUIRE( c[1][0] == complex(50., -49.) );
		REQUIRE( c[0][1] == complex(50., +49.) );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(1., a); // c†=c=aa†
		REQUIRE( c[1][0] == complex(50., -49.) );
		REQUIRE( c[0][1] == complex(50., +49.) );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(1., hermitized(a)); // c†=c=a†a
		assert( c[2][1] == complex(41., +2.) );
		assert( c[1][2] == complex(41., -2.) );
	}
	{
		using multi::blas::herk;
		multi::array<complex, 2> c = herk(a); // c†=c=a†a
		REQUIRE( c[1][0] == complex(50., -49.) );
		REQUIRE( c[0][1] == complex(50., +49.) );
	}
	{
		using multi::blas::herk;
		using multi::blas::hermitized;
		multi::array<complex, 2> c = herk(hermitized(a)); // c†=c=a†a
		assert( c[2][1] == complex(41., +2.) );
		assert( c[1][2] == complex(41., -2.) );
	}
}

TEST_CASE("multi::blas::herk real case", "[report]"){
	multi::array<double, 2> const a = {
		{ 1., 3., 4.},
		{ 9., 7., 1.}
	};
	using multi::blas::triangular;
	using multi::blas::operation;
	{
		static_assert( not boost::multi::blas::is_complex_array<multi::array<double, 2>>{} );
		multi::array<double, 2> c({2, 2}, 9999.);
		syrk(triangular::lower, operation::identity, 1., a, 0., c);//c†=c=aa†=(aa†)†, `c` in lower triangular
	}
	{
		static_assert( not boost::multi::blas::is_complex_array<multi::array<double, 2>>{} );
		multi::array<double, 2> c({2, 2}, 9999.);
		herk(triangular::lower, operation::identity, 1., a, 0., c);//c†=c=aa†=(aa†)†, `c` in lower triangular
	}
}

TEST_CASE("multi::blas::herk complex timing", "[report]"){
	multi::array<complex, 2> const a({4000, 4000}); std::iota(data_elements(a), data_elements(a) + num_elements(a), 0.2);
	multi::array<complex, 2> c({4000, 4000}, 9999.);
	boost::timer::auto_cpu_timer t;
	using multi::blas::herk;
	using multi::blas::hermitized;
	herk(1., hermitized(a), c); // c†=c=a†a
}


#endif
#endif

