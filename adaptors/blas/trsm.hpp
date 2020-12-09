#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXXX $CXXFLAGS $0 -o $0.$X -lboost_unit_test_framework `pkg-config --cflags --libs blas` -lboost_timer&&$0.$X&&rm $0.$X;exit
#endif
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_BLAS_TRSM_HPP
#define MULTI_ADAPTORS_BLAS_TRSM_HPP

#include "../blas/core.hpp"

#include "../blas/operations.hpp" // uplo
#include "../blas/filling.hpp"
#include "../blas/side.hpp"

#include "../../config/NODISCARD.hpp"

namespace boost{
namespace multi{namespace blas{

enum class diagonal : char{
	unit = 'U', 
	non_unit = 'N', general = non_unit
};

template<class A, std::enable_if_t<not is_conjugated<A>{}, int> =0> 
auto trsm_base_aux(A&& a){return base(std::forward<A>(a));}

template<class A, std::enable_if_t<    is_conjugated<A>{}, int> =0>
auto trsm_base_aux(A&& a){return underlying(base(a));}

using core::trsm;

template<class A2D, class B2D>
decltype(auto) trsm_move(filling a_nonz, diagonal a_diag, typename A2D::element_type alpha, A2D const& a, B2D&& b, side s = side::left)
//->decltype(
//	trsm('R', 'X', 'N', 'D', size(rotated(b)), b.size(), alpha, trsm_base_aux(a), stride(a)         , trsm_base_aux(b), b.stride())
//	,
//	std::forward<B2D>(b)
//)
{
	if(s==side::left) assert( size(rotated(a)) == size(b) );
	else              assert( size(rotated(b)) == size(a) );

	if(size(b)==0) return std::forward<B2D>(b);

	auto base_a = trsm_base_aux(a);
	auto base_b = trsm_base_aux(b);

	using core::trsm;
	if(size(rotated(b))==1){
		if(stride(rotated(a))==1) trsm(static_cast<char>(s), static_cast<char>(+a_nonz), 'N', static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, stride(a)         , base_b, stride(b));
		else if(stride(a)==1){		
			if(not is_conjugated<A2D>{}) trsm(static_cast<char>(s), static_cast<char>(-a_nonz), 'T', static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, stride(rotated(a)), base_b, stride(b));
			else                  trsm(static_cast<char>(s), static_cast<char>(-a_nonz), 'C', static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, stride(rotated(a)), base_b, stride(b));
		}
	}else{
		      if(stride(rotated(a))==1 and stride(rotated(b))==1){
			assert(not is_conjugated<A2D>{});
			trsm(static_cast<char>(s), static_cast<char>(+a_nonz), 'N', static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, stride(a)         , base_b, stride(b));
		}else if(stride(a)==1 and stride(rotated(b))==1){
			if(not is_conjugated<A2D>{}) trsm(static_cast<char>(s), static_cast<char>(-a_nonz), 'T', static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, stride(rotated(a)), base_b, stride(b));
			else                  trsm(static_cast<char>(s), static_cast<char>(-a_nonz), 'C', static_cast<char>(a_diag), size(rotated(b)), size(b), alpha, base_a, stride(rotated(a)), base_b, stride(b));
		}
		else if(stride(a)==1 and stride(b)==1){
			assert(not is_conjugated<A2D>{});
			                      trsm(static_cast<char>(swap(s)), static_cast<char>(-a_nonz), 'N', static_cast<char>(a_diag), size(rotated(a)), size(rotated(b)), alpha, base_a, stride(rotated(a)), base_b, stride(rotated(b)));
		}else if(stride(rotated(a))==1 and stride(b)==1){
			if(not is_conjugated<A2D>{}) trsm(static_cast<char>(swap(s)), static_cast<char>(+a_nonz), 'T', static_cast<char>(a_diag), size(a), size(rotated(b)), alpha, base_a, stride(a), base_b, stride(rotated(b)));
			else                  trsm(static_cast<char>(swap(s)), static_cast<char>(+a_nonz), 'C', static_cast<char>(a_diag), size(a), size(rotated(b)), alpha, base_a, stride(a), base_b, stride(rotated(b)));
		}
		else assert(0); // case is not part of BLAS
	}
	return std::forward<B2D>(b);
}

template<typename AA, class A2D, class B2D>
auto trsm(filling a_nonz, diagonal a_diag, AA alpha, A2D const& a, B2D&& b)
->decltype(trsm_move(a_nonz, a_diag, alpha, a, std::forward<B2D>(b)), std::declval<B2D&&>())
{
	if(not is_conjugated<B2D>{}) trsm_move( a_nonz, a_diag, alpha,            a , std::forward<B2D>(b));
	else                         trsm_move(-a_nonz, a_diag, alpha, hermitized(a), rotated(b), side::right);
	return std::forward<B2D>(b);
}

//template<typename AA, class A2D, class B2D>
//NODISCARD("because last argument is const")
//auto trsm(filling a_nonz, diagonal a_diag, AA alpha, A2D const& a, B2D const& b){
//	auto bcpy = decay(b);
//	return trsm_move(a_nonz, a_diag, alpha, a, bcpy);
//}

template<class AA, class A2D, class B2D>
auto trsm(filling a_nonz, AA alpha, A2D const& a, B2D&& b)
->decltype(trsm(a_nonz, diagonal::general, alpha, a, std::forward<B2D>(b))){
	return trsm(a_nonz, diagonal::general, alpha, a, std::forward<B2D>(b));}

//template<class AA, class A2D, class B2D>
//NODISCARD("because input argument is const")
//auto trsm(filling a_nonz, AA alpha, A2D const& a, B2D const& b)
//->decltype(trsm(a_nonz, diagonal::general, alpha, a, std::forward<B2D>(b))){
//	return trsm(a_nonz, diagonal::general, alpha, a, std::forward<B2D>(b));}

template<class A2D, class B2D, class T = typename A2D::element_type>
auto trsm(filling a_nonz, A2D const& a, B2D&& b)
->decltype(trsm(a_nonz, T{1.}, a, std::forward<B2D>(b))){
	return trsm(a_nonz, T{1.}, a, std::forward<B2D>(b));}

//template<class A2D, class B2D, class T = typename A2D::element_type>
//NODISCARD("because last argument is const")
//auto trsm(filling a_nonz, A2D const& a, B2D const& b)
////	return true;
////}
//->decltype(trsm(a_nonz, T{1.}, a, std::forward<B2D>(b))){
//	return trsm(a_nonz, T{1.}, a, std::forward<B2D>(b));}

}}}

#if defined(__INCLUDE_LEVEL__) and not __INCLUDE_LEVEL__

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi.BLAS trsm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>
#include<boost/test/tools/floating_point_comparison.hpp>

#include "../blas/gemm.hpp"

#include "../../array.hpp"

namespace multi = boost::multi;

#include<iostream>
#include<vector>

template<class M> decltype(auto) print(M const& C){
	using boost::multi::size; using std::cout;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j)
			cout<< C[i][j] <<' ';
		cout<<std::endl;
	}
	return cout<<std::endl;
}

namespace utf = boost::unit_test;

#if 0

	using multi::blas::side;
	using multi::blas::filling;
	using multi::blas::diagonal;
		using multi::blas::gemm;




		{
			auto const AT = rotated(A).decay();
			auto const BT = rotated(B).decay();
			using multi::blas::trsm;
		//	auto const Bck=gemm(A, trsm(rotated(AT), rotated(BT)));
		//	for(int i{};i<3;++i)for(int j{};j<size(rotated(B));++j) BOOST_CHECK_SMALL(Bck[i][j]-B[i][j], 0.00001);
		}
		{
			using multi::blas::trsm;
		//	auto const Bck=gemm(A, trsm(A, B));
		//	for(int i{};i<3;++i)for(int j{};j<size(rotated(B));++j) BOOST_CHECK_SMALL(Bck[i][j]-B[i][j], 0.00001);
		}
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_nonsquare_default_diagonal_one_check, *utf::tolerance(0.00001)){
	using complex = std::complex<double>; complex const I{0, 1};
	multi::array<complex, 2> const A = {
		{ 1. + 4.*I,  3.,  4.- 10.*I},
		{ 0.,  7.- 3.*I,  1.},
		{ 0.,  0.,  8.- 2.*I}
	};
	using multi::blas::side;
	using multi::blas::filling;
	using multi::blas::diagonal;
	{
		multi::array<complex, 2> const B = {
			{1. + 1.*I},
			{2. + 1.*I},
			{3. + 1.*I},
		};
		using multi::blas::gemm;
		{
			auto S = trsm(filling::upper, diagonal::general, 1., A, B);
			BOOST_TEST( real(S[2][0]) == 0.323529 );
		}
		{
			auto const BT = +rotated(B);
			auto S = trsm(filling::upper, diagonal::general, 1., A, rotated(BT));
			BOOST_TEST( real(S[2][0]) == 0.323529 );
		}
		{
			auto const AT = +rotated(A);
			auto S = trsm(filling::upper, diagonal::general, 1., rotated(AT), B);
			BOOST_TEST( real(S[2][0]) == 0.323529 );
		}
		{
			auto const AT = +rotated(A);
			auto const BT = +rotated(B);
			auto S = trsm(filling::upper, diagonal::general, 1., rotated(AT), rotated(BT));
			BOOST_TEST( real(S[2][0]) == 0.323529 );
		}
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_trsm_complex_nonsquare_default_diagonal_gemm_check, *utf::tolerance(0.00001)){
	using complex = std::complex<double>; complex const I{0, 1};
	multi::array<complex, 2> const A = {
		{ 1. + 4.*I,  3.,  4.- 10.*I},
		{ 0.,  7.- 3.*I,  1.},
		{ 0.,  0.,  8.- 2.*I}
	};
	using multi::blas::side;
	using multi::blas::filling;
	using multi::blas::diagonal;
	{
		multi::array<complex, 2> const B = {
			{1. + 1.*I, 5. + 3.*I},
			{2. + 1.*I, 9. + 3.*I},
			{3. + 1.*I, 1. - 1.*I},
		};
		using multi::blas::gemm;
		{
			auto S = trsm(filling::upper, diagonal::general, 1., A, B); // S = Ainv.B
			BOOST_TEST( real(S[2][1]) == 0.147059  );
		}
		{
			auto const BT = +rotated(B);
			auto S = trsm(filling::upper, diagonal::general, 1., A, rotated(BT));
			BOOST_TEST( real(S[2][1]) == 0.147059  );
		}
		{
			auto const AT = +rotated(A);
			auto S = trsm(filling::upper, diagonal::general, 1., rotated(AT), B);
			BOOST_TEST( real(S[2][1]) == 0.147059  );
		}
		{
			auto const AT = +rotated(A);
			auto const BT = +rotated(B);
			auto S = trsm(filling::upper, diagonal::general, 1., rotated(AT), rotated(BT));
			BOOST_TEST( real(S[2][1]) == 0.147059  );
		}
	}
}

#endif
#endif
#endif

