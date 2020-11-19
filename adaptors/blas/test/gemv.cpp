#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
echo $X;
mkdir -p build.$X && cd build.$X && cmake .. && make gemv.cppx && ctest;exit
#endif
// © Alfredo A. Correa 2020

// $CXX -D_MULTI_CUBLAS_ALWAYS_SYNC $0 -o $0x `pkg-config --libs blas` -lcudart -lcublas -lboost_unit_test_framework&&$0x&&rm $0x; exit

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS gemv"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<boost/test/output/compiler_log_formatter.hpp>
struct gedit_config{
    struct formatter : boost::unit_test::output::compiler_log_formatter{
        void print_prefix(std::ostream& out, boost::unit_test::const_string file, std::size_t line){
            out<< file <<':'<< line <<": ";
        }
    };
    gedit_config(){boost::unit_test::unit_test_log.set_formatter(new formatter);}
};
BOOST_GLOBAL_FIXTURE(gedit_config);

#include "../../../adaptors/blas/gemv.hpp"
#include "../../../array.hpp"

#include "../../../array.hpp"
#include "../../../utility.hpp"

#include "../../blas/dot.hpp"
#include "../../blas/axpy.hpp"
#include "../../blas/nrm2.hpp"
#include "../../blas/gemm.hpp"

//#include<complex>
//#include<cassert>
//#include<iostream>
//#include<numeric>
//#include<algorithm>
#include<random>

namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

namespace multi = boost::multi;
namespace blas = multi::blas;

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(multi_blas_gemv, *utf::tolerance(0.0001)){

	multi::array<double, 2> const M = {
		{ 9., 24., 30., 9.},
		{ 4., 10., 12., 7.},
		{14., 16., 36., 1.}
	};
	multi::array<double, 1> const v = {1.1, 2.1, 3.1, 4.1};
	{
		multi::array<double, 1>       w(size(M));
		blas::gemv_n(1., begin(M), size(M), begin(v), 0., begin(w));
		BOOST_TEST( w[1] == 91.3 );
		BOOST_TEST( w[2] == +blas::dot(M[2], v) );
	}
	{
		multi::array<double, 1>       w(size(M));
		multi::array<double, 2> const MT = ~M;
		blas::gemv_n(1., begin(~MT), size(~MT), begin(v), 0., begin(w));
		BOOST_TEST( w[1] == 91.3 );
		BOOST_TEST( w[2] == +blas::dot(M[2], v));
	}
	{
		multi::array<double, 1> w(size(M));
		auto mv = blas::gemv(1., M, v);
		copy_n(mv.begin(), mv.size(), w.begin());
		BOOST_TEST( w[1] == 91.3 );
	}
	{
		multi::array<double, 1> w(size(M));
		w = blas::gemv(1., M, v);
		BOOST_TEST( w[1] == 91.3 );
	}
	{
		multi::array<double, 1> w = blas::gemv(1., M, v);
		BOOST_TEST( w[1] == 91.3 );
	}
}

BOOST_AUTO_TEST_CASE(multi_blas_gemv_produce_leak){
	double* p = new double[10000000]; p+=1;
}

BOOST_AUTO_TEST_CASE(multi_blas_gemv_real){
	namespace blas = multi::blas;

	using std::abs;
	multi::array<double, 2> const M = {
		{ 9., 24., 30., 9.},
		{ 4., 10., 12., 7.},
		{14., 16., 36., 1.}
	};
	multi::array<double, 1> const X = {1.1, 2.1, 3.1, 4.1};
//	{
//		multi::array<double, 1> Y = {4.,5.,6.};
//		double const a = 1.1, b = 1.2;
//		blas::gemv(a, M, X, b, Y); // y = a*M*x + b*y

//		multi::array<double, 1> const Y3 = {214.02, 106.43, 188.37};
//		BOOST_REQUIRE( abs(Y[1] - Y3[1]) < 2e-14 );
//	}
//	{
//		blas::gemv_range gr{1.0, M.begin(), M.end(), X.begin()};
//		multi::array<double, 1> Y(gr.begin(), gr.end());
//	}
//	{
//		auto Y = +blas::gemv(M, X);
//		BOOST_TEST( Y[0] == +blas::dot(M[0], X) );
//		BOOST_REQUIRE(std::equal(begin(Y), end(Y), begin(M), [&X](auto&& y, auto&& m){return y==blas::dot(m, X);}));
//	}
//	{
//		multi::array<double, 1> const a = {1., 2., 3.};
//		multi::array<double, 1> const b = {4., 5., 6.};
//		multi::array<double, 1> const dot = blas::gemv(multi::array<double, 2>({a}), b);
//		BOOST_REQUIRE( dot[0] == blas::dot(a, b) );
//	}
//	{
//		multi::array<double, 2> const MT = ~M;
//		using boost::test_tools::tolerance;
//	//	using blas::gemv; BOOST_TEST_REQUIRE( nrm2(blas::axpy(-1., gemv(~+~M, X), gemv(M, X)))() == 0.,  tt::tolerance(1e-13) );
//		using namespace blas;
//		using namespace blas::operators;
//	//	blas::nrm2(X);
//		BOOST_TEST_REQUIRE( (((~+~M)%X - M%X)^2) == 0., tolerance(1e-13) );
//	}
}

//BOOST_AUTO_TEST_CASE(multi_blas_gemv_real_complex){
//	namespace blas = multi::blas;
//	using complex = std::complex<double>; //#define I *std::complex<double>(0, 1)
//	using std::abs;
//	multi::array<complex, 2> const M = {
//		{ 9., 24., 30., 9.},
//		{ 4., 10., 12., 7.},
//		{14., 16., 36., 1.}
//	};
//	multi::array<complex, 1> const X = {1.1, 2.1, 3.1, 4.1};
//	{
//		multi::array<complex, 1> Y = {4., 5., 6.};
//		double const a = 1.1, b = 1.2;
//		blas::gemv(a, M, X, b, Y); // y = a*M*x + b*y
//		
//		multi::array<complex, 1> const Y3 = {214.02, 106.43, 188.37};
//		
//		using namespace blas::operators;
//		BOOST_TEST_REQUIRE( ((Y - Y3)^2)  == 0. , boost::test_tools::tolerance(1e-13) );
//	
//	}
//}

//BOOST_AUTO_TEST_CASE(multi_blas_gemv_complex){
//	
//	namespace blas = multi::blas;
//	using complex = std::complex<double>; std::complex<double> const I{0, 1};
//	
//	using std::abs;
//	multi::array<complex, 2> const M = {{2. + 3.*I, 2. + 1.*I, 1. + 2.*I}, {4. + 2.*I, 2. + 4.*I, 3. + 1.*I}, 
// {7. + 1.*I, 1. + 5.*I, 0. + 3.*I}};
//	multi::array<complex, 1> const X = {1. + 2.*I, 2. + 1.*I, 9. + 2.*I};
//	using namespace blas::operators;
//	BOOST_REQUIRE(( +blas::gemv(   M, X) == multi::array<complex, 1>{4. + 31.*I, 25. + 35.*I, -4. + 53.*I} ));
//	
//	auto MT = +~M;
//	BOOST_REQUIRE(( +blas::gemv(~MT, X) == multi::array<complex, 1>{4. + 31.*I, 25. + 35.*I, -4. + 53.*I} ));
//	
//	auto MH = +*~M;
//	BOOST_REQUIRE( +blas::gemv(~M, X) == (multi::array<complex, 1>{63. + 38.*I, -1. + 62.*I, -4. + 36.*I}) );
//	BOOST_REQUIRE( +blas::gemv(~M, X) == +blas::gemv(MT, X) );// == multi::array<complex, 1>{4. + 31.*I, 25. + 35.*I, -4. + 53.*I} ));
//	
//	BOOST_REQUIRE( +blas::gemv(*M, X) == (multi::array<complex, 1>{26. - 15.*I, 45. - 3.*I, 22. - 23.*I}) );
//	
////	BOOST_REQUIRE( blas::gemv(~*M, X) == (multi::array<complex, 1>{83. + 6.*I, 31. - 46.*I, 18. - 26.*I}) ); // not supported by blas

//}

//BOOST_AUTO_TEST_CASE(multi_blas_gemv_temporary){

//	using complex = std::complex<double>;
//	
//	multi::array<complex, 2> const A = {
//		{1., 0., 0.}, 
//		{0., 1., 0.},
//		{0., 0., 1.}
//	};
//	
//	auto const B = []{
//		auto _ = multi::array<complex, 2>({3, 3});
//		auto rand = [d=std::normal_distribution<>{}, g=std::mt19937{}]()mutable{return complex{d(g), d(g)};};
//		std::generate(_.elements().begin(), _.elements().end(), rand);
//		return _;
//	}();
//	
//	namespace blas = multi::blas;
//	
//	using namespace blas::operators;
//	BOOST_TEST( ( (A%(~B)[0] - ( ~(A*B)  )[0])^2) == 0. );
//	BOOST_TEST( ( (A%(~B)[0] - ((~B)*(~A))[0])^2) == 0. );

//}

//#if 0
//	{
//		auto Y = blas::gemv(M, X);
//		BOOST_REQUIRE((
//			Y == decltype(Y){
//				blas::dot(M[0], X),
//				blas::dot(M[1], X),
//				blas::dot(M[2], X)
//			}
//		));
//		BOOST_REQUIRE(std::equal(begin(Y), end(Y), begin(M), [&X](auto&& y, auto&& m){return y==blas::dot(m, X);}));
//	}
//	{
//		multi::array<double, 1> const a = {1., 2., 3.};
//		multi::array<double, 1> const b = {4., 5., 6.};
//		BOOST_REQUIRE(
//			blas::gemv(multi::array<double, 2>({a}), b)[0] == blas::dot(a, b)
//		);
//	}
//	{
//		multi::array<double, 2> const MT = ~M;
//		using boost::test_tools::tolerance;
//	//	using blas::gemv; BOOST_TEST_REQUIRE( nrm2(blas::axpy(-1., gemv(~+~M, X), gemv(M, X)))() == 0.,  tt::tolerance(1e-13) );
//		using namespace blas;
//		using namespace blas::operators;
//	//	blas::nrm2(X);
//		BOOST_TEST_REQUIRE( (((~+~M|X)-(M|X))^2) == 0., tolerance(1e-13) );
//	}
//#endif


//#if 0
//	{
//		double const M[3][4] = {
//			{ 9., 24., 30., 9.},
//			{ 4., 10., 12., 7.},
//			{14., 16., 36., 1.}
//		};
//		assert( M[2][0] == 14. );
//		double const X[4] = {1.1,2.1,3.1, 4.1};
//		double Y[3] = {4.,5.,6.};
//		double const a = 1.1;
//		double const b = 1.2;
//		gemv('T', a, M, X, b, Y); // y = a*M*x + b*y
//		double Y3[3] = {214.02, 106.43, 188.37};
//		assert( abs(Y[1] - Y3[1]) < 2e-14 );
//	}

//	{
//		multi::array<double, 2> const M = {
//			{ 9., 24., 30., 9.},
//			{ 4., 10., 12., 7.},
//			{14., 16., 36., 1.}
//		};
//		assert( M[2][0] == 14. );
//		multi::array<double, 1> const X = {1.1,2.1,3.1};
//		multi::array<double, 1> Y = {4.,5.,6., 7.};
//		double a = 1.8, b = 1.6;
//		gemv('N', a, M, X, b, Y); // y = a*(M^T)*x + b*y, y^T = a*(x^T)*M + b*y^T
//		multi::array<double, 1> const Y3 = {117.46, 182.6, 315.24, 61.06}; // =1.8 Transpose[{{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}].{1.1, 2.1, 3.1} + 1.6 {4., 5., 6., 7.}
//		assert( abs(Y[2] - Y3[2]) < 1e-13 );
//	}
//	{
//		multi::array<double, 2> const M = {
//			{ 9., 24., 30., 9.},
//			{ 4., 10., 12., 7.},
//			{14., 16., 36., 1.}
//		};
//		assert( M[2][0] == 14. );
//		multi::array<double, 1> const X = {1.1,2.1,3.1, 4.1};
//		multi::array<double, 1> Y = {4.,5.,6.};
//		double a = 1.1, b = 1.2;
//		gemv(a, M, X, b, Y); // y = a*M*x + b*y
//		multi::array<double, 1> const Y3 = {214.02, 106.43, 188.37}; // = 1.1 {{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}.{1.1, 2.1, 3.1, 4.1} + 1.2 {4., 5., 6.}
//		assert( std::abs(Y[1] - Y3[1]) < 2e-14 );
//	}
//	{
//		double const M[3][4] = {
//			{ 9., 24., 30., 9.},
//			{ 4., 10., 12., 7.},
//			{14., 16., 36., 1.}
//		};
//		assert( M[2][0] == 14. );
//		double const X[4] = {1.1,2.1,3.1, 4.1};
//		double Y[3] = {4.,5.,6.};
//		double a = 1.1, b = 1.2;
//		gemv(a, M, X, b, Y); // y = a*M*x + b*y
//		double const Y3[3] = {214.02, 106.43, 188.37};
//		assert( std::abs(Y[1] - Y3[1]) < 2e-14 );
//	}
//	{
//		multi::array<double, 2> const M = {
//			{ 9., 4., 14.},
//			{24., 10., 16.},
//			{30., 12., 36.},
//			{9., 7., 1.}
//		}; assert( M[0][2] == 14. );
//		multi::array<double, 1> const X = {1.1,2.1,3.1, 4.1};
//		multi::array<double, 1> Y = {4.,5.,6.};
//		double a = 1.1, b = 1.2;
//		gemv(a, rotated(M), X, b, Y); // y = a*M*x + b*y

//		multi::array<double, 1> const Y3 = {214.02, 106.43, 188.37}; // = 1.1 {{9., 24., 30., 9.}, {4., 10., 12., 7.}, {14., 16., 36., 1.}}.{1.1, 2.1, 3.1, 4.1} + 1.2 {4., 5., 6.}
//		assert( abs(Y[1] - Y3[1]) < 2e-14 );
//	}
//#endif

