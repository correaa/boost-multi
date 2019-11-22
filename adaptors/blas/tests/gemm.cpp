#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc -x cu --expt-relaxed-constexpr`#$CXX` $0 -o $0x -Wno-deprecated-declarations -lcudart -lcublas -lboost_unit_test_framework `pkg-config --libs blas`&&$0x&&rm $0x $0.cpp; exit
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
	using multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j) std::cout<< C[i][j] <<' ';
		std::cout<<std::endl;
	}
	return std::cout<<std::endl;
}

BOOST_AUTO_TEST_CASE(multi_blas_gemm_square_real){
	multi::array<double, 2> const a = {
		{1, 3, 4},
		{9, 7, 1},
		{1, 2, 3}
	};
	multi::array<double, 2> const b = {	
		{11, 12, 4},
		{ 7, 19, 1},
		{11, 12, 4}
	};
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::operation;
		gemm(operation::identity, operation::identity, 1., a, b, 0., c);
		BOOST_REQUIRE( c[2][1] == 86 );
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::operation;
		gemm(operation::identity, operation::transposition, 1., a, b, 0., c);
		BOOST_REQUIRE( c[2][1] == 48 );
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::operation;
		gemm(operation::transposition, operation::identity, 1., a, b, 0., c);
		BOOST_REQUIRE( c[2][1] == 103 );
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::operation;
		gemm(operation::transposition, operation::transposition, 1., a, b, 0., c);
		BOOST_REQUIRE( c[2][1] == 50 );		
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::gemm;
		gemm(1., a, b, 0., c);
		BOOST_REQUIRE( c[2][1] == 86 );
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::gemm;
		gemm(1., a, rotated(b), 0., c);
		BOOST_REQUIRE( c[2][1] == 48 );
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::gemm;
		gemm(1., rotated(a), b, 0., c);
		BOOST_REQUIRE( c[2][1] == 103 );		
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::gemm;
		gemm(1., rotated(a), rotated(b), 0., c);
		BOOST_REQUIRE( c[2][1] == 50 );		
	}
	{
		multi::array<double, 2> c({size(a), size(rotated(b))}, 9999);
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(2., hermitized(a), hermitized(b), 0., c);
		BOOST_REQUIRE( c[2][1] == 100 );

		multi::array<double, 2> const c_copy = gemm(2., hermitized(a), hermitized(b));
		BOOST_REQUIRE( c == c_copy );
		multi::array<double, 2> const c_copy2 = gemm(hermitized(a), hermitized(b));
		BOOST_REQUIRE( c_copy2[2][1] == 50 );
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
	{
		namespace cuda = multi::cuda;
		cuda::array<complex, 2> const acu = a;
		cuda::array<complex, 2> const bcu = b;
		cuda::array<complex, 2> ccu({2, 4});
		using multi::blas::gemm;
		gemm(1., acu, bcu, 0., ccu);
		BOOST_REQUIRE( ccu[1][2] == complex(112, 12) );
	}
	{
		namespace cuda = multi::cuda;
		cuda::managed::array<complex, 2> const amcu = a;
		cuda::managed::array<complex, 2> const bmcu = b;
		cuda::managed::array<complex, 2> cmcu({2, 4});
		using multi::blas::gemm;
		gemm(1., amcu, bmcu, 0., cmcu);
		BOOST_REQUIRE( cmcu[1][2] == complex(112, 12) );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare_automatic2){
	using complex = std::complex<double>; constexpr complex I{0,1};
	multi::array<complex, 2> const a = {
		{1.-2.*I, 9.-1.*I},
		{3.+3.*I, 7.-4.*I},
		{1.+9.*I, 1.+8.*I}
	};
	multi::array<complex, 2> const b = {	
		{ 11.+1.*I, 12.+1.*I, 4.+1.*I, 8.-2.*I},
		{  7.+8.*I, 19.-2.*I, 2.+1.*I, 7.+1.*I},
		{  5.+1.*I,  3.-1.*I, 3.+8.*I, 1.+1.*I}
	};
	{
		multi::array<complex, 2> c({2, 4});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(a), b, 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );

		multi::array<complex, 2> const c_copy = gemm(1., hermitized(a), b);
		multi::array<complex, 2> const c_copy2 = gemm(hermitized(a), b);
		BOOST_REQUIRE(( c == c_copy and c == c_copy2 ));
	}
	{
		namespace cuda = multi::cuda;
		cuda::array<complex, 2> const acu = a;
		cuda::array<complex, 2> const bcu = b;
		cuda::array<complex, 2> ccu({2, 4});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(acu), bcu, 0., ccu);
		BOOST_REQUIRE( ccu[1][2] == complex(112, 12) );

		cuda::array<complex, 2> const ccu_copy = gemm(1., hermitized(acu), bcu);
		cuda::array<complex, 2> const ccu_copy2 = gemm(hermitized(acu), bcu);
		BOOST_REQUIRE(( ccu_copy == ccu and ccu_copy2 == ccu ));
	}
	{
		namespace cuda = multi::cuda;
		cuda::managed::array<complex, 2> const amcu = a;
		cuda::managed::array<complex, 2> const bmcu = b;
		cuda::managed::array<complex, 2> cmcu({2, 4});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(amcu), bmcu, 0., cmcu);
		BOOST_REQUIRE( cmcu[1][2] == complex(112, 12) );

		cuda::managed::array<complex, 2> const cmcu_copy = gemm(1., hermitized(amcu), bmcu);
		cuda::managed::array<complex, 2> const cmcu_copy2 = gemm(hermitized(amcu), bmcu);
		BOOST_REQUIRE(( cmcu_copy == cmcu and cmcu_copy2 == cmcu ));
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare_automatic3){
	using complex = std::complex<double>; complex const I{0,1};
	multi::array<complex, 2> const a = {
		{1.-2.*I, 9.-1.*I},
		{3.+3.*I, 7.-4.*I},
		{1.+9.*I, 1.+8.*I}
	};
	multi::array<complex, 2> const bH = {	
		{ 11.+1.*I, 12.+1.*I, 4.+1.*I, 8.-2.*I},
		{  7.+8.*I, 19.-2.*I, 2.+1.*I, 7.+1.*I},
		{  5.+1.*I,  3.-1.*I, 3.+8.*I, 1.+1.*I}
	};
	multi::array<complex, 2> const b = multi::blas::hermitized(bH);
	{
		multi::array<complex, 2> c({2, 4});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(a), hermitized(b), 0., c); // c=ab, c⸆=b⸆a⸆
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
	}
	{
		namespace cuda = multi::cuda;
		cuda::array<complex, 2> const acu = a;
		cuda::array<complex, 2> const bcu = b;
		cuda::array<complex, 2> ccu({2, 4});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(acu), hermitized(bcu), 0., ccu);
		BOOST_REQUIRE( ccu[1][2] == complex(112, 12) );
	}
	{
		namespace cuda = multi::cuda;
		cuda::managed::array<complex, 2> const amcu = a;
		cuda::managed::array<complex, 2> const bmcu = b;
		cuda::managed::array<complex, 2> cmcu({2, 4});
		using multi::blas::gemm;
		using multi::blas::hermitized;
		gemm(1., hermitized(amcu), hermitized(bmcu), 0., cmcu);
		BOOST_REQUIRE( cmcu[1][2] == complex(112, 12) );
	}
}

BOOST_AUTO_TEST_CASE(multi_adaptors_blas_gemm_complex_nonsquare_automatic4){
	using complex = std::complex<double>; complex const I{0,1};
	multi::array<complex, 2> c({12, 12});
	{
		multi::array<complex, 2> const a({12, 100}, 1.+2.*I);
		multi::array<complex, 2> const b({12, 100}, 1.+2.*I);
		using multi::blas::hermitized;
		using multi::blas::gemm;
		gemm(1., a, hermitized(b), 0., c);
		BOOST_REQUIRE( real(c[0][0]) > 0);

		auto c_copy = gemm(1., a, hermitized(b));
		BOOST_REQUIRE( c_copy == c );
	}
	{
		multi::array<complex, 2> const a_block({24, 100}, 1.+2.*I);
		multi::array<complex, 2> const b({12, 100}, 1.+2.*I);
		multi::array<complex, 2> c2({12, 12});

		using multi::blas::hermitized;
		using multi::blas::gemm;
		gemm(1., a_block.strided(2), hermitized(b), 0., c2);

		BOOST_REQUIRE( real(c[0][0]) > 0);
		BOOST_REQUIRE( c == c2 );

		auto c2_copy = gemm(1., a_block.strided(2), hermitized(b));
		BOOST_REQUIRE( c2_copy == c2 );
	}
}


