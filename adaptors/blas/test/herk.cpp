#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic -Wfatal-errors -Wno-deprecated-declarations $0 -o $0x -lcudart -lcublas -lboost_unit_test_framework \
`pkg-config --cflags --libs blas`&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../../adaptors/blas/cuda.hpp" // must be included before blas/gemm.hpp
#include "../../../adaptors/blas/herk.hpp"

#include "../../../adaptors/cuda.hpp"
#include "../../../array.hpp"

BOOST_AUTO_TEST_CASE(multi_blas_cuda_herk){
	namespace multi = boost::multi;
	namespace cuda = multi::cuda;

	using complex = std::complex<double>;
	constexpr complex I{0., 1.};

	multi::array<complex, 2> const a = {
		{ 1. + 3.*I, 3.- 2.*I, 4.+ 1.*I},
		{ 9. + 1.*I, 7.- 8.*I, 1.- 3.*I}
	};
	{
		multi::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		using multi::blas::triangular;
		herk(triangular::lower, 1., a, c); // c†=c=aa†
		BOOST_REQUIRE( c[1][0] == complex(50., -49.) );
	//	BOOST_REQUIRE( c[0][1] == complex(50., +49.) );
	}
	cuda::managed::array<complex, 2> const agpu = a;
	BOOST_REQUIRE(a == agpu);
	{
		cuda::managed::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		using multi::blas::triangular;
		herk(triangular::lower, 1., a, c);
		BOOST_REQUIRE( c[1][0] == complex(50., -49.) );
		BOOST_REQUIRE( c[0][1] == 9999. );
	}
	{
		cuda::managed::array<complex, 2> c({2, 2}, 9999.);
		using multi::blas::herk;
		using multi::blas::triangular;
		herk(1., a, c);
		BOOST_REQUIRE( c[1][0] == complex(50., -49.) );
		BOOST_REQUIRE( c[0][1] == conj(c[1][0]) );
	}
	{
		multi::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::hermitized;
		herk(triangular::lower, 1., hermitized(a), c); // c=c†=a†a, `c` in lower triangular
		BOOST_REQUIRE( c[2][1]==complex(41., 2.) );
		BOOST_REQUIRE( c[1][2]==9999. );
	}
	{
		cuda::managed::array<complex, 2> c({3, 3}, 9999.);
		using multi::blas::triangular;
		using multi::blas::hermitized;
	//	auto a = hermitized(agpu);
		herk(triangular::lower, 1., hermitized(agpu), c); // c=c†=a†a, `c` in lower triangular
	//	BOOST_REQUIRE( c[2][1]==complex(41., 2.) );
	//	BOOST_REQUIRE( c[1][2]==9999. );
	}

}

