
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../../../adaptors/cuda/cublas/context.hpp"

#include "../managed/ptr.hpp"
#include "../../../../adaptors/cuda.hpp"
#include "../../../../adaptors/blas/gemm.hpp"

#include<random>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;
namespace blas = multi::blas;

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(multi_cuda_mngd_ptr){
	using T = double;
	static_assert( sizeof(cuda::managed::ptr<T>) == sizeof(T*) );
	static_assert( std::is_convertible<cuda::managed::ptr<T>, T*>{} );
	auto f = [](double* dp){return bool{dp};};
	cuda::managed::ptr<T> p;
	f(p);
}

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(multi_cuda_mngd_ptr_call_gemm){
	using complex = std::complex<double>; complex const I{0, 1};
	boost::multi::cuda::managed::array<complex, 2> m = {
		{ 1. + 2.*I, 3. - 3.*I, 1.-9.*I},
		{ 9. + 1.*I, 7. + 4.*I, 1.-8.*I},
	};
	boost::multi::cuda::managed::array<complex, 2> const b = {
		{ 11.+1.*I, 12.+1.*I, 4.+1.*I, 8.-2.*I},
		{  7.+8.*I, 19.-2.*I, 2.+1.*I, 7.+1.*I},
		{  5.+1.*I,  3.-1.*I, 3.+8.*I, 1.+1.*I}
	};
//	{
//		blas::context ctxt;
//		auto c =+ blas::gemm(&ctxt, 1., m, b);
//		static_assert( std::is_same<decltype(c), multi::cuda::managed::array<complex, 2>>{} );
//		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
//		BOOST_REQUIRE( b[1][2] == 2.+1.*I );
//	}
	{
		multi::cuda::managed::array<complex, 2> c({2, 4});
		multi::cuda::cublas::context ctxt;
		blas::gemm_n(ctxt, 1., begin(m), size(m), begin(b), 0., begin(c));
		cudaDeviceSynchronize();
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
		BOOST_REQUIRE( b[1][2] == 2.+1.*I );
	}
	{
		multi::cuda::cublas::context ctxt;
		auto c =+ blas::gemm(&ctxt, 1., m, b);
		static_assert( std::is_same<decltype(c), multi::cuda::managed::array<complex, 2>>{} );
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
		BOOST_REQUIRE( b[1][2] == 2.+1.*I );
	}
	{
		auto c =+ blas::gemm(1., m, b);//blas::default_context_of(m.base()), 1., m, b);
		static_assert( std::is_same<decltype(c), multi::cuda::managed::array<complex, 2>>{} );
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
		BOOST_REQUIRE( b[1][2] == 2.+1.*I );
	}
	{
		multi::cuda::managed::array<complex, 2> c({2, 4});
		multi::cuda::cublas::context ctxt;
		blas::gemm_n(ctxt, 1., begin(m), size(m), begin(b), 0., begin(c));
		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
		BOOST_REQUIRE( b[1][2] == 2.+1.*I );
	}
//	{
//		auto c =+ blas::gemm(1., m, b);
//		static_assert( std::is_same<decltype(c), multi::cuda::managed::array<complex, 2>>{} );
//		BOOST_REQUIRE( c[1][2] == complex(112, 12) );
//		BOOST_REQUIRE( b[1][2] == 2.+1.*I );
//	}
}

