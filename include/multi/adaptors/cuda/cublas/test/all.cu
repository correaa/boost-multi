// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUBLAS all"
#include<boost/test/unit_test.hpp>

#include <multi/adaptors/cuda/cublas.hpp>

#include <multi/adaptors/blas/asum.hpp>
#include <multi/adaptors/blas/axpy.hpp>
#include <multi/adaptors/blas/copy.hpp>
#include <multi/adaptors/blas/gemm.hpp>
#include <multi/adaptors/blas/nrm2.hpp>
#include <multi/adaptors/blas/scal.hpp>
#include <multi/adaptors/blas/swap.hpp>

#include <multi/adaptors/thrust.hpp>

#include<thrust/complex.h>

#include<numeric>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>

namespace multi = boost::multi;

using complex = thrust::complex<double>;

template<class T = complex, class Alloc = std::allocator<T>>
auto generate_ABx() {
	complex const I{0.0, 1.0};
	multi::array<T, 1, Alloc> x = { 1.0 + I*0.0,  2.0 + I*0.0,  3.0 + I*0.0,  4.0 + I*0.0};

	multi::array<complex, 2, Alloc> A = {
		{ 1.0 + I*0.0,  2.0 + I*0.0,  3.0 + I*0.0,  4.0 + I*0.0},
		{ 5.0 + I*0.0,  6.0 + I*0.0,  7.0 + I*0.0,  8.0 + I*0.0},
		{ 9.0 + I*0.0, 10.0 + I*0.0, 11.0 + I*0.0, 12.0 + I*0.0},
		{13.0 + I*0.0, 14.0 + I*0.0, 15.0 + I*0.0, 16.0 + I*0.0},
	};

	multi::array<complex, 2, Alloc> B = {
		{ 1.0 + I*0.0,  2.0 + I*0.0,  3.0 + I*0.0,  4.0 + I*0.0},
		{ 5.0 + I*0.0,  6.0 + I*0.0,  7.0 + I*0.0,  8.0 + I*0.0},
		{ 9.0 + I*0.0, 10.0 + I*0.0, 11.0 + I*0.0, 12.0 + I*0.0},
		{13.0 + I*0.0, 14.0 + I*0.0, 15.0 + I*0.0, 16.0 + I*0.0},
	};

	return std::make_tuple(std::move(x), std::move(A), std::move(B));
}

BOOST_AUTO_TEST_CASE(cublas_scal_complex_column) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	{
		using T = complex;
		auto [x, A, B] = generate_ABx<T, thrust::cuda::allocator<T> >();
		auto const s = 2.0 + I*3.0;
		blas::scal(s, x);  // x_i <- s*x_i

		{
			auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
			auto xx = +x2;
			blas::scal(s, xx);
			BOOST_REQUIRE(xx == x);
		}
		{
			auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
			using blas::operators::operator*=;
			x2 *= s;
			BOOST_REQUIRE(x == x2);
		}
		{
			auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
			thrust::transform(x2.begin(), x2.end(), x2.begin(), [s] __device__ (T& e) {return s*e;});

			BOOST_REQUIRE(x == x2);
		}
		{
			auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
			thrust::for_each(x2.begin(), x2.end(), [s] __device__ (T& e) {return e*=s;});

			BOOST_REQUIRE(x == x2);
		}
	}
}

BOOST_AUTO_TEST_CASE(cublas_copy_complex) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc = thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> const x = { 1.0 + I*8.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};
	multi::array<T, 1, Alloc> y = { 1.0 + I*9.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};

	blas::copy(x, y);
	BOOST_REQUIRE( static_cast<complex>(y[0]) == 1.0 + I*8.0 );
	{
		thrust::copy(begin(x), end(x), begin(y));
		BOOST_REQUIRE( static_cast<complex>(y[0]) == 1.0 + I*8.0 );
	}
	{
		blas::copy_n(x.begin(), x.size(), y.begin());
		BOOST_REQUIRE( static_cast<complex>(y[0]) == 1.0 + I*8.0 );
	}
	{
		y() = blas::copy(x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == 1.0 + I*8.0 );
	}
	{
		multi::array<T, 1, Alloc> yy = blas::copy(x);
		BOOST_REQUIRE( static_cast<complex>(yy[0]) == 1.0 + I*8.0 );
	}
	{
		y = blas::copy(x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == 1.0 + I*8.0 );
	}
	{
		{
			using blas::operators::operator<<;
			y << x;
		//  BOOST_REQUIRE(( static_cast<complex>(y[0]) == 1.0 + I*8.0 ));  // this can't be used with a free operator<<
		}
		BOOST_REQUIRE(( static_cast<complex>(y[0]) == 1.0 + I*8.0 ));  // this can't be used with a free operator<<
	}
}

#if 1
BOOST_AUTO_TEST_CASE(cublas_swap_complex) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc = thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> x = { 1.0 + I*8.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};
	multi::array<T, 1, Alloc> y = { 1.0 + I*9.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};

	blas::swap(x, y);
	BOOST_REQUIRE( static_cast<complex>(x[0]) == 1.0 + I*9.0 );
	{
		thrust::swap_ranges(begin(x), end(x), begin(y));
		thrust::swap_ranges(begin(x), end(x), begin(y));
		BOOST_REQUIRE( static_cast<complex>(x[0]) == 1.0 + I*9.0 );
	}
	{
		using blas::operator^;
		(x^y);
		(x^y);
		BOOST_REQUIRE( static_cast<complex>(x[0]) == 1.0 + I*9.0 );
	}
}

BOOST_AUTO_TEST_CASE(cublas_asum_complex_column) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc = thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> const x = { 1.0 + I*8.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};

	double res;
	blas::asum_n(x.begin(), x.size(), &res);
	{
		double res2;
		res2 = blas::asum(x);
		BOOST_REQUIRE( res == res2 );
	}
	{
		double res2 = blas::asum(x);
		BOOST_REQUIRE( res == res2 );
	}
	{
		auto res2 = std::transform_reduce(
			x.begin(), x.end(), double{}, std::plus<>{}, [](T const& e) {return std::abs(e.real()) + std::abs(e.imag());}
		);
		BOOST_REQUIRE( res == res2 );
	}
	{
		auto res2 = thrust::transform_reduce(
			x.begin(), x.end(), [] __device__ (T const& e) {return std::abs(e.real()) + std::abs(e.imag());},
			double{}, thrust::plus<>{}
		);
		BOOST_REQUIRE( res == res2 );
	}
	{
		multi::static_array<double, 0, thrust::cuda::allocator<double>> res2({}, 0.0);
		res2.assign( &blas::asum(x) );
		res2 = blas::asum(x);
		BOOST_REQUIRE(( res == static_cast<multi::static_array<double, 0, thrust::cuda::allocator<double>>::element_ref>(res2) ));
		BOOST_REQUIRE(( res == static_cast<double>(res2) ));
	//  BOOST_REQUIRE( res == res2 );
	}
	{
		multi::array<double, 0, thrust::cuda::allocator<double>> res2 = blas::asum(x);
		BOOST_REQUIRE(( res == static_cast<multi::static_array<double, 0, thrust::cuda::allocator<double>>::element_ref>(res2) ));
		BOOST_REQUIRE(( res == static_cast<double>(res2) ));
	//  BOOST_REQUIRE( res == res2 );
	}
	{
		using blas::operators::operator==;
		using blas::operators::operator!=;
		BOOST_REQUIRE( x != 0 );
		BOOST_REQUIRE( not (x == 0) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_nrm2_complex_column) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> const x = { 1.0 + I*8.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};

	double res;
	blas::nrm2(x, res);
	{
		double res2;
		res2 = blas::nrm2(x);
		BOOST_REQUIRE( res == res2 );
	}
	{
		auto res2 = +blas::nrm2(x);
		BOOST_REQUIRE( res == res2 );
	}
	{
		auto res2 = sqrt(thrust::transform_reduce(
			x.begin(), x.end(), [] __device__ (T const& e) {return thrust::norm(e);},
			double{}, thrust::plus<>{}
		));
		BOOST_REQUIRE( res == res2 );
	}
	{
		multi::array<double, 0, thrust::cuda::allocator<double>> res2 = blas::nrm2(x);
		BOOST_REQUIRE(( res == static_cast<double>(res2) ));
	}
}

BOOST_AUTO_TEST_CASE(cublas_dot_complex_column) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> const x = { 1.0 + I*8.0,  2.0 + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};
	multi::array<T, 1, Alloc> const y = { 1.0 + I*2.0,  2.0 + I*3.0,  3.0 + I*5.0,  4.0 + I*7.0};

	{
		T res;
		blas::dot(x, y, res);
		{
			complex res2;
			res2 = blas::dot(x, y);
			BOOST_REQUIRE(res == res2);
		}
		{
			multi::array<complex, 0> res2(complex{1.0, 0.0});
			res2 = blas::dot(x, y);
			BOOST_REQUIRE( static_cast<complex>(res2) == res );
		}
		{
			using blas::operators::operator,;
			auto res2 = +(x, y);
			BOOST_REQUIRE(res == res2);
		}
		{
			auto res2 = +blas::dot(x, y);
			BOOST_REQUIRE(res == res2);
		}
		{
		//  auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
		//  thrust::for_each(x2.begin(), x2.end(), [s] __device__ (T& e) {return e*=s;});
			auto res2 = thrust::inner_product(x.begin(), x.end(), y.begin(), T{});
			BOOST_REQUIRE(res == res2);
		}
	}
	{
		T res;
		blas::dot(blas::C(x), y, res);
		{
			using blas::operators::operator,;
			using blas::operators::operator*;
			auto res2 = +(*x, y);
			BOOST_REQUIRE(res == res2);
		}
		{
			auto res2 = +blas::dot(blas::C(x), y);
			BOOST_REQUIRE(res == res2);
		}
		{
		//  auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
		//  thrust::for_each(x2.begin(), x2.end(), [s] __device__ (T& e) {return e*=s;});
			auto res2 = thrust::inner_product(x.begin(), x.end(), y.begin(), T{}, thrust::plus<>{}, [] __device__ (T const& t1, T const& t2) {return conj(t1)*t2;});
			BOOST_REQUIRE(res == res2);
		}
	}
	{
		T res;
		blas::dot(x, blas::C(y), res);
		{
			using blas::operators::operator,;
			auto res2 = +(x, blas::C(y));
			BOOST_REQUIRE(res == res2);
		}
		{
			auto res2 = +blas::dot(x, blas::C(y));
			BOOST_REQUIRE(res == res2);
		}
		{
		//  auto [x2, A2, B2] = generate_ABx<complex, thrust::cuda::allocator<complex> >();
		//  thrust::for_each(x2.begin(), x2.end(), [s] __device__ (T& e) {return e*=s;});
			auto res2 = thrust::inner_product(x.begin(), x.end(), y.begin(), T{}, thrust::plus<>{}, [] __device__ (T const& t1, T const& t2) {return t1*conj(t2);});
			BOOST_REQUIRE(res == res2);
		}
		{
			BOOST_REQUIRE( blas::dot(blas::C(x), x) == pow(blas::nrm2(x), 2.0) );
			BOOST_REQUIRE( blas::dot(x, blas::C(x)) == pow(blas::nrm2(x), 2.0) );

			using blas::operators::operator,;
			using blas::operators::operator*;
			using blas::operators::abs;
			using blas::operators::norm;
			using blas::operators::operator^;

			BOOST_REQUIRE( (*x, x) == pow(abs(x), 2.0) );
			BOOST_REQUIRE( (*x, x) == pow(abs(x), 2)   );
			BOOST_REQUIRE( (*x, x) == norm(x)          );

			BOOST_REQUIRE( (x, *x) == pow(abs(x), 2.0) );
			BOOST_REQUIRE( (x, *x) == pow(abs(x), 2)   );
			BOOST_REQUIRE( (x, *x) == norm(x)          );

			BOOST_REQUIRE( (*x, x) == (x^2)            );
		}
	}
}
#endif