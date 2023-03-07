// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2023 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUBLAS all"
#include<boost/test/unit_test.hpp>

#include <multi/adaptors/cuda/cublas.hpp>

#include <multi/adaptors/blas/axpy.hpp>
#include <multi/adaptors/blas/gemm.hpp>
#include <multi/adaptors/blas/nrm2.hpp>
#include <multi/adaptors/blas/scal.hpp>

#include <multi/adaptors/thrust.hpp>

#include<thrust/complex.h>

#include<numeric>
#include <thrust/inner_product.h>

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
	}
	// {
	//  T res;
	//  blas::dot(blas::C(x), blas::C(y), res);
	// }
}
