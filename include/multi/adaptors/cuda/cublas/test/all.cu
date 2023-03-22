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
	{
		using blas::operators::contains_nan;
		BOOST_REQUIRE( not contains_nan(x) );
	}
	{
		using blas::operators::isfinite;
		using blas::operators::isinf;
		BOOST_REQUIRE( isfinite(x) );
		BOOST_REQUIRE( not isinf(x) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_asum_complex_nans) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc = thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> const x = { 1.0 + I*8.0,  std::numeric_limits<double>::quiet_NaN() + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};

	double res;
	{
		using blas::operators::contains_nan;
		BOOST_REQUIRE( contains_nan(x) );
	}
	{
		using blas::operators::operator==;
		using blas::operators::operator!=;
		BOOST_REQUIRE( not (x != 0) );
		BOOST_REQUIRE( not (x == 0) );
	}
	{
		using blas::operators::isfinite;
		using blas::operators::isinf;
		BOOST_REQUIRE( not isfinite(x) );
		BOOST_REQUIRE( not isinf(x) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_asum_complex_inf) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc = thrust::cuda::allocator<complex>;

	multi::array<T, 1, Alloc> const x = { 1.0 + I*8.0,  std::numeric_limits<double>::infinity() + I*6.0,  3.0 + I*5.0,  4.0 + I*3.0};

	double res;
	{
		using blas::operators::contains_nan;
		BOOST_REQUIRE( not contains_nan(x) );
	}
	{
		using blas::operators::operator==;
		using blas::operators::operator!=;
		BOOST_REQUIRE(     (x != 0) );
		BOOST_REQUIRE( not (x == 0) );
	}
	{
		using blas::operators::isfinite;
		using blas::operators::isinf;
		BOOST_REQUIRE( not isfinite(x) );
		BOOST_REQUIRE( isinf(x) );
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

BOOST_AUTO_TEST_CASE(cublas_axpy_complex_one) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;

	multi::array<complex, 1, Alloc> const x = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0}, {4.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming

	blas::axpy(1.0, x, y);
	std::cout << y[0] << std::endl;
	BOOST_REQUIRE( static_cast<complex>(y[0]) == 3.2 + I*0.0 );
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		thrust::transform(x.begin(), x.end(), yy.begin(), yy.begin(), [] __device__ (auto const& ex, auto const& ey) {return ex + ey;});
		BOOST_TEST( yy == y , boost::test_tools::per_element() );
	}
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };
		using blas::operators::operator+=;
		yy += x;
		BOOST_REQUIRE( yy == y );
	}
}

BOOST_AUTO_TEST_CASE(cublas_axpy_complex_mone) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;

	multi::array<complex, 1, Alloc> const x = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0}, {4.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming

	blas::axpy(-1.0, x, y);
	std::cout << y[0] << std::endl;
	BOOST_REQUIRE( static_cast<complex>(y[0]) == 1.0 + I*0.0 );
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		thrust::transform(x.begin(), x.end(), yy.begin(), yy.begin(), [] __host__ __device__ (T ex, T ey) {return -1.0*ex + ey;});
		BOOST_TEST( yy == y , boost::test_tools::per_element() );
	}
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };
		using blas::operators::operator-=;
		yy -= x;
		BOOST_REQUIRE( yy == y );
	}
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };
		using blas::operators::operator-=;
		yy -= x;
		yy -= y;
		using blas::operators::norm;
		BOOST_REQUIRE( norm(yy) == 0 );
		using blas::operators::operator==;
		BOOST_REQUIRE( operator==(yy, 0) );
		BOOST_REQUIRE( yy == 0 );
	}
}

BOOST_AUTO_TEST_CASE(cublas_axpy_complex_alpha) {
	namespace blas = multi::blas;
	complex const I{0.0, 1.0};

	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;

	multi::array<complex, 1, Alloc> const x = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0}, {4.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming

	blas::axpy(3.0, x, y);
	std::cout << y[0] << std::endl;
	BOOST_REQUIRE( static_cast<complex>(y[0]) == 5.4 + I*0.0 );
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		thrust::transform(x.begin(), x.end(), yy.begin(), yy.begin(), [aa=3.0] __device__ (T ex, T ey) {return aa*ex + ey;});
		BOOST_TEST( yy == y , boost::test_tools::per_element() );
	}
	{
		multi::array<complex, 1, Alloc> yy = { {2.1, 0.0}, {4.1, 0.0}, {6.1, 0.0}, {11.0, 0.0} };
		using blas::operators::operator+=;
		using blas::operators::operator*;
		yy += 3.0*x;
		BOOST_REQUIRE( yy == y );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemv_conj_complex_zero) {
	namespace blas = multi::blas;
	using T = complex;
	complex const I{0.0, 1.0};
	using Alloc =  thrust::cuda::allocator<complex>;

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{ { 9.0, 0.0}, {24.0, 0.0}, {30.0, 0.0}, {9.0, 0.0} },
		{ { 4.0, 0.0}, {10.0, 0.0}, {12.0, 0.0}, {7.0, 0.0} },
		{ {14.0, 0.0}, {16.0, 0.0}, {36.0, 0.0}, {1.0, 0.0} },
	};
	multi::array<complex, 1, Alloc> const x = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0}, {4.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	blas::gemv(1.0, A, x, 0.0, y);
	{

		multi::array<complex, 1, Alloc> yy = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		std::transform(begin(A), end(A), begin(yy), [&x] (auto const& Ac) {return blas::dot(Ac, x);});

		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		yy = blas::gemv(1.0, A, x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = blas::gemv(1.0, A, x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );

	}
	{
		using blas::operators::operator%;

		multi::array<complex, 1, Alloc> yy = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		yy = A % x;
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemv_complex_conj_zero) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{  9.0 + I*0.0, 24.0 + I* 0.0, 30.0 + I* 0.0, 9.0 + I* 0.0 },
		{  4.0 + I*0.0, 10.0 + I* 0.0, 12.0 + I* 0.0, 7.0 + I* 0.0 },
		{ 14.0 + I*0.0, 16.0 + I* 0.0, 36.0 + I* 0.0, 1.0 + I* 0.0 },
	};
	multi::array<complex, 1, Alloc> const x = { 1.1 + I* 0.0, 2.1 + I* 0.0, 3.1 + I* 0.0};  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
	blas::gemv(1.0, blas::T(A), x, 0.0, y);
	{
		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		using blas::operators::operator*;
		std::transform(begin(transposed(A)), end(transposed(A)), begin(yy), [&x] (auto const& Ac) {return blas::dot(Ac, x);});

		BOOST_REQUIRE_CLOSE( static_cast<complex>(y[0]).real(), static_cast<complex>(yy[0]).real(), 1e-7 );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		yy = blas::gemv(1.0, blas::T(A), x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = blas::gemv(1.0, blas::T(A), x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		using blas::operators::operator%;

		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		yy = ~A % x;
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
}

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(cublas_one_gemv_complex_zero) {
	namespace blas = multi::blas;
	using T = complex;
	complex const I{0.0, 1.0};
	using Alloc =  thrust::cuda::allocator<complex>;

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{ { 9.0, 0.0}, {24.0, 0.0}, {30.0, 0.0}, {9.0, 0.0} },
		{ { 4.0, 0.0}, {10.0, 0.0}, {12.0, 0.0}, {7.0, 0.0} },
		{ {14.0, 0.0}, {16.0, 0.0}, {36.0, 0.0}, {1.0, 0.0} },
	};
	multi::array<complex, 1, Alloc> const x = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0}, {4.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
	blas::gemv(1.0, blas::J(A), x, 0.0, y);
	{
		multi::array<complex, 1, Alloc> yy = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		std::transform(begin(A), end(A), begin(yy), [&x] (auto const& Ac) {
			using blas::operators::operator*;  // nvcc 11.8 needs this to be inside lambda
			return blas::dot(*Ac, x);}
		);

		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		yy = blas::gemv(1.0, blas::J(A), x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = blas::gemv(1.0, blas::J(A), x);
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );

	}
	{
		using blas::operators::operator%;
		using blas::operators::operator*;

		multi::array<complex, 1, Alloc> yy = { {1.1, 0.0}, {2.1, 0.0}, {3.1, 0.0} };  // NOLINT(readability-identifier-length) BLAS naming
		yy = *A % x;
		BOOST_REQUIRE( static_cast<complex>(y[0]) == static_cast<complex>(yy[0]) );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
}


BOOST_AUTO_TEST_CASE(cublas_one_gemv_complex_conjtrans_zero) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  std::allocator<complex>;  // thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{  9.0 + I*0.0, 24.0 + I* 0.0, 30.0 + I* 0.0, 9.0 + I* 0.0 },
		{  4.0 + I*0.0, 10.0 + I* 0.0, 12.0 + I* 0.0, 7.0 + I* 0.0 },
		{ 14.0 + I*0.0, 16.0 + I* 0.0, 36.0 + I* 0.0, 1.0 + I* 0.0 },
	};
	multi::array<complex, 1, Alloc> const x = { 1.1 + I* 0.0, 2.1 + I* 0.0, 3.1 + I* 0.0};  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming

	// blas::gemv(1.0, blas::H(A), x, 0.0, y);

	{
		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		std::transform(begin(transposed(A)), end(transposed(A)), begin(yy), [&x] (auto const& Ac) {
			using blas::operators::operator*;  // nvcc 11.8 needs this to be inside lambda
			return blas::dot(*Ac, x);}
		);

		BOOST_REQUIRE_CLOSE( static_cast<complex>(yy[0]).real() ,  61.7, 1.e-7  );
		BOOST_REQUIRE_CLOSE( static_cast<complex>(yy[1]).real() ,  97.0, 1.e-7  );
		BOOST_REQUIRE_CLOSE( static_cast<complex>(yy[2]).real() , 169.8, 1.e-7  );
		BOOST_REQUIRE_CLOSE( static_cast<complex>(yy[3]).real() ,  27.7, 1.e-7  );

		using blas::operators::operator*;
		BOOST_REQUIRE_CLOSE( static_cast<complex>(yy[0]).real() , (+blas::dot(*(~A)[0], x)).real() , 1.e-7  );
		BOOST_REQUIRE_CLOSE( static_cast<complex>(yy[1]).real() , (+blas::dot(*(~A)[1], x)).real() , 1.e-7  );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemv_complex_trans_one) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{  9.0 + I*0.0, 24.0 + I* 0.0, 30.0 + I* 0.0, 9.0 + I* 0.0 },
		{  4.0 + I*0.0, 10.0 + I* 0.0, 12.0 + I* 0.0, 7.0 + I* 0.0 },
		{ 14.0 + I*0.0, 16.0 + I* 0.0, 36.0 + I* 0.0, 1.0 + I* 0.0 },
	};
	multi::array<complex, 1, Alloc> const x = { 1.1 + I* 0.0, 2.1 + I* 0.0, 3.1 + I* 0.0};  // NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 1, Alloc> y = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
	blas::gemv(3.0 + I*4.0, blas::T(A), x, 1.0, y);
	{
		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		// using blas::operators::operator*;
		std::transform(begin(transposed(A)), end(transposed(A)), begin(yy), begin(yy), [&x,aa=3.0 + I*4.0,bb=1.0] (auto const& Ac, complex e) {return aa*blas::dot(Ac, x) + bb*e;});

		BOOST_REQUIRE_CLOSE( static_cast<complex>(y[0]).real(), static_cast<complex>(yy[0]).real(), 1e-7 );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		yy += blas::gemv(3.0 + I*4.0, blas::T(A), x);

		BOOST_REQUIRE_CLOSE( static_cast<complex>(y[0]).real(), static_cast<complex>(yy[0]).real(), 1e-7 );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
	{
		multi::array<complex, 1, Alloc> yy = { 1.1 + I* 0.0, 2.1 +I* 0.0, 3.1 + I* 0.0, 6.7 + I*0.0 };  // NOLINT(readability-identifier-length) BLAS naming
		using blas::operators::operator*;
		yy += (3.0 + I*4.0)* ~A % x;

		BOOST_REQUIRE_CLOSE( static_cast<complex>(y[0]).real(), static_cast<complex>(yy[0]).real(), 1e-7 );
		BOOST_REQUIRE( static_cast<complex>(y[1]) == static_cast<complex>(yy[1]) );
		BOOST_REQUIRE( static_cast<complex>(y[2]) == static_cast<complex>(yy[2]) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemm_complex_trans_none) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{1.0 - 2.0 * I, 9.0 - 1.0 * I},
		{2.0 + 3.0 * I, 1.0 - 2.0 * I},
	};
	multi::array<complex, 2, Alloc> const B = {
		{3.0 - 4.0 * I, 19.0 - 1.0 * I},
		{1.0 + 5.0 * I,  8.0 - 8.0 * I},
	};
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		blas::gemm({1.0, 0.0}, A, B, {0.0, 0.0}, C);

		// std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
		//  [&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		// );
		std::transform(begin(A), end(A), begin(C_copy), end(C_copy),
			[&B, aa=1.0, bb=0] (auto const& Ar, auto&& Cr) {return blas::gemv(aa, blas::T(B), Ar, bb, std::move(Cr));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C = blas::gemm(1.0 + I*0.0, A, B);

		// std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
		//  [&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		// );
		std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=0.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, blas::T(B), Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C += blas::gemm(1.0 + I*0.0, A, B);

		std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=1.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += A*B;

		std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, blas::T(B), Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemm_complex_trans_second) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{1.0 - 2.0 * I, 9.0 - 1.0 * I},
		{2.0 + 3.0 * I, 1.0 - 2.0 * I},
	};
	multi::array<complex, 2, Alloc> const B = {
		{3.0 - 4.0 * I, 19.0 - 1.0 * I},
		{1.0 + 5.0 * I,  8.0 - 8.0 * I},
	};
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		blas::gemm({1.0, 0.0}, A, blas::T(B), {0.0, 0.0}, C);

		std::transform(begin(B), end(B), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C = blas::gemm(1.0 + I*0.0, A, blas::T(B));

		// std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
		//  [&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		// );
		std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=0.0] (auto const& Ac, auto&& Cr) {
			return blas::gemv(aa, B, Ac, bb, std::move(Cr));
		});

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C += blas::gemm(1.0 + I*0.0, A, blas::T(B));

		std::transform(begin(B), end(B), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=1.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += A * ~B;

		std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, B, Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += 2.0*(A * ~B);

		std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=2.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, B, Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemm_complex_trans_first) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{1.0 - 2.0 * I, 9.0 - 1.0 * I},
		{2.0 + 3.0 * I, 1.0 - 2.0 * I},
	};
	multi::array<complex, 2, Alloc> const B = {
		{3.0 - 4.0 * I, 19.0 - 1.0 * I},
		{1.0 + 5.0 * I,  8.0 - 8.0 * I},
	};
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		blas::gemm({1.0, 0.0}, blas::T(A), B, {0.0, 0.0}, C);

		std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, blas::T(A), Bc, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C = blas::gemm(1.0 + I*0.0, blas::T(A), B);

		// std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
		//  [&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		// );
		std::transform(begin(transposed(A)), end(transposed(A)), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=0.0] (auto const& Ac, auto&& Cr) {
			return blas::gemv(aa, blas::T(B), Ac, bb, std::move(Cr));
		});

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C += blas::gemm(1.0 + I*0.0, blas::T(A), B);

		std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=1.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, blas::T(A), Bc, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += ~A * B;

		std::transform(begin(transposed(A)), end(transposed(A)), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, blas::T(B), Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += 2.0*(~A * B);

		std::transform(begin(transposed(A)), end(transposed(A)), begin(C_copy), begin(C_copy), [&B, aa=2.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, blas::T(B), Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemm_complex_trans_both) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{1.0 - 2.0 * I, 9.0 - 1.0 * I},
		{2.0 + 3.0 * I, 1.0 - 2.0 * I},
	};
	multi::array<complex, 2, Alloc> const B = {
		{3.0 - 4.0 * I, 19.0 - 1.0 * I},
		{1.0 + 5.0 * I,  8.0 - 8.0 * I},
	};
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		blas::gemm({1.0, 0.0}, blas::T(A), blas::T(B), {0.0, 0.0}, C);

		std::transform(begin(B), end(B), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=0.0] (auto const& Br, auto&& Cc) {return blas::gemv(aa, blas::T(A), Br, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C = blas::gemm(1.0 + I*0.0, blas::T(A), blas::T(B));

		// std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
		//  [&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
		// );
		std::transform(begin(transposed(A)), end(transposed(A)), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=0.0] (auto const& Ac, auto&& Cr) {
			return blas::gemv(aa, B, Ac, bb, std::move(Cr));
		});

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		C += blas::gemm(1.0 + I*0.0, blas::T(A), blas::T(B));

		std::transform(begin(B), end(B), begin(transposed(C_copy)), begin(transposed(C_copy)),
			[&A, aa=1.0, bb=1.0] (auto const& Br, auto&& Cc) {return blas::gemv(aa, blas::T(A), Br, bb, std::move(Cc));}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += ~A * ~B;

		std::transform(begin(transposed(A)), end(transposed(A)), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, B, Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto C_copy = C;
		using blas::operators::operator*;
		using blas::operators::operator+=;
		C += 2.0*(~A * ~B);

		std::transform(begin(transposed(A)), end(transposed(A)), begin(C_copy), begin(C_copy), [&B, aa=2.0, bb=1.0] (auto const& Ar, auto&& Cr) {
			return blas::gemv(aa, B, Ar, bb, std::move(Cr));
		}
		);

		BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
		BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	}
}

BOOST_AUTO_TEST_CASE(cublas_one_gemm_complex_conj_second) {
	namespace blas = multi::blas;
	using T = complex;
	using Alloc =  std::allocator<complex>;  // thrust::cuda::allocator<complex>;
	complex const I{0.0, 1.0};

	// NOLINT(readability-identifier-length) BLAS naming
	multi::array<complex, 2, Alloc> const A = {
		{1.0 - 2.0 * I, 9.0 - 1.0 * I},
		{2.0 + 3.0 * I, 1.0 - 2.0 * I},
	};
	multi::array<complex, 2, Alloc> const B = {
		{3.0 - 4.0 * I, 19.0 - 1.0 * I},
		{1.0 + 5.0 * I,  8.0 - 8.0 * I},
	};
	{
		multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
		auto CC = C;
		auto C_copy = CC;
		// blas::gemm({1.0, 0.0}, A, blas::J(B), {0.0, 0.0}, C);
		blas::gemm({1.0, 0.0}, blas::T(B), blas::H(A), {0.0, 0.0}, C_copy);
		{
			auto const [is, js] = C.extensions();
			for(auto i : is) {
				for(auto j : js) {
					C[i][j] *= 0.0;
					for(auto k : B.extension()) {
						C[i][j] += A[i][k]*conj(B[k][j]);
					}
				}
			}
		}
		{
			std::transform(begin(A), end(A), begin(CC), begin(CC), [BT = transposed(B)](auto const& Ar, auto&& Cr) {
				return std::transform(
					begin(BT), end(BT), begin(Cr), begin(Cr), [&Ar](auto const& BCr, auto&& Ce) {
						return 1.0*blas::dot(Ar, blas::C(BCr)) + 0.0*Ce;
					}
				), std::move(Cr);
			});
		}
		BOOST_TEST_REQUIRE( static_cast<complex>(CC[1][0]).real() == static_cast<complex>(C[1][0]).real() );
		BOOST_TEST_REQUIRE( static_cast<complex>(CC[1][0]).imag() == static_cast<complex>(C[1][0]).imag() );

		BOOST_TEST_REQUIRE( static_cast<complex>(CC[0][1]).real() == static_cast<complex>(C[0][1]).real() );
		BOOST_TEST_REQUIRE( static_cast<complex>(CC[0][1]).imag() == static_cast<complex>(C[0][1]).imag() );

		BOOST_TEST_REQUIRE( static_cast<complex>(C_copy[1][0]).real() == +static_cast<complex>(C[0][1]).real() );
		BOOST_TEST_REQUIRE( static_cast<complex>(C_copy[1][0]).imag() == -static_cast<complex>(C[0][1]).imag() );
	}
	//  auto const stl_gemvH = [](auto aa, auto const& A, auto const& x, auto bb, auto&& yy) {
	//      std::transform(begin(transposed(A)), end(transposed(A)), begin(yy), begin(yy), [aa, &x, bb] (auto const& Ac, auto&& ye) {
	//          using blas::operators::operator*;  // nvcc 11.8 needs this to be inside lambda
	//          return aa*static_cast<complex>(blas::dot(*Ac, x)) + bb*ye;}
	//      );
	//      return std::move(yy);
	//  };
	//  auto const stl_gemv = [](auto aa, auto const& A, auto const& x, auto bb, auto&& yy) {
	//      std::transform(begin(A), end(A), begin(yy), begin(yy), [aa, &x, bb] (auto const& Ar, auto&& ye) {
	//          using blas::operators::operator*;  // nvcc 11.8 needs this to be inside lambda
	//          return aa*static_cast<complex>(blas::dot(Ar, x)) + bb*ye;}
	//      );
	//      return std::move(yy);
	//  };

	//  std::transform(begin(A), end(A), begin(C), end(C),
	//      [&B, aa=1.0, bb=0, stl_gemv] (auto const& Ar, auto&& Cr) {return stl_gemv(aa, blas::H(B), Ar, bb, std::move(Cr));}
	//  );

	//  std::transform(
	//      begin(A), end(A), begin(C), end(C),
	//      [&B, aa=1.0, bb=0, stl_gemv] (auto const& Ar, auto&& Cr) {
	//          std::transform(begin(blas::H(B)), end(blas::H(B)), begin(Cr), begin(Cr), [aa, &Ar, bb] (auto const& Bc, auto&& ye) {
	//              using blas::operators::operator*;  // nvcc 11.8 needs this to be inside lambda
	//              return aa*static_cast<complex>(blas::dot(Bc, Ar)) + bb*ye;
	//          });
	//          return std::move(Cr);
	//      }
	//  );

	//  BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
	//  BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	// }
	// {
	//  multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
	//  auto C_copy = C;
	//  C = blas::gemm(1.0 + I*0.0, A, blas::T(B));

	//  // std::transform(begin(transposed(B)), end(transposed(B)), begin(transposed(C_copy)), begin(transposed(C_copy)),
	//  //  [&A, aa=1.0, bb=0.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
	//  // );
	//  std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=0.0] (auto const& Ac, auto&& Cr) {
	//      return blas::gemv(aa, B, Ac, bb, std::move(Cr));
	//  });

	//  BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
	//  BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	// }
	// {
	//  multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
	//  auto C_copy = C;
	//  C += blas::gemm(1.0 + I*0.0, A, blas::T(B));

	//  std::transform(begin(B), end(B), begin(transposed(C_copy)), begin(transposed(C_copy)),
	//      [&A, aa=1.0, bb=1.0] (auto const& Bc, auto&& Cc) {return blas::gemv(aa, A, Bc, bb, std::move(Cc));}
	//  );

	//  BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
	//  BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	// }
	// {
	//  multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
	//  auto C_copy = C;
	//  using blas::operators::operator*;
	//  using blas::operators::operator+=;
	//  C += A * ~B;

	//  std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=1.0, bb=1.0] (auto const& Ar, auto&& Cr) {
	//      return blas::gemv(aa, B, Ar, bb, std::move(Cr));
	//  }
	//  );

	//  BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
	//  BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	// }
	// {
	//  multi::array<complex, 2, Alloc> C({2, 2}, {3.0, 0.0});  // NOLINT(readability-identifier-length) conventional BLAS naming
	//  auto C_copy = C;
	//  using blas::operators::operator*;
	//  using blas::operators::operator+=;
	//  C += 2.0*(A * ~B);

	//  std::transform(begin(A), end(A), begin(C_copy), begin(C_copy), [&B, aa=2.0, bb=1.0] (auto const& Ar, auto&& Cr) {
	//      return blas::gemv(aa, B, Ar, bb, std::move(Cr));
	//  }
	//  );

	//  BOOST_REQUIRE( static_cast<complex>(C_copy[1][0]) == static_cast<complex>(C[1][0]) );
	//  BOOST_REQUIRE( static_cast<complex>(C_copy[0][1]) == static_cast<complex>(C[0][1]) );
	// }
}

#endif
