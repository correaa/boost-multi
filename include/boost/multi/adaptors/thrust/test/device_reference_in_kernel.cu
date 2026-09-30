// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Exposes a surprise when dereferencing Thrust *device* pointers inside kernels:
// `y[i]` is not a raw `double&` but a Thrust synthetic reference (`thrust::device_reference<double>`),
// even though in device code the memory is directly addressable.
// The proxy converts to a value `double`, but not to `double&`, so generic (template) code fails to compile.
// Thrust *universal* pointers do not have this problem, their reference type is a raw `double&`.

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <thrust/complex.h>
#include <thrust/device_allocator.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

// y <- x + y, generic (e.g. to also work with float or complex)
template<class T> __host__ __device__ void xpy(T const& x, T& y) { y = x + y; }

auto main() -> int {
	// basic types, basic operators works without much problems
	{
		multi::thrust::device_array<double, 1> y(1000, 10.0);

		thrust::for_each(
			y.extent().begin(), y.extent().end(),
			[x = 2.0, y = y.home()] __device__(int i) {
				y[i] = x + y[i];  // this works because thrust::reference have some basic conversions and + is a built-in function
				// xpy(x, y[i]);  // doesn't compile, y[i] is a thrust::device_reference<double>, T can't be deduced
				// xpy(x, thrust::raw_reference_cast(y[i]));  // compiles, raw_reference_cast gives a double&
			}
		);

		BOOST_TEST( y[0] == 12.0 );  // there is thrust magic here too
		BOOST_TEST( y[999] == 12.0 );
	}
	{
		multi::thrust::device_array<thrust::complex<double>, 1> y(1000, thrust::complex<double>(10.0, 0.0));

		thrust::for_each(
			y.extent().begin(), y.extent().end(),
			[x = thrust::complex<double>(2.0, 0.0), y = y.home()] __device__(int i) {
				// y[i] = x + y[i];  // doesn't work because operator+(complex) is a template
				// xpy(x, y[i]);  // doesn't compile, y[i] is a thrust::device_reference<double>, T can't be deduced
				xpy(x, thrust::raw_reference_cast(y[i]));  // compiles, raw_reference_cast gives a double&
			}
		);

		BOOST_TEST( thrust::complex<double>(y[0]) == thrust::complex<double>(12.0, 0.0) );
		BOOST_TEST( thrust::complex<double>(y[999]) == thrust::complex<double>(12.0, 0.0) );
	}
	{
		multi::thrust::universal_array<thrust::complex<double>, 1> y(1000, thrust::complex<double>(10.0, 0.0));

		thrust::for_each(
			y.extent().begin(), y.extent().end(),
			[x = thrust::complex<double>(2.0, 0.0), y = y.home()] __device__(int i) {
				// y[i] = x + y[i];  // works because these are raw references directly
				// xpy(x, y[i]);  // works because these are raw references directly
				xpy(x, thrust::raw_reference_cast(y[i]));  // works but unnecessary
			}
		);

		BOOST_TEST( y[0] == thrust::complex<double>(12.0, 0.0) );
		BOOST_TEST( y[999] == thrust::complex<double>(12.0, 0.0) );
	}
	{
		multi::thrust::device_array<thrust::complex<double>, 1> y(1000, thrust::complex<double>(10.0, 0.0));

		thrust::for_each(
			y.extent().begin(), y.extent().end(),
			[x = thrust::complex<double>(2.0, 0.0), y = y.raw_array_cast().home()] __device__(int i) {
				y[i] = x + y[i];  // works because of the conversion above
				// xpy(x, y[i]);  // works because of the conversion above
				// xpy(x, thrust::raw_reference_cast(y[i]));  // works but unnecessary
			}
		);

		BOOST_TEST( thrust::complex<double>(y[0]) == thrust::complex<double>(12.0, 0.0) );
		BOOST_TEST( thrust::complex<double>(y[999]) == thrust::complex<double>(12.0, 0.0) );
	}
	{
		multi::thrust::device_array<thrust::complex<double>, 1> y(1000, thrust::complex<double>(10.0, 0.0));

		thrust::for_each(
			y.extent().begin(), y.extent().end(),
			[x = thrust::complex<double>(2.0, 0.0), y = +y.home()] __device__(int i) {
				y[i] = x + y[i];  // works because of the conversion above
				// xpy(x, y[i]);  // works because of the conversion above
				// xpy(x, thrust::raw_reference_cast(y[i]));  // works but unnecessary
			}
		);

		BOOST_TEST( thrust::complex<double>(y[0]) == thrust::complex<double>(12.0, 0.0) );
		BOOST_TEST( thrust::complex<double>(y[999]) == thrust::complex<double>(12.0, 0.0) );
	}

	return boost::report_errors();
}
