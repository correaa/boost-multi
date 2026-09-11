// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Same as hipfft_minimal.hip, but fw_gpu's underlying storage is allocated
// with extra trailing padding (a flat 1-D buffer larger than the 5x3=15
// elements actually needed), and the 2-D FFT output is only a *view* into
// the first 15 elements of it. If padding makes the later read succeed,
// the FFT write is going out of bounds of a tightly-sized allocation.

#include <boost/multi/adaptors/thrust.hpp>

#if (!(defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__))) && (!defined(__HIPCC__))
#include <boost/multi/adaptors/cufft.hpp>
#else
#include <boost/multi/adaptors/hipfft.hpp>
#endif

#include <thrust/complex.h>

#include <hip/hip_runtime.h>

#include <cstdio>

namespace multi = boost::multi;
using complex   = thrust::complex<double>;

auto main() -> int {
	complex const I{0.0, 1.0};  // NOLINT(readability-identifier-length)

	multi::array<complex, 2> const in_cpu = {
		{ 1.0 + 2.0 * I,  9.0 - 1.0 * I,  2.0 + 4.0 * I},
		{ 3.0 + 3.0 * I,  7.0 - 4.0 * I,  1.0 + 9.0 * I},
		{ 4.0 + 1.0 * I,  5.0 + 3.0 * I,  2.0 + 4.0 * I},
		{ 3.0 - 1.0 * I,  8.0 + 7.0 * I,  2.0 + 1.0 * I},
		{31.0 - 1.0 * I, 18.0 + 7.0 * I, 2.0 + 10.0 * I}
	};

	auto const in_gpu = multi::thrust::cuda::array<complex, 2>{in_cpu};

	constexpr multi::ssize_t rows = 5;
	constexpr multi::ssize_t cols = 3;
	constexpr multi::ssize_t pad  = 256;  // extra elements of slack after the "real" 15

	multi::thrust::cuda::array<complex, 1> fw_flat(rows * cols + pad);  // one padded, flat allocation (NOTE: no inner braces -- `({N})` would be a 1-element initializer_list, not an N-element extent)
	multi::array_ref<complex, 2, decltype(fw_flat.base())> fw_gpu(fw_flat.base(), {rows, cols});  // view over just the first 15 elements

	std::printf("in_gpu   base=%p  extents={%td,%td}\n", static_cast<void const*>(::thrust::raw_pointer_cast(in_gpu.base())), static_cast<std::ptrdiff_t>(in_gpu.size()), static_cast<std::ptrdiff_t>(in_gpu[0].size()));
	std::printf("fw_flat  base=%p  size=%td  (padded, %td real + %td pad)\n", static_cast<void const*>(::thrust::raw_pointer_cast(fw_flat.base())), static_cast<std::ptrdiff_t>(fw_flat.size()), static_cast<std::ptrdiff_t>(rows * cols), static_cast<std::ptrdiff_t>(pad));

	multi::cufft::plan<2>({true, true}, in_gpu.layout(), fw_gpu.layout())
		.execute(in_gpu.base(), fw_gpu.base(), multi::cufft::forward);

	hipError_t const last = hipGetLastError();
	std::printf("hipGetLastError() after execute() = %d (%s)\n", static_cast<int>(last), hipGetErrorString(last));

	hipError_t const sync = hipDeviceSynchronize();
	std::printf("hipDeviceSynchronize() after execute() = %d (%s)\n", static_cast<int>(sync), hipGetErrorString(sync));

	std::printf("about to read fw_gpu[3][2] (padded allocation) ...\n");
	std::fflush(stdout);

	complex const val = fw_gpu[3][2];

	std::printf("got %f %f\n", val.real(), val.imag());

	std::printf("about to read fw_flat[%td] (last padding element, sanity check tail is intact) ...\n", static_cast<std::ptrdiff_t>(rows * cols + pad - 1));
	std::fflush(stdout);

	complex const tail = fw_flat[rows * cols + pad - 1];
	std::printf("tail got %f %f\n", tail.real(), tail.imag());

	return 0;
}
