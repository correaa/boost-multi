// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Minimal isolation of the first FFT block in cufft/test/cufft.cpp (the one
// hipfft.cpp.x aborts in), with diagnostics inserted between plan.execute()
// and the failing device->host read, to find out whether:
//  (a) the FFT plan/execute itself already leaves the HIP context in an
//      error state (would show up as a non-hipSuccess from hipGetLastError()
//      or hipDeviceSynchronize() right after .execute(), even though
//      cufftSafeCall/hipfftSafeCall didn't catch it), or
//  (b) fw_gpu's pointer/extents are sane and the read is what first fails.

#include <boost/multi/adaptors/thrust.hpp>

#if (!(defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__))) && (!defined(__HIPCC__))
#include <boost/multi/adaptors/cufft.hpp>
#else
#include <boost/multi/adaptors/hipfft.hpp>
#endif

#include <thrust/complex.h>
#include <thrust/copy.h>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

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
	auto       fw_gpu = multi::thrust::cuda::array<complex, 2>(extents(in_gpu));

	std::printf("in_gpu  base=%p  extents={%td,%td}\n", static_cast<void const*>(::thrust::raw_pointer_cast(in_gpu.base())), static_cast<std::ptrdiff_t>(in_gpu.size()), static_cast<std::ptrdiff_t>(in_gpu[0].size()));
	std::printf("fw_gpu  base=%p  extents={%td,%td}\n", static_cast<void const*>(::thrust::raw_pointer_cast(fw_gpu.base())), static_cast<std::ptrdiff_t>(fw_gpu.size()), static_cast<std::ptrdiff_t>(fw_gpu[0].size()));

	multi::cufft::plan<2>({true, true}, in_gpu.layout(), fw_gpu.layout())
		.execute(in_gpu.base(), fw_gpu.base(), multi::cufft::forward);

	// (a) did the FFT itself leave the context poisoned?
	hipError_t const last = hipGetLastError();
	std::printf("hipGetLastError() after execute() = %d (%s)\n", static_cast<int>(last), hipGetErrorString(last));

	hipError_t const sync = hipDeviceSynchronize();
	std::printf("hipDeviceSynchronize() after execute() = %d (%s)\n", static_cast<int>(sync), hipGetErrorString(sync));

	if(last != hipSuccess || sync != hipSuccess) {
		std::printf("FFT execution itself left an error -- the read below is not the real bug.\n");
		return 1;
	}

	std::printf("about to bulk-copy fw_gpu (5x3) device->host via thrust::copy ...\n");
	std::fflush(stdout);

	std::vector<complex> fw_host(static_cast<std::size_t>(fw_gpu.num_elements()));
	::thrust::copy(fw_gpu.elements().begin(), fw_gpu.elements().end(), fw_host.begin());

	std::printf("bulk copy ok, fw_host[3*3+2] = %f %f  (expect nonzero real and imag)\n", fw_host[3 * 3 + 2].real(), fw_host[3 * 3 + 2].imag());
	std::fflush(stdout);

	std::printf("about to read fw_gpu[3][2] (single-element proxy read) ...\n");
	std::fflush(stdout);

	// (b) the actual read that crashes in hipfft.cpp.x
	complex const val = fw_gpu[3][2];

	std::printf("got %f %f\n", val.real(), val.imag());

	return 0;
}
