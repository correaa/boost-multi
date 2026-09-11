// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Minimal isolation test: does a single-element H->D write + D->H read of a
// thrust::complex<double> multi::array work at all on this ROCm/rocThrust
// build, independent of hipFFT? If this aborts the same way hipfft.cpp.x
// does ("trivial_device_copy D->H failed: hipErrorInvalidValue"), the bug
// has nothing to do with the FFT plan/execute path; if it passes, the bug
// is specific to whatever hipfft.cpp's FFT execution leaves in fw_gpu.

#include <boost/multi/adaptors/thrust.hpp>

#include <thrust/complex.h>

#include <cstdio>

namespace multi = boost::multi;
using complex   = thrust::complex<double>;

auto main() -> int {
	multi::thrust::cuda::array<complex, 1> arr({4});

	arr[0] = complex{1.0, 2.0};  // H->D write via assignment

	complex const val = arr[0];  // D->H read: same mechanism crashing in hipfft.cpp.x

	std::printf("got %f %f (expected 1.0 2.0)\n", val.real(), val.imag());

	return (val.real() == 1.0 && val.imag() == 2.0) ? 0 : 1;
}
