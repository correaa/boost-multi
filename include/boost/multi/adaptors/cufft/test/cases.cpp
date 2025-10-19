// Copyright 2020-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>

#include <boost/multi/adaptors/fftw.hpp>
#include <boost/multi/array.hpp>

#include <boost/multi/adaptors/cufft.hpp>

#if(!(defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__))) && (!defined(__HIPCC__))
#include <boost/multi/adaptors/cufft.hpp>
#else
#include <boost/multi/adaptors/hipfft.hpp>
#endif

#include <boost/multi/adaptors/fft.hpp>
#include <boost/multi/adaptors/thrust.hpp>

#include <thrust/complex.h>
#include <thrust/transform_reduce.h>

#include <random>

namespace multi = boost::multi;

using complex   = thrust::complex<double>;

template<>
constexpr bool multi::force_element_trivial_default_construction<thrust::complex<double>> = true;

auto main() -> int try {
	complex const I{0.0, 1.0};  // NOLINT(readability-identifier-length)

	auto in_cpu = multi::array<complex, 4>({20, 30, 50, 70});

	std::generate(
		in_cpu.elements().begin(), in_cpu.elements().end(),
		[dist = std::normal_distribution<>{}, gen = std::mt19937(std::random_device{}())] () mutable { return dist(gen); }
	);

	auto const in_gpu = multi::thrust::cuda::array<complex, 4>(in_cpu);

	auto const& in_cpu_view = in_cpu.transposed().rotated().transposed().unrotated().transposed();
	auto const& in_gpu_view = in_gpu.transposed().rotated().transposed().unrotated().transposed();

	auto out_gpu = multi::thrust::cuda::array<complex, 4>(in_gpu_view.extensions());
	auto out_cpu = multi::array<complex, 4>(out_gpu.extensions());

	assert(in_gpu_view.extensions() == out_gpu.extensions());

	multi::fftw::dft_forward({true, true, true, false}, in_cpu_view, out_cpu);
	
	multi::cufft::plan<4>({true, true, true, false}, in_gpu_view.layout(), out_gpu.layout())
		.execute(in_gpu_view.base(), out_gpu.base(), multi::cufft::forward);

	std::cout << out_cpu[2][3][4][5] << ' ' << out_gpu[2][3][4][5] << std::endl;
	BOOST_TEST( thrust::abs(out_cpu[2][3][4][5] - static_cast<complex>(out_gpu[2][3][4][5])) < 1e-6 );

	return boost::report_errors();
} catch(...) {
	throw;
	return 1;
}
