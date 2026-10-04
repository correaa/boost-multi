// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Performance-portable kernel execution: a single piece of code, compiled
// once, that runs (and is meant to run efficiently) on several backends
// depending only on the array type it is called with. Reference shape of
// the kernel (elsewhere expressed with an explicit execution-policy
// template argument):
//
//     EXEC_POL::forall(RangeSegment(0, N), [=] (int i) {
//         y[i] = a * x[i] + y[i];
//     });
//
// with EXEC_POL one of {seq_exec, omp_parallel_for_exec, cuda_exec<BLOCK>}
// selecting both the backend AND the performance characteristics at the
// call site. Below, `daxpy()` is written once; thrust::transform dispatches
// on the iterator system of its arguments, so the backend (and its
// performance) follows from the array type passed in: multi::array runs
// Thrust's sequential host backend, multi::thrust::device_array runs on the
// GPU via CUDA, multi::thrust::omp::array runs multithreaded via OpenMP.
// Each variant below is timed to show that performance, not just the
// result, carries over.

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/adaptors/thrust/omp.hpp>
#include <boost/multi/array.hpp>

#include <thrust/transform.h>  // IWYU pragma: keep

#include <boost/core/lightweight_test.hpp>

#include <chrono>
#include <iostream>

namespace multi = boost::multi;

namespace {

// the single kernel: same code, no backend-specific branch.
// instantiated below with a plain host array, a Thrust/CUDA array, and a
// Thrust/OMP array; the backend (and its performance) follows from Array1D alone.
// the lambda needs an explicit __host__ __device__ (nvcc does not reliably
// auto-detect it as an extended lambda inside a function template here), a
// plain copy-capture `[a]` rather than an init-capture `[a = a]`, and
// explicit (non-`auto`) parameter types: nvcc rejects both init-captures and
// generic-lambda parameters on extended __host__ __device__ lambdas.
template<class Array1D>
auto daxpy(double a, Array1D const& x, Array1D y) -> Array1D {
	thrust::transform(
		x.begin(), x.end(), y.begin(), y.begin(),
		[a] __host__ __device__(double xi, double yi) { return a * xi + yi; }
	);
	return y;
}

template<class F>
auto timed(char const* label, F&& f) {
	auto const  tick = std::chrono::high_resolution_clock::now();
	auto        ret  = std::forward<F>(f)();
	auto const  secs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tick).count();
	std::cout << label << ": " << secs << " s\n";
	return ret;
}

}  // end namespace

auto main() -> int {  // NOLINT(bugprone-exception-escape)
	constexpr auto N = 1 << 20;
	constexpr auto daxpy_a = 2.0;

	multi::array<double, 1> x(N);
	multi::array<double, 1> y0(N);
	for(auto const i : x.extent()) {  // NOLINT(altera-unroll-loops)
		x[i]  = static_cast<double>(i);
		y0[i] = static_cast<double>(i) * 2.0;
	}

	// ground truth, not part of the portability demonstration
	multi::array<double, 1> reference = y0;
	for(auto const i : x.extent()) {  // NOLINT(altera-unroll-loops)
		reference[i] = daxpy_a * x[i] + reference[i];
	}

	{  // cpu: multi::array has no Thrust allocator, thrust::transform falls back to its sequential host backend
		auto const result = timed("cpu ", [&] { return daxpy(daxpy_a, x, y0); });
		for(auto const i : x.extent()) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( result[i] == reference[i] );
		}
	}
	{  // gpu: same daxpy(), called with device-allocated arrays
		multi::thrust::device_array<double, 1> const dx  = x;
		multi::thrust::device_array<double, 1> const dy0 = y0;

		multi::array<double, 1> const result = timed("gpu ", [&] {
			auto ret = daxpy(daxpy_a, dx, dy0);
			cudaDeviceSynchronize();
			return ret;
		});
		for(auto const i : x.extent()) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( result[i] == reference[i] );
		}
	}
#if defined(_OPENMP)  // Thrust's OMP backend static_asserts without compiler OpenMP support (e.g. -fopenmp)
	{  // omp: same daxpy(), called with OpenMP-allocated arrays
		multi::thrust::omp::array<double, 1> const ox  = x;
		multi::thrust::omp::array<double, 1> const oy0 = y0;

		multi::array<double, 1> const result = timed("omp ", [&] { return daxpy(daxpy_a, ox, oy0); });
		for(auto const i : x.extent()) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( result[i] == reference[i] );
		}
	}
#endif

	return boost::report_errors();
}
