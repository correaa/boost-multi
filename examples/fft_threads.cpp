// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Running one multi::fft_plan from several threads.
//
// A plan owns no scratch: `execute()` allocates whatever it needs through the
// allocator it is handed, per call. Everything the plan itself holds (twiddle
// tables, stage factorisations, sub-plans) is written once at construction and
// read-only afterwards. So a `const` plan can be shared by any number of
// threads with no locking, as long as each thread supplies its own scratch.
//
// That is the whole pattern, and the two lines that matter are:
//
//   * ONE plan, `const`, shared by reference;
//   * ONE arena PER THREAD, reused across that thread's transforms.
//
// The arena matters as much as the plan. `execute()`'s default allocator is a
// fresh std::allocator<T> per call, so a loop of transforms would allocate and
// free the scratch every iteration. A monotonic buffer over a caller-owned
// block, rewound (not freed) between calls, removes that entirely -- and it
// gives each thread private scratch, which is what makes sharing the plan safe.
//
// Build (needs pthreads):
//   c++ -std=c++17 -O3 -I../include fft_threads.cpp -o fft_threads.x -lpthread

#include <boost/multi/algorithms/fft.hpp>
#include <boost/multi/array.hpp>

#include <algorithm>  // for max
#include <chrono>
#include <complex>
#include <cstdio>
#include <memory>  // for unique_ptr
#include <memory_resource>
#include <thread>
#include <vector>

namespace multi = boost::multi;

using complex = std::complex<double>;

namespace {

// Scratch for one thread: a monotonic buffer over a block sized to the plan's
// own requirement, rewound after every transform so the block is reused rather
// than reallocated.
//
// The block is RAW, UNINITIALIZED storage, deliberately:
//
//   * `new std::byte[n]` default-initializes, which for a trivial type means
//     no work at all. `std::vector<std::byte>(n)` and
//     `std::make_unique<std::byte[]>(n)` both VALUE-initialize instead, i.e.
//     they memset the whole block -- on a scratch buffer whose every byte is
//     overwritten before it is read, that is pure waste. fft.hpp's own
//     internal arena avoids exactly this (see fft_scratch_arena), so a caller
//     arena that zero-filled would undo the effort.
//   * std::byte, not complex, because this is storage rather than objects:
//     the resource only ever sees `void*` + size, and an array of `complex`
//     would run 2*N field initializations we do not want. `operator new[]`
//     returns memory aligned for any fundamental type, which covers complex.
class thread_arena {
	std::unique_ptr<std::byte[]>             buffer_;  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) raw storage, see above
	std::pmr::monotonic_buffer_resource      resource_;
	std::pmr::polymorphic_allocator<complex> allocator_;

	static auto bytes_for(std::size_t elements) -> std::size_t {
		// One spare element's worth covers any alignment padding the resource
		// may insert when it carves the first block.
		return (elements + 1) * sizeof(complex);
	}

 public:
	template<class Plan>
	explicit thread_arena(Plan const& plan)
	: buffer_(new std::byte[bytes_for(plan.scratch_elements())])  // default-init: no zero-fill
	, resource_(buffer_.get(), bytes_for(plan.scratch_elements()))
	, allocator_(&resource_) {}

	auto allocator() -> std::pmr::polymorphic_allocator<complex>& { return allocator_; }
	void rewind() { resource_.release(); }  // reuse the block; does not free it
};

auto now() { return std::chrono::steady_clock::now(); }

auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
	return std::chrono::duration<double>(now() - start).count();
}

}  // namespace

auto main() -> int {  // NOLINT(bugprone-exception-escape)
	int const transform_size  = 1024;
	int const transform_count = 20000;

	// Many independent transforms -- the shape that parallelises. One plan
	// describes all of them, since they share a shape.
	std::vector<multi::array<complex, 1>> signals;
	signals.reserve(static_cast<std::size_t>(transform_count));
	for(int i = 0; i != transform_count; ++i) {
		multi::array<complex, 1> signal(multi::extents_t<1>{transform_size}, complex{});
		for(int k = 0; k != transform_size; ++k) {
			signal[k] = complex{static_cast<double>((k + i) % 17), static_cast<double>((k * 3 + i) % 11)};
		}
		signals.push_back(std::move(signal));
	}

	multi::fft_plan<1, complex> const plan(multi::extents_t<1>{transform_size}, multi::fft_forward);

	// --- one thread ------------------------------------------------------
	double single_thread_seconds = 0.0;
	{
		thread_arena arena(plan);

		auto const start = now();
		for(auto& signal : signals) {
			plan.execute(signal.home(), arena.allocator());
			arena.rewind();
		}
		single_thread_seconds = seconds_since(start);
	}
	std::printf("%2d thread : %7.3f s   %9.0f transforms/s\n",
	            1, single_thread_seconds, transform_count / single_thread_seconds);

	// --- several threads, same plan ---------------------------------------
	auto const hardware_threads = std::max(2U, std::thread::hardware_concurrency());
	for(unsigned threads = 2; threads <= hardware_threads; threads *= 2) {
		auto const start = now();
		{
			std::vector<std::thread> pool;
			pool.reserve(threads);
			for(unsigned t = 0; t != threads; ++t) {
				pool.emplace_back([&plan, &signals, transform_count, threads, t] {
					thread_arena arena(plan);  // private to this thread
					for(int i = static_cast<int>(t); i < transform_count; i += static_cast<int>(threads)) {
						plan.execute(signals[static_cast<std::size_t>(i)].home(), arena.allocator());
						arena.rewind();
					}
				});
			}
			for(auto& thread : pool) {
				thread.join();
			}
		}
		double const elapsed = seconds_since(start);
		std::printf("%2u threads: %7.3f s   %9.0f transforms/s   %4.2fx\n",
		            threads, elapsed, transform_count / elapsed, single_thread_seconds / elapsed);
	}

	// The same pattern slices an N-D array along an untouched axis: with
	// `{none, forward, forward}` on a (depth, n, n) array, every depth-slice
	// is independent, so a thread can take `plan2d.execute(arr[k].home(), ...)`
	// for its own share of `k` -- build the plan for the ACTIVE axes only
	// (here 2-D) and drive the untouched one yourself.

	return 0;
}
