// Copyright 2024-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// COMPILATION_INSTRUCTIONS:
//   g++ -std=c++17 -O3 -march=native -mtune=native -funroll-loops -fno-math-errno -DNDEBUG \
//     -I../include algorithms_fft.cpp -o algorithms_fft.x -lfftw3 \
//     && ./algorithms_fft.x && gnuplot algorithms_fft_plots.gp
//   (add -DDISABLE_WISDOM to build the wisdom-disabled control variant, which
//   writes to fft_bench_*_nowisdom.dat instead of fft_bench_*.dat; add
//   -DUSE_ESTIMATE as well to additionally use FFTW_ESTIMATE instead of
//   FFTW_MEASURE, writing to fft_bench_*_estimate.dat)
//
// -funroll-loops/-fno-math-errno are pure codegen flags with zero effect on
// numerical results -- deliberately NOT using -ffast-math (or any of its
// components): the fft_ops customization point exists specifically to get
// vectorized performance under strict IEEE semantics, and relaxing that here
// would make the comparison less representative of how the library actually
// ships.

// Size sweep of multi::fft_plan vs FFTW, emitting gnuplot-friendly .dat files
// (one per dimensionality).
//
// Methodology:
//   * Plan-recycled, steady-state EXECUTION ONLY: for both libraries, the plan
//     is built once, outside the timed region, and reused for every timed
//     repetition. Plan-build time is never included in either measurement.
//   * FFTW is allowed to use wisdom (accumulated across sizes within one run,
//     and persisted to a wisdom file across separate runs of this program) and
//     plans with FFTW_MEASURE -- since planning time isn't measured, FFTW is
//     given its best realistic shot at execution speed, not a crippled
//     FFTW_ESTIMATE guess.
//   * Sizes are exactly 2^a * 3^b * 5^c ("5-smooth"), the same family the
//     radix-2/3/4/5/8 kernels (fft.hpp) and the Bluestein convolution-length
//     search (fft.hpp, "cheapest 5-smooth candidate") are built around,
//     including every pure single-prime power (2^a, 3^b, 5^c) that fits in
//     each dimensionality's tested range, so every radix family is exercised
//     in isolation as well as mixed -- not just powers of two.
//   * Both untimed single-shot warm-up executions (absorbs first-call lazy
//     allocation/cache effects for BOTH libraries symmetrically -- an earlier
//     version of this file only warmed up multi::fft_plan, not FFTW) and the
//     timed measurements themselves are INTERLEAVED at the single-repetition
//     granularity (flush, time mine; flush, time FFTW; repeat) rather than
//     run as two separate blocked loops. This cancels out any linear drift
//     (thermal ramp, frequency-scaling transients, background load) that
//     would otherwise systematically favor whichever library happens to run
//     first or last.
//   * The process is pinned to a single CPU core (where supported) to reduce
//     core-migration variance, and a short fixed CPU warm-up runs before any
//     timed work to get past initial turbo-boost ramp-up.
//   * CPU cache flushed before every individual timed call; single-threaded.
//   * A lightweight timing-drift self-check (calibrate at start and end of the
//     sweep) warns on stderr if the machine looks like it throttled mid-run --
//     this is a supplement to, not a substitute for, only starting the sweep
//     on an idle, unthrottled machine.
//   * multi::fft_plan::execute() is passed a std::pmr::polymorphic_allocator
//     backed by a std::pmr::monotonic_buffer_resource over a persistent,
//     caller-owned arena -- one arena per plan, sized to
//     plan.scratch_elements() and allocated once outside the timed region,
//     with resource.release() called after every execute() (timed and
//     warm-up alike) to rewind the arena for reuse without freeing it. This
//     replaces an earlier std::pmr::unsynchronized_pool_resource-per-plan
//     mitigation that was believed to reclaim cleanly on deallocate() but,
//     per fft.NOTES.md §11.14/§11.15 (perf + strace measurement), does NOT:
//     a pool resource only pools blocks up to its internal size cap and
//     delegates anything larger straight to the upstream resource, so a
//     single large (tens-to-hundreds of MB) scratch request every call fell
//     through to a fresh mmap/munmap pair EVERY repetition, inflating the
//     largest sizes in every sweep below. The monotonic-arena replacement
//     measured a 2.03x wall-time reduction on the previously worst case
//     (1-D n=1,048,576, six-step) in the profiling harness; see §11.15 for
//     the mechanism (repeated mmap also means repeated first-touch page
//     faults, not just syscall cost). FFTW is unaffected -- it already
//     reuses its own plan-owned buffers.
#include <boost/multi/algorithms/fft.hpp>
#include <boost/multi/array.hpp>

#include <fftw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory_resource>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace multi = boost::multi;
using complex   = std::complex<double>;

namespace {

char const* const wisdom_filename = "algorithms_fft.wisdom";

struct watch {
	std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
	auto sec() const { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - s).count(); }
};

std::vector<char> g_thrash(64 << 20);  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
void               flush_cache() {
	for(std::size_t i = 0; i < g_thrash.size(); i += 64) { g_thrash[i]++; }
	char volatile x = g_thrash[0];
	(void)x;
}

// Pin the process to a single CPU core, reducing core-migration/cache-affinity
// variance between measurements. Best-effort: silently does nothing on
// non-Linux or if the affinity call fails.
void pin_to_one_cpu() {
#ifdef __linux__
	cpu_set_t allowed;
	CPU_ZERO(&allowed);
	if(sched_getaffinity(0, sizeof(allowed), &allowed) != 0) { return; }
	for(int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
		if(CPU_ISSET(cpu, &allowed)) {
			cpu_set_t one;
			CPU_ZERO(&one);
			CPU_SET(cpu, &one);
			(void)sched_setaffinity(0, sizeof(one), &one);
			return;
		}
	}
#endif
}

// The suite pins itself to one core (see pin_to_one_cpu) so that single-thread
// timings are not perturbed by core migration. A threading sweep obviously
// cannot live with that -- every thread would contend for the same core and
// measure a flat 1.0x -- so it restores the full mask for its own duration and
// puts the pin back afterwards, leaving the rest of the suite unaffected.
class scoped_all_cpus {
#ifdef __linux__
	cpu_set_t saved_{};
	bool      ok_{false};

 public:
	scoped_all_cpus() {
		CPU_ZERO(&saved_);
		ok_ = (sched_getaffinity(0, sizeof(saved_), &saved_) == 0);
		if(!ok_) { return; }
		cpu_set_t all;
		CPU_ZERO(&all);
		for(int cpu = 0; cpu != static_cast<int>(std::thread::hardware_concurrency()); ++cpu) { CPU_SET(cpu, &all); }
		(void)sched_setaffinity(0, sizeof(all), &all);
	}
	scoped_all_cpus(scoped_all_cpus const&)                    = delete;
	scoped_all_cpus(scoped_all_cpus&&)                         = delete;
	auto operator=(scoped_all_cpus const&) -> scoped_all_cpus& = delete;
	auto operator=(scoped_all_cpus&&) -> scoped_all_cpus&      = delete;
	~scoped_all_cpus() {
		if(ok_) { (void)sched_setaffinity(0, sizeof(saved_), &saved_); }
	}
#else
 public:
	scoped_all_cpus() = default;
#endif
};

// Busy loop of real floating-point work, run once before any timed
// measurement, so the CPU is past its initial ramp-up (and out of any
// idle/low-power state) before the sweep's first size -- otherwise the
// earliest sizes are measured from an artificially slow, still-ramping state
// that later sizes never see, and the closing calibration reads the
// difference back as "drift".
//
// TWO things had to be right, and for a long time only the duration was
// tuned: 0.25s measured 17% drift, 2s measured 16-21%, 8s still measured
// 31-60% on an *idle* machine. Duration was never the whole problem.
//
// The ramp is a PACKAGE-level effect. Under the `powersave` governor idle
// cores park (800 MHz here) and the package clock follows the activity of ALL
// of them -- so a warm-up that inherits the suite's one-core pin (main() calls
// pin_to_one_cpu() FIRST) heats one core while the rest stay parked, and the
// package never leaves its low-power state. Restoring the full mask and
// warming every core is what actually flattens the start: 120s on all cores
// took a run from 31-60% drift to -3.4% on this machine, where 8s on one core
// had not.
//
// The two minutes this costs are deliberate. Three runs in one session were
// discarded to the un-diagnosed version.
#ifndef BOOST_MULTI_FFT_BENCH_WARMUP_SECONDS
#define BOOST_MULTI_FFT_BENCH_WARMUP_SECONDS 120.0
#endif
void warm_up_cpu() {
	scoped_all_cpus const all_cores;  // must outlive `pool`: threads inherit the affinity mask at creation

	unsigned const           cores = std::max(1U, std::thread::hardware_concurrency());
	std::vector<std::thread> pool;
	pool.reserve(cores);
	for(unsigned t = 0; t != cores; ++t) {
		pool.emplace_back([] {
			watch           w;
			volatile double x = 1.0;
			while(w.sec() < BOOST_MULTI_FFT_BENCH_WARMUP_SECONDS) {
				for(int i = 0; i != 100000; ++i) { x = std::sin(x) + std::cos(x); }
			}
		});
	}
	for(auto& th : pool) { th.join(); }
}

template<class F> auto time_it(long reps, F f) -> double {
	double t = 0;
	for(long r = 0; r != reps; ++r) {
		flush_cache();
		watch w;
		f();
		t += w.sec();
	}
	return t / reps;
}

// Interleaved pairwise timing: alternates single timed calls to `f`/`g`.
// Each input is restored before its cache flush and outside the timed region;
// the reported times therefore contain only the in-place FFT execution.
// Interleaving still cancels drift from temperature, frequency scaling, and
// background load.
template<class PrepareF, class F, class PrepareG, class G>
auto time_it_interleaved(long reps, PrepareF prepare_f, F f, PrepareG prepare_g, G g) -> std::pair<double, double> {
	double tf = 0;
	double tg = 0;
	for(long r = 0; r != reps; ++r) {
		prepare_f();
		flush_cache();
		{
			watch w;
			f();
			tf += w.sec();
		}
		prepare_g();
		flush_cache();
		{
			watch w;
			g();
			tg += w.sec();
		}
	}
	return {tf / reps, tg / reps};
}

// Persistent scratch arena for one plan: a std::pmr::monotonic_buffer_resource
// over a caller-owned buffer sized to the plan's scratch_elements(), reused
// across every execute() call via release() (rewinds the arena, does not
// free it) -- see the file-header comment and fft.NOTES.md §11.14/§11.15.
template<class Plan>
struct arena_alloc {
	std::vector<std::byte>                   buf;
	std::pmr::monotonic_buffer_resource      mbr;
	std::pmr::polymorphic_allocator<complex> alloc;

	explicit arena_alloc(Plan const& plan)
	: buf(plan.scratch_elements() * sizeof(complex) + 4096), mbr(buf.data(), buf.size()), alloc(&mbr) {}

	void reset() { mbr.release(); }
};

auto reps_for(double n_total) -> long {
	double const work = 5.0 * n_total * std::log2(std::max(n_total, 2.0));
	return std::clamp<long>(static_cast<long>(2e8 / work), 5, 300);
}

// Fixed-cost reference computation used to sanity-check that the machine
// didn't slow down (thermal throttling, background load) partway through the
// sweep: timed once before and once after; a large drift is reported, not
// silently absorbed into the results.
auto calibrate() -> double {
	multi::array<complex, 1>          a(multi::extents_t<1>{16384}, complex{1.0, 0.0});
	multi::fft_plan<1, complex> const plan{multi::extents_t<1>{16384}, multi::fft_forward};
	arena_alloc<decltype(plan)>        arena(plan);
	plan.execute(a.home(), arena.alloc);
	arena.reset();
	return time_it(200, [&] { plan.execute(a.home(), arena.alloc); arena.reset(); });
}

template<std::ptrdiff_t D>
void sweep(std::vector<int> const& sides, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s: in-place multi::fft_plan vs in-place FFTW 3 (plan recycled; input setup and plan-build excluded; interleaved cold-cache timing)\n", label);
	#if defined(BOOST_MULTI_FFT_DISABLE_PACK_CONTIGUOUS_BATCHES)
	std::fprintf(out, "# Multi schedule: direct contiguous fibers (packing disabled)\n");
	#else
	std::fprintf(out, "# Multi schedule: selective packed contiguous batches (n >= 48)\n");
	#endif
	std::fprintf(out, "# multi::fft_plan::execute() uses a std::pmr::monotonic_buffer_resource over a persistent arena, released (not freed) after every call\n");
#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
	std::fprintf(out, "# FFTW: FFTW_ESTIMATE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#elif defined(DISABLE_WISDOM)
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#else
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom allowed (accumulated within and across runs via %s)\n", wisdom_filename);
#endif
	std::fprintf(out, "# sizes are 2^a * 3^b * 5^c, including all pure single-prime powers in range\n");
	std::fprintf(out, "# mflops = 5*N*log2(N)/time_us (benchFFT convention), N = total points\n");
	std::fprintf(out, "# n_side  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long N = 1;
		for(int d = 0; d != D; ++d) { N *= n; }
		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }
		long const reps = reps_for(static_cast<double>(N));

		multi::array<complex, 1> flat(multi::extents_t<1>{N});
		auto                      load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		std::array<multi::ssize_t, D> ext_arr{};
		ext_arr.fill(n);
		multi::fft_plan<D, complex> const plan{ext_arr, multi::fft_forward};  // plan build: not timed

		// One persistent arena per plan, built once and reused (via release())
		// across the warm-up and every timed repetition below -- see the
		// file-header comment and fft.NOTES.md §11.14/§11.15.
		arena_alloc<decltype(plan)> arena(plan);

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
            for(long i = 0; i != N; ++i) {
                in[i][0] = base[static_cast<std::size_t>(i)].real();
                in[i][1] = base[static_cast<std::size_t>(i)].imag();
            }
		};
		loadf();
#ifdef DISABLE_WISDOM
		fftw_forget_wisdom();  // force a cold search/estimate for every size
#endif
#ifdef USE_ESTIMATE
		unsigned const fftw_flag = FFTW_ESTIMATE;
#else
		unsigned const fftw_flag = FFTW_MEASURE;
#endif
		fftw_plan p =  // plan build: not timed; input and output are the same buffer
			D == 1 ? fftw_plan_dft_1d(n, in, in, FFTW_FORWARD, fftw_flag)
			: D == 2 ? fftw_plan_dft_2d(n, n, in, in, FFTW_FORWARD, fftw_flag)
			        : fftw_plan_dft_3d(n, n, n, in, in, FFTW_FORWARD, fftw_flag);
		loadf();  // FFTW_MEASURE overwrites in/out while searching strategies; reload before timing (harmless no-op difference for ESTIMATE)

		double mine = 0.0;
		double ffw  = 0.0;
		if constexpr(D == 1) {
			load();
			plan.execute(flat.home(), arena.alloc);  // untimed warm-up, symmetric with FFTW's below
			arena.reset();
			loadf();
			fftw_execute(p);  // untimed warm-up, symmetric with multi's above
			std::tie(mine, ffw) = time_it_interleaved(
				reps, load, [&] { plan.execute(flat.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });
		} else if constexpr(D == 2) {
			multi::array_ref<complex, 2> v(flat.data_elements(), {n, n});
			load();
			plan.execute(v.home(), arena.alloc);
			arena.reset();
			loadf();
			fftw_execute(p);
			std::tie(mine, ffw) = time_it_interleaved(
				reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });
		} else {
			multi::array_ref<complex, 3> v(flat.data_elements(), {n, n, n});
			load();
			plan.execute(v.home(), arena.alloc);
			arena.reset();
			loadf();
			fftw_execute(p);
			std::tie(mine, ffw) = time_it_interleaved(
				reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });
		}

		fftw_destroy_plan(p);
		fftw_free(in);

		double const work = 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}

// Batched 1-D: `howmany` row-fibers of length `n`, contiguous-row layout
// (row stride == n), against FFTW's advanced (many) interface -- the
// specialized-batching adversary to multi::fft_plan's general per-axis
// direction feature (fft.NOTES.md §10), since {none, forward} on a rank-2
// array IS the "many" interface here (batch axis untouched, fiber axis
// transformed), not a separate code path. This isolates the slab
// gather/scatter machinery (fft_exec_slab's run_fused tile writes when the
// batch axis is contiguous) rather than the single-fiber path the D==1
// sweep above already covers.
void sweep_many(std::vector<int> const& sides, int howmany, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s (howmany=%d): in-place multi::fft_plan{none,forward} vs in-place fftw_plan_many_dft (plan recycled; input setup and plan-build excluded; interleaved cold-cache timing)\n", label, howmany);
	#if defined(BOOST_MULTI_FFT_EXPERIMENT_PACK_CONTIGUOUS_BATCHES)
	std::fprintf(out, "# Multi schedule: experimental packed contiguous batches\n");
	#else
	std::fprintf(out, "# Multi schedule: direct contiguous fibers\n");
	#endif
	std::fprintf(out, "# multi::fft_plan::execute() uses a std::pmr::monotonic_buffer_resource over a persistent arena, released (not freed) after every call\n");
#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
	std::fprintf(out, "# FFTW: FFTW_ESTIMATE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#elif defined(DISABLE_WISDOM)
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#else
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom allowed (accumulated within and across runs via %s)\n", wisdom_filename);
#endif
	std::fprintf(out, "# row layout: %d contiguous fibers of length n (row stride == n, batch axis untouched)\n", howmany);
	std::fprintf(out, "# mflops = 5*howmany*n*log2(n)/time_us (batched benchFFT convention)\n");
	std::fprintf(out, "# n  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long const N = static_cast<long>(n) * howmany;

		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }
		long const reps = reps_for(static_cast<double>(howmany) * static_cast<double>(n));

		multi::array<complex, 1>     flat(multi::extents_t<1>{N});
		multi::array_ref<complex, 2> v(flat.data_elements(), {howmany, n});  // row-major: fiber (axis 1) contiguous, batch (axis 0) untouched
		auto                          load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		multi::fft_plan<2, complex> const plan{
			v.sizes(),
			{{multi::fft_direction::none, multi::fft_direction::forward}}
		};  // plan build: not timed

		arena_alloc<decltype(plan)> arena(plan);

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
            for(long i = 0; i != N; ++i) {
                in[i][0] = base[static_cast<std::size_t>(i)].real();
                in[i][1] = base[static_cast<std::size_t>(i)].imag();
            }
		};
		loadf();
#ifdef DISABLE_WISDOM
		fftw_forget_wisdom();  // force a cold search/estimate for every size
#endif
#ifdef USE_ESTIMATE
		unsigned const fftw_flag = FFTW_ESTIMATE;
#else
		unsigned const fftw_flag = FFTW_MEASURE;
#endif
		int        fftw_n[1] = {n};
		fftw_plan p =  // plan build: not timed; in-place, contiguous rows: istride=ostride=1, idist=odist=n
			fftw_plan_many_dft(1, fftw_n, howmany, in, nullptr, 1, n, in, nullptr, 1, n, FFTW_FORWARD, fftw_flag);
		loadf();  // FFTW_MEASURE overwrites in/out while searching strategies; reload before timing

		load();
		plan.execute(v.home(), arena.alloc);  // untimed warm-up, symmetric with FFTW's below
		arena.reset();
		loadf();
		fftw_execute(p);  // untimed warm-up, symmetric with multi's above
		auto [mine, ffw] = time_it_interleaved(
			reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });

		fftw_destroy_plan(p);
		fftw_free(in);

		double const work = 5.0 * static_cast<double>(howmany) * static_cast<double>(n) * std::log2(static_cast<double>(n));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}

// Batched, STRIDED many: same (n, howmany) total data as sweep_many, but the
// axes are swapped so the TRANSFORMED axis (n) is non-contiguous (stride
// howmany) and the UNTOUCHED axis (howmany) is contiguous -- {forward, none}
// on a (n, howmany) row-major array, instead of sweep_many's {none, forward}
// on (howmany, n). This isolates Multi's genuinely-batched fused execution
// path (fft_exec_slab's batch-axis-contiguous branch, m = min(mb_, howmany),
// no gather/scatter) from the m=1-per-fiber path sweep_many actually
// exercises (its {none, forward} on (howmany, n) has the transformed axis
// LAST/contiguous, which always routes through the per-fiber fast path --
// confirmed separately that this gives ~1.0x internal batch benefit, i.e.
// no batching happens at all in sweep_many's own numbers). FFTW's matching
// call is the "strided batch" advanced-interface pattern: istride=idist
// swapped from sweep_many's contiguous-row pattern (istride=1,idist=n) to
// istride=howmany,idist=1, since element (k,j) of the (n,howmany) array is
// at offset k*howmany+j: fixed j, varying k (one length-n transform) steps
// by howmany; successive j (which transform) steps by 1.
void sweep_many_strided(std::vector<int> const& sides, int howmany, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s (howmany=%d): in-place multi::fft_plan{forward,none} vs in-place fftw_plan_many_dft strided (plan recycled; input setup and plan-build excluded; interleaved cold-cache timing)\n", label, howmany);
	#if defined(BOOST_MULTI_FFT_EXPERIMENT_PACK_CONTIGUOUS_BATCHES)
	std::fprintf(out, "# Multi schedule: experimental packed contiguous batches\n");
	#else
	std::fprintf(out, "# Multi schedule: direct contiguous fibers\n");
	#endif
	std::fprintf(out, "# multi::fft_plan::execute() uses a std::pmr::monotonic_buffer_resource over a persistent arena, released (not freed) after every call\n");
#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
	std::fprintf(out, "# FFTW: FFTW_ESTIMATE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#elif defined(DISABLE_WISDOM)
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#else
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom allowed (accumulated within and across runs via %s)\n", wisdom_filename);
#endif
	std::fprintf(out, "# column layout: %d contiguous columns, transformed axis (n) has stride=howmany\n", howmany);
	std::fprintf(out, "# mflops = 5*howmany*n*log2(n)/time_us (batched benchFFT convention)\n");
	std::fprintf(out, "# n  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long const N = static_cast<long>(n) * howmany;

		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }
		long const reps = reps_for(static_cast<double>(howmany) * static_cast<double>(n));

		multi::array<complex, 1>     flat(multi::extents_t<1>{N});
		multi::array_ref<complex, 2> v(flat.data_elements(), {n, howmany});  // row-major: transformed axis (0) has stride howmany, batch axis (1) contiguous
		auto                          load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		multi::fft_plan<2, complex> const plan{
			v.sizes(),
			{{multi::fft_direction::forward, multi::fft_direction::none}}
		};  // plan build: not timed

		arena_alloc<decltype(plan)> arena(plan);

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
            for(long i = 0; i != N; ++i) {
                in[i][0] = base[static_cast<std::size_t>(i)].real();
                in[i][1] = base[static_cast<std::size_t>(i)].imag();
            }
		};
		loadf();
#ifdef DISABLE_WISDOM
		fftw_forget_wisdom();  // force a cold search/estimate for every size
#endif
#ifdef USE_ESTIMATE
		unsigned const fftw_flag = FFTW_ESTIMATE;
#else
		unsigned const fftw_flag = FFTW_MEASURE;
#endif
		int        fftw_n[1] = {n};
		fftw_plan p =  // plan build: not timed; in-place, strided batch: istride=ostride=howmany, idist=odist=1
			fftw_plan_many_dft(1, fftw_n, howmany, in, nullptr, howmany, 1, in, nullptr, howmany, 1, FFTW_FORWARD, fftw_flag);
		loadf();  // FFTW_MEASURE overwrites in/out while searching strategies; reload before timing

		load();
		plan.execute(v.home(), arena.alloc);  // untimed warm-up, symmetric with FFTW's below
		arena.reset();
		loadf();
		fftw_execute(p);  // untimed warm-up, symmetric with multi's above
		auto [mine, ffw] = time_it_interleaved(
			reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });

		fftw_destroy_plan(p);
		fftw_free(in);

		double const work = 5.0 * static_cast<double>(howmany) * static_cast<double>(n) * std::log2(static_cast<double>(n));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}

// Batched 2-D: `depth` layers of an n x n 2-D FFT, {none, forward, forward}
// on a (depth, n, n) row-major array (batch axis 0 untouched, both trailing
// axes transformed) -- against FFTW's rank-2 advanced (many) interface. Each
// depth-layer is contiguous (n*n elements), layers spaced by n*n (idist ==
// odist == n*n, istride == ostride == 1, no embedding padding), matching the
// plan's row-major layout exactly.
void sweep_many3d(std::vector<int> const& sides, int depth, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s (depth=%d): in-place multi::fft_plan{none,forward,forward} vs in-place fftw_plan_many_dft rank=2 (plan recycled; input setup and plan-build excluded; interleaved cold-cache timing)\n", label, depth);
	#if defined(BOOST_MULTI_FFT_EXPERIMENT_PACK_CONTIGUOUS_BATCHES)
	std::fprintf(out, "# Multi schedule: experimental packed contiguous batches\n");
	#else
	std::fprintf(out, "# Multi schedule: direct contiguous fibers\n");
	#endif
	std::fprintf(out, "# multi::fft_plan::execute() uses a std::pmr::monotonic_buffer_resource over a persistent arena, released (not freed) after every call\n");
#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
	std::fprintf(out, "# FFTW: FFTW_ESTIMATE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#elif defined(DISABLE_WISDOM)
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#else
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom allowed (accumulated within and across runs via %s)\n", wisdom_filename);
#endif
	std::fprintf(out, "# layout: %d contiguous n x n layers (layer stride == n*n, batch axis 0 untouched)\n", depth);
	std::fprintf(out, "# mflops = 5*depth*n*n*log2(n*n)/time_us (batched benchFFT convention)\n");
	std::fprintf(out, "# n  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long const nn = static_cast<long>(n) * n;
		long const N  = nn * depth;

		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }
		long const reps = reps_for(static_cast<double>(depth) * static_cast<double>(nn));

		multi::array<complex, 1>     flat(multi::extents_t<1>{N});
		multi::array_ref<complex, 3> v(flat.data_elements(), {depth, n, n});  // row-major: layers (axis 0) untouched, each n x n layer (axes 1,2) contiguous
		auto                          load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		multi::fft_plan<3, complex> const plan{
			v.sizes(),
			std::array<multi::fft_direction, 3>{{multi::fft_direction::none, multi::fft_direction::forward, multi::fft_direction::forward}}
		};  // plan build: not timed

		arena_alloc<decltype(plan)> arena(plan);

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
            for(long i = 0; i != N; ++i) {
                in[i][0] = base[static_cast<std::size_t>(i)].real();
                in[i][1] = base[static_cast<std::size_t>(i)].imag();
            }
		};
		loadf();
#ifdef DISABLE_WISDOM
		fftw_forget_wisdom();  // force a cold search/estimate for every size
#endif
#ifdef USE_ESTIMATE
		unsigned const fftw_flag = FFTW_ESTIMATE;
#else
		unsigned const fftw_flag = FFTW_MEASURE;
#endif
		int        fftw_n[2] = {n, n};
		fftw_plan p =  // plan build: not timed; in-place, contiguous layers: istride=ostride=1, idist=odist=n*n, no embedding
			fftw_plan_many_dft(2, fftw_n, depth, in, nullptr, 1, n * n, in, nullptr, 1, n * n, FFTW_FORWARD, fftw_flag);
		loadf();  // FFTW_MEASURE overwrites in/out while searching strategies; reload before timing

		load();
		plan.execute(v.home(), arena.alloc);  // untimed warm-up, symmetric with FFTW's below
		arena.reset();
		loadf();
		fftw_execute(p);  // untimed warm-up, symmetric with multi's above
		auto [mine, ffw] = time_it_interleaved(
			reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });

		fftw_destroy_plan(p);
		fftw_free(in);

		double const work = 5.0 * static_cast<double>(depth) * static_cast<double>(nn) * std::log2(static_cast<double>(nn));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}

// Batched 2-D with a GAP axis: {forward, none, forward} on a (n, depth, n)
// row-major array -- the two transformed axes (0 and 2) are NOT adjacent
// (depth sits between them), unlike sweep_many3d's {none,forward,forward}
// (adjacent trailing pair, hits the fused-pair fast path) or
// sweep_many_strided's {forward,none} (single active axis). Since
// dirs_[D-1]=dirs_[2]=forward but dirs_[D-2]=dirs_[1]=none, apply_()'s
// fused-pair guard (needs BOTH non-none) fails -- axis 2 is processed via
// the plain per-fiber (m=1, contiguous) path, then axis 0 via a once-rotated
// view whose recursion (fft_apply_last's mid-stride-driven transposed()
// choice) can still land the batch on axis 2's stride-1 extent, same
// fused-contiguous-batch mechanism sweep_many_strided isolates -- but with
// an extra outer `depth` loop multiplying call count for BOTH passes. FFTW
// comparison needs the guru interface (the two transformed dims are not a
// simple contiguous embedding): dims = {n, stride=depth*n} (axis 0) and
// {n, stride=1} (axis 2); howmany_dims = {depth, stride=n} (axis 1, the gap).
void sweep_gap3d(std::vector<int> const& sides, int depth, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s (depth=%d): in-place multi::fft_plan{forward,none,forward} vs in-place fftw_plan_guru_dft (plan recycled; input setup and plan-build excluded; interleaved cold-cache timing)\n", label, depth);
	#if defined(BOOST_MULTI_FFT_EXPERIMENT_PACK_CONTIGUOUS_BATCHES)
	std::fprintf(out, "# Multi schedule: experimental packed contiguous batches\n");
	#else
	std::fprintf(out, "# Multi schedule: direct contiguous fibers\n");
	#endif
	std::fprintf(out, "# multi::fft_plan::execute() uses a std::pmr::monotonic_buffer_resource over a persistent arena, released (not freed) after every call\n");
#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
	std::fprintf(out, "# FFTW: FFTW_ESTIMATE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#elif defined(DISABLE_WISDOM)
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#else
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom allowed (accumulated within and across runs via %s)\n", wisdom_filename);
#endif
	std::fprintf(out, "# layout: (n, depth=%d, n), transformed axes 0 and 2 separated by untouched axis 1\n", depth);
	std::fprintf(out, "# mflops = 5*depth*n*n*log2(n*n)/time_us (batched benchFFT convention)\n");
	std::fprintf(out, "# n  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long const nn = static_cast<long>(n) * n;
		long const N  = nn * depth;

		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }
		long const reps = reps_for(static_cast<double>(depth) * static_cast<double>(nn));

		multi::array<complex, 1>     flat(multi::extents_t<1>{N});
		multi::array_ref<complex, 3> v(flat.data_elements(), {n, depth, n});  // row-major: axis 0 stride depth*n, axis 1 (gap) stride n, axis 2 stride 1
		auto                          load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		multi::fft_plan<3, complex> const plan{
			v.sizes(),
			std::array<multi::fft_direction, 3>{{multi::fft_direction::forward, multi::fft_direction::none, multi::fft_direction::forward}}
		};  // plan build: not timed

		arena_alloc<decltype(plan)> arena(plan);

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
            for(long i = 0; i != N; ++i) {
                in[i][0] = base[static_cast<std::size_t>(i)].real();
                in[i][1] = base[static_cast<std::size_t>(i)].imag();
            }
		};
		loadf();
#ifdef DISABLE_WISDOM
		fftw_forget_wisdom();  // force a cold search/estimate for every size
#endif
#ifdef USE_ESTIMATE
		unsigned const fftw_flag = FFTW_ESTIMATE;
#else
		unsigned const fftw_flag = FFTW_MEASURE;
#endif
		fftw_iodim dims[2] = {
			{n, static_cast<int>(depth * n), static_cast<int>(depth * n)},  // axis 0: extent n, stride depth*n
			{n, 1, 1}                                                       // axis 2: extent n, stride 1
		};
		fftw_iodim howmany_dims[1] = {
			{depth, n, n}  // axis 1 (the gap): extent depth, stride n
		};
		fftw_plan p =  // plan build: not timed; in-place, guru interface for the non-adjacent transformed pair
			fftw_plan_guru_dft(2, dims, 1, howmany_dims, in, in, FFTW_FORWARD, fftw_flag);
		loadf();  // FFTW_MEASURE overwrites in/out while searching strategies; reload before timing

		load();
		plan.execute(v.home(), arena.alloc);  // untimed warm-up, symmetric with FFTW's below
		arena.reset();
		loadf();
		fftw_execute(p);  // untimed warm-up, symmetric with multi's above
		auto [mine, ffw] = time_it_interleaved(
			reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });

		fftw_destroy_plan(p);
		fftw_free(in);

		double const work = 5.0 * static_cast<double>(depth) * static_cast<double>(nn) * std::log2(static_cast<double>(nn));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}

// Batched 3-D ("many4d"): (depth, n, n, n) array, {none,forward,forward,forward}
// -- the 4-D analog of sweep_many3d, one untouched leading axis (contiguous
// blocks of size n^3) plus a full 3-D transform per block. FFTW comparison:
// fftw_plan_many_dft rank=3, howmany=depth, contiguous n^3 blocks
// (istride=ostride=1, idist=odist=n^3).
void sweep_many4d(std::vector<int> const& sides, int depth, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s (depth=%d): in-place multi::fft_plan{none,forward,forward,forward} vs in-place fftw_plan_many_dft rank=3 (plan recycled; input setup and plan-build excluded; interleaved cold-cache timing)\n", label, depth);
	std::fprintf(out, "# multi::fft_plan::execute() uses a std::pmr::monotonic_buffer_resource over a persistent arena, released (not freed) after every call\n");
#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
	std::fprintf(out, "# FFTW: FFTW_ESTIMATE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#elif defined(DISABLE_WISDOM)
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom DISABLED (fftw_forget_wisdom() before every plan)\n");
#else
	std::fprintf(out, "# FFTW: FFTW_MEASURE, wisdom allowed (accumulated within and across runs via %s)\n", wisdom_filename);
#endif
	std::fprintf(out, "# layout: %d contiguous n x n x n blocks (block stride == n^3, batch axis 0 untouched)\n", depth);
	std::fprintf(out, "# mflops = 5*depth*n^3*log2(n^3)/time_us (batched benchFFT convention)\n");
	std::fprintf(out, "# n  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long const nnn = static_cast<long>(n) * n * n;
		long const N   = nnn * depth;

		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }
		long const reps = reps_for(static_cast<double>(depth) * static_cast<double>(nnn));

		multi::array<complex, 1>     flat(multi::extents_t<1>{N});
		multi::array_ref<complex, 4> v(flat.data_elements(), {depth, n, n, n});
		auto                          load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		multi::fft_plan<4, complex> const plan{
			v.sizes(),
			std::array<multi::fft_direction, 4>{{multi::fft_direction::none, multi::fft_direction::forward, multi::fft_direction::forward, multi::fft_direction::forward}}
		};  // plan build: not timed

		arena_alloc<decltype(plan)> arena(plan);

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
            for(long i = 0; i != N; ++i) {
                in[i][0] = base[static_cast<std::size_t>(i)].real();
                in[i][1] = base[static_cast<std::size_t>(i)].imag();
            }
		};
		loadf();
#ifdef DISABLE_WISDOM
		fftw_forget_wisdom();
#endif
#ifdef USE_ESTIMATE
		unsigned const fftw_flag = FFTW_ESTIMATE;
#else
		unsigned const fftw_flag = FFTW_MEASURE;
#endif
		int       fftw_n[3] = {n, n, n};
		fftw_plan p =  // plan build: not timed; in-place, contiguous blocks: istride=ostride=1, idist=odist=n^3
			fftw_plan_many_dft(3, fftw_n, depth, in, nullptr, 1, nnn, in, nullptr, 1, nnn, FFTW_FORWARD, fftw_flag);
		loadf();

		load();
		plan.execute(v.home(), arena.alloc);
		arena.reset();
		loadf();
		fftw_execute(p);
		auto [mine, ffw] = time_it_interleaved(
			reps, load, [&] { plan.execute(v.home(), arena.alloc); arena.reset(); }, loadf, [&] { fftw_execute(p); });

		fftw_destroy_plan(p);
		fftw_free(in);

		double const work = 5.0 * static_cast<double>(depth) * static_cast<double>(nnn) * std::log2(static_cast<double>(nnn));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}



// Threading: one shared `const` plan, one scratch arena per thread, and the
// batch of independent transforms split across threads. This is a CALLER-side
// pattern -- fft.hpp itself is single-threaded and unchanged -- so what it
// measures is whether the plan's "owns no scratch, hence shareable" design
// actually delivers parallel speedup. FFTW is given its own native threading
// (fftw_plan_with_nthreads on one batched call) as the reference, which is
// how an FFTW user would parallelise the same workload.
void sweep_parallel(std::vector<int> const& sides, int howmany, char const* fname, char const* label) {
	scoped_all_cpus const all_cpus;  // undo the suite's single-core pin for this sweep only

	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s: %d independent 1-D transforms, split across threads\n", label, howmany);
	std::fprintf(out, "# NOTE: this sweep restores full CPU affinity; the rest of the suite runs pinned to one core\n");
	std::fprintf(out, "# Multi: ONE shared const fft_plan + one pmr monotonic arena per thread (fft.hpp is not itself threaded)\n");
	std::fprintf(out, "# FFTW:  fftw_plan_with_nthreads on the equivalent batched fftw_plan_many_dft\n");
	std::fprintf(out, "# plan build excluded from timing; cache flushed before every timed call\n");
	std::fprintf(out, "# n  threads  mine_ms  fftw_ms  mine_speedup  fftw_speedup  ratio_mine_over_fftw\n");

	auto const max_threads = std::max(2U, std::thread::hardware_concurrency());

	for(int n : sides) {
		std::vector<complex>                   base(static_cast<std::size_t>(n) * static_cast<std::size_t>(howmany));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }

		std::vector<multi::array<complex, 1>> sigs;
		sigs.reserve(static_cast<std::size_t>(howmany));
		for(int i = 0; i != howmany; ++i) {
			multi::array<complex, 1> s(multi::extents_t<1>{n}, complex{});
			std::copy(base.begin() + static_cast<std::ptrdiff_t>(i) * n, base.begin() + static_cast<std::ptrdiff_t>(i + 1) * n, s.begin());
			sigs.push_back(std::move(s));
		}

		multi::fft_plan<1, complex> const plan{multi::extents_t<1>{n}, multi::fft_forward};
		long const reps = std::clamp<long>(static_cast<long>(2e8 / (5.0 * n * howmany * std::log2(std::max(n, 2)))), 3, 50);

		auto run_multi = [&](unsigned threads) {
			std::vector<std::thread> pool;
			pool.reserve(threads);
			for(unsigned t = 0; t != threads; ++t) {
				pool.emplace_back([&, t] {
					arena_alloc<decltype(plan)> arena(plan);
					for(int i = static_cast<int>(t); i < howmany; i += static_cast<int>(threads)) {
						plan.execute(sigs[static_cast<std::size_t>(i)].home(), arena.alloc);
						arena.reset();
					}
				});
			}
			for(auto& th : pool) { th.join(); }
		};

		auto* in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * base.size()));
		auto  loadf = [&] {
			for(std::size_t i = 0; i != base.size(); ++i) { in[i][0] = base[i].real(); in[i][1] = base[i].imag(); }
		};

		double mine_1 = 0.0;
		double ffw_1  = 0.0;
		for(unsigned threads = 1; threads <= max_threads; threads *= 2) {
			double const mine = time_it(reps, [&] { run_multi(threads); });

			fftw_plan_with_nthreads(static_cast<int>(threads));
			loadf();
#ifdef DISABLE_WISDOM
			fftw_forget_wisdom();
#endif
#ifdef USE_ESTIMATE
			unsigned const fftw_flag = FFTW_ESTIMATE;
#else
			unsigned const fftw_flag = FFTW_MEASURE;
#endif
			int       nn[] = {n};
			fftw_plan p    = fftw_plan_many_dft(1, nn, howmany, in, nullptr, 1, n, in, nullptr, 1, n, FFTW_FORWARD, fftw_flag);
			double const ffw = time_it(reps, [&] { fftw_execute(p); });
			fftw_destroy_plan(p);

			if(threads == 1) { mine_1 = mine; ffw_1 = ffw; }
			std::fprintf(out, "%8d %8u %12.5f %12.5f %12.3f %12.3f %8.3f\n",
			             n, threads, mine * 1e3, ffw * 1e3, mine_1 / mine, ffw_1 / ffw, mine / ffw);
			std::fflush(out);
			std::fprintf(stderr, "%s n=%d threads=%u done\n", label, n, threads);
		}
		fftw_free(in);
	}
	std::fclose(out);
}

// Multi-only phase timing for the 3-D schedules.  This separates the active
// axis passes from the combined schedules without introducing a second FFTW
// guru-plan comparison; the existing sweeps provide that comparison.
void sweep_3d_phases(std::vector<int> const& sides, int depth, char const* fname) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# 3-D phase timings, Multi only, cold-cache execution (depth=%d)\n", depth);
	std::fprintf(out, "# n  axis0_ms  axis1_ms  axis2_ms  trailing_pair_ms  gap_pair_ms\n");
	for(int n : sides) {
		long const N = static_cast<long>(n) * n * depth;
		multi::array<complex, 1> flat(multi::extents_t<1>{N});
		std::mt19937 gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : flat) { e = complex{dist(gen), dist(gen)}; }
		multi::array_ref<complex, 3> v(flat.data_elements(), {n, depth, n});
		std::array<std::array<multi::fft_direction, 3>, 5> const dirs{{
			{{multi::fft_direction::forward, multi::fft_direction::none, multi::fft_direction::none}},
			{{multi::fft_direction::none, multi::fft_direction::forward, multi::fft_direction::none}},
			{{multi::fft_direction::none, multi::fft_direction::none, multi::fft_direction::forward}},
			{{multi::fft_direction::none, multi::fft_direction::forward, multi::fft_direction::forward}},
			{{multi::fft_direction::forward, multi::fft_direction::none, multi::fft_direction::forward}}
		}};
		std::array<double, 5> times{};
		for(std::size_t p = 0; p != dirs.size(); ++p) {
			multi::fft_plan<3, complex> plan{v.sizes(), dirs[p]};
			arena_alloc<decltype(plan)> arena(plan);
			times[p] = time_it(reps_for(static_cast<double>(N)), [&] { plan.execute(v.home(), arena.alloc); arena.reset(); });
		}
		std::fprintf(out, "%8d %12.5f %12.5f %12.5f %16.5f %12.5f\n", n, times[0]*1e3, times[1]*1e3, times[2]*1e3, times[3]*1e3, times[4]*1e3);
	}
	std::fclose(out);
}

}  // namespace

auto main() -> int {
	pin_to_one_cpu();
	warm_up_cpu();

#ifndef DISABLE_WISDOM
	fftw_import_wisdom_from_filename(wisdom_filename);  // ignore failure: no prior wisdom is fine
#endif

	fftw_init_threads();  // sweep_parallel compares against FFTW's own threading

	double const calib_before = calibrate();
	std::fprintf(stderr, "calibration (before): %.4f ms\n", calib_before * 1e3);

#if defined(DISABLE_WISDOM) && defined(USE_ESTIMATE)
#define BOOST_MULTI_FFT_BENCH_SUFFIX "_estimate"
#elif defined(DISABLE_WISDOM)
#define BOOST_MULTI_FFT_BENCH_SUFFIX "_nowisdom"
#else
#define BOOST_MULTI_FFT_BENCH_SUFFIX ""
#endif

#if defined(BOOST_MULTI_FFT_EXPERIMENT_PACK_CONTIGUOUS_BATCHES)
#define BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX "_packed"
#else
#define BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ""
#endif

	// Sizes are exactly 2^a * 3^b * 5^c: every pure power of 2, 3, and 5 that
	// fits in the tested range, plus mixed 2-/3-way composites at several
	// magnitudes, so every radix-2/3/4/5/8 combination is exercised both in
	// isolation and mixed.
	sweep<1>({53, 97, 125, 128, 144, 180, 200, 243, 256, 512, 625, 729, 1024,
	          1080, 1296, 1600, 2048, 2187, 3125, 4096, 6561, 8192,
	          15625, 16384, 19683, 20250, 24000, 27000, 32768, 59049, 65536,
	          78125, 131072, 172800, 177147, 230400, 250000, 262144, 390625,
	          524288, 531441, 1048576, 1259712, 1594323, 1600000, 1953125, 2097152},
	         "fft_bench_1d" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "1D n");
	// 2D: every pure power of 2/3/5 that fits in [24,2000]
	// (32,64,128,256,512,1024 / 27,81,243,729 / 25,125,625) plus mixed
	// composites, for a denser view of the radix-2/3/4/5/8 kernel space.
	sweep<2>({24, 25, 27, 32, 40, 60, 64, 75, 81, 100, 125, 128,
	          216, 243, 250, 256, 320, 375, 405, 486, 512, 625, 729,
	          1024, 1215, 1350, 1600, 2000},
	         "fft_bench_2d" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "2D n x n");
	// 3D: every pure power of 2/3/5 that fits in [8,300]
	// (16,32,128,256 / 9,81,243 / already had 25,125) plus mixed composites.
	sweep<3>({8, 9, 15, 16, 20, 25, 27, 32, 64, 81, 90, 100,
	          125, 128, 144, 216, 243, 250, 256, 300},
	         "fft_bench_3d" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "3D n x n x n");

	// Batched 1-D ("many"): fiber sizes spanning the radix-2/3/4/5/8 kernel
	// families, at two batch depths, against fftw_plan_many_dft -- see
	// sweep_many's comment. Two depths (moderate and deep) so the plot shows
	// whether the ratio is stable as the batch axis grows, not just a single
	// point.
	sweep_many({32, 64, 81, 100, 125, 128, 243, 256, 512, 625, 729, 1024, 2048, 4096},
	           32, "fft_bench_many_h32" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "many n (howmany=32)");
	sweep_many({32, 64, 81, 100, 125, 128, 243, 256, 512, 625, 729, 1024, 2048, 4096},
	           256, "fft_bench_many_h256" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "many n (howmany=256)");

	// Batched, STRIDED many -- isolates the genuinely-batched fused path
	// (see sweep_many_strided's comment); sweep_many above does NOT exercise
	// this path (its transformed axis is contiguous/last, always m=1).
	sweep_many_strided({32, 64, 81, 100, 125, 128, 243, 256, 512, 625, 729, 1024, 2048, 4096},
	                    32, "fft_bench_many_strided_h32" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "many-strided n (howmany=32)");
	sweep_many_strided({32, 64, 81, 100, 125, 128, 243, 256, 512, 625, 729, 1024, 2048, 4096},
	                    256, "fft_bench_many_strided_h256" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "many-strided n (howmany=256)");

	// Batched 2-D ("many3d"): n x n layer sizes spanning the radix-2/3/4/5/8
	// kernel families, {none, forward, forward} on a (depth, n, n) array vs
	// fftw_plan_many_dft rank=2 -- see sweep_many3d's comment.
	sweep_many3d({8, 9, 16, 20, 25, 27, 32, 64, 81, 100, 125, 128, 243, 256},
	             32, "fft_bench_many3d_h32" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "many3d n x n (depth=32)");
	sweep_3d_phases({8, 16, 32, 64, 81, 100, 128, 243, 256}, 32, "fft_bench_3d_phases_estimate.dat");

	// Batched 2-D with a GAP axis ({forward,none,forward}) -- see
	// sweep_gap3d's comment: non-adjacent transformed pair, needs FFTW's
	// guru interface for a correct comparison.
	sweep_gap3d({8, 9, 16, 20, 25, 27, 32, 64, 81, 100, 125, 128, 243, 256},
	            32, "fft_bench_gap3d_h32" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "gap3d n x n (depth=32)");

	// Batched 3-D ("many4d"): the 4-D analog of many3d -- one untouched
	// leading axis, full 3-D transform per block.
	sweep_many4d({8, 16, 32, 64, 128}, 32, "fft_bench_many4d_h32" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "many4d n x n x n (depth=32)");

	// Threading: caller-side parallelism over a shared plan (see
	// examples/fft_threads.cpp). Deliberately LAST: it is the only sweep that
	// saturates every core, and the heat that generates throttles whatever
	// runs after it -- putting it here kept the suite's calibration drift at
	// its usual few percent instead of the 19% measured when it ran earlier.
	sweep_parallel({64, 256, 1024, 4096}, 4096,
	               "fft_bench_parallel" BOOST_MULTI_FFT_BENCH_SUFFIX BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX ".dat", "parallel n");

#undef BOOST_MULTI_FFT_BENCH_SUFFIX
#undef BOOST_MULTI_FFT_BENCH_VARIANT_SUFFIX

	double const calib_after = calibrate();
	std::fprintf(stderr, "calibration (after):  %.4f ms\n", calib_after * 1e3);
	// Signed, because the two directions mean opposite things and only one of
	// them threatens the results. Slower at the end is throttling or
	// background load, and does bias the later sweeps. FASTER at the end is
	// the machine still ramping when the run began -- and, since sweep_parallel
	// runs last and saturates every core, it also leaves the governor boosted
	// for the closing calibration. That inflates this number (19% observed)
	// without the sweeps themselves being degraded, so it is reported
	// differently rather than as a throttling warning.
	double const drift = (calib_after - calib_before) / calib_before;
	if(drift > 0.15) {
		std::fprintf(stderr,
		              "WARNING: machine SLOWED %.0f%% across the sweep -- thermal throttling or background "
		              "load; later sweeps are biased, consider re-running.\n",
		              drift * 100.0);
	} else if(drift < -0.15) {
		std::fprintf(stderr,
		              "NOTE: machine SPED UP %.0f%% across the sweep (clocks ramping at the start, and the "
		              "multi-threaded final sweep leaves the governor boosted). Not throttling; earliest "
		              "sweeps may read slightly slow. Pre-warming the CPU reduces the first effect.\n",
		              -drift * 100.0);
	}

#ifndef DISABLE_WISDOM
	fftw_export_wisdom_to_filename(wisdom_filename);  // persist accumulated wisdom for the next run
#endif
	return 0;
}
