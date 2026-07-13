// Copyright 2024-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// 2-D FFT benchmark: multi::fft_plan vs FFTW at three planning levels.
// Emits fft_bench_2d_planning_levels.dat (gnuplot-friendly).
//
// COMPILATION:
//   g++ -std=c++17 -O3 -march=native -mtune=native -funroll-loops -fno-math-errno -DNDEBUG \
//     -I../include algorithms_fft_2d_planning_levels.cpp \
//     -o algorithms_fft_2d_planning_levels.x -lfftw3 \
//     && ./algorithms_fft_2d_planning_levels.x \
//     && gnuplot algorithms_fft_2d_planning_levels.gp
//
// The program polls /proc/loadavg and waits until the 1-minute load average
// drops below 0.5 before starting any timing.  FFTW_EXHAUSTIVE can take
// minutes to plan for n >= 256; that planning time is printed to stderr but
// NOT included in the execution timings.  Sizes are capped at n=512 because
// FFTW_EXHAUSTIVE beyond that is impractically slow.
//
// Methodology: identical to algorithms_fft.cpp (flushed cache, interleaved
// timing, persistent monotonic-arena allocator, CPU warm-up, calibration
// drift check).  All three FFTW plans call fftw_forget_wisdom() before
// planning so each level starts from a clean state.

#include <boost/multi/algorithms/fft.hpp>
#include <boost/multi/array.hpp>

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <memory_resource>
#include <random>
#include <tuple>
#include <vector>

#ifdef __linux__
#  include <sched.h>
#  include <unistd.h>
#endif

namespace multi = boost::multi;
using complex   = std::complex<double>;

namespace {

// ── idle guard ──────────────────────────────────────────────────────────────

bool machine_is_idle(double threshold = 0.5) {
#ifdef __linux__
	std::FILE* f = std::fopen("/proc/loadavg", "r");
	if(!f) { return true; }
	char buf[64] = {'9'};  // default: treat empty file as busy
	if(std::fread(buf, 1, sizeof(buf) - 1, f) == 0) { buf[0] = '9'; }
	std::fclose(f);
	return std::strtod(buf, nullptr) < threshold;
#else
	(void)threshold;
	return true;
#endif
}

void wait_for_idle() {
	if(machine_is_idle()) { return; }
	std::fprintf(stderr, "Waiting for idle machine (1-min load >= 0.5)...\n");
	while(!machine_is_idle()) {
#ifdef __linux__
		::sleep(30);
#endif
		std::fprintf(stderr, "  still waiting...\n");
	}
	std::fprintf(stderr, "Machine idle — starting benchmark.\n");
}

// ── infrastructure (mirrors algorithms_fft.cpp) ──────────────────────────────

struct watch {
	std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
	auto sec() const { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - s).count(); }
};

std::vector<char> g_thrash(64 << 20);
void               flush_cache() {
	for(std::size_t i = 0; i < g_thrash.size(); i += 64) { g_thrash[i]++; }
	char volatile x = g_thrash[0];
	(void)x;
}

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

void warm_up_cpu() {
	watch           w;
	volatile double x = 1.0;
	while(w.sec() < 8.0) {
		for(int i = 0; i != 100000; ++i) { x = std::sin(x) + std::cos(x); }
	}
}

auto reps_for(double n_total) -> long {
	double const work = 5.0 * n_total * std::log2(std::max(n_total, 2.0));
	return std::clamp<long>(static_cast<long>(2e8 / work), 5, 300);
}

template<class Plan>
struct arena_alloc {
	std::vector<std::byte>                   buf;
	std::pmr::monotonic_buffer_resource      mbr;
	std::pmr::polymorphic_allocator<complex> alloc;

	explicit arena_alloc(Plan const& plan)
	: buf(plan.scratch_elements() * sizeof(complex) + 4096), mbr(buf.data(), buf.size()), alloc(&mbr) {}

	void reset() { mbr.release(); }
};

// 4-way interleaved timing: one flushed timed call per variant per rep,
// cycling through all four variants to cancel thermal/frequency drift.
template<class PA, class FA, class PB, class FB, class PC, class FC, class PD, class FD>
auto time_4way(long reps, PA pa, FA fa, PB pb, FB fb, PC pc, FC fc, PD pd, FD fd)
	-> std::tuple<double, double, double, double> {
	double ta = 0, tb = 0, tc = 0, td = 0;
	for(long r = 0; r != reps; ++r) {
		pa(); flush_cache(); { watch w; fa(); ta += w.sec(); }
		pb(); flush_cache(); { watch w; fb(); tb += w.sec(); }
		pc(); flush_cache(); { watch w; fc(); tc += w.sec(); }
		pd(); flush_cache(); { watch w; fd(); td += w.sec(); }
	}
	return {ta / reps, tb / reps, tc / reps, td / reps};
}

auto calibrate() -> double {
	multi::array<complex, 1>          a(multi::extents_t<1>{16384}, complex{1.0, 0.0});
	multi::fft_plan<1, complex> const plan{multi::extents_t<1>{16384}, multi::fft_forward};
	arena_alloc<decltype(plan)>       arena(plan);
	flush_cache();
	watch w;
	plan.execute(a.home(), arena.alloc);
	arena.reset();
	return w.sec();
}

}  // namespace

auto main() -> int {
	wait_for_idle();
	pin_to_one_cpu();

	std::FILE* out = std::fopen("fft_bench_2d_planning_levels.dat", "w");
	std::fprintf(out,
	             "# 2-D FFT: multi::fft_plan vs FFTW at three planning levels\n"
	             "# in-place n×n complex<double>, cold cache, interleaved timing\n"
	             "# FFTW: fftw_forget_wisdom() before each plan (clean state per level)\n"
	             "# planning time excluded from all timings; sizes capped at 512 (EXHAUSTIVE)\n"
	             "# mflops = 5*N*log2(N)/time_us  (benchFFT convention), N = n*n\n"
	             "# n  N  mine_ms  est_ms  mea_ms  exh_ms"
	             "  mine_mflops  est_mflops  mea_mflops  exh_mflops\n");

	std::fprintf(stderr, "calibration (before): ");
	double const calib_before = calibrate();
	std::fprintf(stderr, "%.4f ms\n", calib_before * 1e3);

	warm_up_cpu();

	// 5-smooth sizes ≤ 512; EXHAUSTIVE beyond 512 is impractically slow.
	for(int n : {24, 25, 27, 32, 40, 60, 64, 75, 81, 100, 125, 128,
	             216, 243, 250, 256, 320, 375, 405, 486, 512}) {
		long const N    = static_cast<long>(n) * n;
		long const reps = reps_for(static_cast<double>(N));

		std::vector<complex>                   base(static_cast<std::size_t>(N));
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) { e = complex{dist(gen), dist(gen)}; }

		multi::array<complex, 1>     flat(multi::extents_t<1>{N});
		multi::array_ref<complex, 2> v(flat.data_elements(), {n, n});
		auto                          load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };

		std::array<multi::ssize_t, 2> const ext2{n, n};
		multi::fft_plan<2, complex> const plan{ext2, multi::fft_forward};
		arena_alloc<decltype(plan)>        arena(plan);

		auto* fw = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * static_cast<std::size_t>(N)));
		auto  loadf = [&] {
			for(long i = 0; i != N; ++i) {
				fw[i][0] = base[static_cast<std::size_t>(i)].real();
				fw[i][1] = base[static_cast<std::size_t>(i)].imag();
			}
		};

		// Build three FFTW plans — each from a clean slate.
		loadf();
		fftw_forget_wisdom();
		watch t_est;
		fftw_plan p_est = fftw_plan_dft_2d(n, n, fw, fw, FFTW_FORWARD, FFTW_ESTIMATE);
		std::fprintf(stderr, "n=%d ESTIMATE plan:   %.2f s\n", n, t_est.sec());

		loadf();
		fftw_forget_wisdom();
		watch t_mea;
		fftw_plan p_mea = fftw_plan_dft_2d(n, n, fw, fw, FFTW_FORWARD, FFTW_MEASURE);
		std::fprintf(stderr, "n=%d MEASURE plan:    %.2f s\n", n, t_mea.sec());

		loadf();
		fftw_forget_wisdom();
		watch t_exh;
		fftw_plan p_exh = fftw_plan_dft_2d(n, n, fw, fw, FFTW_FORWARD, FFTW_EXHAUSTIVE);
		std::fprintf(stderr, "n=%d EXHAUSTIVE plan: %.2f s\n", n, t_exh.sec());

		// Warm-up (one untimed call per variant, symmetric).
		load();  plan.execute(v.home(), arena.alloc);  arena.reset();
		loadf(); fftw_execute(p_est);
		loadf(); fftw_execute(p_mea);
		loadf(); fftw_execute(p_exh);

		auto [tm, te, tme, tex] = time_4way(
			reps,
			load,  [&] { plan.execute(v.home(), arena.alloc); arena.reset(); },
			loadf, [&] { fftw_execute(p_est); },
			loadf, [&] { fftw_execute(p_mea); },
			loadf, [&] { fftw_execute(p_exh); });

		fftw_destroy_plan(p_est);
		fftw_destroy_plan(p_mea);
		fftw_destroy_plan(p_exh);
		fftw_free(fw);

		double const work = 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
		std::fprintf(out, "%5d %10ld %10.4f %10.4f %10.4f %10.4f %12.1f %12.1f %12.1f %12.1f\n",
		             n, N,
		             tm * 1e3, te * 1e3, tme * 1e3, tex * 1e3,
		             work / (tm * 1e6), work / (te * 1e6), work / (tme * 1e6), work / (tex * 1e6));
		std::fflush(out);
		std::fprintf(stderr, "n=%d done (reps=%ld)\n\n", n, reps);
	}

	std::fclose(out);

	std::fprintf(stderr, "calibration (after):  ");
	double const calib_after = calibrate();
	std::fprintf(stderr, "%.4f ms\n", calib_after * 1e3);
	double const drift = std::abs(calib_after - calib_before) / calib_before;
	if(drift > 0.15) {
		std::fprintf(stderr,
		             "WARNING: calibration drifted %.0f%% -- results may be affected by "
		             "thermal throttling or background load; consider re-running.\n",
		             drift * 100.0);
	}
}
