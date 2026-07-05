// Copyright 2024-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifdef COMPILATION_INSTRUCTIONS
g++ - std = c++ 17 - O3 - march = native - DNDEBUG - I../ include $0 - o $0.x - lfftw3 &&./ $0.x && gnuplot algorithms_fft_plots.gp;
exit
#endif

// Size sweep of multi::fft_plan vs FFTW, emitting gnuplot-friendly .dat files
// (one per dimensionality). Methodology: both libraries build their plan once
// (untimed) and recycle it; only execution is timed; CPU cache flushed before
// every repetition; FFTW_ESTIMATE, no wisdom; single thread.
#include <boost/multi/algorithms/fft.hpp>
#include <boost/multi/array.hpp>

#include <fftw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

	namespace multi = boost::multi;
using complex       = std::complex<double>;

struct watch {
	std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
	auto                                           sec() const { return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - s).count(); }
};
static std::vector<char> g_thrash(64 << 20);
void                     flush_cache() {
    for(std::size_t i = 0; i < g_thrash.size(); i += 64) {
        g_thrash[i]++;
    }
    char volatile x = g_thrash[0];
    (void)x;
}
template<class F> double time_it(long reps, F f) {
	double t = 0;
	for(long r = 0; r != reps; ++r) {
		flush_cache();
		watch w;
		f();
		t += w.sec();
	}
	return t / reps;
}

auto reps_for(double n_total) -> long {
	double const work = 5.0 * n_total * std::log2(std::max(n_total, 2.0));
	return std::clamp<long>(static_cast<long>(2e8 / work), 5, 300);
}

template<std::ptrdiff_t D>
void sweep(std::vector<int> const& sides, char const* fname, char const* label) {
	std::FILE* out = std::fopen(fname, "w");
	std::fprintf(out, "# %s: multi::fft_plan vs FFTW 3 (plan recycled, exec only, cache flushed/rep, single thread, FFTW_ESTIMATE)\n", label);
	std::fprintf(out, "# mflops = 5*N*log2(N)/time_us (benchFFT convention), N = total points\n");
	std::fprintf(out, "# n_side  N_total  mine_ms  fftw_ms  mine_mflops  fftw_mflops  ratio_mine_over_fftw\n");
	for(int n : sides) {
		long N = 1;
		for(int d = 0; d != D; ++d) {
			N *= n;
		}
		std::vector<complex>                   base(N);
		std::mt19937                           gen(42);
		std::uniform_real_distribution<double> dist(-1.0, 1.0);
		for(auto& e : base) {
			e = complex{dist(gen), dist(gen)};
		}
		long const reps = reps_for(static_cast<double>(N));

		multi::array<complex, 1> flat(multi::extensions_t<1>{N});
		auto                     load = [&] { std::copy(base.begin(), base.end(), flat.begin()); };
		double                   mine = 0.0;
		{
			std::array<multi::ssize_t, D> ext_arr{};
			ext_arr.fill(n);
			multi::fft_plan<complex, D> const plan{ext_arr, multi::fft_forward};
			if constexpr(D == 1) {
				load();
				plan(flat);
				mine = time_it(reps, [&] { load(); plan(flat); });
			} else if constexpr(D == 2) {
				multi::array_ref<complex, 2> v(flat.data_elements(), {n, n});
				load();
				plan(v);
				mine = time_it(reps, [&] { load(); plan(v); });
			} else {
				multi::array_ref<complex, 3> v(flat.data_elements(), {n, n, n});
				load();
				plan(v);
				mine = time_it(reps, [&] { load(); plan(v); });
			}
		}

		auto* in    = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
		auto* fo    = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * N));
		auto  loadf = [&] { for(long i = 0; i != N; ++i) { in[i][0] = base[i].real(); in[i][1] = base[i].imag(); } };
		fftw_forget_wisdom();
		fftw_plan p = D == 1 ? fftw_plan_dft_1d(n, in, fo, FFTW_FORWARD, FFTW_ESTIMATE)
					: D == 2 ? fftw_plan_dft_2d(n, n, in, fo, FFTW_FORWARD, FFTW_ESTIMATE)
							 : fftw_plan_dft_3d(n, n, n, in, fo, FFTW_FORWARD, FFTW_ESTIMATE);
		loadf();
		double ffw = time_it(reps, [&] { loadf(); fftw_execute(p); });
		fftw_destroy_plan(p);
		fftw_free(in);
		fftw_free(fo);

		double const work = 5.0 * static_cast<double>(N) * std::log2(static_cast<double>(N));
		std::fprintf(out, "%8d %10ld %12.5f %12.5f %12.1f %12.1f %8.3f\n", n, N, mine * 1e3, ffw * 1e3, work / (mine * 1e6), work / (ffw * 1e6), mine / ffw);
		std::fflush(out);
		std::fprintf(stderr, "%s n=%d done (reps=%ld)\n", label, n, reps);
	}
	std::fclose(out);
}

int main() {
	sweep<1>({16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152}, "fft_bench_1d.dat", "1D n");
	sweep<2>({16, 32, 64, 128, 256, 512, 1024, 2048}, "fft_bench_2d.dat", "2D n x n");
	sweep<3>({8, 16, 32, 64, 128, 256}, "fft_bench_3d.dat", "3D n x n x n");
	return 0;
}
