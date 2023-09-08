#ifdef COMPILATION// sudo cpupower frequency-set --governor performance && sudo apt install libbenchmark-dev
${CXX:-c++} -O3 -DNDEBUG `#-DNOEXCEPT_ASSIGNMENT` -I../../../../../include/ $0 -o $0x `pkg-config --libs benchmark fftw3`&&$0x&&rm $0x;exit
#endif

#include <benchmark/benchmark.h>

#include <multi/array.hpp>
#include <multi/adaptors/fftw/memory.hpp>
#include <multi/adaptors/fftw.hpp>

#include<complex>

namespace multi = boost::multi;

using complex = std::complex<double>;

template <class Alloc>
static void Allocation(benchmark::State& state){

    multi::array<complex, 2, Alloc> in({state.range(0), state.range(0)*2}, 1.2);
	multi::array<complex, 2, Alloc> out(extensions(in), 3.1);

    std::vector<double> v(state.range(0)*3.14);
    benchmark::DoNotOptimize(v);

    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);

    benchmark::ClobberMemory();

    multi::fftw::plan p(std::array<bool, 2>{true, true}, in, out, multi::fftw::forward, multi::fftw::estimate);
	for(auto _ : state){
		benchmark::DoNotOptimize(in);
		benchmark::DoNotOptimize(out);
		// benchmark::ClobberMemory();

        p();
	}

    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
}

BENCHMARK(Allocation<std        ::allocator<complex>>)->DenseRange(100, 500, 28);
BENCHMARK(Allocation<multi::fftw::allocator<complex>>)->DenseRange(100, 500, 28);

template <class Alloc>
static void Allocation1D(benchmark::State& state){

    multi::array<complex, 1, Alloc> in({state.range(0)}, 1.2);
	multi::array<complex, 1, Alloc> out(extensions(in), 3.1);

    std::vector<double> v(state.range(0)*3.14);
    benchmark::DoNotOptimize(v);

    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);

    benchmark::ClobberMemory();

    multi::fftw::plan p(std::array<bool, 1>{true}, in, out, multi::fftw::forward, multi::fftw::estimate);
	for(auto _ : state){
		benchmark::DoNotOptimize(in);
		benchmark::DoNotOptimize(out);
		// benchmark::ClobberMemory();

        p();
	}

    benchmark::DoNotOptimize(in);
    benchmark::DoNotOptimize(out);
    benchmark::ClobberMemory();
}

BENCHMARK(Allocation1D<std        ::allocator<complex>>)->RangeMultiplier(2)->Range(128, 128*1024);
BENCHMARK(Allocation1D<multi::fftw::allocator<complex>>)->RangeMultiplier(2)->Range(128, 128*1024);

BENCHMARK_MAIN();
