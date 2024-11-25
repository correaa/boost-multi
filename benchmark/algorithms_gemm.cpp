#ifdef COMPILATION  // sudo cpupower frequency-set --governor performance; sudo apt search libbenchmark-dev
set -x;
c++ -std=c++17 -DNDEBUG -Ofast -mfpmath=sse -march=native -funroll-loops $0 -o $0x -I../include `pkg-config --cflags --libs benchmark tbb openblas` && $0x && rm $0x;
exit
#endif

// Copyright 2019-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/algorithms/gemm.hpp>
#include <boost/multi/array.hpp>
#include <boost/multi/adaptors/blas.hpp>

#include <benchmark/benchmark.h>

#include <boost/core/lightweight_test.hpp>

#include <numeric>
#include <random>
#include <thread>

namespace multi = boost::multi;

static void BM_System(benchmark::State& state) {
	std::random_device               rd;
	std::mt19937                     gen(rd());
	std::uniform_real_distribution<> dis(-1.0, +1.0);

	auto const N = 1024;

	auto const A = [&] {
		multi::array<double, 2> _({N, N});
		std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
		return _;
	}();

	auto const B = [&] {
		multi::array<double, 2> _({(~A).size(), N});
		std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
		return _;
	}();

	multi::array<double, 2> C({A.size(), (~B).size()}, 0.0);

	for(auto _ : state) {
		multi::blas::gemm(2.1, A, B, 1.1, C);
		benchmark::DoNotOptimize(C);
	}
}
BENCHMARK(BM_System);

static void BM_Naive(benchmark::State& state) {
	std::random_device               rd;
	std::mt19937                     gen(rd());
	std::uniform_real_distribution<> dis(-1.0, +1.0);

	auto const N = 1024;

	auto const A = [&] {
		multi::array<double, 2> _({N, N});
		std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
		return _;
	}();

	auto const B = [&] {
		multi::array<double, 2> _({(~A).size(), N});
		std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
		return _;
	}();

	multi::array<double, 2> C({A.size(), (~B).size()}, 0.0);

	for(auto _ : state) {
		multi::detail::naive_gemm(2.1, A, B, 1.1, C);
		benchmark::DoNotOptimize(C);
	}
}
BENCHMARK(BM_Naive);

static void BM_Chunked(benchmark::State& state) {
	std::random_device               rd;
	std::mt19937                     gen(rd());
	std::uniform_real_distribution<> dis(-1.0, +1.0);

	auto const N = 1024;

	auto const A = [&] {
		multi::array<double, 2> _({N, N});
		std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
		return _;
	}();

	auto const B = [&] {
		multi::array<double, 2> _({(~A).size(), N});
		std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
		return _;
	}();

	multi::array<double, 2> C({A.size(), (~B).size()}, 0.0);

	for(auto _ : state) {
		multi::gemm(2.1, A, B, 1.1, C);
		benchmark::DoNotOptimize(C);
	}
}
BENCHMARK(BM_Chunked);

#define BOOST_AUTO_TEST_CASE(CasenamE) /**/

auto main(int argc, char** argv) -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	::benchmark::Initialize(&argc, argv);

	BOOST_AUTO_TEST_CASE(algorithm_gemm) {

		std::random_device               rd;
		std::mt19937                     gen(rd());
		std::uniform_real_distribution<> dis(-1.0, +1.0);

		auto const N = 256;

		auto const A = [&] {
			multi::array<double, 2> _({N, N});
			std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
			return _;
		}();

		auto const B = [&] {
			multi::array<double, 2> _({(~A).size(), N});
			std::generate(begin(elements(_)), end(elements(_)), [&] { return dis(gen); });
			return _;
		}();

		// zero init, beta = zero multiplication
		{
			multi::array<double, 2> C_blas({A.size(), (~B).size()}, 0.0);
			multi::array<double, 2> C_gold({A.size(), (~B).size()}, 0.0);
			multi::array<double, 2> C = C_gold;

			multi::blas::gemm(1.0, A, B, 0.0, C_blas);
			multi::detail::naive_gemm(1.0, A, B, 0.0, C_gold);
			multi::gemm(1.0, A, B, 0.0, C);

			BOOST_TEST( std::abs(C[123][121] - C_gold[123][121]) < 1E-12 );
			BOOST_TEST( std::abs(C[123][121] - C_blas[123][121]) < 1E-12 );
		}

		// non-zero init, beta = zero multiplication
		{
			multi::array<double, 2> C_gold({A.size(), (~B).size()}, 0.);
			std::generate(begin(elements(C_gold)), end(elements(C_gold)), [&] { return dis(gen); });

			multi::array<double, 2> C = C_gold;

			multi::detail::naive_gemm(1., A, B, 0., C_gold);
			multi::gemm(1., A, B, 0., C);

			BOOST_TEST( std::abs( C[123][121] - C_gold[123][121] ) < 1E-12 );
		}
		// non-zero init, beta = one multiplication
		{
			multi::array<double, 2> C_gold({A.size(), (~B).size()}, 0.);
			std::generate(begin(elements(C_gold)), end(elements(C_gold)), [&] { return dis(gen); });

			multi::array<double, 2> C = C_gold;

			multi::detail::naive_gemm(1.0, A, B, 1.0, C_gold);
			multi::gemm(1.0, A, B, 1.0, C);

			BOOST_TEST( std::abs( C[123][121] - C_gold[123][121]) < 1E-12 );
		}
		{
			multi::array<double, 2> C_gold({A.size(), (~B).size()}, 0.);
			std::generate(begin(elements(C_gold)), end(elements(C_gold)), [&] { return dis(gen); });

			multi::array<double, 2> C = C_gold;

			multi::detail::naive_gemm(1., A, B, 0.3, C_gold);
			multi::gemm(1.0, A, B, 0.3, C);

			BOOST_TEST( std::abs( C[123][121] - C_gold[123][121] ) < 1E-12 );
		}
		{
			multi::array<double, 2> C_gold({A.size(), (~B).size()}, 0.);
			std::generate(begin(elements(C_gold)), end(elements(C_gold)), [&] { return dis(gen); });

			multi::array<double, 2> C = C_gold;

			multi::detail::naive_gemm(2., A, B, 0., C_gold);
			multi::gemm(2.0, A, B, 0.0, C);

			BOOST_TEST( std::abs( C[123][121] - C_gold[123][121] ) < 1E-12 );
		}
		{
			multi::array<double, 2> C_gold({A.size(), (~B).size()}, 0.);
			std::generate(begin(elements(C_gold)), end(elements(C_gold)), [&] { return dis(gen); });

			multi::array<double, 2> C = C_gold;

			multi::detail::naive_gemm(2., A, B, 0.3, C_gold);
			multi::gemm(2., A, B, 0.3, C);

			BOOST_TEST( std::abs( C[123][121] - C_gold[123][121] ) < 1E-12 );
		}
	}

	benchmark::RunSpecifiedBenchmarks();
	benchmark::Shutdown();

	return boost::report_errors();
}
