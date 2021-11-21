// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4-*-
// © Alfredo A. Correa 2019-2021

#include <benchmark/benchmark.h>

#include "../array.hpp"
#include "../adaptors/blas/gemm.hpp"

#include<numeric>  // for inner_product and iota
#include <execution>  // for par

namespace multi = boost::multi;

template<class MatrixA, class MatrixB, class MatrixC>
auto naive_product(MatrixA const& A, MatrixB const& B, double beta, MatrixC&& C) -> MatrixC&& {
	assert(   C .size() ==   A. size() );
	assert( (~C).size() == (~B).size() );
	for(auto i : extension(C)) {
		for(auto j : extension(~C)) {
			C[i][j] = std::inner_product(begin(A[i]), end(A[i]), begin((~B)[j]), beta*C[i][j]);
		}
	}
	return std::forward<MatrixC>(C);
}

template<class MatrixA, class MatrixB, class MatrixC>
inline auto naive_product2(MatrixA const& A, MatrixB const& B, double beta, MatrixC&& C) -> MatrixC&& {
	assert( C.size() == A.size() );
	assert( (~C).size() == (~B).size() );

	std::transform(std::execution::unseq,
		begin(A), end(A), begin(C), begin(C),
		[&](auto const& arowi, auto&& crowi) {
			std::transform(std::execution::unseq,
				begin(crowi), end(crowi), begin(~B), begin(crowi),
				[&](auto const& c, auto const& b) {
					return std::transform_reduce(std::execution::unseq,
						begin(arowi), end(arowi), begin(b), beta*c);
				}
		);
		return std::move(crowi);
	});
	return std::forward<MatrixC>(C);
//	}
}

template<class MatrixA, class MatrixB, class MatrixC>
auto naive_product3(MatrixA const& A, MatrixB const& B, double beta, MatrixC&& C) -> MatrixC&& {
	assert( C.size() == A.size() );
	assert( (~C).size() == (~B).size() );

	std::transform(std::execution::par,
		begin(~B), end(~B), begin(~C), begin(~C), [beta, begin_A = begin(A)](auto const& bcolj_ref, auto&& ccolj) {
		auto const bcolj = +bcolj_ref;
		std::transform(std::execution::seq,
			begin(ccolj), end(ccolj), begin_A, begin(ccolj),
			[&](auto const& c, auto const& a) {
				return std::transform_reduce(std::execution::unseq, begin(a), end(a), begin(bcolj), beta*c);
			}
		);
		return std::move(ccolj);
	});

	return std::forward<MatrixC>(C);
}

//template<class MatrixA, class MatrixB, class MatrixC>
//auto product(MatrixA const& A, MatrixB const& B, MatrixC&& C) {
//	assert(   C .size() ==   A .size() );
//	assert( (~C).size() == (~B).size() );

//	auto const N = 100;

//	std::for_each(std::execution::par, multi::iextension{0, extension( C).size()/N}.begin(), multi::iextension{0, extension( C).size()/N}.end(), [&](auto nblock) {
//		auto const& Ahblock = A.chunked(N)[nblock];
//		std::for_each(std::execution::par, multi::iextension{0, extension(~C).size()/N}.begin(), multi::iextension{0, extension(~C).size()/N}.end(), [&](auto mblock) {
//			auto const& Bvblock = (~B).chunked(N)[mblock];
//			~(~C.chunked(N)[nblock]).chunked(N)[mblock] =
//				std::inner_product(
//					(~Ahblock).chunked(N).begin(), (~Ahblock).chunked(N).end(),
//					(~Bvblock).chunked(N).begin(),
//					~(~C.chunked(N)[nblock]).chunked(N)[mblock],
//					[](auto&& a, auto const& b) {return naive_product2(+~std::get<0>(b), ~+~std::get<1>(b), std::move(a));},
//					[](auto const&... as) {return std::tie(as...);}
//				)
//			;
//		});
//	});
//}

template<class MatrixA, class MatrixB, class MatrixC>
auto product(MatrixA const& A, MatrixB const& B, MatrixC&& C) {
	assert(   C .size() ==   A .size() );
	assert( (~C).size() == (~B).size() );

	constexpr auto N = 128;
	std::transform(std::execution::par, begin(A.chunked(N)), end(A.chunked(N)), begin(C.chunked(N)), begin(C.chunked(N)), [N, &B](auto const& Afatrow, auto&& Cfatrow) {
		auto const& Ablocks = (~Afatrow).chunked(N);
		auto const& Bfatcols = (~B).chunked(N);
		auto&& Cblocks = (~Cfatrow).chunked(N);
		std::transform(std::execution::par, begin(Bfatcols), end(Bfatcols), begin(Cblocks), begin(Cblocks), [N, &Ablocks](auto const& Bfatcol, auto const& Cblock) {
			auto const& Bblocks = (~Bfatcol).chunked(N);
			return
				~std::inner_product(
					begin(Ablocks), end(Ablocks), begin(Bblocks),
					~Cblock,
					[](auto&& ret, auto const& prod) {return prod(std::move(ret));},
					[](auto const& Ablock, auto const& Bblock) {
						return [Ab = +~Ablock, Bb = +~Bblock](auto&& into){return naive_product2(Ab, ~Bb, 1., std::move(into));};
					}
				)
			;
		});
		return std::move(Cfatrow);
	});
}


auto const N = 1024;

static void Bnaive0(benchmark::State& _) {
	auto const A = []{
		multi::array<double, 2> A({       N,        N}    );
		std::iota(begin(A.elements()), end(A.elements()), 0.1);
		return A;
	}();

	auto const B = [&]{
		multi::array<double, 2> B({(~A).size(),        N}    );
		std::iota(begin(B.elements()), end(B.elements()), 0.2);
		return B;
	}();

	multi::array<double, 2> C1({  A.size() , (~B).size()}, 0.);
	multi::array<double, 2> C2({  A.size() , (~B).size()}, 0.);
	multi::array<double, 2> C3({  A.size() , (~B).size()}, 0.);
	multi::array<double, 2> C ({  A.size() , (~B).size()}, 0.);
	multi::array<double, 2> C4({  A.size() , (~B).size()}, 0.);

	while(_.KeepRunning()) {
		naive_product (A, B, 0., C1);
		naive_product2(A, B, 0., C2);
		naive_product3(A, B, 0., C3);
		product(A, B, C);
		multi::blas::gemm(1., A, B, 0., C4);
	}
	std::cerr<<"**********"<< C1[789][657] <<' '<< C2[789][657] <<' '<< C3[789][657] <<' '<< C4[789][657] <<' ' << C1[789][657] - C2[789][657] <<' '<< C1[789][657] - C3[789][657] <<' '<< C4[789][657] - C3[789][657] <<' '<< C4[789][657] - C[789][657]  <<std::endl;
}

static void Bnaive1(benchmark::State& _) {
	auto const A = []{
		multi::array<double, 2> A({       N,        N}    );
		std::iota(begin(A.elements()), end(A.elements()), 0.1);
		return A;
	}();

	auto const B = [&]{
		multi::array<double, 2> B({(~A).size(),        N}    );
		std::iota(begin(B.elements()), end(B.elements()), 0.2);
		return B;
	}();

	multi::array<double, 2> C({  A.size() , (~B).size()}, 0.);

	while(_.KeepRunning()) {
		naive_product(A, B, 0., C);
	}
}

static void Bnaive2(benchmark::State& state) {
	auto const A = []{
		multi::array<double, 2> A({       N,        N}    );
		std::iota(begin(A.elements()), end(A.elements()), 0.1);
		return A;
	}();

	auto const B = [&]{
		multi::array<double, 2> B({(~A).size(),        N}    );
		std::iota(begin(B.elements()), end(B.elements()), 0.2);
		return B;
	}();

	multi::array<double, 2> C({  A.size() , (~B).size()}, 0.);

	for(auto _ : state) {
		naive_product2(A, B, 0., C);
	}
}

static void Bnaive3(benchmark::State& state) {
	auto const A = []{
		multi::array<double, 2> A({       N,        N}    );
		std::iota(begin(A.elements()), end(A.elements()), 0.1);
		return A;
	}();

	auto const B = [&]{
		multi::array<double, 2> B({(~A).size(),        N}    );
		std::iota(begin(B.elements()), end(B.elements()), 0.2);
		return B;
	}();

	multi::array<double, 2> C({  A.size() , (~B).size()}, 0.);

	for(auto _ : state) {
		naive_product3(A, B, 0., C);
		benchmark::DoNotOptimize(C[1][1]);
	    benchmark::ClobberMemory(); // Force 42 to be written to memory.
	}
}

static void Bnaive_product(benchmark::State& state) {
	auto const A = []{
		multi::array<double, 2> A({       N,        N}    );
		std::iota(begin(A.elements()), end(A.elements()), 0.1);
		return A;
	}();

	auto const B = [&]{
		multi::array<double, 2> B({(~A).size(),        N}    );
		std::iota(begin(B.elements()), end(B.elements()), 0.2);
		return B;
	}();

	multi::array<double, 2> C({  A.size() , (~B).size()}, 0.);

	for(auto _ : state) {
		product(A, B, C);
		benchmark::DoNotOptimize(C);
	    benchmark::ClobberMemory(); // Force 42 to be written to memory.
	}
}

static void Bgemm(benchmark::State& state) {
	auto const A = []{
		multi::array<double, 2> A({       N,        N}    );
		std::iota(begin(A.elements()), end(A.elements()), 0.1);
		return A;
	}();

	auto const B = [&]{
		multi::array<double, 2> B({(~A).size(),        N}    );
		std::iota(begin(B.elements()), end(B.elements()), 0.2);
		return B;
	}();

	multi::array<double, 2> C({  A.size() , (~B).size()}, 0.);

	for(auto _ : state) {
		multi::blas::gemm(A, B, C);
		benchmark::DoNotOptimize(C);
	    benchmark::ClobberMemory(); // Force 42 to be written to memory.
	}
}


BENCHMARK(Bnaive0);
BENCHMARK(Bnaive1);
BENCHMARK(Bnaive2);
BENCHMARK(Bnaive3);
BENCHMARK(Bnaive_product);
BENCHMARK(Bgemm);
BENCHMARK(Bnaive0);

BENCHMARK_MAIN();
