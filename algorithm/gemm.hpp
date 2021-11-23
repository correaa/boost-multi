// Copyright 2021 Alfredo A. Correa
// this header contains a generic gemm algorithm (not the blas one)
// it is 3 times slower than blas::gemm but it is more generic in the type and in the operations
// when compiled using -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -DNDEBUG -mfpmath=sse -march=native -funroll-loops -fargument-noalias"

#ifndef MULTI_ALGORITHM_GEMM_HPP
#define MULTI_ALGORITHM_GEMM_HPP

#include <execution>  // for par  // needs linking to TBB library
#include <numeric>  // for inner_product and transform_reduce

namespace boost {
namespace multi {

namespace detail {

template<class Talpha, class MatrixA, class MatrixB, class Tbeta, class MatrixC>
inline auto naive_gemm(Talpha const& alpha, MatrixA const& A, MatrixB const& B, Tbeta const& beta, MatrixC&& C) -> MatrixC&& {
	assert( C.size() == A.size() );
	assert( (~C).size() == (~B).size() );

	std::transform(std::execution::unseq,
		begin(A), end(A), begin(C), begin(C),
		[&](auto const& arowi, auto&& crowi) {
			std::transform(std::execution::unseq,
				begin(crowi), end(crowi), begin(~B), begin(crowi),
				[&](auto const& c, auto const& brow) {
					return alpha*std::transform_reduce(std::execution::unseq, begin(arowi), end(arowi), begin(brow), beta*c);
				}
		);
		return std::move(crowi);  // NOLINT(bugprone-move-forwarding-reference)
	});
	return std::forward<MatrixC>(C);
}

}  // end namespace detail

template<class Talpha, class MatrixA, class MatrixB, class Tbeta, class MatrixC>
auto gemm(Talpha const& alpha, MatrixA const& A, MatrixB const& B, Tbeta const& /*beta*/, MatrixC&& C) -> MatrixC&& {
	assert(   C .size() ==   A .size() );
	assert( (~C).size() == (~B).size() );

	constexpr auto N = 128;

	assert(   A .size() % N == 0);
	assert( (~A).size() % N == 0);
	assert( (~B).size() % N == 0);
	assert(   B .size() % N == 0);

	std::transform(std::execution::par, begin(A.chunked(N)), end(A.chunked(N)), begin(C.chunked(N)), begin(C.chunked(N)), [&](auto const& Afatrow, auto&& Cfatrow) {
		auto const& Bfatcols = (~B).chunked(N);
		auto const& Ablocks = (~Afatrow).chunked(N);
		auto&& Cblocks = (~Cfatrow).chunked(N);
		std::transform(std::execution::par, begin(Bfatcols), end(Bfatcols), begin(Cblocks), begin(Cblocks), [&](auto const& Bfatcol, auto const& /*Cblock*/) {
			auto const& Bblocks = (~Bfatcol).chunked(N);
			return
				+~std::inner_product(
					begin(Ablocks), end(Ablocks), begin(Bblocks),
					multi::array<double, 2>({N, N}, 0.),
					[&](auto const& ret, auto const& prod) {multi::array<double, 2> r({N, N}, 0.); std::transform(begin(elements(ret)), end(elements(ret)), begin(elements(prod)), begin(elements(r)), std::plus<>{}); return r;},
					[&](auto const& Ablock, auto const& Bblock) {
						return detail::naive_gemm(alpha, +~Ablock, ~+~Bblock, 0., multi::array<double, 2>({N, N}, 0.));
					//	return [&, Ab = +~Ablock, Bb = +~Bblock](auto&& into){return detail::naive_gemm(alpha, Ab, ~Bb, beta, std::move(into));};  // NOLINT(bugprone-move-forwarding-reference)
					}
				)
			;
		});
		return std::move(Cfatrow);  // NOLINT(bugprone-move-forwarding-reference)
	});
	return std::forward<MatrixC>(C);
}

}  // end namespace multi
}  // end namespace boost
#endif
