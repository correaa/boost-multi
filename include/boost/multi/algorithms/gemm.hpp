// Copyright 2021-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// this header contains a generic gemm algorithm (not the blas one)
// it is ~3 times slower than blas::gemm but it is more generic in the type and in the operations
// when compiled using -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -DNDEBUG -mfpmath=sse -march=native -funroll-loops"

#ifndef BOOST_MULTI_ALGORITHM_GEMM_HPP
#define BOOST_MULTI_ALGORITHM_GEMM_HPP

#include <boost/multi/array.hpp>
#include <boost/multi/detail/static_allocator.hpp>  // TODO(correaa) export IWYU

#include <execution>  // for par  // needs linking to TBB library
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>  // for inner_product and transform_reduce

namespace boost {
namespace multi {

namespace detail {

template<class Talpha, class MatrixA, class MatrixB, class Tbeta, class MatrixC, class BinarySum, class BinaryProd>
inline auto naive_gemm(Talpha const& alpha, MatrixA const& A, MatrixB const& B, Tbeta const& beta, MatrixC&& C, BinarySum sum2, BinaryProd prod2) -> MatrixC&& {
	assert( C.size() == A.size() );
	assert( (~C).size() == (~B).size() );

	std::transform(// std::execution::par,  // intel (and others?) cannot simd this level
		begin(A), end(A), begin(C), begin(C),
		[&](auto const& arowi, auto&& crowi) -> decltype(auto) {
			std::transform(std::execution::unseq,
				begin(crowi), end(crowi), begin(~B), begin(crowi),
				[&](auto&& c, auto const& brow) {
					// cppcheck-suppress cppcheckError  .. internal cppcheck error
					return sum2(alpha*std::transform_reduce(std::execution::unseq,
						begin(arowi), end(arowi), begin(brow), decltype(+c){0.}, sum2, prod2), beta*std::forward<decltype(c)>(c)
					);
				}
		);
		return std::forward<decltype(crowi)>(crowi);
	});
	return std::forward<MatrixC>(C);
}

template<class Talpha, class MatrixA, class MatrixB, class Tbeta, class MatrixC>
inline auto naive_gemm(Talpha const& alpha, MatrixA const& A, MatrixB const& B, Tbeta const& beta, MatrixC&& C) -> MatrixC&& {
	return naive_gemm(alpha, A, B, beta, std::forward<MatrixC>(C), std::plus<>{}, std::multiplies<>{});
}

}  // end namespace detail

template<class Talpha, class MatrixA, class MatrixB, class Tbeta, class MatrixC, class BinarySum, class BinaryProd>
auto gemm(Talpha const& alpha, MatrixA const& A, MatrixB const& B, Tbeta const& beta, MatrixC&& C, BinarySum sum2, BinaryProd prod2) -> MatrixC&& {
	assert(   C .size() ==   A .size() );
	assert( (~C).size() == (~B).size() );

	auto const N = std::min(128L, C.size());

	assert(   A .size() % N == 0);
	assert( (~A).size() % N == 0);
	assert( (~B).size() % N == 0);
	assert(   B .size() % N == 0);

	std::transform(std::execution::seq,
		begin(A.chunked(N)), end(A.chunked(N)), begin(C.chunked(N)), begin(C.chunked(N)), [&](auto const& Afatrow, auto&& Cfatrow) -> decltype(auto) {
		auto const& BfatcolsT = (~B).chunked(N);
		auto const& AblocksT = (~Afatrow).chunked(N);
		auto&& CblocksT = (~Cfatrow).chunked(N);
		std::transform(std::execution::seq,
			begin(BfatcolsT), end(BfatcolsT), begin(CblocksT), begin(CblocksT), [&](auto const& BfatcolT, auto&& CblockTR) {
			auto const& Bblocks = (~BfatcolT).chunked(N);
			auto Cblock = +~CblockTR;
			// multi::array<typename MatrixA::element, 2> Cblock = ~CblockTR;
			//multi::static_array<typename MatrixA::element, 2, multi::detail::static_allocator<typename MatrixA::element, 128L*128L>>
			// 	Cblock = ~CblockTR;
			std::transform(std::execution::unseq,
				begin(Cblock.elements()), end(Cblock.elements()), begin(Cblock.elements()), [&](auto&& c) {return beta*std::forward<decltype(c)>(c);}
			);
			return
				+~std::inner_product(
					begin(AblocksT), end(AblocksT), begin(Bblocks),
					std::move(Cblock),
					[&](auto&& ret, auto const& prod) {return prod(std::forward<decltype(ret)>(ret));},
					[&](auto const& AblockT, auto const& Bblock) {
						return [&, AbR = +~AblockT, BbTR = +~Bblock](auto&& into) {return detail::naive_gemm(alpha, AbR, ~BbTR, 1., std::forward<decltype(into)>(into), sum2, prod2);};
					}
				)
			;
		});
		return std::forward<decltype(Cfatrow)>(Cfatrow);
	});
	return std::forward<MatrixC>(C);
}

template<class Talpha, class MatrixA, class MatrixB, class Tbeta, class MatrixC>
auto gemm(Talpha const& alpha, MatrixA const& A, MatrixB const& B, Tbeta const& beta, MatrixC&& C) -> MatrixC&& {
	return gemm(alpha, A, B, beta, std::forward<MatrixC>(C), std::plus<>{}, std::multiplies<>{});
}


}  // end namespace multi
}  // end namespace boost
#endif  // BOOST_MULTI_ALGORITHM_GEMM_HPP
