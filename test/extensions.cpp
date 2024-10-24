// Copyright 2021-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>

#include <boost/multi/array.hpp>

// #include <algorithm>  // for equal
// #include <numeric>    // for accumulate
// #include <vector>     // for vector

template <typename F, typename Tuple, std::size_t... I>
constexpr decltype(auto) apply_impl(F&& f, Tuple&& t, std::index_sequence<I...>) {
    return std::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))...);
}

template <typename F, typename Tuple>
constexpr decltype(auto) Apply(F&& f, Tuple&& t) {
    return apply_impl(std::forward<F>(f), std::forward<Tuple>(t),
                      std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

namespace std {
//   template <typename _Fn, size_t... _Idx>
//     constexpr decltype(auto)
//     APPLY_impl(_Fn&& __f, boost::multi::extensions_t<2> const& __t, std::index_sequence<_Idx...>)
//     {
//       return std::__invoke(std::forward<_Fn>(__f),
// 			   std::get<_Idx>(__t)...);
//     }

//   template <typename _Fn, boost::multi::dimensionality_type D>
//     constexpr decltype(auto)
//     apply(_Fn&& __f, boost::multi::extensions_t<D> const& __t)
//     noexcept
//     {
// 		return __t.apply(__f);
//     }
}

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(bugprone-exception-escape)

	multi::array<int, 2> A2D({5, 7}, 1.0);
	auto const A2Dx = A2D.extension();

	BOOST_TEST( A2Dx[0] == 0 );

	BOOST_TEST( &A2D() == &A2D(A2Dx) );

	auto const A2Dxs = A2D.extensions();

	BOOST_TEST( std::get<0>(A2Dxs) == A2Dx );
	BOOST_TEST( std::get<1>(A2Dxs) == A2D[0].extension() );

	BOOST_TEST( &A2D() == &A2D(std::get<0>(A2D.extensions())) );
	BOOST_TEST( &A2D() == &A2D(std::get<0>(A2D.extensions()), std::get<1>(A2D.extensions())) );

	auto a1 = std::get<0>(A2D.extensions());
	auto a2 = std::get<0>(A2Dxs);

	BOOST_TEST( a1 == a2 );

	static_assert( std::tuple_size_v<decltype(A2Dxs)> == 2 );
	static_assert( std::is_same_v<std::tuple_element_t<0, decltype(A2Dxs)>, decltype(A2Dx)> );
	static_assert( std::is_same_v<std::tuple_element_t<1, decltype(A2Dxs)>, decltype(A2Dx)> );

	BOOST_TEST( std::apply([](auto x1, auto x2) noexcept {return x1.size() + x2.size();}, std::make_tuple(std::get<0>(A2Dxs), std::get<1>(A2Dxs))) == 12 );
	BOOST_TEST( std::apply([](auto x1, auto x2) noexcept {return x1.size() + x2.size();}, A2Dxs) == 12 );

	// BOOST_TEST( std::apply([](auto x1, auto x2) noexcept {return x1.size() + x2.size();}, A2Dxs) == 12 );

	BOOST_TEST( &A2D() == &(std::apply(A2D, A2Dxs)) );

	// BOOST_TEST( A2Dxs.begin() != A2Dxs.end() );
	BOOST_TEST( A2Dxs.size() == A2D.size() );
	std::tuple<long, long> t1{1, 2};
	std::tuple<long, long> t2{1, 2};
	BOOST_TEST( t1 == t2 );

	BOOST_TEST( A2Dxs.sizes() == A2D.sizes() );

	auto const [ni, nj] = A2Dxs.sizes();
	for(int i = 0; i != ni; ++i) {
		for(int j = 0; j != nj; ++j) {
			auto const [first, second] = A2Dxs[i][j];
			BOOST_TEST( first == i );
			BOOST_TEST( second == j );
		}
		std::cout << std::endl;
	}

	auto const [is, js] = A2Dxs;
	for(auto const i : is) {
		for(auto const j : js) {
			auto const [first, second] = A2Dxs[i][j];
			BOOST_TEST( first == i );
			BOOST_TEST( second == j );
		}
	}

	return boost::report_errors();
}
