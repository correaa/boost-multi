// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#if (__cplusplus > 202002L || (defined(_MSVC_LANG) && _MSVC_LANG > 202002L))  // nvc++ (26.5) reports 202100L for -std=c++23
#if __has_include(<mdspan>)
#include <mdspan>
#endif
#if __has_include(<linalg>)
#include <linalg>
#if (defined(__cpp_lib_linalg) && (__cpp_lib_linalg >= 202311L)) || defined(__NVCOMPILER)  // nvc++ ships <linalg> (Kokkos stdBLAS) without the feature macro
#define BOOST_MULTI_TEST_HAS_LINALG 1
#endif
#endif
#endif

namespace multi = boost::multi;

#ifdef BOOST_MULTI_HAS_MDSPAN
template<
	class MultiArray,
	typename T                   = std::remove_reference_t<typename std::decay_t<MultiArray>::element_cref>,
	multi::dimensionality_type D = std::decay_t<MultiArray>::dimensionality>
auto to_strided_mdspan(MultiArray&& arr) {
	using std::apply;
	auto shape = apply(
		[](auto... sizes) { return std::dextents<std::size_t, D>{static_cast<std::size_t>(sizes)...}; },
		arr.sizes()
	);

	auto strides = apply(
		[](auto... strds) { return std::array<std::size_t, D>{static_cast<std::size_t>(strds)...}; },
		arr.strides()
	);

	return std::mdspan<T, std::dextents<std::size_t, D>, std::layout_stride>{
		arr.base(), std::layout_stride::mapping{shape, strides}
	};
}

auto fun_const(std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mds) -> int const& {
	return mds[0, 0];
}

auto fun(std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mds) -> int& {
	return mds[0, 0];
}

#endif

auto main() -> int {
	{
		multi::array<int, 2> const carr = {
			{ 1,  2,  3,  4},
			{ 5,  6,  7,  8},
			{ 9, 10, 11, 12},
			{13, 14, 15, 16},
		};

		BOOST_TEST( carr[1][1] == 6 );

#ifdef BOOST_MULTI_HAS_MDSPAN
		std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_arr_const = carr;
		BOOST_TEST(( &mdspan_arr_const[0, 0] == &carr[0][0] ));
		fun_const(mdspan_arr_const);
		fun_const(carr);

		auto const& center = carr({1, 3}, {1, 3});
		BOOST_TEST( &center[0][0] == &carr[1][1] );

		std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_center = center;
		BOOST_TEST(( &mdspan_center[0, 0] == &center[0][0] ));
		fun_const(mdspan_center);
		fun_const(center);

		std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_center_rv = carr({1, 3}, {1, 3});
		BOOST_TEST(( &mdspan_center_rv[0, 0] == &carr({1, 3}, {1, 3})[0][0] ));
		fun_const(mdspan_center_rv);
		fun_const(carr({1, 3}, {1, 3}));
#endif
	}
	{
		multi::array<int, 2> arr = {
			{ 1,  2,  3,  4},
			{ 5,  6,  7,  8},
			{ 9, 10, 11, 12},
			{13, 14, 15, 16},
		};

		BOOST_TEST( arr[1][1] == 6 );

#ifdef BOOST_MULTI_HAS_MDSPAN
		std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_arr_const = arr;
		BOOST_TEST(( &mdspan_arr_const[0, 0] == &arr[0][0] ));
		fun_const(mdspan_arr_const);
		fun_const(arr);

		std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_arr = arr;
		BOOST_TEST(( &mdspan_arr[0, 0] == &arr[0][0] ));
		fun_const(mdspan_arr);
		fun_const(arr);

		fun(mdspan_arr);
		fun(arr);

		auto&& center = arr({1, 3}, {1, 3});
		BOOST_TEST( &center[0][0] == &arr[1][1] );

		std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_center_const = center;
		BOOST_TEST(( &mdspan_center_const[0, 0] == &center[0][0] ));
		fun_const(mdspan_center_const);
		fun_const(center);

		std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_center = center;
		BOOST_TEST(( &mdspan_center[0, 0] == &center[0][0] ));
		fun_const(mdspan_center);
		fun_const(center);

		fun(mdspan_center);
		fun(center);

		std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_center_rv = arr({1, 3}, {1, 3});
		BOOST_TEST(( &mdspan_center_rv[0, 0] == &arr({1, 3}, {1, 3})[0][0] ));
		fun_const(mdspan_center_rv);
		fun_const(arr({1, 3}, {1, 3}));

		fun(mdspan_center_rv);
		fun(arr({1, 3}, {1, 3}));

		std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_center_crv = arr({1, 3}, {1, 3});
		BOOST_TEST(( &mdspan_center_crv[0, 0] == &arr({1, 3}, {1, 3})[0][0] ));
		fun_const(mdspan_center_crv);
		fun_const(arr({1, 3}, {1, 3}));

		// fun(mdspan_center_crv);  // doesn't compile, good
		fun(arr({1, 3}, {1, 3}));
#endif
	}
	{
#ifdef BOOST_MULTI_HAS_MDSPAN
		multi::array<int, 2> arr({3, 2}, 5);

		std::mdspan<int, std::dextents<std::size_t, 2>> mdspan_arr = arr;
		BOOST_TEST(( &mdspan_arr[1, 1] == &arr[1, 1] ));
#endif
	}
	{
#if defined(BOOST_MULTI_HAS_MDSPAN) && defined(BOOST_MULTI_TEST_HAS_LINALG)
		multi::array<double, 2> A = {
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
		};
		multi::array<double, 2> B = {
			{ 7.0,  8.0},
			{ 9.0, 10.0},
			{11.0, 12.0},
		};
		multi::array<double, 2> C({2, 2}, 0.0);

		using mdspan2D = std::mdspan<double, std::dims<2>, std::layout_stride>;

		mdspan2D A_mds = A;
		mdspan2D B_mds = B;
		mdspan2D C_mds = C;

		// C = A * B, written through the mdspan into the multi array
		std::linalg::matrix_product(A_mds, B_mds, C_mds);
		BOOST_TEST( C[0][0] ==  58.0 );
		BOOST_TEST( C[0][1] ==  64.0 );
		BOOST_TEST( C[1][0] == 139.0 );
		BOOST_TEST( C[1][1] == 154.0 );

		// in-place scaling is visible in the multi array
		std::linalg::scale(2.0, A_mds);
		BOOST_TEST( A[0][0] == 2.0 );
		BOOST_TEST( A[1][2] == 12.0 );

		// scaling a strided sub-block leaves the rest untouched
		multi::array<double, 2> D({3, 3}, 1.0);
		mdspan2D center_mds = D({1, 3}, {1, 3});
		std::linalg::scale(3.0, center_mds);
		BOOST_TEST( D[0][0] == 1.0 );
		BOOST_TEST( D[1][1] == 3.0 );
		BOOST_TEST( D[2][2] == 3.0 );
		BOOST_TEST( D[0][2] == 1.0 );

		// norm of a strided sub-block
		multi::array<double, 2> E = {
			{9.0, 9.0, 9.0},
			{9.0, 3.0, 0.0},
			{9.0, 0.0, 4.0},
		};
		mdspan2D E_center = E({1, 3}, {1, 3});
		BOOST_TEST( std::linalg::matrix_frob_norm(E_center) == 5.0 );

		// transposed multi view (non-unit leading stride) as operand: G = F^T * F
		multi::array<double, 2> F = {
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
		};
		multi::array<double, 2> G({3, 3}, 0.0);
		mdspan2D Ft_mds = F.transposed();
		mdspan2D F_mds  = F;
		mdspan2D G_mds  = G;
		std::linalg::matrix_product(Ft_mds, F_mds, G_mds);
		BOOST_TEST( G[0][0] == 17.0 );
		BOOST_TEST( G[1][2] == 36.0 );
		BOOST_TEST( G[2][2] == 45.0 );
#endif
	}
	// #ifdef BOOST_MULTI_HAS_MDSPAN
	// 	std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_arr_const = arr;
	// 	BOOST_TEST(( &mdspan_arr_const[0, 0] == &arr[0][0] ));
	// 	BOOST_TEST( &fun(arr) == &arr[0][0] );

	// 	// std::mdspan<int const, std::dextents<std::size_t, 1>, std::layout_stride> mds1D = center[0];
	// 	// BOOST_TEST(( &mds1D[0] == &center[0][0] ));

	// 	std::mdspan<int, std::dextents<std::size_t, 2>, std::layout_stride> mdspan_arr = arr;
	// 	BOOST_TEST(( &mdspan_arr[0, 0] == &arr[0][0] ));
	// #endif

	// 	auto const& center = arr({1, 3}, {1, 3});
	// 	BOOST_TEST( &center[0][0] == &arr[1][1] );

	// #ifdef BOOST_MULTI_HAS_MDSPAN
	// 	std::mdspan<int const, std::dextents<std::size_t, 2>, std::layout_stride> mds = center;
	// 	BOOST_TEST(( &mds[0, 0] == &center[0][0] ));

	// 	BOOST_TEST( &fun_const(center) == &center[0][0] );

	// 	std::mdspan<int const, std::dextents<std::size_t, 1>, std::layout_stride> mds1D = center[0];
	// 	BOOST_TEST(( &mds1D[0] == &center[0][0] ));
	// #endif

	return boost::report_errors();
}
