// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#if (__cplusplus >= 202302L || (defined(_MSVC_LANG) && _MSVC_LANG > 202002L))
#if __has_include(<mdspan>)
#include <mdspan>
#endif
#endif

namespace multi = boost::multi;

#if defined(__cpp_lib_mdspan) && (__cpp_lib_mdspan >= 202207L)
template<class MultiArray, multi::dimensionality_type D = std::decay_t<MultiArray>::dimensionality>
auto to_strided_mdspan(MultiArray&& arr) {
	using std::apply;
	auto const shape = apply(
		[](auto... sizes) { return std::dextents<std::size_t, D>{static_cast<std::size_t>(sizes)...}; },
		arr.sizes()
	);

	auto const strides = apply(
		[](auto... strds) { return std::array<std::size_t, D>{static_cast<std::size_t>(strds)...}; },
		arr.strides()
	);

	return std::mdspan<int, std::dextents<std::size_t, D>, std::layout_stride>{
		arr.base(), std::layout_stride::mapping{shape, strides}
	};
}
#endif

auto main() -> int {
	multi::array<int, 2> arr = {
		{ 1,  2,  3,  4},
		{ 5,  6,  7,  8},
		{ 9, 10, 11, 12},
		{13, 14, 15, 16}
	};

	auto const& center = arr({1, 3}, {1, 3});
	BOOST_TEST( &center[0][0] == &arr[1][1] );

	#if defined(__cpp_lib_mdspan) && (__cpp_lib_mdspan >= 202207L)
	auto mds = to_strided_mdspan(center);
	BOOS_TEST( &mds[0, 0] == &center[0][0] );
	#endif

	return boost::report_errors();
}
