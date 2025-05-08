// Copyright 2018-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#if(__cplusplus >= 202002L)
#include <ranges>
#endif

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

namespace apl {

namespace {

template<multi::dimensionality_type D, class Extensions = multi::extensions_t<D>>
constexpr auto iota(Extensions const& exts) {
	auto beg = multi::extension_t(0L, exts.num_elements()).begin();  // std::views::iota(0, exts.num_elements()).begin();
	return multi::array_ref<typename decltype(beg)::value_type, Extensions::dimensionality, decltype(beg)>(exts, beg);
}

template<class... Es>
constexpr auto iota(Es... es) {
	return iota<sizeof...(Es)>(multi::extensions_t<static_cast<multi::dimensionality_type>(sizeof...(Es))>{es...});
}

// template<multi::dimensionality_type D>
// auto iota(multi::extensions_t<D> const& exts) {
//  auto beg = multi::extension_t(0L, exts.num_elements()).begin();  // std::views::iota(0, exts.num_elements()).begin();
//  return multi::array_ref<typename decltype(beg)::value_type, D, decltype(beg)>(exts, beg);
// }


template<class... Es> auto ι(Es... es) { return iota(es...); }
// #if defined(__clang_major__)
constexpr auto const θ = iota(0L);
// #endif

}  // end anonymous namespace

}  // end namespace apl

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	BOOST_TEST(( apl::iota<2>({2, 3}) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( apl::iota(2, 3) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( apl::iota(4) == multi::array{0, 1, 2, 3} ));

	using apl::ι;
	using apl::θ;

	BOOST_TEST(( ι(2, 3) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( ι(2, 3) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( θ == ι(0) ));

	return boost::report_errors();
}
