// Copyright 2018-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#if (__cplusplus >= 202002L)
#   include <ranges>
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

template<class... Es> [[maybe_unused]] auto ι(Es... es) { return iota(es...); }

[[maybe_unused]] constexpr auto const Zilde = iota(0L);

[[maybe_unused]] constexpr auto const& Ɵ = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& θ = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& Ө = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& ϑ = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& Ø = Zilde;

#if defined(__clang__)
#   pragma clang diagnostic ignored "-Wc99-compat"
#endif
[[maybe_unused]] constexpr auto const& ϴ = Zilde;  // NOLINT(misc-confusable-identifiers)

[[maybe_unused]] constexpr struct {
#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	template<class U, class... Us>
	[[maybe_unused]]
#   if (__cpp_multidimensional_subscript >= 202211L)
	static
#   endif
		constexpr auto operator[](U u, Us... us)
#   if !(__cpp_multidimensional_subscript >= 202211L)
			const
#   endif
	{
		if constexpr(std::is_same_v<U, int>) {
			return multi::array<U, 1>{u, us...};
		} else {
			return multi::array<typename U::element_type, U::dimensionality + 1>{u, us...};
		}
	}
#endif
} _;

// template<class T>
// auto oo(std::initializer_list<T> il) {
//  return multi::array{il};
// }
// template<class T>
// auto oo(std::initializer_list<std::initializer_list<T>> il) {
//  return multi::array{il};
// }

}  // end anonymous namespace

}  // end namespace apl

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	BOOST_TEST(( apl::iota<2>({2, 3}) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( apl::iota(2, 3) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( apl::iota(4) == multi::array{0, 1, 2, 3} ));

	using apl::_;
	using apl::θ;
	using apl::ι;

#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)

	BOOST_TEST(( ι(4)    == _[0, 1, 2, 3] ));
	BOOST_TEST(( ι(2, 3) == _[ _[0, 1, 2], _[3, 4, 5] ] ));
	BOOST_TEST(( θ == ι(0) ));

#endif

	return boost::report_errors();
}
