// Copyright 2018-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>

#if (__cplusplus >= 202002L)
#   include <ranges>
#endif
#include <boost/core/lightweight_test.hpp>

#include <cstddef>
#include <type_traits>

namespace multi = boost::multi;

namespace apl {

namespace {

template<multi::dimensionality_type D, class Extensions = multi::extensions_t<D>>
constexpr auto iota(Extensions const& exts) {
	auto beg = multi::extension_t(std::ptrdiff_t{0}, exts.num_elements()).begin();  // std::views::iota(0, exts.num_elements()).begin();
	return multi::array_ref<typename decltype(beg)::value_type, Extensions::dimensionality, std::remove_const_t<decltype(beg)>>(exts, beg);
}

template<class... Es>
constexpr auto iota(Es... es) {
	return iota<sizeof...(Es)>(multi::extensions_t<static_cast<multi::dimensionality_type>(sizeof...(Es))>{es...});
}

}  // end namespace

namespace symbols {

namespace {
// cppcheck-suppress [syntaxError] -begin
template<class... Es> [[maybe_unused]] auto ι(Es... es) { return iota(es...); }
// cppcheck-suppress [syntaxError] -end
}  // end namespace

}  // end namespace symbols

[[maybe_unused]] constexpr auto const Zilde = iota<1>(multi::extensions_t<1>{std::ptrdiff_t{0}});

namespace symbols {

namespace {
[[maybe_unused]] constexpr auto const& Ɵ = Zilde;  // NOLINT(misc-confusable-identifiers)
#if !defined(_MSC_VER)
[[maybe_unused]] constexpr auto const& θ = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& Ө = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& ϑ = Zilde;  // NOLINT(misc-confusable-identifiers)
[[maybe_unused]] constexpr auto const& Ø = Zilde;

#   if defined(__clang__)
#       pragma clang diagnostic ignored "-Wc99-compat"
#   endif
[[maybe_unused]] constexpr auto const& ϴ = Zilde;  // NOLINT(misc-confusable-identifiers)
#endif

struct underscore_t {
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
};

#if!defined(_MSC_VER)
[[maybe_unused]] constexpr underscore_t _;
#endif

}  // end namespace

}  // end namespace symbols

}  // end namespace apl

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	BOOST_TEST(( apl::iota<2>({2, 3}) == multi::array{{0, 1, 2}, {3, 4, 5}} ));

#if !defined(_MSC_VER)
	BOOST_TEST(( apl::iota(2, 3) == multi::array{{0, 1, 2}, {3, 4, 5}} ));
	BOOST_TEST(( apl::iota(4) == multi::array{0, 1, 2, 3} ));

#   if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	// NOLINTNEXTLINE(google-build-using-namespace)
	using namespace apl::symbols;  // NOLINT(build/namespaces)

	BOOST_TEST(( ι(4)    == _[0, 1, 2, 3] ));
	BOOST_TEST(( ι(2, 3) == _[ _[0, 1, 2], _[3, 4, 5] ] ));
	BOOST_TEST(( Ɵ == ι(0) ));
#   endif
#endif

	return boost::report_errors();
}
