// Copyright 2018-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_LAYOUT_HPP
#define BOOST_MULTI_DETAIL_LAYOUT_HPP

#include <boost/multi/detail/config/NODISCARD.hpp>
#include <boost/multi/detail/config/NO_UNIQUE_ADDRESS.hpp>
#include <boost/multi/detail/index_range.hpp>    // IWYU pragma: export  // for index_extension, extension_t, tuple, intersection, range, operator!=, operator==
#include <boost/multi/detail/operators.hpp>      // IWYU pragma: export  // for equality_comparable
#include <boost/multi/detail/serialization.hpp>  // IWYU pragma: export  // for archive_traits
#include <boost/multi/detail/tuple_zip.hpp>      // IWYU pragma: export  // for get, tuple, tuple_prepend, tail, tuple_prepend_t, ht_tuple
#include <boost/multi/detail/types.hpp>          // IWYU pragma: export  // for dimensionality_type, index, size_type, difference_type, size_t

// #include "types.hpp"

#include <algorithm>         // for max
#include <array>             // for array
#include <cassert>           // for assert
#include <cstddef>           // for size_t, ptrdiff_t, __GLIBCXX__
#include <cstdlib>           // for abs
#include <initializer_list>  // for initializer_list
#include <iterator>
#include <memory>       // for swap
#include <tuple>        // for tuple_element, tuple, tuple_size, tie, make_index_sequence, index_sequence
#include <type_traits>  // for enable_if_t, integral_constant, decay_t, declval, make_signed_t, common_type_t
#include <utility>      // for forward

// clang-format off
namespace boost::multi { template <boost::multi::dimensionality_type D, typename SSize = multi::size_type> struct layout_t; }
namespace boost::multi::detail { template <class ...Ts> class tuple; }
// clang-format on

#if defined(__NVCC__)
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

namespace boost::multi {

template<typename Stride>
struct stride_traits;

template<>
struct stride_traits<std::ptrdiff_t> {
	using category = std::random_access_iterator_tag;
};

template<typename Stride>
struct stride_traits {
	using category = typename Stride::category;
};

template<typename Integer>
struct stride_traits<std::integral_constant<Integer, 1>> {
#if defined(__cplusplus) && (__cplusplus >= 202002L) && (!defined(__clang__) || __clang_major__ != 10)
	using category = std::contiguous_iterator_tag;
#else
	using category = std::random_access_iterator_tag;
#endif
};

namespace detail {

template<class Tuple, std::size_t... Ns>
constexpr auto tuple_tail_impl(Tuple&& tup, std::index_sequence<Ns...> /*012*/) {
	(void)tup;  // workaround bug warning in nvcc
	using boost::multi::detail::get;
	return boost::multi::detail::tuple{std::forward<decltype(get<Ns + 1U>(std::forward<Tuple>(tup)))>(get<Ns + 1U>(std::forward<Tuple>(tup)))...};
}

template<class Tuple>
constexpr auto tuple_tail(Tuple&& t)  // NOLINT(readability-identifier-length) std naming
	-> decltype(tuple_tail_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>> - 1U>())) {
	return tuple_tail_impl(std::forward<Tuple>(t), std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>> - 1U>());
}

}  // end namespace detail

// template<dimensionality_type D, typename SSize=multi::size_type> struct layout_t;

template<dimensionality_type D>
struct extensions_t : boost::multi::detail::tuple_prepend_t<index_extension, typename extensions_t<D - 1>::base_> {
	using base_ = boost::multi::detail::tuple_prepend_t<index_extension, typename extensions_t<D - 1>::base_>;

 private:
	base_ impl_;

 public:
	static constexpr dimensionality_type dimensionality = D;

	extensions_t()    = default;
	using nelems_type = multi::index;

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(multi::size_t size)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : allow terse syntax
		: extensions_t{index_extension{size}} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(index_extension ext1)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{ext1} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 2, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr extensions_t(index_extension ext1, index_extension ext2)
		: base_{ext1, ext2} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 3, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3)
		: base_{ext1, ext2, ext3} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 4, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3, index_extension ext4) noexcept
		: base_{ext1, ext2, ext3, ext4} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 5, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3, index_extension ext4, index_extension ext5)
		: base_{ext1, ext2, ext3, ext4, ext5} {}

	template<class T = void, std::enable_if_t<sizeof(T*) && D == 6, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr extensions_t(index_extension ext1, index_extension ext2, index_extension ext3, index_extension ext4, index_extension ext5, index_extension ext6)
		: base_{ext1, ext2, ext3, ext4, ext5, ext6} {}

	template<class T1, class T = void, class = decltype(base_{tuple<T1>{}}), std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(detail::tuple<T1> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		: base_{std::move(extensions)} {}

	template<class T1, class T = void, class = decltype(base_{::std::tuple<T1>{}}), std::enable_if_t<sizeof(T*) && D == 1, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(::std::tuple<T1> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T = void, class = decltype(base_{tuple<T1, T2>{}}), std::enable_if_t<sizeof(T*) && D == 2, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(detail::tuple<T1, T2> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T = void, class = decltype(base_{::std::tuple<T1, T2>{}}), std::enable_if_t<sizeof(T*) && D == 2, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(::std::tuple<T1, T2> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T = void, class = decltype(base_{tuple<T1, T2, T3>{}}), std::enable_if_t<sizeof(T*) && D == 3, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(tuple<T1, T2, T3> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T = void, class = decltype(base_{::std::tuple<T1, T2, T3>{}}), std::enable_if_t<sizeof(T*) && D == 3, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(::std::tuple<T1, T2, T3> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T = void, class = decltype(base_{tuple<T1, T2, T3, T4>{}}), std::enable_if_t<sizeof(T*) && D == 4, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(tuple<T1, T2, T3, T4> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T = void, class = decltype(base_{::std::tuple<T1, T2, T3, T4>{}}), std::enable_if_t<sizeof(T*) && D == 4, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(::std::tuple<T1, T2, T3, T4> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T5, class T = void, class = decltype(base_{tuple<T1, T2, T3, T4, T5>{}}), std::enable_if_t<sizeof(T*) && D == 5, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(tuple<T1, T2, T3, T4, T5> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) allow terse syntax
		: base_{std::move(extensions)} {}

	template<class T1, class T2, class T3, class T4, class T5, class T = void, class = decltype(base_{::std::tuple<T1, T2, T3, T4, T5>{}}), std::enable_if_t<sizeof(T*) && D == 5, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(::std::tuple<T1, T2, T3, T4, T5> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		: base_{std::move(extensions)} {}

	template<class... Ts>
	constexpr explicit extensions_t(tuple<Ts...> const& tup)
		: extensions_t(tup, std::make_index_sequence<static_cast<std::size_t>(D)>()) {}

	constexpr extensions_t(index_extension const& extension, typename layout_t<D - 1>::extensions_type const& other)
		: extensions_t(multi::detail::ht_tuple(extension, other.base())) {}

	constexpr auto base() const& -> base_ const& { return *this; }  // impl_;}
	constexpr auto base() & -> base_& { return *this; }             // impl_;}

	friend constexpr auto operator*(index_extension const& extension, extensions_t const& self) -> extensions_t<D + 1> {
		// return extensions_t<D + 1>(tuple(extension, self.base()));
		return extensions_t<D + 1>(extension, self);
	}

	friend BOOST_MULTI_HD auto operator==(extensions_t const& self, extensions_t const& other) { return self.base() == other.base(); }
	friend BOOST_MULTI_HD auto operator!=(extensions_t const& self, extensions_t const& other) { return self.base() != other.base(); }

	using index        = multi::index;
	using indices_type = multi::detail::tuple_prepend_t<index, typename extensions_t<D - 1>::indices_type>;

	[[nodiscard]] constexpr auto from_linear(nelems_type const& n) const -> indices_type {
		auto const sub_num_elements = extensions_t<D - 1>{static_cast<base_ const&>(*this).tail()}.num_elements();
#if !(defined(__NVCC__) || defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__))
		assert(sub_num_elements != 0);  // clang hip doesn't allow assert in host device functions
#endif
		return multi::detail::ht_tuple(n / sub_num_elements, extensions_t<D - 1>{static_cast<base_ const&>(*this).tail()}.from_linear(n % sub_num_elements));
	}

	friend constexpr auto operator%(nelems_type idx, extensions_t const& extensions) { return extensions.from_linear(idx); }

	constexpr explicit operator bool() const { return !layout_t<D>{*this}.empty(); }

	template<class... Indices>
	constexpr auto to_linear(index const& idx, Indices const&... rest) const {
		auto const sub_extensions = extensions_t<D - 1>{this->base().tail()};
		return (idx * sub_extensions.num_elements()) + sub_extensions.to_linear(rest...);
	}
	template<class... Indices>
	constexpr auto operator()(index idx, Indices... rest) const { return to_linear(idx, rest...); }

	constexpr auto operator[](index idx) const
		-> decltype(std::declval<base_ const&>()[idx]) {
		return static_cast<base_ const&>(*this)[idx];
	}

	template<class... Indices>
	constexpr auto next_canonical(index& idx, Indices&... rest) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		if(extensions_t<D - 1>{this->base().tail()}.next_canonical(rest...)) {
			++idx;
		}
		if(idx == this->base().head().last()) {
			idx = this->base().head().first();
			return true;
		}
		return false;
	}
	template<class... Indices>
	constexpr auto prev_canonical(index& idx, Indices&... rest) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		if(extensions_t<D - 1>{this->base().tail()}.prev_canonical(rest...)) {
			--idx;
		}
		if(idx < static_cast<index>(this->base().head().first())) {
			idx = static_cast<index>(this->base().head().back());
			return true;
		}
		return false;
	}

	auto size() const { return this->get<0>().size(); }
	auto sizes() const {
		return this->apply([](auto const&... xs) { return multi::detail::mk_tuple(xs.size()...); });
	}

 private:
	template<class Archive, std::size_t... I>
	void serialize_impl_(Archive& arxiv, std::index_sequence<I...> /*unused012*/) {
		using boost::multi::detail::get;
		(void)std::initializer_list<unsigned>{(arxiv & multi::archive_traits<Archive>::make_nvp("extension", get<I>(this->base())), 0U)...};
	}

 public:
	template<class Archive>
	void serialize(Archive& arxiv, unsigned int const /*version*/) {
		serialize_impl_(arxiv, std::make_index_sequence<static_cast<std::size_t>(D)>());
	}

 private:
	template<class Array, std::size_t... I, typename = decltype(base_{boost::multi::detail::get<I>(std::declval<Array const&>())...})>
	constexpr extensions_t(Array const& tup, std::index_sequence<I...> /*unused012*/)
		: base_{boost::multi::detail::get<I>(tup)...} {}

	static constexpr auto multiply_fold_() -> size_type { return static_cast<size_type>(1U); }
	static constexpr auto multiply_fold_(size_type const& size) -> size_type { return size; }
	template<class... As>
	static constexpr auto multiply_fold_(size_type const& size, As const&... rest) -> size_type { return size * static_cast<size_type>(multiply_fold_(rest...)); }

	template<std::size_t... I> constexpr auto num_elements_impl_(std::index_sequence<I...> /*unused012*/) const -> size_type {
		using boost::multi::detail::get;
		return static_cast<size_type>(multiply_fold_(static_cast<size_type>(get<I>(this->base()).size())...));
	}

 public:
	constexpr auto num_elements() const -> size_type {
		return static_cast<size_type>(num_elements_impl_(std::make_index_sequence<static_cast<std::size_t>(D)>()));
	}
	friend constexpr auto intersection(extensions_t const& self, extensions_t const& other) -> extensions_t {
		using boost::multi::detail::get;
		return extensions_t{
			multi::detail::ht_tuple(
				index_extension{intersection(get<0>(self.base()), get<0>(other.base()))},
				intersection(extensions_t<D - 1>{self.base().tail()}, extensions_t<D - 1>{other.base().tail()}).base()
			)
		};
	}

	template<std::size_t Index, std::enable_if_t<(Index < D), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(extensions_t const& self) -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(self.base());
	}

	template<std::size_t Index, std::enable_if_t<(Index < D), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr auto get() const -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(this->base());
	}

	template<class Fn>
	constexpr auto apply(Fn&& fn) const -> decltype(auto) {
		return std::apply(std::forward<Fn>(fn), this->base());
	}
};

template<> struct extensions_t<0> : tuple<> {
	using base_ = tuple<>;

 private:
	// base_ impl_;

 public:
	static constexpr dimensionality_type dimensionality = 0;  // TODO(correaa): consider deprecation

	using rank = std::integral_constant<dimensionality_type, 0>;

	using nelems_type = index;

	explicit extensions_t(tuple<> const& tup)
		: base_{tup} {}

	extensions_t() = default;

	constexpr auto base() const& -> base_ const& { return *this; }
	constexpr auto base() & -> base_& { return *this; }

	template<class Archive> static void serialize(Archive& /*ar*/, unsigned /*version*/) { /*noop*/ }

	static constexpr auto num_elements() /*const*/ -> size_type { return 1; }

	using indices_type = tuple<>;

	[[nodiscard]] static constexpr auto from_linear(nelems_type const& n) /*const*/ -> indices_type {
		assert(n == 0);
		(void)n;  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : constexpr function
		return indices_type{};
	}
	friend constexpr auto operator%(nelems_type const& n, extensions_t const& /*s*/) -> tuple<> { return /*s.*/ from_linear(n); }

	static constexpr auto to_linear() /*const*/ -> difference_type { return 0; }
	constexpr auto        operator()() const { return to_linear(); }

	constexpr void operator[](index) const = delete;

	static constexpr auto next_canonical() /*const*/ -> bool { return true; }
	static constexpr auto prev_canonical() /*const*/ -> bool { return true; }

	friend constexpr auto intersection(extensions_t const& /*x1*/, extensions_t const& /*x2*/) -> extensions_t { return {}; }

	constexpr BOOST_MULTI_HD auto operator==(extensions_t const& /*other*/) const { return true; }
	constexpr BOOST_MULTI_HD auto operator!=(extensions_t const& /*other*/) const { return false; }

	template<std::size_t Index>  // TODO(correaa) = detele ?
	friend constexpr auto get(extensions_t const& self) -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(self.base());
	}

	template<std::size_t Index>  // TODO(correaa) = detele ?
	constexpr auto get() const -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(this->base());
	}
};

template<> struct extensions_t<1> : tuple<multi::index_extension> {
	using base_ = tuple<multi::index_extension>;

	static constexpr auto dimensionality = 1;  // TODO(correaa): consider deprecation

	class elements_t {
		multi::index_range rng_;

	 public:
		class iterator : multi::index_range::iterator {
			friend class elements_t;  // enclosing class is friend automatically?
			explicit iterator(multi::index_range::iterator it)
				: multi::index_range::iterator{it} {}

			auto base_() const -> multi::index_range::iterator const& { return *this; }
			auto base_() -> multi::index_range::iterator& { return *this; }

		 public:
			using value_type      = std::tuple<multi::index_range::iterator::value_type>;
			using difference_type = multi::index_range::iterator::difference_type;
			// using pointer = void;
			// using reference = value_type;

			auto operator*() const { return value_type(*base_()); }

			auto operator++() -> iterator& {
				++base_();
				return *this;
			}
			auto operator--() -> iterator& {
				--base_();
				return *this;
			}

			auto operator+=(difference_type n) -> iterator& {
				base_() += n;
				return *this;
			}
			auto operator-=(difference_type n) -> iterator& {
				base_() -= n;
				return *this;
			}

			auto operator+(difference_type n) const { return iterator{*this} += n; }
			auto operator-(difference_type n) const { return iterator{*this} -= n; }

			auto operator==(iterator const& other) const { return base_() == other.base_(); }
			auto operator!=(iterator const& other) const { return base_() != other.base_(); }

			auto operator[](difference_type n) const { return *((*this) + n); }
		};
		// using const_iterator = iterator;

		auto begin() const { return iterator{rng_.begin()}; }
		auto end() const { return iterator{rng_.end()}; }

		explicit elements_t(multi::index_range rng)
			: rng_{rng} {}
	};

	auto elements() const {
		using std::get;
		// auto rng = get<0>(static_cast<tuple<multi::index_extension> const&>(*this));
		return elements_t{get<0>(static_cast<tuple<multi::index_extension> const&>(*this))};
	}

	using nelems_type = index;

	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax (compatible with std::vector(int) constructor
	constexpr extensions_t(multi::size_t size)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		: base_(multi::index_extension{0, size}) {}

	template<class T1>
	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int>  // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(tuple<T1> extensions)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		: base_{static_cast<multi::index_extension>(extensions.head())} {}

	// cppcheck-suppress noExplicitConstructor ; to allow passing tuple<int, int> // NOLINTNEXTLINE(runtime/explicit)
	constexpr extensions_t(multi::index_extension const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		: base_{other} {}

	constexpr explicit extensions_t(base_ tup)
		: base_{tup} {}

	extensions_t() = default;

	constexpr auto base() const& -> base_ const& { return *this; }
	constexpr auto base() & -> base_& { return *this; }

	BOOST_MULTI_HD constexpr auto operator==(extensions_t const& other) const -> bool { return base() == other.base(); }  // when compiling as cuda code, this needs --expt-relaxed-constexpr
	BOOST_MULTI_HD constexpr auto operator!=(extensions_t const& other) const -> bool { return base() != other.base(); }

	constexpr auto num_elements() const -> size_type { return this->base().head().size(); }

	using indices_type = multi::detail::tuple<multi::index>;

	[[nodiscard]] constexpr auto from_linear(nelems_type const& n) const -> indices_type {  // NOLINT(readability-convert-member-functions-to-static) TODO(correaa)
		return indices_type{n};
	}

	friend constexpr auto operator%(nelems_type idx, extensions_t const& extensions)
		-> multi::detail::tuple<multi::index> {
		return extensions.from_linear(idx);
	}

	static constexpr auto to_linear(index const& idx) -> difference_type { return idx; }

	constexpr auto operator[](index idx) const {
		using std::get;
		return multi::detail::tuple<multi::index>{get<0>(this->base())[idx]};
	}
	constexpr auto operator()(index idx) const { return idx; }
	// constexpr auto operator()(index const& /*idx*/) const -> difference_type { return to_linear(42); }

	template<class... Indices>
	constexpr auto next_canonical(index& idx) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		// using boost::multi::detail::get;
		if(idx == ::boost::multi::detail::get<0>(this->base()).back()) {
			idx = ::boost::multi::detail::get<0>(this->base()).first();
			return true;
		}
		++idx;
		return false;
	}
	constexpr auto prev_canonical(index& idx) const -> bool {  // NOLINT(google-runtime-references) idx is mutated
		using boost::multi::detail::get;
		if(idx == get<0>(this->base()).first()) {
			// idx = 42;  // TODO(correaa) implement and test
			idx = get<0>(this->base()).back();
			return true;
		}
		--idx;
		return false;
	}

	friend auto intersection(extensions_t const& self, extensions_t const& other) {
		return extensions_t{
			intersection(
				boost::multi::detail::get<0>(self.base()),
				boost::multi::detail::get<0>(other.base())
			)
		};
	}
	template<class Archive>
	void serialize(Archive& arxiv, unsigned /*version*/) {
		using boost::multi::detail::get;
		auto&  extension_ = get<0>(this->base());
		arxiv& multi::archive_traits<Archive>::make_nvp("extension", extension_);
	}

	template<std::size_t Index, std::enable_if_t<(Index < 1), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	constexpr auto get() const -> decltype(auto) {                       // -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(this->base());
	}

	template<std::size_t Index, std::enable_if_t<(Index < 1), int> = 0>      // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(extensions_t const& self) -> decltype(auto) {  // -> typename std::tuple_element<Index, base_>::type {
		using boost::multi::detail::get;
		return get<Index>(self.base());
	}
};

template<dimensionality_type D> using iextensions = extensions_t<D>;

template<boost::multi::dimensionality_type D>
constexpr auto array_size_impl(boost::multi::extensions_t<D> const&)
	-> std::integral_constant<std::size_t, static_cast<std::size_t>(D)>;

extensions_t(multi::size_t) -> extensions_t<1>;
extensions_t(multi::size_t, multi::size_t) -> extensions_t<2>;
extensions_t(multi::size_t, multi::size_t, multi::size_t) -> extensions_t<3>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<4>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<5>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<6>;
extensions_t(multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t, multi::size_t) -> extensions_t<7>;

}  // end namespace boost::multi

// Some versions of Clang throw warnings that stl uses class std::tuple_size instead
// of struct std::tuple_size like it should be
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

template<boost::multi::dimensionality_type D>
struct std::tuple_size<boost::multi::extensions_t<D>>  // NOLINT(cert-dcl58-cpp) to implement structured binding
	: std::integral_constant<std::size_t, static_cast<std::size_t>(D)> {};

template<>
struct std::tuple_element<0, boost::multi::extensions_t<0>> {  // NOLINT(cert-dcl58-cpp) to implement structured binding
	using type = void;
};

template<std::size_t Index, boost::multi::dimensionality_type D>
struct std::tuple_element<Index, boost::multi::extensions_t<D>> {  // NOLINT(cert-dcl58-cpp) to implement structured binding
	using type = typename std::tuple_element<Index, typename boost::multi::extensions_t<D>::base_>::type;
};

namespace std {  // NOLINT(cert-dcl58-cpp)

// clang wants tuple_size to be a class, not a struct with -Wmismatched-tags
#if !defined(__GLIBCXX__) || (__GLIBCXX__ <= 20190406)
template<> struct tuple_size<boost::multi::extensions_t<0>> : std::integral_constant<boost::multi::dimensionality_type, 0> {};
template<> struct tuple_size<boost::multi::extensions_t<1>> : std::integral_constant<boost::multi::dimensionality_type, 1> {};
template<> struct tuple_size<boost::multi::extensions_t<2>> : std::integral_constant<boost::multi::dimensionality_type, 2> {};
template<> struct tuple_size<boost::multi::extensions_t<3>> : std::integral_constant<boost::multi::dimensionality_type, 3> {};
template<> struct tuple_size<boost::multi::extensions_t<4>> : std::integral_constant<boost::multi::dimensionality_type, 4> {};
template<> struct tuple_size<boost::multi::extensions_t<5>> : std::integral_constant<boost::multi::dimensionality_type, 5> {};
#else
template<> class tuple_size<boost::multi::extensions_t<0>> : public std::integral_constant<boost::multi::dimensionality_type, 0> {};
template<> class tuple_size<boost::multi::extensions_t<1>> : public std::integral_constant<boost::multi::dimensionality_type, 1> {};
template<> class tuple_size<boost::multi::extensions_t<2>> : public std::integral_constant<boost::multi::dimensionality_type, 2> {};
template<> class tuple_size<boost::multi::extensions_t<3>> : public std::integral_constant<boost::multi::dimensionality_type, 3> {};
template<> class tuple_size<boost::multi::extensions_t<4>> : public std::integral_constant<boost::multi::dimensionality_type, 4> {};
template<> class tuple_size<boost::multi::extensions_t<5>> : public std::integral_constant<boost::multi::dimensionality_type, 5> {};
#endif

#if !defined(_MSC_VER) && (!defined(__GLIBCXX__) || (__GLIBCXX__ <= 20240707))
template<std::size_t N, ::boost::multi::dimensionality_type D>
constexpr auto get(::boost::multi::extensions_t<D> const& tp)  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get, gcc workaround
	-> decltype(tp.template get<N>()) {
	return tp.template get<N>();
}

// template<std::size_t N>  // , boost::multi::dimensionality_type D>
// constexpr auto get(boost::multi::extensions_t<2> const& tp)  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get, gcc workaround
// // ->decltype(tp.template get<N>()) {
// -> decltype(auto) {
//  return tp.template get<N>(); }

template<std::size_t N, ::boost::multi::dimensionality_type D>
constexpr auto get(::boost::multi::extensions_t<D>& tp)  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get, gcc workaround
	-> decltype(tp.template get<N>()) {
	return tp.template get<N>();
}

template<std::size_t N, boost::multi::dimensionality_type D>
constexpr auto get(::boost::multi::extensions_t<D>&& tp)  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple get, gcc workaround
	-> decltype(std::move(tp).template get<N>()) {
	return std::move(tp).template get<N>();
}
#endif

template<typename Fn, boost::multi::dimensionality_type D>
constexpr auto
apply(Fn&& fn, boost::multi::extensions_t<D> const& xs) noexcept -> decltype(auto) {  // NOLINT(cert-dcl58-cpp) I have to specialize std::apply as a workaround
	return xs.apply(std::forward<Fn>(fn));
}

}  // end namespace std

namespace boost::multi {

struct monostate : equality_comparable<monostate> {
	friend BOOST_MULTI_HD constexpr auto operator==(monostate const& /*self*/, monostate const& /*other*/) { return true; }
};

template<typename SSize = multi::index>
class stride_t {
	difference_type stride_;

 public:
	// using difference_type = SSize;

	constexpr auto operator()() const -> difference_type { return stride_; }

	template<class Ptr>
	constexpr auto operator()(Ptr ptr) const -> Ptr { return ptr + stride_; }

	using category = std::random_access_iterator_tag;
};

template<typename SSize = multi::index>
class contiguous_stride_t {
 public:
	// using difference_type = SSize;

	constexpr auto operator()() const -> SSize { return 1; }

	template<class Ptr>
	constexpr auto operator()(Ptr const& ptr) const -> Ptr { return ptr + 1; }

#if (__cplusplus >= 202002L)
	using category = std::random_access_iterator_tag;  // std::contiguous_iterator_tag;
#else
	using category = std::random_access_iterator_tag;
#endif
};

using multi::detail::tuple;

template<typename SSize = multi::index>
class contiguous_layout {

 public:
	using dimensionality_type                           = multi::dimensionality_t;
	using rank                                          = std::integral_constant<dimensionality_type, 1>;
	static constexpr auto                rank_v         = rank::value;
	static constexpr dimensionality_type dimensionality = rank_v;

	using size_type  = SSize;
	using sizes_type = typename boost::multi::detail::tuple<size_type>;

	using difference_type = SSize;

	using index           = size_type;
	using index_range     = multi::range<index>;
	using index_extension = multi::extension_t<index>;

	using indexes = tuple<index>;

	using extension_type  = multi::extension_t<index>;
	using extensions_type = multi::extensions_t<1>;

	using stride_type  = std::integral_constant<int, 1>;
	using strides_type = typename boost::multi::detail::tuple<stride_type>;

	using offset_type = std::integral_constant<int, 0>;

	using nelems_type = SSize;

	using sub_type = layout_t<0, SSize>;

 private:
	// BOOST_MULTI_NO_UNIQUE_ADDRESS sub_type sub_;
	// BOOST_MULTI_NO_UNIQUE_ADDRESS stride_type stride_;
	size_type nelems_;

	template<std::size_t N, class Tup>
	static constexpr auto get_(Tup&& tup) {
		using std::get;
		return get<N>(std::forward<Tup>(tup));
	}

 public:
	constexpr explicit contiguous_layout(multi::extensions_t<1> xs)
		: nelems_{get_<0>(xs).size()} {}

	BOOST_MULTI_HD constexpr contiguous_layout(
		sub_type /*sub*/,
		stride_type /*stride*/,
		offset_type /*offset*/,
		nelems_type nelems
	)
		: /*sub_{sub}, stride_{} offset_{},*/ nelems_{nelems} {}

 private:
	constexpr auto at_aux_(index /*idx*/) const {
		return sub_type{};  // sub_.sub_, sub_.stride_, sub_.offset_ + offset_ + (idx*stride_), sub_.nelems_}();
	}

 public:
	constexpr auto operator[](index idx) const { return at_aux_(idx); }

	template<typename... Indices>
	constexpr auto operator()(index idx, Indices... rest) const { return operator[](idx)(rest...); }
	constexpr auto operator()(index idx) const { return at_aux_(idx); }
	constexpr auto operator()() const { return *this; }

	constexpr auto stride() const { return std::integral_constant<int, 1>{}; }
	constexpr auto offset() const { return std::integral_constant<int, 0>{}; }
	constexpr auto extension() const { return extension_type{0, nelems_}; }

	constexpr auto num_elements() const { return nelems_; }

	constexpr auto size() const { return nelems_; }
	constexpr auto sizes() const { return sizes_type{size()}; }

	constexpr auto nelems() const { return nelems_; }

	constexpr auto extensions() const { return multi::extensions_t<1>{extension()}; }

	constexpr auto is_empty() const -> bool { return nelems_ == 0; }

	BOOST_MULTI_NODISCARD("empty checks for emptyness, it performs no action. Use `is_empty()` instead")
	constexpr auto empty() const { return is_empty(); }

	constexpr auto sub() const { return layout_t<0, SSize>{}; }

	constexpr auto is_compact() const { return std::true_type{}; }

	BOOST_MULTI_HD constexpr auto drop(difference_type count) const {
		assert(count <= this->size());

		return contiguous_layout{
			this->sub(),
			this->stride(),
			this->offset(),
			this->stride() * (this->size() - count)
		};
	}

	BOOST_MULTI_HD constexpr auto slice(index first, index last) const {
		return contiguous_layout{
			this->sub(),
			this->stride(),
			this->offset(),
			(this->is_empty()) ? 0 : this->nelems() / this->size() * (last - first)
		};
	}
};

template<dimensionality_type D>
struct bilayout {
	using size_type       = multi::size_t;  // SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index           = difference_type;

	using stride1_type    = difference_type;
	using stride2_type    = difference_type;
	// using bistride_type = std::pair<index, index>;
	using sub_type      = layout_t<D - 1>;

	using dimensionality_type    = typename sub_type::dimensionality_type;
	using rank                   = std::integral_constant<dimensionality_type, sub_type::rank::value + 1>;
	constexpr static auto rank_v = rank::value;

	constexpr static auto dimensionality() { return rank_v; }

 private:
	stride1_type stride1_;
	size_type nelems1_;
	stride2_type stride2_;
	size_type nelems2_;
	sub_type  sub_;

 public:
	bilayout(
		stride1_type stride1,  // NOLINT(bugprone-easily-swappable-parameters)
		size_type    nelems1,
		stride2_type stride2,  // NOLINT(bugprone-easily-swappable-parameters)
		size_type    nelems2,
		sub_type     sub
	)
	: stride1_{stride1}, nelems1_{nelems1}
	, stride2_{stride2}, nelems2_{nelems2}
	, sub_{std::move(sub)} {}

	using offset_type     = std::ptrdiff_t;
	using stride_type     = std::pair<stride1_type, stride2_type>;
	using index_range     = void;
	using strides_type    = void;
	using extension_type  = void;
	using extensions_type = void;
	using sizes_type      = void;
	using indexes         = void;

	// auto stride() const = delete;
	auto stride() const {
		class stride_t {
			stride1_type stride1_;
			stride2_type stride2_;
			size_type nelems2_;

		 public:
			explicit stride_t(stride1_type stride1, stride2_type stride2, size_type size)  // NOLINT(bugprone-easily-swappable-parameters)
				: stride1_{stride1}, stride2_{stride2}, nelems2_{size} {}
			auto operator*(std::ptrdiff_t nn) const { return stride_t{stride1_, nn * stride2_, nelems2_}; }
			auto operator-(offset_type /*unused*/) const { return *this; }
#if (defined(__clang__) && (__clang_major__ >= 16)) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
			auto operator+(double* ptr) { return ptr + (stride2_ % nelems2_) + ((stride2_/nelems2_)*stride1_); }  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic,clang-diagnostic-unsafe-buffer-usage)
#if (defined(__clang__) && (__clang_major__ >= 16)) && !defined(__INTEL_LLVM_COMPILER)
#pragma clang diagnostic pop
#endif
		};
		return stride_t{stride1_, stride2_, nelems2_};
	}
	auto num_elements() const = delete;
	auto offset() const { return offset_type{}; }
	auto size() const       { return (nelems2_ / stride2_) * (nelems1_ / stride1_); }
	auto nelems() const     = delete;
	void extension() const  = delete;
	auto extensions() const = delete;
	auto is_empty() const   = delete;
	auto empty() const      = delete;
	auto sub() const        = delete;
	auto sizes() const      = delete;

	auto is_compact() const = delete;

	using index_extension = multi::index_extension;
};

template<dimensionality_type D, typename SSize>
struct layout_t
	: multi::equality_comparable<layout_t<D, SSize>> {
	auto flatten() const {
		return bilayout<D - 1>{
			stride(),
			nelems(),
			sub().stride(),
			sub().nelems(),
			sub().sub()
		};
	}

	using dimensionality_type = multi::dimensionality_type;
	using rank                = std::integral_constant<dimensionality_type, D>;

	using sub_type        = layout_t<D - 1>;
	using size_type       = SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index           = difference_type;

	using index_extension = multi::index_extension;
	using index_range     = multi::range<index>;

	using stride_type = index;
	using offset_type = index;
	using nelems_type = index;

	using strides_type = typename boost::multi::detail::tuple_prepend<stride_type, typename sub_type::strides_type>::type;
	using offsets_type = typename boost::multi::detail::tuple_prepend<offset_type, typename sub_type::offsets_type>::type;
	using nelemss_type = typename boost::multi::detail::tuple_prepend<nelems_type, typename sub_type::nelemss_type>::type;

	using extension_type = index_extension;  // not index_range!

	using extensions_type = extensions_t<rank::value>;
	using sizes_type      = typename boost::multi::detail::tuple_prepend<size_type, typename sub_type::sizes_type>::type;

	using indexes = typename boost::multi::detail::tuple_prepend<index, typename sub_type::indexes>::type;

	static constexpr dimensionality_type rank_v         = rank::value;
	static constexpr dimensionality_type dimensionality = rank_v;  // TODO(correaa): consider deprecation

	[[deprecated("for compatibility with Boost.MultiArray, use static `dimensionality` instead")]]
	static constexpr auto num_dimensions() { return dimensionality; }  // NOSONAR(cpp:S1133)

	friend constexpr auto dimensionality(layout_t const& /*self*/) { return rank_v; }

 private:
	sub_type    sub_;
	stride_type stride_;  // =  1;  // or std::numeric_limits<stride_type>::max()?
	offset_type offset_;
	nelems_type nelems_;

	template<dimensionality_type, typename> friend struct layout_t;

 public:
	layout_t() = default;

	template<
		class OtherLayout,
		class = decltype(sub_type{std::declval<OtherLayout const&>().sub()}),
		class = decltype(stride_type{std::declval<OtherLayout const&>().stride()}),
		class = decltype(offset_type{std::declval<OtherLayout const&>().offset()}),
		class = decltype(nelems_type{std::declval<OtherLayout const&>().nelems()})>
	BOOST_MULTI_HD constexpr explicit layout_t(OtherLayout const& other)
		: sub_{other.sub()}
		, stride_{other.stride()}
		, offset_{other.offset()}
		, nelems_{other.nelems()} {}

	BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& extensions)
		: sub_{std::apply([](auto const&... subexts) { return multi::extensions_t<D - 1>{subexts...}; }, detail::tail(extensions.base()))}
		, stride_{sub_.num_elements() ? sub_.num_elements() : 1}
		, offset_{boost::multi::detail::get<0>(extensions.base()).first() * stride_}
		, nelems_{boost::multi::detail::get<0>(extensions.base()).size() * sub().num_elements()} {}

	BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& extensions, strides_type const& strides)
		: sub_{std::apply([](auto const&... subexts) { return multi::extensions_t<D - 1>{subexts...}; }, detail::tail(extensions.base())), detail::tail(strides)}
		, stride_{boost::multi::detail::get<0>(strides)}
		, offset_{boost::multi::detail::get<0>(extensions.base()).first() * stride_}
		, nelems_{boost::multi::detail::get<0>(extensions.base()).size() * sub().num_elements()} {}

	BOOST_MULTI_HD constexpr explicit layout_t(sub_type const& sub, stride_type stride, offset_type offset, nelems_type nelems)  // NOLINT(bugprone-easily-swappable-parameters)
		: sub_{sub}
		, stride_{stride}
		, offset_{offset}
		, nelems_{nelems} {}

	BOOST_MULTI_HD constexpr explicit layout_t(sub_type const& sub, stride_type stride, offset_type offset /*, nelems_type nelems*/)  // NOLINT(bugprone-easily-swappable-parameters)
		: sub_{sub}
		, stride_{stride}
		, offset_{offset} /*, nelems_{nelems}*/ {}  // this leaves nelems_ uninitialized

	constexpr auto origin() const { return sub_.origin() - offset_; }

 private:
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"
#endif

	constexpr auto at_aux_(index idx) const {
		return sub_type{sub_.sub_, sub_.stride_, sub_.offset_ + offset_ + (idx * stride_), sub_.nelems_}();
	}

 public:
	constexpr auto operator[](index idx) const { return at_aux_(idx); }

	template<typename... Indices>
	constexpr auto operator()(index idx, Indices... rest) const { return operator[](idx)(rest...); }
	constexpr auto operator()(index idx) const { return at_aux_(idx); }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) can it be returned by reference?
#endif

	constexpr auto operator()() const { return *this; }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

	BOOST_MULTI_HD constexpr auto        sub() & -> sub_type& { return sub_; }
	BOOST_MULTI_HD constexpr auto        sub() const& -> sub_type const& { return sub_; }
	friend BOOST_MULTI_HD constexpr auto sub(layout_t const& self) -> sub_type const& { return self.sub(); }

	BOOST_MULTI_HD constexpr auto        nelems() & -> nelems_type& { return nelems_; }
	BOOST_MULTI_HD constexpr auto        nelems() const& -> nelems_type const& { return nelems_; }
	friend BOOST_MULTI_HD constexpr auto nelems(layout_t const& self) -> nelems_type const& { return self.nelems(); }

	constexpr BOOST_MULTI_HD auto nelems(dimensionality_type dim) const { return (dim != 0) ? sub_.nelems(dim - 1) : nelems_; }

	friend BOOST_MULTI_HD constexpr auto operator==(layout_t const& self, layout_t const& other) -> bool {
		return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) == std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	friend BOOST_MULTI_HD constexpr auto operator!=(layout_t const& self, layout_t const& other) -> bool {
		return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) != std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	constexpr BOOST_MULTI_HD auto operator<(layout_t const& other) const -> bool {
		return std::tie(sub_, stride_, offset_, nelems_) < std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	constexpr auto reindex() const { return *this; }
	constexpr auto reindex(index idx) const {
		return layout_t{
			sub(),
			stride(),
			idx * stride(),
			nelems()
		};
	}
	template<class... Indexes>
	constexpr auto reindexed(index first, Indexes... idxs) const {
		return ((reindexed(first).rotate()).reindexed(idxs...)).unrotate();
	}

	// template<class... Indices>
	// constexpr auto reindex(index idx, Indices... rest) const {
	//  return reindex(idx).rotate().ceindex(rest...).unrotate();
	// }

	constexpr auto        num_elements() const noexcept -> size_type { return size() * sub_.num_elements(); }
	friend constexpr auto num_elements(layout_t const& self) noexcept -> size_type { return self.num_elements(); }

	constexpr auto        is_empty() const noexcept { return nelems_ == 0; }  // mull-ignore: cxx_eq_to_ne
	friend constexpr auto is_empty(layout_t const& self) noexcept { return self.is_empty(); }

	constexpr auto empty() const noexcept { return is_empty(); }

	friend constexpr auto size(layout_t const& self) noexcept -> size_type { return self.size(); }
	constexpr auto        size() const noexcept -> size_type {
        if(nelems_ == 0) {
            return 0;
        }
        // BOOST_MULTI_ACCESS_ASSERT(stride_);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
        // if(nelems_ != 0) {MULTI_ACCESS_ASSERT(stride_ != 0);}
        // return nelems_ == 0?0:nelems_/stride_;
        // assert(stride_ != 0);
        return nelems_ / stride_;
	}

	constexpr BOOST_MULTI_HD auto stride() -> stride_type& { return stride_; }
	constexpr BOOST_MULTI_HD auto stride() const -> stride_type const& { return stride_; }

	friend BOOST_MULTI_HD constexpr auto stride(layout_t const& self) -> index { return self.stride(); }

	BOOST_MULTI_HD constexpr auto        strides() const -> strides_type { return strides_type{stride(), sub_.strides()}; }
	friend BOOST_MULTI_HD constexpr auto strides(layout_t const& self) -> strides_type { return self.strides(); }

	constexpr BOOST_MULTI_HD auto        offset(dimensionality_type dim) const -> index { return (dim != 0) ? sub_.offset(dim - 1) : offset_; }
	BOOST_MULTI_HD constexpr auto        offset() const -> index { return offset_; }
	friend BOOST_MULTI_HD constexpr auto offset(layout_t const& self) -> index { return self.offset(); }
	constexpr BOOST_MULTI_HD auto        offsets() const { return boost::multi::detail::tuple{offset(), sub_.offsets()}; }
	constexpr BOOST_MULTI_HD auto        nelemss() const { return boost::multi::detail::tuple{nelems(), sub_.nelemss()}; }

	constexpr auto base_size() const {
		using std::max;
		return max(nelems_, sub_.base_size());
	}

	constexpr auto        is_compact() const& { return base_size() == num_elements(); }
	friend constexpr auto is_compact(layout_t const& self) { return self.is_compact(); }

	constexpr auto        shape() const& -> decltype(auto) { return sizes(); }
	friend constexpr auto shape(layout_t const& self) -> decltype(auto) { return self.shape(); }

	constexpr BOOST_MULTI_HD auto sizes() const noexcept { return multi::detail::ht_tuple(size(), sub_.sizes()); }

	friend constexpr auto        extension(layout_t const& self) { return self.extension(); }
	[[nodiscard]] constexpr auto extension() const -> extension_type {
		if(nelems_ == 0) {
			return index_extension{};
		}
		// assert(stride_ != 0);  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		assert(offset_ % stride_ == 0);
		assert(nelems_ % stride_ == 0);
		return index_extension{offset_ / stride_, (offset_ + nelems_) / stride_};
	}

	constexpr auto        extensions() const { return extensions_type{multi::detail::ht_tuple(extension(), sub_.extensions().base())}; }  // tuple_cat(make_tuple(extension()), sub_.extensions().base())};}
	friend constexpr auto extensions(layout_t const& self) -> extensions_type { return self.extensions(); }

	[[deprecated("use get<d>(m.extensions()")]]  // TODO(correaa) redeprecate, this is commented to give a smaller CI output
	constexpr auto
	extension(dimensionality_type dim) const {
		return std::apply([](auto... extensions) { return std::array<index_extension, static_cast<std::size_t>(D)>{extensions...}; }, extensions().base()).at(static_cast<std::size_t>(dim));
	}  // cppcheck-suppress syntaxError ; bug in cppcheck 2.14
	   //  [[deprecated("use get<d>(m.strides())  ")]]  // TODO(correaa) redeprecate, this is commented to give a smaller CI output
	constexpr auto stride(dimensionality_type dim) const {
		return std::apply([](auto... strides) { return std::array<stride_type, static_cast<std::size_t>(D)>{strides...}; }, strides()).at(static_cast<std::size_t>(dim));
	}
	//  [[deprecated("use get<d>(m.sizes())    ")]]  // TODO(correaa) redeprecate, this is commented to give a smaller CI output
	//  constexpr auto size     (dimensionality_type dim) const {return std::apply([](auto... sizes     ) {return std::array<size_type      , static_cast<std::size_t>(D)>{sizes     ...};}, sizes     ()       ).at(static_cast<std::size_t>(dim));}

	BOOST_MULTI_HD constexpr auto drop(difference_type count) const {
		assert(count <= this->size());

		return layout_t{
			this->sub(),
			this->stride(),
			this->offset(),
			this->stride() * (this->size() - count)
		};
	}

	BOOST_MULTI_HD constexpr auto slice(index first, index last) const {
		return layout_t{
			this->sub(),
			this->stride(),
			this->offset(),
			(this->is_empty()) ? 0 : this->nelems() / this->size() * (last - first)
		};
	}

	template<typename Size>
	constexpr auto partition(Size const& count) -> layout_t& {
		stride_ *= count;
		nelems_ *= count;
		sub_.partition(count);
		return *this;
	}

	template<class TT>
	constexpr static void ce_swap(TT& t1, TT& t2) {
		TT tmp = std::move(t1);
		t1     = std::move(t2);
		t2     = tmp;
	}

	constexpr auto transpose() const {
		return layout_t(
			sub_type(
				sub().sub(),
				stride(),
				offset(),
				nelems()
			),
			sub().stride(),
			sub().offset(),
			sub().nelems()
		);
	}

	// constexpr auto transpose() -> layout_t& {
	//  // using std::swap;
	//  ce_swap(stride_, sub_.stride_);
	//  ce_swap(offset_, sub_.offset_);
	//  ce_swap(nelems_, sub_.nelems_);
	//  return *this;
	// }

	constexpr auto reverse() const {
		auto ret = unrotate();
		return layout_t(
			ret.sub().reverse(),
			ret.stride(),
			ret.offset(),
			ret.nelems()
		);
	}

	// constexpr auto reverse() -> layout_t& {
	//  *this = creverse();
	//  return *this;
	// }

	constexpr auto rotate() const {
		if constexpr(D > 1) {
			auto const ret = transpose();
			return layout_t(
				ret.sub().rotate(),
				ret.stride(),
				ret.offset(),
				ret.nelems()
			);
		} else {
			return *this;
		}
	}

	constexpr auto unrotate() const {
		if constexpr(D > 1) {
			auto const ret = layout_t(
				sub().unrotate(),
				stride(),
				offset(),
				nelems()
			);
			return ret.transpose();
		} else {
			return *this;
		}
	}

	// [[deprecated]] constexpr auto   rotate() -> layout_t& {if constexpr(D > 1) {transpose(); sub_.  rotate();} return *this;}
	// constexpr auto unrotate() -> layout_t& {if constexpr(D > 1) {sub_.unrotate(); transpose();} return *this;}

	constexpr auto hull_size() const -> size_type {
		if(is_empty()) {
			return 0;
		}
		return std::abs(size() * stride()) > std::abs(sub_.hull_size()) ? size() * stride() : sub_.hull_size();
	}

	[[deprecated("use two arg version")]] constexpr auto scale(size_type factor) const {
		return layout_t{sub_.scale(factor), stride_ * factor, offset_ * factor, nelems_ * factor};
	}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) use checked span
#endif

	auto take(size_type n) const {
		return layout_t(
			this->sub(),
			this->stride(),
			this->offset(),
			this->stride() * n
		);
	}

	BOOST_MULTI_HD constexpr auto halve() const {
		assert(this->size() % 2 == 0);
		return layout_t<D + 1>(
			this->take(this->size() / 2),
			this->nelems() / 2,
			0,
			this->nelems()
		);
	}

	constexpr auto scale(size_type num, size_type den) const {
		assert((stride_ * num) % den == 0);
		assert(offset_ == 0);  // TODO(correaa) implement ----------------vvv
		return layout_t{sub_.scale(num, den), stride_ * num / den, offset_ /* *num/den */, nelems_ * num / den};
	}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};

template<typename SSize>
struct layout_t<0, SSize>
	: multi::equality_comparable<layout_t<0, SSize>> {
	using dimensionality_type = multi::dimensionality_type;
	using rank                = std::integral_constant<dimensionality_type, 0>;

	using size_type       = SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index           = difference_type;
	using index_extension = multi::index_extension;
	using index_range     = multi::range<index>;

	using sub_type    = monostate;
	using stride_type = monostate;
	using offset_type = index;
	using nelems_type = index;

	using strides_type = tuple<>;
	using offsets_type = tuple<>;
	using nelemss_type = tuple<>;

	using extension_type = void;

	using extensions_type = extensions_t<rank::value>;
	using sizes_type      = tuple<>;
	using indexes         = tuple<>;

	static constexpr dimensionality_type rank_v         = rank::value;
	static constexpr dimensionality_type dimensionality = rank_v;  // TODO(correaa) : consider deprecation

	friend constexpr auto dimensionality(layout_t const& /*self*/) { return rank_v; }

 private:
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif
	BOOST_MULTI_NO_UNIQUE_ADDRESS sub_type    sub_;
	BOOST_MULTI_NO_UNIQUE_ADDRESS stride_type stride_;  // TODO(correaa) padding struct 'boost::multi::layout_t<0>' with 1 byte to align 'stride_' [-Werror,-Wpadded]

	offset_type offset_;

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

	nelems_type nelems_;

	template<dimensionality_type, typename> friend struct layout_t;

 public:
	layout_t() = default;

	BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& /*nil*/)
		: offset_{0}
		, nelems_{1} {}

	// BOOST_MULTI_HD constexpr explicit layout_t(extensions_type const& /*nil*/, strides_type const& /*nil*/) {}

	BOOST_MULTI_HD constexpr layout_t(sub_type sub, stride_type stride, offset_type offset, nelems_type nelems)  // NOLINT(bugprone-easily-swappable-parameters)
		: sub_{sub}
		, stride_{stride}
		, offset_{offset}
		, nelems_{nelems} {}

	[[nodiscard]] constexpr auto extensions() const { return extensions_type{}; }
	friend constexpr auto        extensions(layout_t const& self) { return self.extensions(); }

	[[nodiscard]] constexpr auto num_elements() const { return nelems_; }
	friend constexpr auto        num_elements(layout_t const& self) { return self.num_elements(); }

	[[nodiscard]] constexpr auto sizes() const { return tuple<>{}; }
	friend constexpr auto        sizes(layout_t const& self) { return self.sizes(); }

	[[nodiscard]] constexpr auto strides() const { return strides_type{}; }
	[[nodiscard]] constexpr auto offsets() const { return offsets_type{}; }
	[[nodiscard]] constexpr auto nelemss() const { return nelemss_type{}; }

	constexpr auto operator()() const { return offset_; }
	// constexpr explicit operator offset_type() const {return offset_;}

	constexpr auto stride() const -> stride_type = delete;
	constexpr auto offset() const -> offset_type { return offset_; }
	constexpr auto nelems() const -> nelems_type { return nelems_; }
	constexpr auto sub() const -> sub_type = delete;

	constexpr auto size() const -> size_type           = delete;
	constexpr auto extension() const -> extension_type = delete;

	constexpr auto is_empty() const noexcept { return nelems_ == 0; }

	BOOST_MULTI_NODISCARD("empty checks for emptyness, it performs no action. Use `is_empty()` instead")
	constexpr auto empty() const noexcept { return nelems_ == 0; }

	friend constexpr auto empty(layout_t const& self) noexcept { return self.empty(); }

	[[deprecated("is going to be removed")]]
	constexpr auto is_compact() const -> bool = delete;

	constexpr auto base_size() const -> size_type { return 0; }
	constexpr auto origin() const -> offset_type { return 0; }

	constexpr auto reverse() const { return *this; }
	// constexpr auto reverse()          -> layout_t& {return *this;}

	BOOST_MULTI_HD constexpr auto take(size_type /*n*/) const {
		return layout_t<0, SSize>{};
	}

	BOOST_MULTI_HD constexpr auto halve() const {
		return layout_t<1, SSize>(*this, 0, 0, 0);
	}

	// [[deprecated("use two arg version")]] constexpr auto scale(size_type /*size*/) const {return *this;}
	constexpr auto scale(size_type /*num*/, size_type /*den*/) const { return *this; }

	//  friend constexpr auto operator!=(layout_t const& self, layout_t const& other) {return not(self == other);}
	friend BOOST_MULTI_HD constexpr auto operator==(layout_t const& self, layout_t const& other) {
		return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) == std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	friend BOOST_MULTI_HD constexpr auto operator!=(layout_t const& self, layout_t const& other) {
		return std::tie(self.sub_, self.stride_, self.offset_, self.nelems_) != std::tie(other.sub_, other.stride_, other.offset_, other.nelems_);
	}

	constexpr auto operator<(layout_t const& other) const -> bool {
		return std::tie(offset_, nelems_) < std::tie(other.offset_, other.nelems_);
	}

	constexpr auto rotate() const { return *this; }
	constexpr auto unrotate() const { return *this; }

	// constexpr auto   rotate() -> layout_t& {return *this;}
	// constexpr auto unrotate() -> layout_t& {return *this;}

	constexpr auto hull_size() const -> size_type { return num_elements(); }  // not in bytes
};

constexpr auto
operator*(layout_t<0>::index_extension const& extensions_0d, layout_t<0>::extensions_type const& /*zero*/)
	-> typename layout_t<1>::extensions_type {
	return typename layout_t<1>::extensions_type{tuple<layout_t<0>::index_extension>{extensions_0d}};
}

constexpr auto operator*(extensions_t<1> const& extensions_1d, extensions_t<1> const& self) {
	using boost::multi::detail::get;
	return extensions_t<2>({get<0>(extensions_1d.base()), get<0>(self.base())});
}

}  // end namespace boost::multi

namespace boost::multi::detail {

template<class Tuple>
struct convertible_tuple : Tuple {
	using Tuple::Tuple;
	explicit convertible_tuple(Tuple const& other)
		: Tuple(other) {}

 public:
	using array_type = std::array<std::ptrdiff_t, std::tuple_size<Tuple>::value>;
	auto to_array() const noexcept {
		return std::apply([](auto... es) noexcept {
			return std::array<std::common_type_t<decltype(es)...>, sizeof...(es)>{{static_cast<size_type>(es)...}};
		},
						  static_cast<Tuple const&>(*this));
	}

	/*explicit*/ operator array_type() const& noexcept { return to_array(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	/*explicit*/ operator array_type() && noexcept { return to_array(); }      // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-stack-address"
#endif
	[[deprecated("This is here for nominal compatiblity with Boost.MultiArray, this would be a dangling conversion")]]
	operator std::ptrdiff_t const*() const&&;  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
											   /*{ return to_array().data(); }*/
#ifdef __clang__
#pragma clang diagnostic pop
#endif

	template<std::size_t Index, std::enable_if_t<(Index < std::tuple_size_v<Tuple>), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(convertible_tuple const& self) -> typename std::tuple_element<Index, Tuple>::type {
		using std::get;
		return get<Index>(static_cast<Tuple const&>(self));
	}
};

template<class Array>
struct decaying_array : Array {
	using Array::Array;
	explicit decaying_array(Array const& other)
		: Array(other) {}

	[[deprecated("possible dangling conversion, use `std::array<T, D> p` instead of `auto* p`")]]
	constexpr operator std::ptrdiff_t const*() const { return Array::data(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

	template<std::size_t Index, std::enable_if_t<(Index < std::tuple_size_v<Array>), int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa)
	friend constexpr auto get(decaying_array const& self) -> typename std::tuple_element<Index, Array>::type {
		using std::get;
		return get<Index>(static_cast<Array const&>(self));
	}
};
}  // end namespace boost::multi::detail

template<class Tuple> struct std::tuple_size<boost::multi::detail::convertible_tuple<Tuple>> : std::integral_constant<std::size_t, std::tuple_size_v<Tuple>> {};  // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple size
template<class Array> struct std::tuple_size<boost::multi::detail::decaying_array<Array>> : std::integral_constant<std::size_t, std::tuple_size_v<Array>> {};     // NOLINT(cert-dcl58-cpp) normal idiom to defined tuple size

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_DETAIL_LAYOUT_HPP
