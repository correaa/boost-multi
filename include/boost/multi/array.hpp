// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ARRAY_HPP_
#define BOOST_MULTI_ARRAY_HPP_

#include <boost/multi/array_ref.hpp>  // IWYU pragma: export
#include <boost/multi/detail/adl.hpp>
#include <boost/multi/detail/config/NO_UNIQUE_ADDRESS.hpp>
#include <boost/multi/detail/is_trivial.hpp>
#include <boost/multi/detail/memory.hpp>
#include <boost/multi/detail/static_allocator.hpp>  // TODO(correaa) export IWYU

#include <iterator>  // for std::sentinel_for
#include <memory>    // for std::allocator_traits
#include <stdexcept>
#include <type_traits>  // for std::common_reference
#include <utility>      // for std::move

#if __has_include(<memory_resource>)
#include <memory_resource>
// Apple clang provides the header but not the compiled library prior to version 16
#if (defined(__cpp_lib_memory_resource) && (__cpp_lib_memory_resource >= 201603)) && !(defined(__APPLE__) && defined(__clang_major__) && __clang_major__ <= 15) && (!defined(_LIBCPP_VERSION) || !(_LIBCPP_VERSION <= 160001))
#define BOOST_MULTI_HAS_MEMORY_RESOURCE
#endif
#endif

#if defined(__cplusplus) && (__cplusplus >= 202002L) && __has_include(<concepts>) && __has_include(<ranges>)
#include <concepts>  // for constructible_from  // NOLINT(misc-include-cleaner)  // IWYU pragma: keep
#include <ranges>    // IWYU pragma: keep
#endif

// TODO(correaa) or should be (__CUDA__) or CUDA__ || HIP__
#ifdef __NVCC__
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4626)  // assignment operator was implicitly defined as deleted
#endif

namespace boost::multi {

namespace detail {

template<class Allocator>
struct array_allocator {
	using allocator_type = Allocator;
	array_allocator()    = default;

 private:
	BOOST_MULTI_NO_UNIQUE_ADDRESS allocator_type alloc_;

	using allocator_traits = multi::allocator_traits<allocator_type>;

	using size_type_ = typename allocator_traits::size_type;  // NOLINT(readability-redundant-typename) typename needed in C++17
	using pointer_   = typename allocator_traits::pointer;    // NOLINT(readability-redundant-typename) typename needed in C++17

 protected:
	constexpr auto alloc() & -> auto& { return alloc_; }
	constexpr auto alloc() const& -> allocator_type const& { return alloc_; }

	constexpr explicit array_allocator(allocator_type const& alloc) : alloc_{alloc} {}  // NOLINT(modernize-pass-by-value)

	constexpr auto allocate(size_type_ n) -> pointer_ {
		return n ? allocator_traits::allocate(alloc_, n) : pointer_{nullptr};
	}
	constexpr auto allocate(size_type_ n, typename allocator_traits::const_void_pointer hint) -> pointer_ {  // NOLINT(readability-redundant-typename) typename needed in C++17
		return n ? allocator_traits::allocate(alloc_, n, hint) : pointer_{nullptr};
	}

	constexpr auto uninitialized_fill_n(pointer_ first, size_type_ count, typename allocator_traits::value_type const& value) {  // NOLINT(readability-redundant-typename) typename needed in C++17
		return adl_alloc_uninitialized_fill_n(alloc_, first, count, value);
	}

	template<typename It>
	auto uninitialized_copy_n(It first, size_type count, pointer_ d_first) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename std::pointer_traits<pointer_>::element_type> && !multi::force_element_trivial_default_construction<typename std::pointer_traits<pointer_>::element_type>) {
			adl_alloc_uninitialized_default_construct_n(alloc_, d_first, count);
		}
		return adl_copy_n(first, count, d_first);
#else
		return adl_alloc_uninitialized_copy_n(alloc_, first, count, d_first);
#endif
	}

	template<typename It>
	auto uninitialized_move_n(It first, size_type count, pointer_ d_first) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename std::pointer_traits<pointer_>::element_type> && !multi::force_element_trivial_default_construction<typename std::pointer_traits<pointer_>::element_type>) {
			adl_alloc_uninitialized_default_construct_n(alloc_, d_first, count);
		}
		return adl_copy_n(std::make_move_iterator(first), count, d_first);
#else
		return adl_alloc_uninitialized_move_n(alloc_, first, count, d_first);
#endif
	}

	template<class EP, typename It>
	auto uninitialized_copy_n(EP&& ep, It first, size_type count, pointer_ d_first) {
		return adl_uninitialized_copy_n(std::forward<EP>(ep), first, count, d_first);
	}

	template<typename It>
	auto destroy_n(It first, size_type n) { return adl_alloc_destroy_n(this->alloc(), first, n); }

 public:
	BOOST_MULTI_HD constexpr auto get_allocator() const noexcept -> allocator_type { return alloc_; }
};

}  // end namespace detail

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, dimensionality_type D, class DummyAlloc = std::allocator<T>>  // DummyAlloc mechanism allows using the convention array<T, an_allocator<>>, is an_allocator supports void template argument
struct dynamic_array                                                            // NOLINT(fuchsia-multiple-inheritance) : multiple inheritance used for composition
: protected detail::array_allocator<
	  typename allocator_traits<DummyAlloc>::template rebind_alloc<T>>
, public array_ref<T, D, typename multi::allocator_traits<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::pointer>
, boost::multi::random_iterable<dynamic_array<T, D, typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>> {
	static_assert(
		std::is_same_v<
			std::remove_const_t<typename multi::allocator_traits<DummyAlloc>::value_type>,
			typename dynamic_array::element_type> ||
			std::is_same_v<
				std::remove_const_t<typename multi::allocator_traits<DummyAlloc>::value_type>,
				void>,  // allocator template can be redundant or void (which can be a default for the allocator)
		"allocator value type must match array value type"
	);

 protected:
	using array_alloc = detail::array_allocator<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>;

 public:
	using detail::array_allocator<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::get_allocator;

	using allocator_type = typename detail::array_allocator<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::allocator_type;
	using decay_type     = array<T, D, allocator_type>;
	using layout_type    = typename array_ref<T, D, typename multi::allocator_traits<allocator_type>::pointer>::layout_type;

	using ref = array_ref<
		T, D,
		typename multi::allocator_traits<typename multi::allocator_traits<allocator_type>::template rebind_alloc<T>>::pointer>;

	auto operator new(std::size_t count) -> void* { return ::operator new(count); }
	auto operator new(std::size_t count, void* ptr) -> void* { return ::operator new(count, ptr); }

	void operator delete(void* ptr) noexcept { ::operator delete(ptr); }  // this overrides the deleted delete operator in reference (base) class subarray

 protected:
	using alloc_traits = typename multi::allocator_traits<allocator_type>;

	auto uninitialized_value_construct() {
		return adl_alloc_uninitialized_value_construct_n(dynamic_array::alloc(), this->base_, this->num_elements());
	}

	constexpr auto uninitialized_default_construct() {
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			return adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->base_, this->num_elements());
		}
	}

	template<typename It> auto uninitialized_copy_elements(It first) {
		return array_alloc::uninitialized_copy_n(first, this->num_elements(), this->data_elements());
	}

	// template<typename It> auto uninitialized_move_elements(It first) {
	//  return array_alloc::uninitialized_move_n(first, this->num_elements(), this->data_elements());
	// }

	template<class EP, typename It> auto uninitialized_copy_elements(EP&& ep, It first) {
		return array_alloc::uninitialized_copy_n(std::forward<EP>(ep), first, this->num_elements(), this->data_elements());
	}

	constexpr void destroy() {
		if constexpr(!(std::is_trivially_destructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_destruction<typename dynamic_array::element_type>)) {
			array_alloc::destroy_n(this->data_elements(), this->num_elements());
		}
	}

	void allocate() {
		this->base_ = array_alloc::allocate(static_cast<typename multi::allocator_traits<typename dynamic_array::allocator_type>::size_type>(this->dynamic_array::num_elements()));
	}

 public:
	using value_type = typename std::conditional_t<
		(D > 1),  // this parenthesis is needed
		array<typename dynamic_array::element_type, D - 1, allocator_type>,
		typename dynamic_array::element_type>;

	using typename ref::difference_type;
	using typename ref::size_type;

	explicit dynamic_array(allocator_type const& alloc) : array_alloc{alloc}, ref(nullptr, {}) {}

	using ref::operator();
	BOOST_MULTI_HD constexpr auto operator()() && -> decltype(auto) { return ref::element_moved(); }

	using ref::taked;

	constexpr auto taked(difference_type n) && -> decltype(auto) { return ref::taked(n).element_moved(); }

	using ref::dropped;

	constexpr auto dropped(difference_type n) && -> decltype(auto) { return ref::dropped(n).element_moved(); }

	constexpr dynamic_array(dynamic_array&& other)
	: array_alloc{other.alloc()},
	  ref{
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())),
		  other.extensions()
	  } {
		adl_alloc_uninitialized_move_n(
			this->alloc(),
			other.data_elements(),
			other.num_elements(),
			this->data_elements()
		);
		(void)std::move(other);
	}

	constexpr dynamic_array(decay_type&& other, allocator_type const& alloc) noexcept
	: array_alloc{alloc}, ref(std::exchange(other.base_, nullptr), other.extensions()) {
		std::move(other).layout_mutable() = typename dynamic_array::layout_type(typename dynamic_array::extensions_type{});  // = {};  careful! this is the place where layout can become invalid
	}

	constexpr explicit dynamic_array(decay_type&& other) noexcept
	: dynamic_array(std::move(other), allocator_type{}) {}  // 6b

#if __cplusplus >= 202002L && (!defined(__clang_major__) || (__clang_major__ != 10))
	template<class It, std::sentinel_for<It> Sentinel = It, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>
	constexpr explicit dynamic_array(It const& first, Sentinel const& last, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(layout_type{index_extension(adl_distance(first, last)) * multi::extensions(*first)}.num_elements())),
		  index_extension(adl_distance(first, last)) * multi::extensions(*first)
	  ) {
#if defined(__clang__) && defined(__CUDACC__)
		// TODO(correaa) add workaround for non-default constructible type and use adl_alloc_uninitialized_default_construct_n
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), ref::data_elements(), ref::num_elements());
		}
		adl_copy_n(first, last - first, ref::begin());
#else
		adl_alloc_uninitialized_copy(dynamic_array::alloc(), first, last, ref::begin());
#endif
	}
#else
	template<class It, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>
	constexpr explicit dynamic_array(It const& first, It const& last, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(layout_type{index_extension(adl_distance(first, last)) * multi::extensions(*first)}.num_elements())),
		  index_extension(adl_distance(first, last)) * multi::extensions(*first)
	  ) {
#if defined(__clang__) && defined(__CUDACC__)
		// TODO(correaa) add workaround for non-default constructible type and use adl_alloc_uninitialized_default_construct_n
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), ref::data_elements(), ref::num_elements());
		}
		adl_copy_n(first, last - first, ref::begin());
#else
		adl_alloc_uninitialized_copy(dynamic_array::alloc(), first, last, ref::begin());
#endif
	}
#endif

#if __cplusplus >= 202002L && (!defined(__clang_major__) || (__clang_major__ != 10))
	template<class It, std::sentinel_for<It> Sentinel, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>
	constexpr explicit dynamic_array(It const& first, Sentinel const& last)
	: dynamic_array(first, last, allocator_type{}) {}
#else
	template<class It, class = typename std::iterator_traits<std::decay_t<It>>::difference_type>
	constexpr explicit dynamic_array(It const& first, It const& last) : dynamic_array(first, last, allocator_type{}) {}
#endif

#if defined(__cpp_lib_ranges) && (__cpp_lib_ranges >= 201911L)  //  && !defined(_MSC_VER)
 private:
	void extent_(typename dynamic_array::extensions_type const& extensions) {
		auto new_layout = typename dynamic_array::layout_t{extensions};
		if(new_layout.num_elements() == 0) {
			return;
		}
		this->layout_mutable() = new_layout;  // typename array::layout_t{extensions};
		this->base_            = this->dynamic_array::array_alloc::allocate(
            static_cast<typename multi::allocator_traits<typename dynamic_array::allocator_type>::size_type>(
                new_layout.num_elements()
            ),
            this->data_elements()  // used as hint
        );
	}

 public:
	template<
		class Range, class = std::enable_if_t<!std::is_base_of<dynamic_array, std::decay_t<Range>>{}>,
		class = decltype(std::declval<Range const&>().begin()),
		class = decltype(std::declval<Range const&>().end()),
		// class = decltype(/*dynamic_array*/ (std::declval<Range const&>().begin() - std::declval<Range const&>().end())),  // instantiation of dynamic_array here gives a compiler error in 11.0, partially defined type?
		class = std::enable_if_t<!is_subarray<Range const&>::value>>                                                                                                 // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
	requires std::is_convertible_v<std::ranges::range_reference_t<std::decay_t<std::ranges::range_reference_t<Range>>>, T> explicit dynamic_array(Range const& rng)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax  // NOSONAR
	: dynamic_array() {
		if(rng.size() == 0) {
			return;
		}
		auto outer_it = std::ranges::begin(rng);
		static_assert(D == 2);

		auto const& outer_ref   = *outer_it;
		auto        common_size = static_cast<size_type>(outer_ref.size());
		extent_({static_cast<size_type>(rng.size()), common_size});

		auto [is, js] = this->extensions();
		{
			index const i        = 0;
			auto        inner_it = std::ranges::begin(outer_ref);
			for(auto j : js) {              // NOLINT(altera-unroll-loops) TODO(correa) change to algorithm applied on elements
				(*this)[i][j] = *inner_it;  // rng[i][j];
				++inner_it;
			}
			++outer_it;
		}

		for(index i = 1; i != is.last(); ++i) {
			auto const& outer_ref2 = *outer_it;
			assert(static_cast<multi::size_t>(outer_ref2.size()) == common_size);

			auto inner_it = std::ranges::begin(outer_ref2);
			for(auto j : js) {              // NOLINT(altera-unroll-loops) TODO(correa) change to algorithm applied on elements
				(*this)[i][j] = *inner_it;  // rng[i][j];
				++inner_it;
			}
			++outer_it;
		}
	}
#endif

	template<
		class Range, class = std::enable_if_t<!std::is_base_of<dynamic_array, std::decay_t<Range>>{}>,
		class = decltype(std::declval<Range const&>().begin()),
		class = decltype(std::declval<Range const&>().end()),
		// class = decltype(/*dynamic_array*/ (std::declval<Range const&>().begin() - std::declval<Range const&>().end())),  // instantiation of dynamic_array here gives a compiler error in 11.0, partially defined type?
		class = std::enable_if_t<!is_subarray<Range const&>::value>>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
	// cppcheck-suppress noExplicitConstructor ; because I want to use equal for lazy assigments form range-expressions // NOLINTNEXTLINE(runtime/explicit)
	dynamic_array(Range const& rng)                     // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) : to allow terse syntax  // NOSONAR
	: dynamic_array(std::begin(rng), std::end(rng)) {}  // Sonar: Prefer free functions over member functions when handling objects of generic type "Range".

	template<class TT>
	auto uninitialized_fill_elements(TT const& value) {
		return array_alloc::uninitialized_fill_n(this->data_elements(), this->num_elements(), value);
	}

	template<class TT, class... As>
	dynamic_array(array_ref<TT, D, As...> const& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref{
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())),
		  other.extensions()
	  } {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
		}
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
#else
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.data_elements(), other.num_elements(), this->data_elements());
#endif
	}

	dynamic_array(typename dynamic_array::extensions_type extensions, typename dynamic_array::element_type const& elem, allocator_type const& alloc)  // 2
	: array_alloc{alloc}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{extensions}.num_elements()), nullptr), extensions} {
		array_alloc::uninitialized_fill_n(this->data_elements(), static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()), elem);
	}

	template<class Element>
	explicit dynamic_array(
		Element const& elem, allocator_type const& alloc,
		std::enable_if_t<std::is_convertible_v<Element, typename dynamic_array::element_type> && (D == 0), int> /*dummy*/ = 0  // NOLINT(fuchsia-default-arguments-declarations) for classic sfinae, needed by MSVC?
	)
	: dynamic_array(typename dynamic_array::extensions_type{}, elem, alloc) {}

	constexpr dynamic_array(typename dynamic_array::extensions_type exts, typename dynamic_array::element_type const& elem)
	: array_alloc{},
	  array_ref<T, D, typename multi::allocator_traits<typename multi::allocator_traits<DummyAlloc>::template rebind_alloc<T>>::pointer>(
		  exts,
		  array_alloc::allocate(
			  static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t(exts).num_elements()),
			  nullptr
		  )
	  ) {
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type>) {
			array_alloc::uninitialized_fill_n(this->base(), static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()), elem);
		} else {  // this workaround allows constexpr arrays for simple types
			adl_fill_n(this->base(), static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()), elem);
		}
	}

	template<class ValueType, class = decltype(std::declval<ValueType>().extensions()), std::enable_if_t<std::is_convertible_v<ValueType, typename dynamic_array::value_type>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value, allocator_type const& alloc)                                                 // fill constructor
	: array_alloc{alloc}, ref(array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t(extension * value.extensions()).num_elements())), extension * value.extensions()) {
		static_assert(std::is_trivially_default_constructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_default_construction<typename dynamic_array::element_type>);  // TODO(correaa) not implemented for non-trivial types,
		adl_fill_n(this->begin(), this->size(), value);                                                                                                                                             // TODO(correaa) implement via .elements()? substitute with uninitialized version of fill, uninitialized_fill_n?
	}

	template<class ValueType, class = decltype(std::declval<ValueType>().extensions()), std::enable_if_t<std::is_convertible_v<ValueType, typename dynamic_array::value_type>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value)                                                                              // fill constructor
	: dynamic_array(extension, value, allocator_type{}) {}

	explicit dynamic_array(::boost::multi::extensions_t<D> const& extensions, allocator_type const& alloc)
	: array_alloc{alloc}, ref(array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{extensions}.num_elements())), extensions) {
		uninitialized_default_construct();
	}

	explicit dynamic_array(::boost::multi::extensions_t<D> const& exts)
	: dynamic_array(exts, allocator_type{}) {}

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                            // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_trivially_default_constructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unused*/, allocator_type const& alloc)
	: dynamic_array(extensions, alloc) {}

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                              // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	[[deprecated("****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****")]]
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unusued*/) = delete /*[["****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****")]]*/;

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                              // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	[[deprecated("****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****")]]
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unused*/, allocator_type const& /*alloc*/) = delete /*[["****element type cannot be partially formed (uninitialized), if you insists that this type should be treated as trivially constructible, consider opting-in to multi::force_trivial_default_construction at your own risk****"]]*/;

	template<class UninitilazedTag, std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_same_v<UninitilazedTag, ::boost::multi::uninitialized_elements_t>), int> = 0,                                                                            // NOLINT(modernize-use-constraints) for C++20
			 std::enable_if_t<sizeof(UninitilazedTag*) && (std::is_trivially_default_constructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_default_construction<typename dynamic_array::element_type>), int> = 0>  // NOLINT(modernize-use-constraints) for C++20
	explicit constexpr dynamic_array(::boost::multi::extensions_t<D> const& extensions, UninitilazedTag /*unusued*/) : dynamic_array(extensions) {}

	template<class OtherT, class OtherEP, class OtherLayout, class = std::enable_if_t<std::is_assignable<typename ref::element_ref, typename multi::subarray<OtherT, D, OtherEP, OtherLayout>::element_type>{}>, class = decltype(adl_copy(std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().begin(), std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	constexpr dynamic_array(multi::const_subarray<OtherT, D, OtherEP, OtherLayout> const& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{other.extensions()}.num_elements())),
		  other.extensions()
	  ) {
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.elements().begin(), this->num_elements(), this->data_elements());
	}

	template<class F>  // TODO(correaa) make more generic, e.g.: take ArrayWithElementsLike
	constexpr dynamic_array(multi::restriction<D, F> const& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{other.extensions()}.num_elements())),
		  other.extensions()
	  ) {
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.elements().begin(), this->num_elements(), this->data_elements());
	}

	template<class F>  // ArrayElementsLike, class = typename ArrayElementsLike::elements_t>
	// cppcheck-suppress noExplicitConstructor  // NOLINTNEXTLINE(runtime/explicit)
	constexpr dynamic_array(multi::restriction<D, F> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) to allow terse syntax
	: dynamic_array(other, allocator_type{}) {}

	template<class OtherT, class OtherEP, class OtherLayout, class = std::enable_if_t<std::is_assignable<typename ref::element_ref, typename multi::subarray<OtherT, D, OtherEP, OtherLayout>::element_type>{}>, class = decltype(adl_copy(std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().begin(), std::declval<multi::subarray<OtherT, D, OtherEP, OtherLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	constexpr dynamic_array(multi::subarray<OtherT, D, OtherEP, OtherLayout>&& other, allocator_type const& alloc)
	: array_alloc{alloc},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{other.extensions()}.num_elements())),
		  other.extensions()
	  ) {
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), std::move(other).elements().begin(), this->num_elements(), this->data_elements());
	}

	template<
		class TT, class EElementPtr, class LLayout, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout>&>().base()), T>, int> = 0,
		class = decltype(adl_copy(std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().begin(), std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	explicit dynamic_array(multi::const_subarray<TT, D, EElementPtr, LLayout> const& other)
	: dynamic_array(other, allocator_type{}) {}

	template<
		class TT, class EElementPtr, class LLayout, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<multi::subarray<TT, D, EElementPtr, LLayout> const&>().base()), T>, int> = 0,
		class = decltype(adl_copy(std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().begin(), std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	// cppcheck-suppress noExplicitConstructor  // NOLINTNEXTLINE(runtime/explicit)
	constexpr /*implicit*/ dynamic_array(multi::const_subarray<TT, D, EElementPtr, LLayout> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: dynamic_array(other, allocator_type{}) {}

	template<
		class TT, class EElementPtr, class LLayout, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<multi::subarray<TT, D, EElementPtr, LLayout> const&>().base()), T>, int> = 0,
		class = decltype(adl_copy(std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().begin(), std::declval<multi::const_subarray<TT, D, EElementPtr, LLayout> const&>().end(), std::declval<typename dynamic_array::iterator>()))>
	// cppcheck-suppress noExplicitConstructor  // NOLINTNEXTLINE(runtime/explicit)
	constexpr /*implicit*/ dynamic_array(multi::subarray<TT, D, EElementPtr, LLayout>&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: dynamic_array(std::move(other), allocator_type{}) {}

	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr dynamic_array(multi::subarray<T, D, typename dynamic_array::element_ptr, typename dynamic_array::layout_type> const&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: dynamic_array(other, allocator_type{}) {}

	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr dynamic_array(multi::const_subarray<T, D, typename dynamic_array::element_ptr, typename dynamic_array::layout_type> const&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
	: dynamic_array(other, allocator_type{}) {}

	// cppcheck-suppress noExplicitConstructor ; see below
	constexpr dynamic_array(multi::subarray<T, D, typename dynamic_array::element_ptr, typename dynamic_array::layout_type>&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: dynamic_array(std::move(other), allocator_type{}) {}

	template<class TT, class... Args, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
																																											   // cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	/*mplct*/ dynamic_array(array_ref<TT, D, Args...>& other)                                                                                                                  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		dynamic_array::uninitialized_copy_elements(other.data_elements());
	}

	template<class TT, class... Args, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(array_ref<TT, D, Args...>& other)                                                                                                                    // moLINT(fuchsia-default-arguments-declarations)
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		dynamic_array::uninitialized_copy_elements(other.data_elements());
	}

	// NOLINTNEXTLINE(modernize-use-constraints) TODO(correaa) for C++20
	template<class TT, class... Args, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&&>().base()), T>, int> = 0>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	/*mplct*/ dynamic_array(array_ref<TT, D, Args...>&& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(std::move(other).data_elements());
	}

	template<class TT, class... Args, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...>&&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(array_ref<TT, D, Args...>&& other)                                                                                                                    // NOLINT(fuchsia-default-arguments-declarations)
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(std::move(other).data_elements());
	}

	// NOLINTNEXTLINE(modernize-use-constraints) TODO(correaa) for C++20
	template<class TT, class... Args, std::enable_if_t<multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...> const&>().base()), T>, int> = 0>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax
	/*mplct*/ dynamic_array(array_ref<TT, D, Args...> const& other)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)  // NOSONAR
	: array_alloc{}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())), other.extensions()} {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(other.data_elements());
	}

	template<class TT, class... Args, std::enable_if_t<!multi::detail::is_implicitly_convertible_v<decltype(*std::declval<array_ref<TT, D, Args...> const&>().base()), T>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(array_ref<TT, D, Args...> const& other)                                                                                                                    // NOLINT(fuchsia-default-arguments-declarations)
	: array_alloc{},
	  ref(
		  array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())),
		  other.extensions()
	  ) {
		assert(this->stride() != 0);
		dynamic_array::uninitialized_copy_elements(std::move(other).data_elements());
	}

	constexpr dynamic_array(dynamic_array const& other)  // 5b
	: array_alloc{
		  multi::allocator_traits<allocator_type>::select_on_container_copy_construction(other.alloc())
	  },
	  ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements())  //,
		  ),
		  other.extensions()} {
		assert(this->stride() != 0);
		uninitialized_copy_elements(other.data_elements());
	}

	template<class ExecutionPolicy, std::enable_if_t<!std::is_convertible_v<ExecutionPolicy, typename dynamic_array::extensions_type>, int> = 0>  // NOLINT(modernize-use-constraints,modernize-type-traits) TODO(correaa) for C++20
	dynamic_array(ExecutionPolicy&& policy, dynamic_array const& other)
	: array_alloc{multi::allocator_traits<allocator_type>::select_on_container_copy_construction(other.alloc())}, ref{array_alloc::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements()), other.data_elements()), extensions(other)} {
		assert(this->stride() != 0);
		uninitialized_copy_elements(std::forward<ExecutionPolicy>(policy), other.data_elements());
	}

	// cppcheck-suppress noExplicitConstructor ; to allow assignment-like construction of nested arrays
	constexpr dynamic_array(std::initializer_list<typename dynamic_array<T, D>::value_type> values)
	: dynamic_array{(values.size() == 0) ? array<T, D>() : array<T, D>(values.begin(), values.end())} {}  // construct all with default constructor and copy to special memory at the end

	// template<class TT, std::enable_if_t<std::is_same_v<T, TT> || (D == 2), int> =0>
	// constexpr dynamic_array(std::initializer_list<std::initializer_list<TT>> il)
	// : dynamic_array{(il.size() == 0) ? array<T, D>() : array<T, D>(il.begin(), il.end())} {}  // construct all with default constructor and copy to special memory at the end

	dynamic_array(
		std::initializer_list<typename dynamic_array<T, D>::value_type> values,
		allocator_type const&                                           alloc
	)
	: dynamic_array{(values.size() == 0) ? dynamic_array<T, D>() : dynamic_array<T, D>(values.begin(), values.end()), alloc} {}

	template<class TT, std::size_t N>
	constexpr explicit dynamic_array(TT (&array)[N])  // @SuppressWarnings(cpp:S5945) NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backward compatibility // NOSONAR
	: dynamic_array(std::begin(array), std::end(array)) {
		assert(this->stride() != 0);
	}

	constexpr auto begin() const& noexcept -> typename dynamic_array::const_iterator { return ref::begin(); }
	constexpr auto end() const& noexcept -> typename dynamic_array::const_iterator { return ref::end(); }

	constexpr auto begin() && noexcept -> typename dynamic_array::move_iterator { return ref::begin(); }
	constexpr auto end() && noexcept -> typename dynamic_array::move_iterator { return ref::end(); }

	constexpr auto begin() & noexcept -> typename dynamic_array::iterator { return ref::begin(); }
	constexpr auto end() & noexcept -> typename dynamic_array::iterator { return ref::end(); }

	using ref::operator[];

	BOOST_MULTI_HD constexpr auto operator[](index idx) && -> decltype(auto) {
		return multi::move(ref::operator[](idx));
	}

	constexpr auto max_size() const noexcept { return static_cast<typename dynamic_array::size_type>(multi::allocator_traits<allocator_type>::max_size(this->alloc())); }  // TODO(correaa)  divide by nelements in under-dimensions?

 protected:
	constexpr void deallocate() {
		assert(this->stride() != 0);
		if(this->num_elements()) {
			multi::allocator_traits<allocator_type>::deallocate(this->alloc(), this->base_, static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()));
		}
	}

	void clear() noexcept {
		this->destroy();
		deallocate();
		this->layout_mutable() = typename dynamic_array::layout_type(typename dynamic_array::extensions_type{});
		assert(this->stride() != 0);
	}

 public:
	constexpr dynamic_array() noexcept
	: array_alloc{}, ref(nullptr, typename dynamic_array::extensions_type{}) {
		assert(this->stride() != 0);
		assert(this->size() == 0);
	}

#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
	constexpr ~dynamic_array()
#else
	~dynamic_array()
#endif
	{
		assert(this->stride() != 0);
		destroy();
		assert(this->stride() != 0);
		deallocate();
	}

	using element_const_ptr = typename std::pointer_traits<typename dynamic_array::element_ptr>::template rebind<typename dynamic_array::element_type const>;
	using element_move_ptr  = multi::move_ptr<typename dynamic_array::element_ptr>;

	using reference = std::conditional_t<
		(D > 1),
		subarray<typename dynamic_array::element_type, D - 1, typename dynamic_array::element_ptr>,
		std::conditional_t<
			D == 1,
			typename std::iterator_traits<typename dynamic_array::element_ptr>::reference,
			void>>;

	using const_reference = std::conditional_t<
		(D > 1),
		const_subarray<typename dynamic_array::element_type, D - 1, typename dynamic_array::element_ptr>,  // TODO(correaa) should be const_reference, but doesn't work witn rangev3?
		std::conditional_t<
			D == 1,
			decltype(*std::declval<typename dynamic_array::element_const_ptr>()),
			void>>;

	using iterator       = multi::array_iterator<T, D, typename dynamic_array::element_ptr>;
	using const_iterator = multi::array_iterator<T, D, typename dynamic_array::element_ptr, true>;

	friend auto get_allocator(dynamic_array const& self) -> allocator_type { return self.get_allocator(); }

	// cppcheck-suppress duplInheritedMember ; to override
	BOOST_MULTI_HD constexpr auto data_elements() const& -> element_const_ptr { return this->base_; }

	// cppcheck-suppress duplInheritedMember ; to override
	BOOST_MULTI_HD constexpr auto data_elements() & -> typename dynamic_array::element_ptr { return this->base_; }

	// cppcheck-suppress duplInheritedMember ; to override
	BOOST_MULTI_HD constexpr auto data_elements() && -> typename dynamic_array::element_move_ptr { return std::make_move_iterator(this->base_); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(dynamic_array const& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(dynamic_array& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(dynamic_array&& self) { return std::move(self).data_elements(); }

	constexpr auto base() & -> typename dynamic_array::element_ptr { return ref::base(); }
	constexpr auto base() const& -> typename dynamic_array::element_const_ptr { return typename dynamic_array::element_const_ptr{ref::base()}; }

	constexpr auto origin() & -> typename dynamic_array::element_ptr { return ref::origin(); }
	constexpr auto origin() const& -> typename dynamic_array::element_const_ptr { return ref::origin(); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array& self) -> typename dynamic_array::element_ptr { return self.origin(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array const& self) -> typename dynamic_array::element_const_ptr { return self.origin(); }

	template<class TT, typename EElementPtr, class LLayout>
	auto operator=(multi::const_subarray<TT, D, EElementPtr, LLayout> const& other) -> dynamic_array& {
		ref::operator=(other);  // TODO(correaa) : protect for self assigment
		assert(this->stride() != 0);
		return *this;
	}
	auto operator=(dynamic_array const& other) & -> dynamic_array& {
		if(std::addressof(other) == this) {
			return *this;
		}  // cert-oop54-cpp
		assert(other.extensions() == this->extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		assert(this->stride() != 0);
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	constexpr auto operator=(dynamic_array&& other) noexcept -> dynamic_array& {                               // lints  (cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
		assert(extensions(other) == dynamic_array::extensions());                                              // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : allow a constexpr-friendly assert
		adl_move(other.data_elements(), other.data_elements() + other.num_elements(), this->data_elements());  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) there is no std::move_n algorithm
		assert(this->stride() != 0);
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	template<class TT, class... As>
	auto operator=(dynamic_array<TT, D, As...> const& other) & -> dynamic_array& {
		assert(extensions(other) == dynamic_array::extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

	template<class Archive>
	// cppcheck-suppress duplInheritedMember ; to override
	void serialize(Archive& arxiv, unsigned int const version) { ref::serialize(arxiv, version); }

 private:
	void swap_(dynamic_array& other) noexcept { operator()().swap(other()); }

 public:
	friend void swap(dynamic_array& lhs, dynamic_array& rhs) noexcept { lhs.swap_(rhs); }
};

template<typename T, dimensionality_type D, class Alloc = std::allocator<T>>
using static_array [[deprecated("static_array has been renamed to dynamics_array (uses dynamic memory)")]] = dynamic_array<T, D, Alloc>;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<typename T, class Alloc>
struct dynamic_array<T, ::boost::multi::dimensionality_type{0}, Alloc>  // NOLINT(fuchsia-multiple-inheritance) : design
: protected detail::array_allocator<Alloc>
, public array_ref<T, 0, typename multi::allocator_traits<typename detail::array_allocator<Alloc>::allocator_type>::pointer> {
	static_assert(std::is_same_v<typename multi::allocator_traits<Alloc>::value_type, typename dynamic_array::element_type>, "allocator value type must match array value type");

 private:
	using array_alloc = detail::array_allocator<Alloc>;

 public:
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	// NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() && -> dynamic_array* = delete;  // NOSONAR(cpp:S877) NOLINT(google-runtime-operator) : delete to avoid taking address of temporary
	// NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() & -> dynamic_array* { return this; }  // NOSONAR(cpp:S877) NOLINT(google-runtime-operator) : override from base
	// NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() const& -> dynamic_array const* { return this; }  // NOSONAR(cpp:S877) NOLINT(google-runtime-operator) : override from base
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	using array_alloc::get_allocator;
	using allocator_type = typename dynamic_array::allocator_type;
	using decay_type     = array<T, 0, Alloc>;

	template<class Ptr>
	void assign(Ptr data) & {
		if(data) {
			assert(this->num_elements() == 1);
			adl_copy_n(data, this->num_elements(), this->base());
		}
	}

	template<
		class Singleton, std::enable_if_t<!std::is_base_of_v<dynamic_array, Singleton> && !std::is_same_v<Singleton, typename dynamic_array::element_type>, int> = 0,
		class = decltype(adl_copy_n(&std::declval<Singleton>(), 1, typename dynamic_array::element_ptr{}))>
	auto operator=(Singleton const& single) -> dynamic_array& {
		assign(&single);
		return *this;
	}

 protected:
	using alloc_traits = typename multi::allocator_traits<allocator_type>;
	using ref          = array_ref<T, 0, typename multi::allocator_traits<typename multi::allocator_traits<Alloc>::template rebind_alloc<T>>::pointer>;

	auto uninitialized_value_construct() {
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			return adl_alloc_uninitialized_value_construct_n(dynamic_array::alloc(), this->base_, this->num_elements());
		}
	}

	template<typename It> auto uninitialized_copy(It first) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(this->alloc(), this->data_elements(), this->num_elements());
		}
		return adl_copy(first, this->num_elements(), this->data_elements());
#else
		return adl_alloc_uninitialized_copy_n(this->alloc(), first, this->num_elements(), this->data_elements());
#endif
	}

	template<typename It>
	auto uninitialized_move(It first) {
		return adl_alloc_uninitialized_move_n(this->alloc(), first, this->num_elements(), this->data_elements());
	}

	constexpr void destroy() {
		if constexpr(!(std::is_trivially_destructible_v<typename dynamic_array::element_type> || multi::force_element_trivial_destruction<typename dynamic_array::element_type>)) {
			array_alloc::destroy_n(this->data_elements(), this->num_elements());
		}
	}

 public:
	using typename ref::difference_type;
	using typename ref::size_type;
	using typename ref::value_type;
	constexpr explicit dynamic_array(allocator_type const& alloc) : array_alloc{alloc} {}

	constexpr dynamic_array(decay_type&& other, allocator_type const& alloc)  // 6b
	: array_alloc{alloc}, ref{other.base_, other.extensions()} {
		std::move(other).ref::layout_t::operator=({});
	}

	using ref::operator==;
	using ref::operator!=;

	dynamic_array(
		typename dynamic_array::extensions_type const& extensions,
		typename dynamic_array::element const& elem, allocator_type const& alloc
	)
	: array_alloc{alloc},
	  ref(dynamic_array::allocate(
			  static_cast<typename multi::allocator_traits<allocator_type>::size_type>(
				  typename dynamic_array::layout_t{extensions}.num_elements()
			  )
		  ),
		  extensions) {
		uninitialized_fill(elem);
	}

	dynamic_array(typename dynamic_array::element_type const& elem, allocator_type const& alloc)
	: dynamic_array(typename dynamic_array::extensions_type{}, elem, alloc) {}

	template<typename OtherT, typename OtherEPtr, class OtherLayout>
	explicit dynamic_array(multi::const_subarray<OtherT, 0, OtherEPtr, OtherLayout> const& other, allocator_type const& alloc)
	: array_alloc{alloc}, ref(dynamic_array::allocate(other.num_elements()), extensions(other)) {
		assert(other.num_elements() <= 1);
		if(other.num_elements()) {
#if defined(__clang__) && defined(__CUDACC__)
			if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
				adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
			}
			adl_copy(other.base(), other.base() + other.num_elements(), this->base());
#else
			adl_alloc_uninitialized_copy(dynamic_array::alloc(), other.base(), other.base() + other.num_elements(), this->base());
#endif
		}
	}

	template<class TT, class... Args>
	explicit dynamic_array(multi::dynamic_array<TT, 0, Args...> const& other, allocator_type const& alloc)  // TODO(correaa) : call other constructor (above)
	: array_alloc{alloc}, ref(dynamic_array::allocate(static_cast<typename std::allocator_traits<Alloc>::size_type>(other.num_elements())), extensions(other)) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
		}
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
#else
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), other.data_elements(), other.num_elements(), this->data_elements());
#endif
	}

	template<class TT, class... Args>
	explicit dynamic_array(multi::dynamic_array<TT, 0, Args...> const& other)
	: dynamic_array(other, allocator_type{}) {}

	auto uninitialized_fill(typename dynamic_array::element_type const& elem) {
		array_alloc::uninitialized_fill_n(
			this->base_,
			static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()),
			elem
		);
	}

	template<class TT, class... Args>
	auto operator=(multi::const_subarray<TT, 0, Args...> const& other) -> dynamic_array& {
		adl_copy_n(other.base(), 1, this->base());
		return *this;
	}

	dynamic_array(
		typename dynamic_array::extensions_type const& extensions,
		typename dynamic_array::element_type const&    elem
	)  // 2
	: array_alloc{}, ref(dynamic_array::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(typename dynamic_array::layout_t{extensions}.num_elements()), nullptr), extensions) {
		uninitialized_fill(elem);
	}

	dynamic_array() : dynamic_array(multi::iextensions<0>{}) {}  // TODO(correaa) a noexcept will force a partially formed state for zero dimensional arrays

	explicit dynamic_array(typename dynamic_array::element_type const& elem)
	: dynamic_array(multi::iextensions<0>{}, elem) {}

	template<
		class Singleton, std::enable_if_t<!std::is_base_of_v<dynamic_array, Singleton> && !std::is_same_v<Singleton, typename dynamic_array::element_type>, int> = 0,  // NOLINT(modernize-type-traits) for C++20
		class = decltype(adl_copy_n(&std::declval<Singleton>(), 1, typename dynamic_array::element_ptr{}))>
	// cppcheck-suppress noExplicitConstructor ; to allow terse syntax  // NOLINTNEXTLINE(runtime/explicit)
	/*implict*/ dynamic_array(Singleton const& single)  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) this is used by the
	: ref(dynamic_array::allocate(1), typename dynamic_array::extensions_type{}) {
#if defined(__clang__) && defined(__CUDACC__)
		if constexpr(!std::is_trivially_default_constructible_v<typename dynamic_array::element_type> && !multi::force_element_trivial_default_construction<typename dynamic_array::element_type>) {
			adl_alloc_uninitialized_default_construct_n(dynamic_array::alloc(), this->data_elements(), this->num_elements());
		}
		adl_copy_n(&single, typename multi::allocator_traits<Alloc>::size_type{1}, this->data_elements());
#else
		adl_alloc_uninitialized_copy_n(dynamic_array::alloc(), &single, typename multi::allocator_traits<Alloc>::size_type{1}, this->data_elements());
#endif
	}

	template<class ValueType, typename = std::enable_if_t<std::is_same_v<ValueType, value_type>>>                                          // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value, allocator_type const& alloc)  // 3
	: dynamic_array(extension * extensions(value), alloc) {
		assert(this->stride() != 0);
		using std::fill;
		fill(this->begin(), this->end(), value);
	}

	template<class ValueType, typename = std::enable_if_t<std::is_same_v<ValueType, value_type>>>             // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	explicit dynamic_array(typename dynamic_array::index_extension const& extension, ValueType const& value)  // 3
	: dynamic_array(extension, value, allocator_type{}) {}
	// : dynamic_array(extension * extensions(value)) {  // TODO(correaa) : call other constructor (above)
	//  assert(this->stride() != 0);
	//  using std::fill;
	//  fill(this->begin(), this->end(), value);
	// }

	explicit dynamic_array(typename dynamic_array::extensions_type const& extensions, allocator_type const& alloc)  // 3
	: array_alloc{alloc}, ref(dynamic_array::allocate(typename dynamic_array::layout_t{extensions}.num_elements()), extensions) {
		// assert(this->stride() != 0);
		uninitialized_value_construct();
	}
	explicit dynamic_array(typename dynamic_array::extensions_type const& extensions)  // 3
	: dynamic_array(extensions, allocator_type{}) {
		// assert(this->stride() != 0);
	}

	dynamic_array(dynamic_array const& other, allocator_type const& alloc)  // 5b
	: array_alloc{alloc}, ref(dynamic_array::allocate(other.num_elements()), extensions(other)) {
		assert(this->stride() != 0);
		uninitialized_copy_(other.data_elements());
	}

	dynamic_array(dynamic_array const& other)  // 5b
	: array_alloc{other.get_allocator()}, ref{dynamic_array::allocate(other.num_elements(), other.data_elements()), {}} {
		assert(this->stride() != 0);
		uninitialized_copy(other.data_elements());
	}

	dynamic_array(dynamic_array&& other) noexcept
	: array_alloc{other.get_allocator()}, ref(std::exchange(other.base_, nullptr), other.extensions()) {
		other.layout_mutable() = {};
		// other.layout_t<0>::operator=({});
		// , ref(dynamic_array::allocate(static_cast<typename multi::allocator_traits<allocator_type>::size_type>(other.num_elements()), other.data_elements()), other.extensions()) {
		//  adl_alloc_uninitialized_move_n(
		//      this->alloc(),
		//      other.data_elements(),
		//      other.num_elements(),
		//      this->data_elements()
		//  );
		(void)std::move(other);
	}

 protected:
	void deallocate() {  // TODO(correaa) : move this to detail::array_allocator
		if(this->num_elements() && this->base_) {
			multi::allocator_traits<allocator_type>::deallocate(this->alloc(), this->base_, static_cast<typename multi::allocator_traits<allocator_type>::size_type>(this->num_elements()));
		}
	}
	void clear() noexcept {
		this->destroy();
		deallocate();
		layout_t<0>::operator=({});
	}

 public:
	~dynamic_array() noexcept {
		this->destroy();
		deallocate();
	}
	using element_const_ptr = typename std::pointer_traits<typename dynamic_array::element_ptr>::template rebind<typename dynamic_array::element_type const>;

	BOOST_MULTI_FRIEND_CONSTEXPR auto get_allocator(dynamic_array const& self) -> allocator_type { return self.get_allocator(); }

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto base() & -> typename dynamic_array::element_ptr { return ref::base(); }
	constexpr auto base() const& -> typename dynamic_array::element_const_ptr { return ref::base(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	BOOST_MULTI_FRIEND_CONSTEXPR auto base(dynamic_array& self) -> typename dynamic_array::element_ptr { return self.base(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto base(dynamic_array const& self) -> typename dynamic_array::element_const_ptr { return self.base(); }

	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto origin() & -> typename dynamic_array::element_ptr { return ref::origin(); }
	constexpr auto origin() const& -> typename dynamic_array::element_const_ptr { return ref::origin(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array& self) -> typename dynamic_array::element_ptr { return self.origin(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto origin(dynamic_array const& self) -> typename dynamic_array::element_const_ptr { return self.origin(); }

	// NOSONAR
	constexpr operator typename std::iterator_traits<typename dynamic_array::element_const_ptr>::reference() const& {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return *(this->base_);
	}

	// NOSONAR
	constexpr operator std::add_rvalue_reference_t<typename std::iterator_traits<typename dynamic_array::element_ptr>::reference>() && {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return std::move(*(this->base_));
	}

	// NOSONAR
	constexpr operator typename std::iterator_traits<typename dynamic_array::element_ptr>::reference() & {  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return *(this->base_);
	}

	template<class OtherElement, std::enable_if_t<std::is_convertible_v<typename dynamic_array::element_type, OtherElement>, int> = 0>  // NOLINT(modernize-use-constraints)
	// cppcheck-suppress duplInheritedMember ; to overwrite
	constexpr explicit operator OtherElement() const {
		return static_cast<OtherElement>(*(this->base_));
	}

	constexpr auto rotated() const& {  // cppcheck-suppress duplInheritedMember ; to overwrite
		typename dynamic_array::layout_t new_layout = this->layout();
		new_layout.rotate();
		return subarray<T, 0, typename dynamic_array::element_const_ptr>{new_layout, this->base_};
	}

	constexpr auto rotated() & {  // cppcheck-suppress duplInheritedMember ; to overwrite
		typename dynamic_array::layout_t new_layout = this->layout();
		new_layout.rotate();
		return subarray<T, 0, typename dynamic_array::element_ptr>{new_layout, this->base_};
	}

	constexpr auto rotated() && {  // cppcheck-suppress duplInheritedMember ; to overwrite
		typename dynamic_array::layout_t new_layout = this->layout();
		new_layout.rotate();
		return subarray<T, 0, typename dynamic_array::element_ptr>{new_layout, this->base_};
	}

	friend constexpr auto rotated(dynamic_array& self) -> decltype(auto) { return self.rotated(); }
	friend constexpr auto rotated(dynamic_array const& self) -> decltype(auto) { return self.rotated(); }

 private:
	constexpr auto unrotated_aux_() {
		typename dynamic_array::layout_t new_layout = *this;
		new_layout.unrotate();
		return subarray<T, 0, typename dynamic_array::element_const_ptr>{new_layout, this->base_};
	}

 public:
	// cppcheck-suppress-begin duplInheritedMember ; to overwrite
	constexpr auto unrotated() & { return unrotated_aux_(); }
	constexpr auto unrotated() const& { return unrotated_aux_().as_const(); }
	// cppcheck-suppress-end duplInheritedMember ; to overwrite

	friend constexpr auto unrotated(dynamic_array& self) -> decltype(auto) { return self.unrotated(); }
	friend constexpr auto unrotated(dynamic_array const& self) -> decltype(auto) { return self.unrotated(); }

	constexpr auto operator=(dynamic_array const& other) -> dynamic_array& {
		assert(extensions(other) == dynamic_array::extensions());  // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : allow a constexpr-friendly assert
		if(this == &other) {
			return *this;
		}  // lints (cert-oop54-cpp) : handle self-assignment properly
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

 private:
	constexpr auto equal_extensions_if_(std::true_type /*true */, dynamic_array const& other) { return this->extensions() == extensions(other); }
	constexpr auto equal_extensions_if_(std::false_type /*false*/, dynamic_array const& /*other*/) { return true; }

 public:
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

	constexpr auto operator=(dynamic_array&& other) noexcept -> dynamic_array& {
		assert(equal_extensions_if_(std::integral_constant<bool, (dynamic_array::rank_v != 0)>{}, other));     // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : allow a constexpr-friendly assert
		adl_move(other.data_elements(), other.data_elements() + other.num_elements(), this->data_elements());  // there is no std::move_n algorithm  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
		return *this;
	}

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	template<class TT, class... As, class = std::enable_if_t<std::is_assignable<typename dynamic_array::element_ref, TT>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	auto operator=(dynamic_array<TT, 0, As...> const& other) & -> dynamic_array& {
		assert(extensions(other) == dynamic_array::extensions());
		adl_copy_n(other.data_elements(), other.num_elements(), this->data_elements());
		return *this;
	}

	constexpr explicit operator subarray<value_type, 0, typename dynamic_array::element_const_ptr, typename dynamic_array::layout_t>() & {  // cppcheck-suppress duplInheritedMember ; to overwrite
		// cppcheck-suppress duplInheritedMember ; to overwrite
		return this->template dynamic_array_cast<value_type, typename dynamic_array::element_const_ptr>();  // cppcheck-suppress duplInheritedMember ; to overwrite
																											// return dynamic_array_cast<typename dynamic_array::value_type, typename dynamic_array::element_const_ptr>(*this);
	}

	template<class Archive>
	void serialize(Archive& arxiv, unsigned int const version) {  // cppcheck-suppress duplInheritedMember ; to overwrite
		ref::serialize(arxiv, version);
	}
};

template<class T, multi::dimensionality_type D, std::size_t Capacity = 4UL * 4UL>
using inplace_array = multi::dynamic_array<T, D, multi::detail::static_allocator<T, Capacity>>;

template<typename T, class Alloc>
struct array<T, 0, Alloc> : dynamic_array<T, 0, Alloc> {
	using dynamic_array<T, 0, Alloc>::dynamic_array;

	using dynamic_array<T, 0, Alloc>::operator=;

#if defined(__cpp_multidimensional_subscript) && (__cpp_multidimensional_subscript >= 202110L)
	BOOST_MULTI_HD constexpr auto operator[]() const& { return *(this->base()); }
#endif

#if !defined(__NVCOMPILER) || (__NVCOMPILER_MAJOR__ > 22 || (__NVCOMPILER_MAJOR__ == 22 && __NVCOMPILER_MINOR__ > 5))  // bug in nvcc 22.5: error: "operator=" has already been declared in the current scope
	template<class TT, class... Args>
	auto operator=(multi::array<TT, 0, Args...> const& other) & -> array& {
		if(other.base()) {
			adl_copy_n(other.base(), other.num_elements(), this->base());
		}
		return *this;
	}

	template<class TT, class... Args>
	auto operator=(multi::array<TT, 0, Args...> const& other) && -> array&& {  // NOLINT(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator) should assigment return auto& ?
		if(other.base()) {
			adl_copy_n(other.base(), other.num_elements(), this->base());
		}
		return std::move(*this);
	}
#endif

	auto reextent(typename array::extensions_type const& /*empty_extensions*/) -> array& {
		return *this;
	}

	// cppcheck-suppress duplInheritedMember ; to overwrite  // NOLINTNEXTLINE(runtime/operator)
	constexpr auto operator&() && -> array* = delete;  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, ::boost::multi::dimensionality_type D, class Alloc>
struct array : dynamic_array<T, D, Alloc> {
	using static_ = dynamic_array<T, D, Alloc>;

	static_assert(
		std::is_same_v<typename multi::allocator_traits<Alloc>::value_type, T> || std::is_same_v<typename multi::allocator_traits<Alloc>::value_type, void>,
		"only exact type of array element or void (default) is allowed as allocator value type"
	);

	// cppcheck-suppress duplInheritedMember ; to override  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() && -> array* = delete;  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary
	// cppcheck-suppress duplInheritedMember ; to override  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() & -> array* { return this; }  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary
	// cppcheck-suppress duplInheritedMember ; to override  // NOLINTNEXTLINE(runtime/operator)
	BOOST_MULTI_HD constexpr auto operator&() const& -> array const* { return this; }  // NOLINT(google-runtime-operator) //NOSONAR delete operator&& defined in base class to avoid taking address of temporary

	template<class Archive, class ArTraits = multi::archive_traits<Archive>>
	void serialize(Archive& arxiv, unsigned int const version) {  // cppcheck-suppress duplInheritedMember ; to override
		auto extensions_ = this->extensions();

		arxiv& ArTraits::make_nvp("extensions", extensions_);  // don't try `using ArTraits::make_nvp`, make_nvp is a static member
		if(this->extensions() != extensions_) {
			clear();
			this->reextent(extensions_);
		}
		static_::serialize(arxiv, version);
	}

	// vvv workaround for MSVC 14.3 and ranges, TODO(correaa) good solution would be to inherit from const_subarray
	BOOST_MULTI_HD operator subarray<T, D, typename array::element_const_ptr, typename array::layout_type> const&() const {     // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
		return reinterpret_cast<subarray<T, D, typename array::element_const_ptr, typename array::layout_type> const&>(*this);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
	}

	// move this to dynamic_array
	template<
		class Range, std::enable_if_t<!has_extensions<std::decay_t<Range>>::value, int> = 0,
		class = decltype(Range{std::declval<typename array::const_iterator>(), std::declval<typename array::const_iterator>()})>
	constexpr explicit operator Range() const {  // cppcheck-suppress duplInheritedMember ; to overwrite
		// vvv Range{...} needed by Windows GCC?
		return Range{this->begin(), this->end()};  // NOLINT(fuchsia-default-arguments-calls) e.g. std::vector(it, it, alloc = {})
	}

	// move this to dynamic_array
	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>                          // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN const&() const& { return this->template to_carray_<TTN>(); }  // cppcheck-suppress duplInheritedMember ; to override

	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>                // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN&() && { return this->template to_carray_<TTN>(); }  // cppcheck-suppress duplInheritedMember ; to override

	template<class TTN, std::enable_if_t<std::is_array_v<TTN>, int> = 0>               // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
	constexpr explicit operator TTN&() & { return this->template to_carray_<TTN>(); }  // cppcheck-suppress duplInheritedMember ; to override

	// NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved) false positive in clang-tidy 17-20 ?
	using dynamic_array<T, D, Alloc>::dynamic_array;  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) passing c-arrays to base
	using typename dynamic_array<T, D, Alloc>::value_type;

	// #if defined(_MSC_VER)
	//  explicit array(typename array::extensions_type const& exts)  // NOTE(correaa) if you think you need to implement this for MSVC is because MSVC needs to be compiled in permissive- mode for C++17
	//  : dynamic_array<T, D, Alloc>(exts) { }
	// #endif

	// cppcheck-suppress noExplicitConstructor ; to allow assignment-like construction of nested arrays
	constexpr array(std::initializer_list<typename dynamic_array<T, D>::value_type> ilv)
	: static_{
		  (ilv.size() == 0) ? array<T, D>()
							: array<T, D>(ilv.begin(), ilv.end())
	  } {
	}

	template<
		class OtherT,
		class = std::enable_if_t<std::is_constructible_v<typename dynamic_array<T, D>::value_type, OtherT> && !std::is_convertible_v<OtherT, typename dynamic_array<T, D>::value_type> && (D == 1)>>  // NOLINT(modernize-use-constraints,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) TODO(correaa) for C++20
	constexpr explicit array(std::initializer_list<OtherT> ilv)                                                                                                                                       // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) inherit explicitness of conversion from the elements
	: static_{
		  (ilv.size() == 0) ? array<T, D>()()
							: array<T, D>(ilv.begin(), ilv.end()).element_transformed([](auto const& elem) noexcept { return static_cast<T>(elem); })
	  } {}

	array()             = default;
	array(array const&) = default;

	~array() = default;

	auto reshape(typename array::extensions_type extensions) & -> array& {
		typename array::layout_t const new_layout{extensions};  // TODO(correaa) implement move-reextent in terms of reshape
		assert(new_layout.num_elements() == this->num_elements());
		this->layout_mutable() = new_layout;
		assert(this->stride() != 0);
		return *this;
	}

	auto clear() noexcept -> array& {  // cppcheck-suppress duplInheritedMember ; to override
		static_::clear();
		assert(this->stride() != 0);
		return *this;
	}
	friend auto clear(array& self) noexcept -> array& { return self.clear(); }

	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(array const& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(array& self) { return self.data_elements(); }
	BOOST_MULTI_FRIEND_CONSTEXPR auto data_elements(array&& self) { return std::move(self).data_elements(); }

	friend BOOST_MULTI_HD constexpr auto move(array& self) -> decltype(auto) { return std::move(self); }
	friend BOOST_MULTI_HD constexpr auto move(array&& self) -> decltype(auto) { return std::move(self); }

	BOOST_MULTI_HD constexpr array(array&& other, Alloc const& alloc) noexcept
	: dynamic_array<T, D, Alloc>{std::move(other), alloc} {
		assert(this->stride() != 0);
	}

	BOOST_MULTI_HD constexpr array(array&& other) noexcept : array{std::move(other), other.get_allocator()} {
		assert(this->stride() != 0);
	}

	// friend auto get_allocator(array const& self) -> typename array::allocator_type { return self.get_allocator(); }

	void swap(array& other) noexcept {
		using std::swap;
		if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_swap::value) {
			swap(this->alloc(), other.alloc());
		}
		swap(this->base_, other.base_);
		swap(
			this->layout_mutable(),
			other.layout_mutable()
		);
		assert(this->stride() != 0);
	}

#ifndef NOEXCEPT_ASSIGNMENT
	auto operator=(array&& other) noexcept -> array& {
		if(this == std::addressof(other)) {
			return *this;
		}
		clear();
		this->base_ = other.base_;
		if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_move_assignment::value) {
			this->alloc() = std::move(other.alloc());
		}
		this->layout_mutable() = std::exchange(other.layout_mutable(), typename array::layout_type(typename array::extensions_type{}));
		assert(this->stride() != 0);
		assert(other.stride() != 0);
		return *this;
	}

	auto operator=(array const& other) -> array& {
		if(array::extensions() == other.extensions()) {
			if(this == &other) {
				return *this;
			}  // required by cert-oop54-cpp
			if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_copy_assignment::value) {
				this->alloc() = other.alloc();
			}
			static_::operator=(other);
		} else {
			clear();
			if constexpr(multi::allocator_traits<typename array::allocator_type>::propagate_on_container_copy_assignment::value) {
				this->alloc() = other.alloc();
			}
			this->layout_mutable() = other.layout();
			array::allocate();
			array::uninitialized_copy_elements(other.data_elements());
		}
		return *this;
	}
#else
	auto operator=(array o) noexcept -> array& { return swap(o), *this; }
#endif

	template<typename OtherT, typename OtherEP, class OtherLayout>
	auto operator=(multi::const_subarray<OtherT, D, OtherEP, OtherLayout> const& other) -> array& {
		if(array::extensions() == other.extensions()) {
			static_::operator=(other);  // TODO(correaa) : protect for self assigment
		} else {
			operator=(array{other});
		}
		return *this;
	}

	template<class TT, class AAlloc>
	auto operator=(multi::array<TT, D, AAlloc> const& other) -> array& {
		if(array::extensions() == other.extensions()) {
			static_::operator=(other);
		} else if(this->num_elements() == other.extensions().num_elements()) {
			reshape(other.extensions());
			static_::operator=(other);
		} else {
			operator=(static_cast<array>(other));
		}
		return *this;
	}

	template<
		class Range, class = decltype(std::declval<static_&>().operator=(std::declval<Range&&>())),
		std::enable_if_t<!has_data_elements<std::decay_t<Range>>::value, int> = 0,
		std::enable_if_t<has_extensions<std::decay_t<Range>>::value, int>     = 0,
		std::enable_if_t<!std::is_base_of_v<array, std::decay_t<Range>>, int> = 0>  // NOLINT(modernize-use-constraints,modernize-type-traits) for C++20
	auto operator=(Range&& other) -> array& {
		if(array::extensions() == other.extensions()) {
			this->operator()() = std::forward<Range>(other);
		} else if(this->num_elements() == other.extensions().num_elements()) {
			reshape(other.extensions());
			this->operator()() = std::forward<Range>(other);
		} else {
			operator=(static_cast<array>(std::forward<Range>(other)));
		}
		return *this;
	}

	template<
		class Range, class = decltype(std::declval<static_&>().operator=(std::declval<Range&&>())),
		std::enable_if_t<!std::is_base_of_v<array, std::decay_t<Range>>, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	auto from(Range&& other) -> array& {                                            // TODO(correaa) : check that LHS is not read-only?
		if(array::extensions() == other.extensions()) {
			this->operator()() = other;
		} else if(this->num_elements() == other.extensions().num_elements()) {
			reshape(other.extensions());
			this->operator()() = other;
		} else {
			operator=(static_cast<array>(std::forward<Range>(other)));
		}
		return *this;
	}

	friend void swap(array& self, array& other) noexcept(true /*noexcept(self.swap(other))*/) { self.swap(other); }

	void assign(typename array::extensions_type extensions, typename array::element_type const& elem) {
		if(array::extensions() == extensions) {
			adl_fill_n(this->base_, this->num_elements(), elem);
		} else {
			this->clear();
			(*this).array::layout_t::operator=(layout_t<D>{extensions});
			this->base_ = this->static_::array_alloc::allocate(this->num_elements(), nullptr);
			adl_alloc_uninitialized_fill_n(this->alloc(), this->base_, this->num_elements(), elem);
		}
	}

	template<class It>
	auto assign(It first, It last) -> array& {  // cppcheck-suppress duplInheritedMember ; to overwrite
		using std::all_of;
		using std::next;
		if(adl_distance(first, last) == this->size()) {
			static_::ref::assign(first);
		} else {
			this->operator=(array(first, last));
		}
		return *this;
	}
	void assign(std::initializer_list<value_type> values) {
		if(values.size() != 0) {
			assign(values.begin(), values.end());
		}
	}

	template<class Range> auto assign(Range&& other) & -> decltype(assign(adl_begin(std::forward<Range>(other)), adl_end(std::forward<Range>(other)))) {
		return assign(adl_begin(std::forward<Range>(other)), adl_end(std::forward<Range>(other)));  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved)
	}

	auto operator=(std::initializer_list<value_type> values) -> array& {
		if(values.size() == 0) {
			this->clear();
		} else {
			assign(values.begin(), values.end());
		}
		return *this;
	}

	auto reextent(typename array::extensions_type const& extensions) && -> array&& {
		if(extensions == this->extensions()) {
			return std::move(*this);
		}

		auto new_layout = typename array::layout_t{extensions};

		if(new_layout.num_elements() != this->layout().num_elements()) {
			this->destroy();
			this->deallocate();

			this->layout_mutable() = new_layout;  // typename array::layout_t{extensions};
			this->base_            = this->static_::array_alloc::allocate(
                static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(
                    new_layout.num_elements()
                ),
                this->data_elements()  // used as hint
            );

			if constexpr(!(std::is_trivially_default_constructible_v<typename array::element_type> || multi::force_element_trivial_default_construction<typename array::element_type>)) {
				adl_alloc_uninitialized_value_construct_n(this->alloc(), this->base_, this->num_elements());
			}
		} else {
			this->layout_mutable() = new_layout;
		}

		return std::move(*this);
	}

	auto reextent(typename array::extensions_type const& extensions) & -> array& {
		if(extensions == this->extensions()) {
			return *this;
		}
		auto&& tmp = typename array::ref(
			this->static_::array_alloc::allocate(
				static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(
					typename array::layout_t{extensions}.num_elements()
				),
				this->data_elements()  // used as hint
			),
			extensions
		);
		if constexpr(!(std::is_trivially_default_constructible_v<typename array::element_type> || multi::force_element_trivial_default_construction<typename array::element_type>)) {
			adl_alloc_uninitialized_value_construct_n(this->alloc(), tmp.data_elements(), tmp.num_elements());
		}
		auto const is = intersection(this->extensions(), extensions);
		tmp.apply(is) = this->apply(is);  // TODO(correaa) : use (and implement) `.move();`
		this->destroy();
		this->deallocate();
		this->base_            = tmp.base();
		this->layout_mutable() = tmp.layout();
		return *this;
	}

	[[nodiscard]] constexpr auto operator+() const& { return array{*this}; }  // cppcheck-suppress duplInheritedMember ; to overwrite
	[[nodiscard]] constexpr auto operator+() && { return array{*this}; }      // cppcheck-suppress duplInheritedMember ; to overwrite

	auto reextent(typename array::extensions_type const& exs, typename array::element_type const& elem) & -> array& {
		if(exs == this->extensions()) {
			return *this;
		}

		// array tmp(x, e, this->get_allocator());  // TODO(correaa) opportunity missed to use hint allocation
		// auto const is = intersection(this->extensions(), x);
		// tmp.apply(is) = this->apply(is);
		// swap(tmp);

		// implementation with hint
		auto&& tmp = typename array::ref(
			this->static_::array_alloc::allocate(
				static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(typename array::layout_t{exs}.num_elements()),
				this->data_elements()  // use as hint
			),
			exs
		);
		this->uninitialized_fill_n(tmp.data_elements(), static_cast<typename multi::allocator_traits<typename array::allocator_type>::size_type>(tmp.num_elements()), elem);
		auto const is = intersection(this->extensions(), exs);
		tmp.apply(is) = this->apply(is);
		this->destroy();
		this->deallocate();
		this->base_            = tmp.base();  // TODO(correaa) : use (and implement) `.move();`
		this->layout_mutable() = tmp.layout();
		//  (*this).array::layout_t::operator=(tmp.layout());

		return *this;
	}
	// template<class... Indices> constexpr auto reindex(Indices... idxs) && -> array&& {
	//  this->layout_mutable() = this->layout_mutable().creindex(idxs...);
	//  return std::move(*this);
	// }
	// template<class... Indices> constexpr auto reindex(Indices... idxs) & -> array& {
	//  this->layout_mutable() = this->layout_mutable().creindex(idxs...);
	//  // this->layout_mutable().reindex(idxs...);
	//  return *this;
	// }

	// ~array() {
	//  assert(this->stride() != 0);
	// }
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __cpp_deduction_guides

#define BOOST_MULTI_IL std::initializer_list  // NOLINT(cppcoreguidelines-macro-usage) saves a lot of typing, TODO(correaa) use template typedef instead of macro

// vvv MSVC 14.3 in c++17 mode needs paranthesis in dimensionality_type(d)
template<class T> dynamic_array(BOOST_MULTI_IL<T>) -> dynamic_array<T, static_cast<dimensionality_type>(1U), std::allocator<T>>;  // MSVC needs the allocator argument error C2955: 'boost::multi::dynamic_array': use of class template requires template argument list
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<T>>) -> dynamic_array<T, static_cast<dimensionality_type>(2U), std::allocator<T>>;
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>) -> dynamic_array<T, static_cast<dimensionality_type>(3U), std::allocator<T>>;
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>) -> dynamic_array<T, static_cast<dimensionality_type>(4U), std::allocator<T>>;
template<class T> dynamic_array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>>) -> dynamic_array<T, static_cast<dimensionality_type>(5U), std::allocator<T>>;

// TODO(correaa) add zero dimensional case?
template<class T> array(BOOST_MULTI_IL<T>) -> array<T, static_cast<dimensionality_type>(1U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<T>>) -> array<T, static_cast<dimensionality_type>(2U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>) -> array<T, static_cast<dimensionality_type>(3U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>) -> array<T, static_cast<dimensionality_type>(4U)>;
template<class T> array(BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<BOOST_MULTI_IL<T>>>>>) -> array<T, static_cast<dimensionality_type>(5U)>;

#undef BOOST_MULTI_IL

template<class T> array(T[]) -> array<T, static_cast<dimensionality_type>(1U)>;  // NOSONAR(cpp:S5945) NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

//  vvv these are necessary to catch {n, m, ...} notation (or single integer notation)
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<0>, T) -> array<T, static_cast<dimensionality_type>(0U)>;  // TODO(correaa) use some std::allocator_traits instead of is_allocator
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<1>, T) -> array<T, static_cast<dimensionality_type>(1U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<2>, T) -> array<T, static_cast<dimensionality_type>(2U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<3>, T) -> array<T, static_cast<dimensionality_type>(3U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<4>, T) -> array<T, static_cast<dimensionality_type>(4U)>;
template<class T, class = std::enable_if_t<!multi::is_allocator_v<T>>> array(iextensions<5>, T) -> array<T, static_cast<dimensionality_type>(5U)>;

// generalization, will not work with naked {n, m, ...} notation (or single integer notation)
template<dimensionality_type D, class T, class = std::enable_if_t<!boost::multi::is_allocator_v<T>>>
array(iextensions<D>, T) -> array<T, D>;

template<class MatrixRef, class DT = typename MatrixRef::decay_type, class T = typename DT::element_type, dimensionality_type D = DT::rank_v, class Alloc = typename DT::allocator_type>
array(MatrixRef) -> array<T, D, Alloc>;

template<class MatValues, class T = typename MatValues::element, dimensionality_type D = MatValues::rank_v>
array(MatValues) -> array<T, D>;

template<class MatValues, class T = typename MatValues::element, dimensionality_type D = MatValues::rank_v, class Alloc = std::allocator<T>, class = std::enable_if_t<multi::is_allocator_v<Alloc>>>  /// , class Alloc = typename DT::allocator_type>
array(MatValues, Alloc) -> array<T, D, Alloc>;

template<typename T, dimensionality_type D, typename P> array(subarray<T, D, P>) -> array<T, D>;

template<
	class Range, std::enable_if_t<!has_extensions<Range>::value, int> = 0,
	typename V = decltype(*::std::begin(std::declval<Range const&>()))
	// typename V = typename std::iterator_traits<decltype(::std::begin(std::declval<Range const&>()))>::value_type
	>
array(Range) -> array<V, 1>;

template<class Reference>
auto operator+(Reference&& ref) -> decltype(array(std::forward<Reference>(ref))) {
	return array(std::forward<Reference>(ref));
}

#endif  // ends defined(__cpp_deduction_guides)

template<class T, std::size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
auto decay(T const (&arr)[N]) noexcept -> multi::array<std::remove_all_extents_t<T[N]>, std::rank_v<T[N]>> {
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	return multi::array_cref<std::remove_all_extents_t<T[N]>, std::rank_v<T[N]>>(data_elements(arr), extensions(arr));
}

template<class T, std::size_t N>
struct array_traits<T[N], void, void> {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	using reference  = T&;
	using element    = std::remove_all_extents_t<T[N]>;  // NOSONAR(cpp:S5945) NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	using decay_type = multi::array<T, 1>;
};

}  // end namespace boost::multi

namespace boost::multi::pmr {

#ifdef BOOST_MULTI_HAS_MEMORY_RESOURCE
template<class T, boost::multi::dimensionality_type D>
using array = boost::multi::array<T, D, std::pmr::polymorphic_allocator<T>>;
#else
template<class T, boost::multi::dimensionality_type D>
struct [[deprecated("no PMR allocator")]] array;  // your version of C++ doesn't provide polymorphic_allocators
#endif

}  // end namespace boost::multi::pmr

// common_reference for compatibility with ranges
#if defined(__cpp_lib_common_reference) || defined(__cpp_lib_ranges)
// TODO(correaa) achieve this by normal inheritance
// NOLINTBEGIN(cert-dcl58-cpp)
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array     &&,          ::boost::multi::array<T, D, A...>                         &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array     &&,          ::boost::multi::array<T, D, A...>                    const&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                         &, typename ::boost::multi::array<T, D, A...>::basic_const_array     &&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                    const&, typename ::boost::multi::array<T, D, A...>::basic_const_array     &&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array     &&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array       ,          ::boost::multi::array<T, D, A...>                         &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array       ; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                    const&, typename ::boost::multi::array<T, D, A...>::basic_const_array const&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array const&; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array const&,          ::boost::multi::array<T, D, A...>                    const&> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array const&; };

// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<typename ::boost::multi::array<T, D, A...>::basic_const_array      &,          ::boost::multi::array<T, D, A...>                         &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array      &; };
// template<class T, ::boost::multi::dimensionality_type D, class... A> struct std::common_reference<         ::boost::multi::array<T, D, A...>                         &, typename ::boost::multi::array<T, D, A...>::basic_const_array      &> { using type = typename ::boost::multi::array<T, D, A...>::basic_const_array      &; };
// NOLINTEND(cert-dcl58-cpp)
#endif

namespace boost::serialization {

template<typename T> struct version;  // in case serialization was not included before

template<typename T, boost::multi::dimensionality_type D, class A>
struct version<boost::multi::array<T, D, A>> {
	using type = std::integral_constant<int, BOOST_MULTI_SERIALIZATION_ARRAY_VERSION>;  // TODO(correaa) use constexpr variable here, not a macro
	// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class) for backward compatibility with Boost Serialization
	enum /*class value_t*/ { value = type::value };  // NOSONAR(cpp:S3642)  // https://community.sonarsource.com/t/suppress-issue-in-c-source-file/43154/24
};

}  // end namespace boost::serialization

#ifdef _MSC_VER
#pragma warning(pop)
#endif

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_ARRAY_HPP_
