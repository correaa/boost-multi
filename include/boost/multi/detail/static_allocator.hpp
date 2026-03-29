// Copyright 2023-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_STATIC_ALLOCATOR_HPP
#define BOOST_MULTI_DETAIL_STATIC_ALLOCATOR_HPP
// #pragma once

#include <boost/multi/detail/config/NODISCARD.hpp>
#include <boost/multi/detail/config/NO_UNIQUE_ADDRESS.hpp>

#include <cstddef>
#include <new>
#include <type_traits>

namespace boost::multi::detail {

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, std::size_t N>
class static_allocator {  // NOSONAR(cpp:S4963) this allocator has special semantics
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // structure padded due to alignment
#endif

	BOOST_MULTI_NO_UNIQUE_ADDRESS
	union storage_t_ {
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
		// In C++20, initialize typed_buffer_ as the active union member so constexpr
		// context can access it. The container overwrites elements via std::construct_at.
		T typed_buffer_[N];  // NOLINT(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,cppcoreguidelines-pro-type-union-access)
		constexpr storage_t_() : typed_buffer_{} {}
		constexpr ~storage_t_() {}
#else
		alignas(T) std::byte buffer_[sizeof(T) * N];  // NOLINT(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays)
		constexpr storage_t_() : buffer_{} {}
		~storage_t_() {}
#endif
	} storage_;

#ifdef _MSC_VER
#pragma warning(pop)
#endif

 public:
	using value_type = T;
	using pointer    = T*;
	// using const_pointer = T const*;
	using size_type = std::size_t;

	template<class TT> struct rebind {
		using other = static_allocator<TT, N>;
	};

 private:

 	static constexpr void check_size_(size_type n) {
		if(n > N) {
			assert(false);
			throw std::bad_alloc{};
		}
	}

 public:

	static constexpr auto max_size() noexcept -> size_type { return N; }
	static constexpr auto capacity() noexcept -> size_type { return N; }

	constexpr static_allocator() = default;

	template<class TT, std::size_t NN>
	explicit constexpr static_allocator(static_allocator<TT, NN> const& /*other*/) {  // NOLINT(hicpp-explicit-conversions,google-explicit-constructor) follow std::allocator  // NOSONAR
		static_assert(NN == N);
	}

	constexpr static_allocator(static_allocator const& /*other*/) noexcept {}  // std::vector makes a copy right away; do not copy buffer

	constexpr static_allocator(static_allocator&& /*other*/) noexcept {}  // called by elements during move construction of a vector; do not move buffer

	[[deprecated("don't assign a container with static_allocator")]]
	auto operator=(static_allocator const&) -> static_allocator& = delete;

	[[deprecated("don't assign a container with static_allocator")]]
	auto operator=(static_allocator&&) -> static_allocator& = delete;

#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
	constexpr
#endif
		~static_allocator() = default;

	auto select_on_container_copy_construction() noexcept -> static_allocator = delete;

	using propagate_on_container_move_assignment = std::false_type;
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_swap            = std::false_type;

	using is_always_equal = std::true_type;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4068)
#endif
	BOOST_MULTI_NODISCARD("because otherwise it will generate a memory leak")
	constexpr auto allocate(size_type n) -> pointer {
		check_size_(n);
#if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
		return std::data(storage_.typed_buffer_);
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"  // buffer_ is aligned as T
		return reinterpret_cast<pointer>(storage_.buffer_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
#pragma GCC diagnostic pop
#endif
	}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

	static constexpr void deallocate(pointer /*ptr*/, size_type n) {
		check_size_(n);
	}
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<class T, std::size_t N, class U>
constexpr auto operator==(static_allocator<T, N> const& /*a1*/, static_allocator<U, N> const& /*a2*/) noexcept { return true; }

template<class T, std::size_t N, class U>
constexpr auto operator!=(static_allocator<T, N> const& /*a1*/, static_allocator<U, N> const& /*a2*/) noexcept { return false; }

template<class T, std::size_t N>
[[deprecated("don't swap containers with static_allocator")]]
void swap(static_allocator<T, N>&, static_allocator<T, N>&) noexcept = delete;

namespace detail {
template<class T, std::size_t N>
auto is_static_allocator_aux(static_allocator<T, N> const&) -> std::true_type;
auto is_static_allocator_aux(...) -> std::false_type;
}

template<class T>
struct is_static_allocator : decltype(detail::is_static_allocator_aux(std::declval<T>())){};

}  // end namespace boost::multi::detail
#endif  // BOOST_MULTI_DETAIL_STATIC_ALLOCATOR_HPP
