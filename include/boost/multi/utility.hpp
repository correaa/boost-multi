// Copyright 2018-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_UTILITY_HPP
#define BOOST_MULTI_UTILITY_HPP
#pragma once

#include <boost/multi/detail/implicit_cast.hpp>  // IWYU pragma: export
#include <boost/multi/detail/layout.hpp>

#include <functional>   // for std::invoke
#include <iterator>     // for std::size (in c++17)
#include <memory>       // for allocator<>
#include <type_traits>  // for std::invoke_result

#if defined(__NVCC__)
#define BOOST_MULTI_HD __host__ __device__
#else
#define BOOST_MULTI_HD
#endif

namespace boost::multi {

template<class T, class Ptr = T*>
struct move_ptr : private std::move_iterator<Ptr> {
	using difference_type   = typename std::iterator_traits<std::move_iterator<Ptr>>::difference_type;
	using value_type        = typename std::iterator_traits<std::move_iterator<Ptr>>::value_type;
	using pointer           = Ptr;
	using reference         = typename std::move_iterator<Ptr>::reference;
	using iterator_category = typename std::iterator_traits<std::move_iterator<Ptr>>::iterator_category;

	template<class U> using rebind = std::conditional_t<
		std::is_const_v<U>,
		typename std::pointer_traits<Ptr>::template rebind<U>,
		move_ptr<U, U*>>;

	using std::move_iterator<Ptr>::move_iterator;

	BOOST_MULTI_HD constexpr /**/ operator Ptr() const { return std::move_iterator<Ptr>::base(); }  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) // NOSONAR(cpp:S1709) decay to lvalue should be easy

	BOOST_MULTI_HD constexpr auto operator+=(difference_type n) -> move_ptr& {
		static_cast<std::move_iterator<Ptr>&>(*this) += n;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator-=(difference_type n) -> move_ptr& {
		static_cast<std::move_iterator<Ptr>&>(*this) -= n;
		return *this;
	}

	BOOST_MULTI_HD constexpr auto operator+(difference_type n) const -> move_ptr {
		move_ptr ret{*this};
		ret += n;
		return ret;
	}
	BOOST_MULTI_HD constexpr auto operator-(difference_type n) const -> move_ptr {
		move_ptr ret{*this};
		ret -= n;
		return ret;
	}

	BOOST_MULTI_HD constexpr auto operator-(move_ptr const& other) const -> difference_type { return static_cast<std::move_iterator<Ptr> const&>(*this) - static_cast<std::move_iterator<Ptr> const&>(other); }

	BOOST_MULTI_HD constexpr auto operator<(move_ptr const& other) const -> bool { return static_cast<std::move_iterator<Ptr> const&>(*this) < static_cast<std::move_iterator<Ptr> const&>(other); }

	constexpr auto                operator*() const -> decltype(auto) { return *static_cast<std::move_iterator<Ptr> const&>(*this); }
	BOOST_MULTI_HD constexpr auto operator[](difference_type n) const -> decltype(auto) { return *((*this) + n); }

	BOOST_MULTI_HD constexpr auto operator==(move_ptr const& other) const -> bool { return static_cast<std::move_iterator<Ptr> const&>(*this) == static_cast<std::move_iterator<Ptr> const&>(other); }
	BOOST_MULTI_HD constexpr auto operator!=(move_ptr const& other) const -> bool { return static_cast<std::move_iterator<Ptr> const&>(*this) != static_cast<std::move_iterator<Ptr> const&>(other); }
};

template<class T> struct ref_add_const {
	using type = T const;
};  // this is not the same as std::add_const

template<class T> struct ref_add_const<T const> {
	using type = T const;
};
template<class T> struct ref_add_const<T const&> {
	using type = T const&;
};
template<class T> struct ref_add_const<T&> {
	using type = T const&;
};

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

template<class T, class UF, class Ptr, class Ref = std::invoke_result_t<UF const&, typename std::iterator_traits<Ptr>::reference>>
struct transform_ptr {
	using difference_type   = typename std::iterator_traits<Ptr>::difference_type;
	using value_type        = std::decay_t<Ref>;  // typename std::iterator_traits<std::move_iterator<Ptr>>::value_type;
	using pointer           = Ptr;
	using reference         = Ref;
	using iterator_category = typename std::iterator_traits<Ptr>::iterator_category;

	template<class U> using rebind =
		transform_ptr<
			std::remove_cv_t<U>,
			UF, Ptr,
			std::conditional_t<
				std::is_const_v<U>,
				typename ref_add_const<Ref>::type,
				Ref>
			// typename std::conditional<
			//  std::is_const_v<U>,
			//  typename ref_add_const<Ref>::type,
			//  Ref
			// >::type
			>;

#if defined(__GNUC__) && (__GNUC__ < 9)
	constexpr explicit transform_ptr(std::nullptr_t nil) : p_{nil} /*, f_{}*/ {}  // seems to be necessary for gcc 7
#endif

	constexpr transform_ptr(pointer ptr, UF fun) : p_{ptr}, f_(std::move(fun)) {}

	template<class Other, class P = typename Other::pointer, decltype(detail::implicit_cast<pointer>(std::declval<P>()))* = nullptr>
	// cppcheck-suppress noExplicitConstructor
	constexpr /*mplc*/ transform_ptr(Other const& other) : p_{other.p_}, f_{other.f_} {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions) // NOSONAR(cpp:S1709)

	template<class Other, class P = typename Other::pointer, decltype(detail::explicit_cast<pointer>(std::declval<P>()))* = nullptr>
	constexpr explicit transform_ptr(Other const& other) : p_{other.p_}, f_{other.f_} {}

	// constexpr auto functor() const -> UF {return f_;}
	constexpr auto base() const -> Ptr const& { return p_; }
	constexpr auto operator*() const -> reference {  // NOLINT(readability-const-return-type) in case synthesis reference is a `T const`
		// invoke allows for example to use .transformed( &member) instead of .transformed( std::mem_fn(&member) )
		return std::invoke(f_, *p_);  // NOLINT(readability-const-return-type) in case synthesis reference is a `T const`
	}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"  // TODO(correaa) use checked span
#endif

	constexpr auto operator+=(difference_type n) -> transform_ptr& {
		p_ += n;
		return *this;
	}
	constexpr auto operator-=(difference_type n) -> transform_ptr& {
		p_ -= n;
		return *this;
	}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

	constexpr auto operator+(difference_type n) const -> transform_ptr { return transform_ptr{*this} += n; }
	constexpr auto operator-(difference_type n) const -> transform_ptr { return transform_ptr{*this} -= n; }

	constexpr auto friend operator+(difference_type n, transform_ptr const& self) { return self + n; }

	constexpr auto operator-(transform_ptr const& other) const -> difference_type { return p_ - other.p_; }

	constexpr auto operator[](difference_type n) const -> reference { return *((*this) + n); }  // NOLINT(readability-const-return-type) transformed_view might return by const value.

	constexpr auto operator==(transform_ptr const& other) const -> bool { return p_ == other.p_; }
	constexpr auto operator!=(transform_ptr const& other) const -> bool { return p_ != other.p_; }

	constexpr auto operator==(std::nullptr_t const& nil) const -> bool { return p_ == nil; }
	constexpr auto operator!=(std::nullptr_t const& nil) const -> bool { return p_ != nil; }

	constexpr auto operator<(transform_ptr const& other) const -> bool { return p_ < other.p_; }

 private:
	Ptr p_;
	UF  f_;  // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members) technically this type can be const

	template<class, class, class, class> friend struct transform_ptr;
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

template<class Array, typename Reference = void, typename Element = void>
struct array_traits;

template<class Array, typename Reference, typename Element>
struct array_traits {
	using reference              = typename Array::reference;
	using element                = typename Array::element;
	using element_ptr            = typename Array::element_ptr;
	using decay_type             = typename Array::decay_type;
	using default_allocator_type = typename Array::default_allocator_type;
};

template<class T, typename = typename T::rank>
auto        has_rank_aux(T const&) -> std::true_type;
inline auto has_rank_aux(...) -> std::false_type;

template<class T> struct has_rank : decltype(has_rank_aux(std::declval<T>())){};

template<typename T> struct rank;

template<class T, typename = decltype(std::declval<T&>().move())>
auto        has_member_move_aux(T const&) -> std::true_type;
inline auto has_member_move_aux(...) -> std::false_type;

template<class T> struct has_member_move : decltype(has_member_move_aux(std::declval<T>())){};

template<typename T, typename = std::enable_if_t<has_rank<T>{}>>
constexpr auto rank_aux(T const&) -> typename T::rank;

template<typename T, typename = std::enable_if_t<!has_rank<T>::value>>
constexpr auto rank_aux(T const&) -> std::integral_constant<size_t, std::rank_v<T>>;

template<typename T> struct rank : decltype(rank_aux(std::declval<T>())){};

template<class Pointer, std::enable_if_t<std::is_pointer<Pointer>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) special sfinae trick
constexpr auto stride(Pointer /*ptr*/) -> std::ptrdiff_t { return 1; }

template<class Pointer, std::enable_if_t<std::is_pointer<Pointer>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) special sfinae trick
constexpr auto base(Pointer ptr) -> Pointer { return ptr; }

template<class TPointer, class U>
constexpr auto reinterpret_pointer_cast(U* other)                                                 // name taken from thrust::reinterpret_pointer_cast, which is difference from std::reinterpret_pointer_cast(std::shared_ptr<T>)
	-> decltype(reinterpret_cast<TPointer>(other)) { return reinterpret_cast<TPointer>(other); }  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) : unavoidalbe implementation?

template<class T, std::size_t N>
constexpr auto size(T const (& /*array*/)[N]) noexcept { return static_cast<multi::size_type>(N); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, typename = typename T::get_allocator>
auto        has_get_allocator_aux(T const&) -> std::true_type;
inline auto has_get_allocator_aux(...) -> std::false_type;

template<class T, std::size_t N>
constexpr auto get_allocator(T (& /*array*/)[N]) noexcept -> std::allocator<std::decay_t<std::remove_all_extents_t<T[N]>>> { return {}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T>
constexpr auto get_allocator(T* const& /*t*/)
	-> decltype(std::allocator<typename std::iterator_traits<T*>::value_type>{}) {
	return std::allocator<typename std::iterator_traits<T*>::value_type>{};
}

template<class T>
constexpr auto default_allocator_of(T* /*unused*/) {
	return std::allocator<typename std::iterator_traits<T*>::value_type>{};
}

template<class T>
constexpr auto to_address(T* const& ptr)
	-> decltype(ptr) {
	return ptr;
}

template<class T>
auto has_get_allocator_aux(T const& cont) -> decltype(cont.get_allocator(), std::true_type{});

template<class T> struct has_get_allocator : decltype(has_get_allocator_aux(std::declval<T>())){};

// template<class T1, class T2, typename Ret = T1>  // std::common_type_t<T1, T2>>
// auto common(T1 const& val1, T2 const& val2) -> Ret {
//  return val1 == val2?
//      val1:
//      Ret{}
//  ;
// }

template<class T>
auto        has_num_elements_aux(T const& /*array*/) -> decltype(std::declval<T const&>().num_elements() + 1, std::true_type{});
inline auto has_num_elements_aux(...) -> decltype(std::false_type{});
template<class T> struct has_num_elements : decltype(has_num_elements_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class A, typename = std::enable_if_t<has_num_elements<A>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
constexpr auto num_elements(A const& arr)
	-> std::make_signed_t<decltype(arr.num_elements())> {
	return static_cast<std::make_signed_t<decltype(arr.num_elements())>>(arr.num_elements());
}

template<class T>
auto        has_size_aux(T const& cont) -> decltype(std::size(cont), std::true_type{});
inline auto has_size_aux(...) -> decltype(std::false_type{});
template<class T> struct has_size : decltype(has_size_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T>
auto        has_data_elements_aux(T&& array) -> decltype(array.data_elements() + 1, std::true_type{});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) why +1?
inline auto has_data_elements_aux(...) -> decltype(std::false_type{});
template<class T> struct has_data_elements : decltype(has_data_elements_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T>
auto        has_base_aux(T&& array) -> decltype(array.base() + 1, std::true_type{});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) why +1?
inline auto has_base_aux(...) -> decltype(std::false_type{});
template<class T> struct has_base : decltype(has_base_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

namespace detail {
template<class T>
auto        has_data_aux(T&& cont) -> decltype(cont.data_elements() + 1, std::true_type{});  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic) TODO(correaa) why +1?
inline auto has_data_aux(...) -> decltype(std::false_type{});
}  // end namespace detail
template<class T> struct has_data : decltype(detail::has_data_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class Array, std::enable_if_t<has_data<std::decay_t<Array>>::value && !has_data_elements<std::decay_t<Array>>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto data_elements(Array& arr) { return std::data(arr); }

template<class Array, std::enable_if_t<has_data<std::decay_t<Array>>::value && !has_data_elements<std::decay_t<Array>>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto data_elements(Array const& arr) { return std::data(arr); }                                                                           // .data();}

template<class A, std::enable_if_t<!has_num_elements<A>::value && has_size<A>::value && has_data<A>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
constexpr auto num_elements(A const& arr) -> std::make_signed_t<decltype(std::size(arr))> {
	return static_cast<std::make_signed_t<decltype(std::size(arr))>>(std::size(arr));  // (arr.size());
}

template<class A, std::enable_if_t<has_data_elements<A>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
constexpr auto data_elements(A const& arr)
	-> decltype(arr.data_elements()) {
	return arr.data_elements();
}

template<class T, std::enable_if_t<!std::is_array_v<std::decay_t<T>> && !has_data_elements<std::decay_t<T>>::value && !has_data<std::decay_t<T>>::value, int> = 0>  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays,modernize-use-constraints) for C++20
constexpr auto data_elements(T& value) -> decltype(&value) { return &value; }

template<class A> struct num_elements_t : std::integral_constant<std::ptrdiff_t, 1> {};

template<class T, std::size_t N> struct num_elements_t<T[N]> : std::integral_constant<std::ptrdiff_t, (N * num_elements_t<T>{})> {};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, std::size_t N> struct num_elements_t<T (&)[N]> : num_elements_t<T[N]> {};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, std::size_t N>
constexpr auto num_elements(T const (& /*array*/)[N]) noexcept { return num_elements_t<T[N]>{}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class Vector, typename = std::enable_if_t<std::is_same_v<typename Vector::pointer, decltype(std::declval<Vector>().data())>>, class = decltype(Vector{}.resize(1))>
auto data_elements(Vector const& vec)
	-> decltype(vec.data()) {
	return vec.data();
}

template<class T, std::size_t N>
constexpr auto stride(T const (& /*array*/)[N]) noexcept -> std::ptrdiff_t { return num_elements_t<T>{}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, std::size_t N>
constexpr auto is_compact(T const (& /*t*/)[N]) noexcept -> bool { return true; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, std::size_t N>
constexpr auto offset(T const (& /*t*/)[N]) noexcept -> std::ptrdiff_t { return 0; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, std::size_t N>
[[deprecated("use data_elements instead")]]  // this name is bad because when the element belongs to std:: then std::data is picked up by ADL and the
constexpr auto
data(T (&array)[N]) noexcept { return data(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, std::size_t N>
constexpr auto data_elements(T (&array)[N]) noexcept { return data_elements(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T>
auto        has_dimensionality_aux(T const& /*array*/) -> decltype(T::rank_v, std::true_type{});
inline auto has_dimensionality_aux(...) -> decltype(std::false_type{});
template<class T> struct has_dimensionality : decltype(has_dimensionality_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class Container, std::enable_if_t<has_dimensionality<Container>{}, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
constexpr auto dimensionality(Container const& /*container*/)
	-> std::decay_t<decltype(typename Container::rank{} + 0)> {
	return Container::rank_v;
}

template<class T>
auto        has_dimensionaliy_member_aux(T const& /*array*/) -> decltype(static_cast<void>(static_cast<boost::multi::dimensionality_type>(T::rank_v)), std::true_type{});
inline auto has_dimensionaliy_member_aux(...) -> decltype(std::false_type{});
template<class T> struct has_dimensionality_member : decltype(has_dimensionaliy_member_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T, typename = std::enable_if_t<!has_dimensionality_member<T>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto dimensionality(T const& /*, void* = nullptr*/) { return 0; }

template<class T, std::size_t N>
constexpr auto dimensionality(T const (&array)[N]) { return 1 + dimensionality(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility

template<class T, class Ret = decltype(std::declval<T const&>().sizes())>
constexpr auto sizes(T const& arr) noexcept -> Ret { return arr.sizes(); }

template<class T, std::enable_if_t<!has_dimensionality<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
constexpr auto sizes(T const& /*unused*/) noexcept { return tuple<>{}; }

template<class T, std::size_t N>
constexpr auto sizes(T const (&array)[N]) noexcept {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility
													  //  using std::size; // this line needs c++17
	return multi::detail::ht_tuple(multi::size(array), multi::sizes(array[0]));
}

template<class T, std::size_t N>
constexpr auto base(T (&array)[N]) noexcept {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility
	return data_elements(array);
}

template<class T, std::size_t N>
constexpr auto base(T (*&array)[N]) noexcept { return base(*array); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, typename = std::enable_if_t<!std::is_array_v<T>>>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto base(T const* ptr) noexcept { return ptr; }

template<class T, typename = std::enable_if_t<!std::is_array_v<T>>>  // NOLINT(modernize-use-constraints) TODO(correaa)
constexpr auto base(T* ptr) noexcept { return ptr; }

template<class T>
constexpr auto corigin(T const& value) { return &value; }

template<class T, std::size_t N>
constexpr auto corigin(T const (&array)[N]) noexcept { return corigin(array[0]); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) for backwards compatibility

template<class T, typename = decltype(std::declval<T>().extension())>
auto        has_extension_aux(T const&) -> std::true_type;
inline auto has_extension_aux(...) -> std::false_type;
template<class T> struct has_extension : decltype(has_extension_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class Container, class = std::enable_if_t<!has_extension<Container>::value>>  // NOLINT(modernize-use-constraints) TODO(correaa)
auto extension(Container const& cont)                                                  // TODO(correaa) consider "extent"
	-> decltype(multi::extension_t<std::make_signed_t<decltype(size(cont))>>(0, static_cast<std::make_signed_t<decltype(size(cont))>>(size(cont)))) {
	return multi::extension_t<std::make_signed_t<decltype(size(cont))>>(0, static_cast<std::make_signed_t<decltype(size(cont))>>(size(cont)));
}

template<dimensionality_type Rank, class Container, std::enable_if_t<!has_extension<Container>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++20
auto extensions(Container const& cont) {
	if constexpr(Rank == 0) {
		return multi::extensions_t<0>{};
	} else {
		using std::size;
		return multi::extension_t<std::make_signed_t<decltype(size(cont))>>(0, static_cast<std::make_signed_t<decltype(size(cont))>>(size(cont))) * extensions<Rank - 1>(cont.front());
	}
}

// template<class T, typename = decltype(std::declval<T>().shape())>
//        auto has_shape_aux(T const&) -> std::true_type;
// inline auto has_shape_aux(...     ) -> std::false_type;

template<class T> struct has_shape : decltype(has_shape_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) trick

template<class T, typename = decltype(std::declval<T const&>().extensions())>
auto        has_extensions_aux(T const&) -> std::true_type;
inline auto has_extensions_aux(...) -> std::false_type;

template<class T> struct has_extensions : decltype(has_extensions_aux(std::declval<T>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) trick

template<class T, std::enable_if_t<has_extensions<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
[[nodiscard]] constexpr auto extensions(T const& array) -> std::decay_\
t<decltype(array.extensions())> {
	return array.extensions();
}

template<class BoostMultiArray, std::size_t... I>
constexpr auto extensions_aux2(BoostMultiArray const& arr, std::index_sequence<I...> /*012*/) {
	return boost::multi::extensions_t<BoostMultiArray::dimensionality>(
		boost::multi::iextension{static_cast<multi::index>(arr.index_bases()[I]), static_cast<multi::index>(arr.index_bases()[I]) + static_cast<multi::index>(arr.shape()[I])}...  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	);
}

template<class Element, class T, std::enable_if_t<has_extensions<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
[[nodiscard]] auto extensions_of(T const& array) {
	if constexpr(std::is_convertible_v<T const&, Element>) {
		return boost::multi::extensions_t<0>{};
	}
	if constexpr(std::is_convertible_v<typename T::reference, Element>) {
		return boost::multi::extensions_t<1>{array.extension()};
	}
}

template<class... Ts> auto what() -> std::tuple<Ts&&...>        = delete;
template<class... Ts> auto what(Ts&&...) -> std::tuple<Ts&&...> = delete;  // NOLINT(cppcoreguidelines-missing-std-forward)

template<class Arr2D>
auto transposed(Arr2D&& arr)
	-> decltype(std::forward<Arr2D>(arr).transposed()) {
	return std::forward<Arr2D>(arr).transposed();
}

// template<class BoostMultiArray, std::enable_if_t<has_shape<BoostMultiArray>::value && !has_extensions<BoostMultiArray>::value, int> =0>
// constexpr auto extensions(BoostMultiArray const& array) {
//  return extensions_aux2(array, std::make_index_sequence<BoostMultiArray::dimensionality>{});
// }

template<class T, std::enable_if_t<!has_extensions<T>::value /*&& !has_shape<T>::value*/, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
constexpr auto extensions(T const& /*unused*/) -> multi::layout_t<0>::extensions_type { return {}; }

template<class T, std::size_t N>
constexpr auto extensions(T (&array)[N]) {  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) : for backwards compatibility
	return index_extension{N} * extensions(array[0]);
}

template<dimensionality_type D>
struct extensions_aux {
	template<class T>
	static auto call(T const& array) {
		return array.extension() * extensions<D - 1>(array);
		// return tuple_cat(std::make_tuple(array.extension()), extensions<D-1>(array));
	}
};

template<> struct extensions_aux<0> {
	template<class T> static auto call(T const& /*unused*/) { return multi::extensions_t<0>{}; }  // std::make_tuple();}
};

template<dimensionality_type D, class T, std::enable_if_t<has_extension<T>::value, int> = 0>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
auto extensions(T const& array) {
	return extensions_aux<D>::call(array);
}

template<class T1> struct extensions_t_aux;

template<class T1, class T2> auto extensions_me(T2 const& array) {
	return extensions_t_aux<T1>::call(array);
}

template<class T1> struct extension_t_aux {
	static auto call(T1 const& /*unused*/) { return std::make_tuple(); }
	template<class T2>
	static auto call(T2 const& array) { return tuple_cat(std::make_tuple(array.extension()), extensions_me<T1>(*begin(array))); }
};

template<class T, typename = decltype(std::declval<T const&>().layout())>
auto        has_layout_member_aux(T const&) -> std::true_type;
inline auto has_layout_member_aux(...) -> std::false_type;

template<class T>
struct has_layout_member : decltype(has_layout_member_aux(std::declval<T const&>())){};  // NOLINT(cppcoreguidelines-pro-type-vararg,hicpp-vararg,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)

template<class T, typename = std::enable_if_t<has_layout_member<T const&>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
auto layout(T const& array)
	-> decltype(array.layout()) {
	return array.layout();
}

template<class T, typename = std::enable_if_t<!has_layout_member<T const&>{}>>  // NOLINT(modernize-use-constraints) TODO(correaa) in C++20
auto layout(T const& /*unused*/) -> layout_t<0> { return {}; }

template<class T, std::size_t N>
constexpr auto layout(T (&array)[N]) { return multi::layout_t<std::rank_v<T[N]>>{multi::extensions(array)}; }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): for backward compatibility

template<class T, std::size_t N>
constexpr auto strides(T (&array)[N]) { return layout(array).strides(); }  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): for backward compatibility

template<class T, std::size_t N>
struct array_traits<std::array<T, N>> {
	static constexpr auto dimensionality() -> dimensionality_type { return 1; }

	using reference   = T&;
	using value_type  = std::decay_t<T>;
	using pointer     = T*;
	using element     = value_type;
	using element_ptr = pointer;
	using decay_type  = std::array<value_type, N>;
};

template<class T, std::size_t N, std::size_t M>
struct array_traits<std::array<std::array<T, M>, N>> {
	static constexpr auto dimensionality() -> dimensionality_type { return 1 + array_traits<std::array<T, M>>::dimensionality(); }

	using reference   = std::array<T, M>&;
	using value_type  = std::array<std::decay_t<T>, M>;
	using pointer     = std::array<T, M>*;
	using element     = typename array_traits<std::array<T, M>>::element;
	using element_ptr = typename array_traits<std::array<T, M>>::element;
	using decay_type  = std::array<value_type, M>;
};

template<class T, std::size_t N> constexpr auto                data_elements(std::array<T, N>& arr) noexcept { return arr.data(); }
template<class T, std::size_t M, std::size_t N> constexpr auto data_elements(std::array<std::array<T, M>, N>& arr) noexcept { return data_elements(arr[0]); }

template<class T, std::size_t N> constexpr auto                data_elements(std::array<T, N> const& arr) noexcept { return arr.data(); }
template<class T, std::size_t M, std::size_t N> constexpr auto data_elements(std::array<std::array<T, M>, N> const& arr) noexcept { return data_elements(arr[0]); }

template<class T, std::size_t N> constexpr auto data_elements(std::array<T, N>&& arr) noexcept { return std::move(arr).data(); }

template<class T, std::size_t M, std::size_t N>
constexpr auto data_elements(std::array<std::array<T, M>, N>&& arr) noexcept { return data_elements(std::move(arr)[0]); }

template<class T, std::size_t N> constexpr auto num_elements(std::array<T, N> const& /*unused*/) noexcept
	-> std::ptrdiff_t { return N; }

template<class T, std::size_t M, std::size_t N>
constexpr auto num_elements(std::array<std::array<T, M>, N> const& arr)
	-> std::ptrdiff_t { return static_cast<std::ptrdiff_t>(N) * num_elements(arr[0]); }

template<class T, std::size_t N>
constexpr auto dimensionality(std::array<T, N> const& /*unused*/) -> boost::multi::dimensionality_type { return 1; }

template<class T, std::size_t M, std::size_t N>
constexpr auto dimensionality(std::array<std::array<T, M>, N> const& arr) -> boost::multi::dimensionality_type {
	return 1 + dimensionality(arr[0]);
}

template<class T, std::size_t N>
constexpr auto extensions(std::array<T, N> const& /*arr*/) {
	return multi::extensions_t<1>{multi::index_extension(0, N)};
}

template<class T, std::size_t N, std::size_t M>
auto extensions(std::array<std::array<T, N>, M> const& arr) {
	return multi::iextension{M} * extensions(arr[0]);
}

template<class T, std::size_t N>
constexpr auto stride(std::array<T, N> const& /*arr*/) {
	return static_cast<multi::size_type>(1U);  // multi::stride_type?
}

template<class T, std::size_t N, std::size_t M>
constexpr auto stride(std::array<std::array<T, N>, M> const& arr) {
	return num_elements(arr[0]);
}

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wlarge-by-value-copy"  // TODO(correaa) use checked span
#endif

template<class T, std::size_t N>
constexpr auto layout(std::array<T, N> const& arr) {
	return multi::layout_t<multi::array_traits<std::array<T, N>>::dimensionality()>{multi::extensions(arr)};
}

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace detail {
inline auto valid_mull(int age) -> bool {
	return age >= 21;
}
}  // end namespace detail

}  // end namespace boost::multi

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_UTILITY_HPP
