// Copyright 2018-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array_ref.hpp>  // for array_ptr, array_ref, subarray

#include <boost/core/lightweight_test.hpp>

#include <array>        // for array
#include <cstdint>      // for int16_t, int32_t
#include <iterator>     // for iterator_traits
#include <memory>       // for allocator
#include <type_traits>  // for is_same, is_convertible, enable...

namespace multi = boost::multi;

namespace minimalistic {

template<class T>
/// minimalistic pointer
class ptr : public std::iterator_traits<T*> {  // NOLINT(misc-use-internal-linkage)
	using underlying_type = T*;
	underlying_type impl_;
	template<class> friend class ptr;

 public:
	ptr() = default;  // cppcheck-suppress uninitMemberVar ;

	constexpr explicit ptr(T* impl) : impl_{impl} {}

	template<class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	// cppcheck-suppress noExplicitConstructor ;
	constexpr ptr(ptr<U> const& other) : impl_{other.impl_} {}  //  NOLINT(*-explicit-constructor, hicpp-explicit-conversions)  // NOSONAR(cpp:S1709)
	using typename std::iterator_traits<T*>::reference;
	using typename std::iterator_traits<T*>::difference_type;

	// NOLINTNEXTLINE(fuchsia-overloaded-operator, fuchsia-trailing-return): operator* used because this class simulates a pointer, trailing return helps
	constexpr auto operator*() const -> reference { return *impl_; }

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

	// NOLINTNEXTLINE(fuchsia-overloaded-operator, cppcoreguidelines-pro-bounds-pointer-arithmetic): operator+ is overloaded to simulate a pointer
	constexpr auto operator+(difference_type n) const { return ptr{impl_ + n}; }
	// NOLINTNEXTLINE(fuchsia-overloaded-operator, cppcoreguidelines-pro-bounds-pointer-arithmetic): operator+ is overloaded to simulate a pointer
	constexpr auto operator-(difference_type n) const { return ptr{impl_ - n}; }

	friend constexpr auto operator+(difference_type n, ptr const& self) { return self + n; }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	//  T& operator[](difference_type n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;

	template<class T2> auto operator==(ptr<T2> const& other) const& { return impl_ == other.impl_; }
	template<class> friend class ptr2;
};

template<class T>
/// minimalistic pointer
class ptr2 : public std::iterator_traits<T*> {  // NOLINT(misc-use-internal-linkage)
	T* impl_;

 public:
	constexpr explicit ptr2(T* impl) : impl_{impl} {}
	constexpr explicit ptr2(ptr<T> const& other) : impl_{other.impl_} {}
	template<class U, class = std::enable_if_t<std::is_convertible_v<U*, T*>>>  // NOLINT(modernize-use-constraints) TODO(correaa) for C++20
	// cppcheck-suppress [noExplicitConstructor, unmatchedSuppression]
	constexpr ptr2(ptr2<U> const& other) : impl_{other.impl_} {}  // NOLINT(*-explicit-constructor, hicpp-explicit-conversions)  // NOSONAR(cpp:S1709)

	using typename std::iterator_traits<T*>::reference;
	using typename std::iterator_traits<T*>::difference_type;

	// NOLINTNEXTLINE(fuchsia-overloaded-operator, fuchsia-trailing-return): operator* used because this class simulates a pointer, trailing return helps
	constexpr auto operator*() const -> reference { return *impl_; }

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

	// NOLINTNEXTLINE(fuchsia-overloaded-operator, cppcoreguidelines-pro-bounds-pointer-arithmetic): operator+ is overloaded to simulate a pointer
	constexpr auto operator+(difference_type n) const { return ptr2{impl_ + n}; }
	// NOLINTNEXTLINE(fuchsia-overloaded-operator, cppcoreguidelines-pro-bounds-pointer-arithmetic): operator+ is overloaded to simulate a pointer
	constexpr auto operator-(difference_type n) const { return ptr2{impl_ - n}; }

	friend constexpr auto operator+(difference_type n, ptr2 const& self) { return self + n; }

#ifdef __clang__
#pragma clang diagnostic pop
#endif

	//  T& operator[](std::ptrdiff_t n) const{return impl_[n];}  // optional
	using default_allocator_type = std::allocator<T>;
};

}  // end namespace minimalistic

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// BOOST_AUTO_TEST_CASE(test_minimalistic_ptr)
	{
		std::array<int, 400> buffer{};
		BOOST_TEST( buffer.size() == 400 );  // cppcheck-suppress knownConditionTrueFalse ; for test

		using pointer_type = minimalistic::ptr<int>;

		auto const CCP = &multi::array_ref<int, 2, pointer_type>({20, 20}, pointer_type{buffer.data()});

		(*CCP)[2];  // cppcheck-suppress danglingTemporaryLifetime
		(*CCP)[1][1];
		(*CCP)[1][1] = 9;

		BOOST_TEST(  (*CCP)[1][1] == 9 );  // cppcheck-suppress knownConditionTrueFalse ; for test
		BOOST_TEST( &(*CCP)[1][1] == &buffer[21] );

		// auto&& CC2 = (*CCP).static_array_cast<double, minimalistic::ptr2<double>>();
		auto&& CC2 = CCP->static_array_cast<int, minimalistic::ptr2<int>>();
		BOOST_TEST( &CC2[1][1] == &(*CCP)[1][1] );

		static_assert(std::is_convertible<int*, int const*>{}, "!");  // NOLINT(readability-trailing-comma) bug in clang-tidy

		minimalistic::ptr<int> const       pd{nullptr};
		minimalistic::ptr<int const> const pcd = pd;
		BOOST_TEST( pcd == pd );

		{
			auto&& REF = *CCP;  // cppcheck-suppress danglingTempReference ;
			(void)REF;          // cppcheck-suppress danglingTempReference ;

			// cppcheck-suppress danglingTempReference ;
			static_assert(std::is_same_v<decltype(REF.partitioned(2).partitioned(2).base()), minimalistic::ptr<int>>);
		}
		{
			auto const& REF = *CCP;  // cppcheck-suppress danglingTempReference ;
			(void)REF;               // cppcheck-suppress danglingTempReference ;

			// cppcheck-suppress danglingTempReference ;
			static_assert(std::is_same_v<decltype(REF.partitioned(2).partitioned(2).base()), minimalistic::ptr<int const>>);
		}
		{
			int data[2][5] = {  // NOLINT(*-avoid-c-arrays,misc-const-correctness)
				{10, 11, 12, 13, 14},
				{20, 21, 22, 23, 24},
			};

			minimalistic::ptr<int const> const p0{&data[0][0]};

			multi::array_ref<int const, 2, minimalistic::ptr<int const>> const arr(p0, {2, 5});

			auto&& marr = arr.const_array_cast<int>();

			static_assert(std::is_same_v<std::decay_t<decltype(marr[0][0])>, int>);

			marr[1][2] = 99;  // exercise the "mutable view" the cast is meant to provide

			BOOST_TEST( data[1][2] == 99 );
			BOOST_TEST( marr[0][0] == 10 );
			BOOST_TEST( marr[1][4] == 24 );
		}
		{
			// same class of UB as above, but in reinterpret_array_cast(count) (array_ref.hpp:2194)
			// instead of const_array_cast(): `arr` must be a const object to select
			// const_subarray::reinterpret_array_cast(size_type) const&, whose fancy-pointer
			// branch does reinterpret_cast<P2 const&>(this->base_) with P2 =
			// minimalistic::ptr<short const> != ElementPtr = minimalistic::ptr<int const>.
			//
			// Unlike const_array_cast's warning, GCC's default -Wstrict-aliasing (level 3,
			// what pre-push's `.build.g++.release` job and test/CMakeLists.txt both use)
			// does NOT flag this one; it only shows up at -Wstrict-aliasing=1 or =2:
			//   array_ref.hpp:2194:67: warning: dereferencing type-punned pointer might
			//     break strict-aliasing rules [-Wstrict-aliasing]      (at level 1)
			//   array_ref.hpp:2194:67: warning: type-punning to incomplete type might
			//     break strict-aliasing rules [-Wstrict-aliasing]      (at level 2)
			std::int32_t data[2][5] = {  // NOLINT(*-avoid-c-arrays)
				{10, 11, 12, 13, 14},
				{20, 21, 22, 23, 24},
			};

			minimalistic::ptr<std::int32_t const> const p0{&data[0][0]};

			multi::array_ref<std::int32_t const, 2, minimalistic::ptr<std::int32_t const>> const arr(p0, {2, 5});

			auto&& marr = arr.reinterpret_array_cast<std::int16_t const>(2);

			static_assert(std::is_same_v<std::decay_t<decltype(marr[0][0][0])>, std::int16_t>);

			BOOST_TEST( marr.size() == 2 );
			BOOST_TEST( static_cast<void const*>(&marr[1][2][0]) == static_cast<void const*>(&data[1][2]) );
		}
		{
			// same class of UB again, but in subarray::reinterpret_pointer_cast_()
			// (array_ref.hpp:2758), the private helper used by the *mutable*
			// (non-const) subarray::reinterpret_array_cast(size_type) & overload --
			// as opposed to const_subarray's, exercised above. `arr` must be a
			// non-const lvalue here to select that overload (a const object would
			// pick const_subarray::reinterpret_array_cast() const& instead, whose
			// own fancy-pointer branch was already fixed with bit_cast_).
			//
			// Of the three UB sites found this session, this is the most elusive
			// for GCC's -Wstrict-aliasing: it only fires at level 1, not level 2
			// (what test/CMakeLists.txt currently uses, chosen to catch the other
			// two without the false positives level 1 is documented to add) nor
			// the default level 3:
			//   array_ref.hpp:2758:32: warning: dereferencing type-punned pointer
			//     might break strict-aliasing rules [-Wstrict-aliasing]  (level 1 only)
			std::int32_t data[2][5] = {  // NOLINT(*-avoid-c-arrays)
				{10, 11, 12, 13, 14},
				{20, 21, 22, 23, 24},
			};

			minimalistic::ptr<std::int32_t> const p0{&data[0][0]};

			multi::array_ref<std::int32_t, 2, minimalistic::ptr<std::int32_t>> arr(p0, {2, 5});  // NOLINT(misc-const-correctness) intentionally non-const

			auto&& marr = arr.reinterpret_array_cast<std::int16_t>(2);

			static_assert(std::is_same_v<std::decay_t<decltype(marr[0][0][0])>, std::int16_t>);

			BOOST_TEST( marr.size() == 2 );
			BOOST_TEST( static_cast<void const*>(&marr[1][2][0]) == static_cast<void const*>(&data[1][2]) );
		}
	}

	return boost::report_errors();
}
