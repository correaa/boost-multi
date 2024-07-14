// Copyright 2019-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wold-style-cast"
	#pragma clang diagnostic ignored "-Wundef"
	#pragma clang diagnostic ignored "-Wconversion"
	#pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wundef"
	#pragma GCC diagnostic ignored "-Wconversion"
	#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

#ifndef BOOST_TEST_MODULE
	#define BOOST_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>

#if defined(__clang__)
	#pragma clang diagnostic pop
#elif defined(__GNUC__)
	#pragma GCC diagnostic pop
#endif

#include <boost/multi/array.hpp>  // for layout_t, apply, subarray, array...  // IWYU pragma: keep  // bug in iwyu 8.22

#include <algorithm>  // for equal
#include <array>      // for array  // IWYU pragma: keep  // bug in iwyu 8.22
// IWYU pragma: no_include  <memory>     // for __alloc_traits<>::value_type
// IWYU pragma: no_include <type_traits>  // for decay_t
#include <utility>  // for as_const, addressof, exchange, move
#include <vector>   // for vector

namespace multi = boost::multi;

// NOLINTNEXTLINE(fuchsia-trailing-return): trailing return helps readability
template<class T> auto fwd_array(T&& array) -> T&& { return std::forward<T>(array); }

BOOST_AUTO_TEST_CASE(multi_array_ptr_equality) {
	multi::array<int, 2> arr = {
		{ 10, 20, 30 },
		{ 40, 50, 60 },
		{ 70, 80, 90 },
		{ 10, 20, 30 },
	};
	BOOST_REQUIRE(  arr[2] ==  arr[2] );
	BOOST_REQUIRE( &arr[2] == &arr[2] );
	BOOST_REQUIRE( !(&arr[2] == &(arr[2]({0, 2}))) );

	BOOST_REQUIRE( arr[2].base() == arr[2]({0, 2}).base() );
	BOOST_REQUIRE( arr[2].layout() != arr[2]({0, 2}).layout() );

	// what( arr[2], arr[2].sliced(0, 2), &(arr[2].sliced(0, 2)) );
	BOOST_REQUIRE(    &arr[2] != &(arr[2].sliced(0, 2))  );

	BOOST_REQUIRE( !( &arr[2] == &std::as_const(arr)[2]({0, 2})) );
	BOOST_REQUIRE( &arr[2] == &fwd_array(arr[2]) );
	BOOST_REQUIRE( &fwd_array(arr[2]) == &arr[2] );

	auto arr_ptr = &arr[2];
	BOOST_REQUIRE( arr_ptr == arr_ptr );

	auto& arr_ptr_ref = arr_ptr;
	arr_ptr           = arr_ptr_ref;
	arr_ptr           = std::move(arr_ptr_ref);

	auto arr_ptr2 = &std::as_const(arr)[2];
	BOOST_REQUIRE( arr_ptr == arr_ptr2 );
	BOOST_REQUIRE( arr_ptr2 == arr_ptr );
	BOOST_REQUIRE( !(arr_ptr != arr_ptr) );

	auto& arr_ptr2_ref = arr_ptr2;
	arr_ptr2           = arr_ptr2_ref;
	arr_ptr2_ref       = arr_ptr2;

	auto const& carr2 = arr[2];
	BOOST_REQUIRE( carr2[0] == arr[2][0] );
	BOOST_REQUIRE( carr2.base() == arr[2].base() );
	BOOST_REQUIRE( &carr2 == &std::as_const(arr)[2] );
	BOOST_REQUIRE( &carr2 == &              arr [2] );

	auto const& ac2 = carr2;  // fwd_array(A[2]);
	BOOST_REQUIRE( &ac2 == &std::as_const(arr)[2] );
	BOOST_REQUIRE( &std::as_const(arr)[2] == &ac2 );
	BOOST_REQUIRE( &ac2 == &              arr [2] );
}

BOOST_AUTO_TEST_CASE(subarray_ptr_1D) {
	multi::subarray_ptr<double, 1> s = nullptr;
	BOOST_REQUIRE(( s == multi::subarray_ptr<double, 1>{nullptr} ));
}

BOOST_AUTO_TEST_CASE(subarray_ptr_2D) {
	multi::subarray_ptr<double, 2> s = nullptr;
	BOOST_REQUIRE(( s == multi::subarray_ptr<double, 2>{nullptr} ));
}

BOOST_AUTO_TEST_CASE(multi_array_ptr) {
	{
		// clang-format off
		std::array<std::array<double, 5>, 4> arr{
			{{{0.0, 1.0, 2.0, 3.0, 4.0}},
			 {{5.0, 6.0, 7.0, 8.0, 9.0}},
			 {{10.0, 11.0, 12.0, 13.0, 14.0}},
			 {{15.0, 16.0, 17.0, 18.0, 19.0}}},
		};
		// clang-format on

		multi::array_ptr<double, 2> const arrP{ &arr };


		static_assert( std::is_trivially_copy_assignable_v<multi::array_ptr<double, 2>> );
		static_assert( std::is_trivially_copyable_v<multi::array_ptr<double, 2>> );

	#ifndef _MSVER
		static_assert( std::is_trivially_default_constructible_v<multi::layout_t<0>> )
		static_assert( std::is_trivially_default_constructible_v<multi::layout_t<1>> );
		static_assert( std::is_trivially_default_constructible_v<multi::layout_t<2>> );
	#endif

		static_assert( std::is_trivially_copyable_v<multi::layout_t<0>> );
		static_assert( std::is_trivially_copyable_v<multi::layout_t<1>> );
		static_assert( std::is_trivially_copyable_v<multi::layout_t<2>> );

	#ifndef _MSVER
		static_assert( std::is_trivially_default_constructible_v<multi::subarray_ptr<double, 2>> );
	#endif
		static_assert( std::is_trivially_copy_assignable_v<multi::subarray_ptr<double, 2>> );
		static_assert( std::is_trivially_copyable_v<multi::subarray_ptr<double, 2>> );

		BOOST_REQUIRE( arrP->extensions() == multi::extensions(arr) );
		BOOST_REQUIRE( extensions(*arrP) == multi::extensions(arr) );

		using multi::extensions;
		BOOST_REQUIRE( extensions(*arrP) == extensions(arr) );
		BOOST_REQUIRE( &arrP->operator[](1)[1] == &arr[1][1] );

		multi::array_ptr<double, 2> const arrP2{ &arr };
		BOOST_REQUIRE( arrP == arrP2 );
		BOOST_REQUIRE( !(arrP != arrP2) );

		std::array<std::array<double, 5>, 4> arr2{};
		multi::array_ptr<double, 2>          arr2P{ &arr2 };
		BOOST_REQUIRE( arr2P != arrP );
		BOOST_REQUIRE( !(arr2P == arrP) );

		arr2P = arrP;
		BOOST_REQUIRE(  arrP ==  arr2P );
		BOOST_REQUIRE( *arrP == *arr2P );
		BOOST_REQUIRE(  arrP->operator==(*arrP) );

		auto&& arrR = *arrP;
		BOOST_REQUIRE( &arrR[1][1] == &arr[1][1] );
		BOOST_REQUIRE( arrR == *arrP );
		BOOST_REQUIRE( std::equal(arrR.begin(), arrR.end(), arrP->begin(), arrP->end()) );
		BOOST_REQUIRE( size(arrR) == arrP->size() );
	}
	{
		// clang-format off
		std::array<std::array<int, 5>, 4> arr = {{
			std::array<int, 5>{ { 00, 10, 20, 30, 40 } },
			std::array<int, 5>{ { 50, 60, 70, 80, 90 } },
			std::array<int, 5>{ { 100, 110, 120, 130, 140 } },
			std::array<int, 5>{ { 150, 160, 170, 180, 190 } },
		}};
		// clang-format on

		std::vector<multi::array_ptr<int, 1>> ptrs;
		ptrs.emplace_back(&arr[0][0], 5);  // NOLINT(readability-container-data-pointer) test access
		ptrs.emplace_back(arr[2].data(), 5);
		ptrs.emplace_back(&arr[3][0], 5);  // NOLINT(readability-container-data-pointer) test access

		BOOST_REQUIRE( &(*ptrs[2])[4] == &arr[3][4]     );
		BOOST_REQUIRE(  (*ptrs[2])[4] == 190            );
		BOOST_REQUIRE(    ptrs[2]->operator[](4) == 190 );
	}
	{
		std::vector<int>       v1(100, 30);  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls)
		std::vector<int> const v2(100, 40);  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls)

		multi::array_ptr<int, 2> const  v1P2D(v1.data(), { 10, 10 });
		multi::array_cptr<int, 2> const v2P2D(v2.data(), { 10, 10 });

		*v1P2D = *v2P2D;
		v1P2D->operator=(*v2P2D);

		BOOST_REQUIRE( v1[8] == 40 );
	}
}

BOOST_AUTO_TEST_CASE(span_like) {
	std::vector<int> vec = { 00, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls)

	using my_span = multi::array_ref<int, 1>;

	auto aP = &my_span{ vec.data() + 2, { 5 } };  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
	BOOST_REQUIRE( aP->size() == 5 );
	BOOST_REQUIRE( (*aP)[0] == 20 );

	auto const& aCRef = *aP;
	BOOST_REQUIRE(  aCRef.size() == 5 );

	BOOST_REQUIRE( &aCRef[0] == &vec[2] );
	BOOST_REQUIRE(  aCRef[0] == 20     );

	auto&& aRef = *aP;
	// what(aP, aRef);
	// (*aP)[0] = 990;
	aRef[0]     = 990;
	BOOST_REQUIRE( vec[2] == 990 );
}

BOOST_AUTO_TEST_CASE(multi_array_ptr_assignment) {
	multi::array<double, 2> arr = {
		{ 1.0, 2.0, 3.0 },
		{ 4.0, 5.0, 6.0 },
		{ 7.0, 8.0, 9.0 },
		{ 1.0, 2.0, 3.0 },
	};
	{
		auto rowP = &arr[2];

		rowP = *std::addressof(rowP);

		auto rowP2 = rowP;
		rowP2      = rowP;  // self assigment

		BOOST_REQUIRE( rowP == rowP2 );
		BOOST_REQUIRE( !(rowP != rowP2) );

		auto rowP0 = &arr[0];

		BOOST_REQUIRE( rowP0 != rowP2 );
		BOOST_REQUIRE( !(rowP0 == rowP2) );

		rowP2 = decltype(rowP2){ nullptr };
		BOOST_REQUIRE( !rowP2 );

		auto rowP3 = std::exchange(rowP, nullptr);
		BOOST_REQUIRE( rowP3 == &arr[2] );
		BOOST_REQUIRE( rowP == nullptr );
		// BOOST_REQUIRE( !rowP );
	}
	{
		auto rowP = &arr();

		rowP = *std::addressof(rowP);

		decltype(rowP) rowP2;
		rowP2 = rowP;

		BOOST_REQUIRE( rowP == rowP2 );

		rowP2 = decltype(rowP2){ nullptr };
		BOOST_REQUIRE( !rowP2 );

		auto rowP3 = std::exchange(rowP, nullptr);
		BOOST_REQUIRE( rowP3 == &arr() );
		BOOST_REQUIRE( rowP == nullptr );
		BOOST_REQUIRE( !rowP );
	}
}
