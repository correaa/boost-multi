// Copyright 2019-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for layout_t, apply, subarray, array...  // IWYU pragma: keep  // bug in iwyu 8.22

#include <boost/core/lightweight_test.hpp>

#include <algorithm>    // for equal
#include <array>        // for array  // IWYU pragma: keep  // bug in iwyu 8.22
#include <memory>       // for addressof
#include <type_traits>  // for is_trivially_copy_assignable_v
#include <utility>      // for as_const, addressof, exchange, move
#include <vector>       // for vector

namespace {
// NOLINTNEXTLINE(fuchsia-trailing-return): trailing return helps readability
template<class T> auto fwd_array(T&& array) -> T&& { return std::forward<T>(array); }
}  // end unnamed namespace

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	namespace multi = boost::multi;

	// BOOST_AUTO_TEST_CASE(multi_array_ptr_equality)
	{
		multi::array<int, 2> arr = {
			{10, 20, 30},
			{40, 50, 60},
			{70, 80, 90},
			{10, 20, 30},
		};
		BOOST_TEST(  arr[2] ==  arr[2] );
		BOOST_TEST( &arr[2] == &arr[2] );
		BOOST_TEST( !(&arr[2] == &(arr[2]({0, 2}))) );

		BOOST_TEST( arr[2].base() == arr[2]({0, 2}).base() );
		BOOST_TEST( arr[2].layout() != arr[2]({0, 2}).layout() );

		// what( arr[2], arr[2].sliced(0, 2), &(arr[2].sliced(0, 2)) );
		BOOST_TEST(    &arr[2] != &(arr[2].sliced(0, 2))  );

		// comparison of different provenance is undefined
		// BOOST_TEST( !( &arr[2] == &std::as_const(arr)[2]({0, 2})) );
		BOOST_TEST( &arr[2] == &fwd_array(arr[2]) );
		BOOST_TEST( &fwd_array(arr[2]) == &arr[2] );

		auto arr_ptr = &arr[2];
		BOOST_TEST( arr_ptr == arr_ptr );

		auto& arr_ptr_ref = arr_ptr;
		arr_ptr           = arr_ptr_ref;

		auto arr_ptr2 = &std::as_const(arr)[2];
		BOOST_TEST( arr_ptr == arr_ptr2 );
		BOOST_TEST( arr_ptr2 == arr_ptr );
		BOOST_TEST( !(arr_ptr != arr_ptr) );

		auto& arr_ptr2_ref = arr_ptr2;
		arr_ptr2           = arr_ptr2_ref;
		arr_ptr2_ref       = arr_ptr2;

		auto const& carr2 = arr[2];
		BOOST_TEST( carr2[0] == arr[2][0] );
		BOOST_TEST( carr2.base() == arr[2].base() );
		BOOST_TEST( &carr2 == &std::as_const(arr)[2] );
		BOOST_TEST( &carr2 == &              arr [2] );
		BOOST_TEST( &carr2 == &              arr [2] );

		// comparing array-pointer of different provenance is undefined
		// BOOST_TEST(   &carr2 != &              arr [2]({0, 2})  );
		// BOOST_TEST( !(&carr2 == &              arr [2]({0, 2})) );

		auto const& ac2 = carr2;  // fwd_array(A[2]);
		BOOST_TEST( &ac2 == &std::as_const(arr)[2] );
		BOOST_TEST( &std::as_const(arr)[2] == &ac2 );
		BOOST_TEST( &ac2 == &              arr [2] );

		auto pac2  = &ac2;
		auto parr2 = &arr[2];
		BOOST_TEST( pac2 == parr2 );

		pac2 = nullptr;
		BOOST_TEST( pac2 != parr2 );

		parr2 = nullptr;
		BOOST_TEST( pac2 == parr2 );
	}

	// BOOST_AUTO_TEST_CASE(multi_array_ptr)
	{
		{
			// clang-format off
		std::array<std::array<double, 5>, 4> arr{
			{{{0.0, 1.0, 2.0, 3.0, 4.0}},
			 {{5.0, 6.0, 7.0, 8.0, 9.0}},
			 {{10.0, 11.0, 12.0, 13.0, 14.0}},
			 {{15.0, 16.0, 17.0, 18.0, 19.0}}},
		};
			// clang-format on

			multi::array_ptr<double, 2> const arrP{&arr};

			static_assert(std::is_trivially_copy_assignable_v<multi::array_ptr<double, 2>>);
			static_assert(std::is_trivially_copyable_v<multi::array_ptr<double, 2>>);

			// #ifndef _MSC_VER
			static_assert(std::is_trivially_default_constructible_v<multi::layout_t<0>>);
			static_assert(std::is_trivially_default_constructible_v<multi::layout_t<1>>);
			static_assert(std::is_trivially_default_constructible_v<multi::layout_t<2>>);
			// #endif

			static_assert(std::is_trivially_copyable_v<multi::layout_t<0>>);
			static_assert(std::is_trivially_copyable_v<multi::layout_t<1>>);
			static_assert(std::is_trivially_copyable_v<multi::layout_t<2>>);

			// static_assert(std::is_trivially_copy_assignable_v<multi::subarray_ptr<double, 2>>);
			// static_assert(std::is_trivially_copyable_v<multi::subarray_ptr<double, 2>>);

			BOOST_TEST( (*arrP).extensions() == multi::extensions(arr) );
			BOOST_TEST( arrP->extensions() == multi::extensions(arr) );
			BOOST_TEST( extensions(*arrP) == multi::extensions(arr) );

			using multi::extensions;
			BOOST_TEST( extensions(*arrP) == extensions(arr) );

			BOOST_TEST( &(*arrP).operator[](1)[1] == &arr[1][1] );
			BOOST_TEST( &arrP->operator[](1)[1] == &arr[1][1] );

			multi::array_ptr<double, 2> const arrP2{&arr};
			BOOST_TEST( arrP == arrP2 );
			BOOST_TEST( !(arrP != arrP2) );

			std::array<std::array<double, 5>, 4> arr2{};
			multi::array_ptr<double, 2>          arr2P{&arr2};
			BOOST_TEST( arr2P != arrP );
			BOOST_TEST( !(arr2P == arrP) );

			arr2P = arrP;
			BOOST_TEST(  arrP ==  arr2P );
			BOOST_TEST( *arrP == *arr2P );

			BOOST_TEST(  (*arrP).operator==(*arrP) );
			BOOST_TEST(  arrP->operator==(*arrP) );

			auto&& arrR = *arrP;
			BOOST_TEST( &arrR[1][1] == &arr[1][1] );
			BOOST_TEST( arrR == *arrP );

			BOOST_TEST( std::equal(arrR.begin(), arrR.end(), (*arrP).begin(), (*arrP).end()) );
			BOOST_TEST( std::equal(arrR.begin(), arrR.end(), arrP->begin(), arrP->end()) );

			BOOST_TEST( arrR.size() == (*arrP).size() );
			BOOST_TEST( size(arrR) == arrP->size() );
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

			BOOST_TEST( &(*ptrs[2])[4] == &arr[3][4]     );
			BOOST_TEST(  (*ptrs[2])[4] == 190            );
			BOOST_TEST(    ptrs[2]->operator[](4) == 190 );
		}
		{
			std::vector<int>       v1(100, 30);  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls)
			std::vector<int> const v2(100, 40);  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls)

			multi::array_ptr<int, 2> const  v1P2D(v1.data(), {10, 10});
			multi::array_cptr<int, 2> const v2P2D(v2.data(), {10, 10});

			*v1P2D = *v2P2D;
			(*v1P2D).operator=(*v2P2D);
			BOOST_TEST( v1[8] == 40 );

			v1P2D->operator=(*v2P2D);
			BOOST_TEST( v1[8] == 40 );
		}
	}

	// BOOST_AUTO_TEST_CASE(span_like)
	{
		std::vector<int> vec = {00, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls)

		using my_span = multi::array_ref<int, 1>;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif

		auto aP = &my_span{vec.data() + 2, {5}};  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

		BOOST_TEST( (*aP).size() == 5 );
		BOOST_TEST( aP->size() == 5 );  // doesn't work on MSVC?

		BOOST_TEST( (*aP)[0] == 20 );

		auto const& aCRef = *aP;
		BOOST_TEST(  aCRef.size() == 5 );

		BOOST_TEST( &aCRef[0] == &vec[2] );
		BOOST_TEST(  aCRef[0] == 20     );

		auto&& aRef = *aP;
		// what(aP, aRef);
		// (*aP)[0] = 990;
		aRef[0] = 990;
		BOOST_TEST( vec[2] == 990 );
	}

	// BOOST_AUTO_TEST_CASE(multi_array_ptr_assignment)
	{
		multi::array<double, 2> arr = {
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
			{7.0, 8.0, 9.0},
			{1.0, 2.0, 3.0},
		};
		{
			auto rowP = &arr[2];

			rowP = *std::addressof(rowP);

			auto rowP2 = rowP;
			rowP2      = rowP;  // self assigment

			BOOST_TEST( rowP == rowP2 );
			BOOST_TEST( !(rowP != rowP2) );

			auto rowP0 = &arr[0];

			BOOST_TEST( rowP0 != rowP2 );
			BOOST_TEST( !(rowP0 == rowP2) );

			rowP2 = decltype(rowP2){nullptr};
			BOOST_TEST( !rowP2 );

			auto rowP3 = std::exchange(rowP, nullptr);
			BOOST_TEST( rowP3 == &arr[2] );
			BOOST_TEST( rowP == nullptr );
			// BOOST_TEST( !rowP );
		}
		{
			auto rowP = &arr();

			rowP = *std::addressof(rowP);

			decltype(rowP) rowP2;
			rowP2 = rowP;

			BOOST_TEST( rowP == rowP2 );

			rowP2 = decltype(rowP2){nullptr};
			BOOST_TEST( !rowP2 );

			auto rowP3 = std::exchange(rowP, nullptr);
			BOOST_TEST( rowP3 == &arr() );
			BOOST_TEST( rowP == nullptr );
			BOOST_TEST( !rowP );
		}
	}

	return boost::report_errors();
}
