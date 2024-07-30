// Copyright 2019-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for apply, operator!=, operator==

#include <algorithm>   // for is_sorted, stable_sort
#include <array>       // for array
#include <functional>  // for __cpp_lib_ranges  // IWYU pragma: keep
#include <iterator>    // for begin, end
#include <vector>      // for vector
#include <version>     // for __cpp_lib_ranges  // IWYU pragma: keep  // bug in iwyu 0.22/18.1.8 with LLVM stdlib?

#if defined(__cpp_lib_ranges)
	#include <concepts>
#endif

namespace multi = boost::multi;

#include <boost/core/lightweight_test.hpp>
#define BOOST_AUTO_TEST_CASE(CasenamE) [[maybe_unused]] void* CasenamE;

int main() {
	BOOST_AUTO_TEST_CASE(array_1D_partial_order_syntax) {
		multi::array<int, 1> const tt = {1, 1, 1};
		multi::array<int, 1> const uu = {2, 2, 2};

		BOOST_TEST(     tt <  uu   );
		BOOST_TEST(   !(tt >  uu)  );
		BOOST_TEST(     tt <= uu   );
		BOOST_TEST(   !(tt >= uu)  );
		BOOST_TEST(   !(tt == uu)  );
		BOOST_TEST(    (tt != uu)  );
		BOOST_TEST(   !(uu <  tt)  );
		BOOST_TEST(    (uu >  tt)  );
		BOOST_TEST(   !(uu <= tt)  );
		BOOST_TEST(    (uu >= tt)  );
	}

#if defined(__cpp_lib_ranges)
	BOOST_AUTO_TEST_CASE(sort_2D) {
		multi::array<int, 2> A = {
			{3, 3, 3},
			{2, 2, 2},
			{1, 1, 1},
		};
		BOOST_TEST( !std::ranges::is_sorted(A) );

		using it = boost::multi::array_iterator<int, 2, int*>;

		static_assert(std::forward_iterator<it>);

		using In  = it;
		using Out = it;

		static_assert(std::indirectly_readable<In>);

		*A.begin() = A[0];

		// const_cast<const std::iter_reference_t<Out>&&>(*A.begin()) = A[0];  // std::forward<T>(t);
		// const_cast<const std::iter_reference_t<Out>&&>(*std::forward<Out>(o)) =
		//             std::forward<T>(t);

		static_assert(std::indirectly_writable<Out, multi::subarray<int, 1>>);
		static_assert(std::indirectly_writable<Out, std::iter_rvalue_reference_t<In>>);
		static_assert(std::indirectly_movable<In, Out>);
		static_assert(std::indirectly_writable<Out, std::iter_value_t<In>>);
		static_assert(std::movable<std::iter_value_t<In>>);
		static_assert(std::constructible_from<std::iter_value_t<In>, std::iter_rvalue_reference_t<In>>);
		static_assert(std::assignable_from<std::iter_value_t<In>&, std::iter_rvalue_reference_t<In>>);

		static_assert(std::indirectly_movable_storable<it, it>);
		static_assert(std::indirectly_swappable<it>);
		static_assert(std::permutable<it>);

		// std::sort(A.begin(), A.end());
		std::ranges::sort(A);

		BOOST_TEST(  std::ranges::is_sorted(A) );
	}

	BOOST_AUTO_TEST_CASE(sort_strings) {
		auto A = multi::array<char, 2>{
			{'S', 'e', 'a', 'n', ' ', ' '},
			{'A', 'l', 'e', 'x', ' ', ' '},
			{'B', 'j', 'a', 'r', 'n', 'e'},
		};
		BOOST_TEST( !std::ranges::is_sorted(A) );

		std::ranges::sort(A);

		BOOST_TEST(  std::ranges::is_sorted(A));

		BOOST_TEST((
		A == multi::array<char, 2>{
			{'A', 'l', 'e', 'x', ' ', ' '},
			{'B', 'j', 'a', 'r', 'n', 'e' },
			{'S', 'e', 'a', 'n', ' ', ' '},
		}
	));

		std::ranges::sort(~A);
		BOOST_TEST(std::ranges::is_sorted(~A));

		static_assert(std::permutable<boost::multi::array_iterator<int, 2, int*>>);
	}
#endif

	BOOST_AUTO_TEST_CASE(multi_array_stable_sort) {
		std::vector<double> vec = {1.0, 2.0, 3.0};  // NOLINT(fuchsia-default-arguments-calls)
		BOOST_TEST( std::is_sorted(begin(vec), end(vec)) );

		multi::array<double, 2> d2D = {
			{150.0, 16.0, 17.0, 18.0, 19.0},
			{ 30.0,  1.0,  2.0,  3.0,  4.0},
			{100.0, 11.0, 12.0, 13.0, 14.0},
			{ 50.0,  6.0,  7.0,  8.0,  9.0},
		};
		BOOST_TEST( !std::is_sorted(begin(d2D), end(d2D) ) );

		std::stable_sort(begin(d2D), end(d2D));
		BOOST_TEST( std::is_sorted( begin(d2D), end(d2D) ) );

		BOOST_TEST((
		d2D == decltype(d2D){
			{ 30.0,  1.0,  2.0,  3.0,  4.0},
			{ 50.0,  6.0,  7.0,  8.0,  9.0},
			{100.0, 11.0, 12.0, 13.0, 14.0},
			{150.0, 16.0, 17.0, 18.0, 19.0},
		}
	));

		BOOST_TEST( !std::is_sorted( begin(d2D.rotated()), end(d2D.rotated()) ) );

		std::stable_sort(begin(d2D.rotated()), end(d2D.rotated()));
		BOOST_TEST( std::is_sorted( begin(d2D.rotated()), end(d2D.rotated()) ) );
		BOOST_TEST( std::is_sorted( begin(d2D          ), end(d2D          ) ) );

		BOOST_TEST((
		d2D == decltype(d2D){
			{ 1.0,  2.0,  3.0,  4.0,  30.0},
			{ 6.0,  7.0,  8.0,  9.0,  50.0},
			{11.0, 12.0, 13.0, 14.0, 100.0},
			{16.0, 17.0, 18.0, 19.0, 150.0},
		}
	));
	}

	BOOST_AUTO_TEST_CASE(multi_array_ref_stable_sort) {
		std::vector<double> vec = {1.0, 2.0, 3.0};  // NOLINT(fuchsia-default-arguments-calls)
		BOOST_TEST( std::is_sorted(begin(vec), end(vec)) );

		// clang-format off
	std::array<std::array<double, 5>, 4> d2D {{
		{{150.0, 16.0, 17.0, 18.0, 19.0}},
		{{ 30.0,  1.0,  2.0,  3.0,  4.0}},
		{{100.0, 11.0, 12.0, 13.0, 14.0}},
		{{ 50.0,  6.0,  7.0,  8.0,  9.0}}
	}};
		// clang-format on

		auto&& d2D_ref = *multi::array_ptr<double, 2>(&d2D[0][0], {4, 5});  // NOLINT(readability-container-data-pointer) test access

		BOOST_TEST( !std::is_sorted(begin(d2D_ref), end(d2D_ref) ) );
		std::stable_sort(begin(d2D_ref), end(d2D_ref));
		BOOST_TEST( std::is_sorted( begin(d2D_ref), end(d2D_ref) ) );

		BOOST_TEST( !std::is_sorted( begin(d2D_ref.rotated()), end(d2D_ref.rotated()) ) );
		std::stable_sort(begin(d2D_ref.rotated()), end(d2D_ref.rotated()));
		BOOST_TEST( std::is_sorted( begin(d2D_ref.rotated()), end(d2D_ref.rotated()) ) );
	}

	BOOST_AUTO_TEST_CASE(lexicographical_compare) {
		multi::array<char, 1> const name1 = {'a', 'b', 'c'};
		multi::array<char, 1> const name2 = {'a', 'c', 'c'};

		BOOST_TEST(  name1 != name2 );
		BOOST_TEST(  name1 <  name2);
		BOOST_TEST(  name1 <= name2);
		BOOST_TEST(!(name1 >  name2));
		BOOST_TEST(!(name1 >  name2));
	}

	BOOST_AUTO_TEST_CASE(lexicographical_compare_offset) {
		multi::array<char, 1> const name1 = {'a', 'b', 'c'};
		// clang-format off
	multi::array<char, 1>       name2({{ 1, 4 }}, '\0');
		// clang-format on

		BOOST_TEST(  name2.size() == 3 );
		BOOST_TEST(( name2.extension() == multi::extension_t<multi::index>{1, 4} ));
		BOOST_TEST(( name2.extension() == multi::extension_t{multi::index{1}, multi::index{4}} ));

		// BOOST_TEST(( name2.extension() == multi::extension_t{1L, 4L} ));

		BOOST_TEST(( name2.extension() == multi::extension_t<>{1, 4} ));
		// BOOST_TEST(( name2.extension() == multi::extension_t{1 , 4 } )); TODO(correaa) solve ambiguity

		name2[1] = 'a';
		name2[2] = 'b';
		name2[3] = 'c';

		BOOST_TEST(  name2 != name1 );
		BOOST_TEST(!(name2 == name1));

		BOOST_TEST(  name2 <  name1 );
		BOOST_TEST(  name2 <= name1 );

		BOOST_TEST(!(name2 >  name1));
		BOOST_TEST(!(name2 >= name1));

		// BOOST_TEST(!(name1 > name2));
		// BOOST_TEST(!(name1 > name2));
	}

	BOOST_AUTO_TEST_CASE(lexicographical_compare_offset_2d) {
		multi::array<char, 2> const name1 = {
			{'a', 'b'},
			{'b', 'c'},
			{'c', 'd'}
		};

		// clang-format off
	multi::array<char, 2> name2({{1, 4}, {0, 2}}, '\0');
		// clang-format on

		BOOST_TEST(  name2.size() == 3 );
		BOOST_TEST(( name2.extension() == multi::extension_t<multi::index>{1, 4} ));
		BOOST_TEST(( name2.extension() == multi::extension_t<>{1, 4} ));
		// BOOST_TEST(( name2.extension() == multi::extension_t{1 , 4 } )); TODO(correaa) solve ambiguity

		name2[1][0] = 'a';
		name2[1][1] = 'a';
		name2[2][0] = 'b';
		name2[2][1] = 'a';
		name2[3][0] = 'c';
		name2[3][1] = 'a';

		BOOST_TEST(  name2 != name1 );
		BOOST_TEST(!(name2 == name1));

		BOOST_TEST(  name2 <  name1 );
		BOOST_TEST(  name2 <= name1 );

		// BOOST_TEST(!(name2 >  name1));
		// BOOST_TEST(!(name2 >= name1));

		BOOST_TEST( name1 > name2 );
		BOOST_TEST(!(name1 < name2));
	}
	return boost::report_errors();
}
