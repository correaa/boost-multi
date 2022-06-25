// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2018-2022 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi element access"
#include<boost/test/unit_test.hpp>

#include "multi/array.hpp"

#include <deque>
#include <numeric>  // for iota

namespace multi = boost::multi;

template<class T> void what(T&&) = delete;

namespace test_bee {
	struct bee{};

	template<class Array> auto paren(Array&& arr, bee const&/*unused*/) -> decltype(auto) {
		return std::forward<Array>(arr)(0);
	}
}  // end namespace test_bee

BOOST_AUTO_TEST_CASE(overload_paren) {
	multi::array<double, 1> arr({10});
	test_bee::bee zero;
	BOOST_REQUIRE( &arr(0) == &arr(zero) );
}

BOOST_AUTO_TEST_CASE(empty_intersection) {
	multi::array<double, 1> arr({10});
	multi::array<double, 1> arr2;

	auto const is = intersection(arr.extension(), arr2.extension());
	BOOST_REQUIRE( arr(is).is_empty() );
	arr2(is) = arr(is);

	BOOST_REQUIRE( arr2(is) == arr(is) );
}

BOOST_AUTO_TEST_CASE(multi_tests_element_access_with_tuple) {
	multi::array<double, 2> m({3, 3}, 44.);
	std::array<int, 2> p = {{1, 2}};

	BOOST_REQUIRE( m[p[0]][p[1]] == m(1, 2) );
	BOOST_REQUIRE( &m(p[0], p[1]) == &m[p[0]][p[1]] );

	BOOST_REQUIRE( &m[p[0]][p[1]] == &m(p[0], p[1]) );
	BOOST_REQUIRE( &m(p[0], p[1]) == &m.apply(p) );

#if not defined(__circle_build__)
	BOOST_REQUIRE( &m[p[0]][p[1]] == &std::apply(m, p) );
	BOOST_REQUIRE( &m[p[0]][p[1]] == &     apply(m, p) );
#endif
}

BOOST_AUTO_TEST_CASE(multi_tests_extension_with_tuple) {
	{
		multi::array<double, 2>::extensions_type x = {3, 4};
		multi::array<double, 2> A(x, 44.);
		BOOST_REQUIRE( size(A) == 3 );
	}
	{
		auto const t = std::make_tuple(3, 4);
		auto const [n, m] = t;
		multi::array<double, 2> A({n, m}, 44.);
		BOOST_REQUIRE( size(A) == 3 );
	}
	{
		auto const t = std::make_tuple(3, 4);
		auto A = std::apply([](auto const&... e) {return multi::array<double, 2>({e...}, 55.);}, t);
		BOOST_REQUIRE( size(A) == 3 );
	}
}

BOOST_AUTO_TEST_CASE(multi_test_constness_reference) {
	multi::array<double, 2> const m({10, 10}, 99.);

	BOOST_REQUIRE( size( m(1, {0, 3}) ) == 3 );

	BOOST_REQUIRE( m(1, {0, 3})[1] == 99. );
	static_assert( decltype( m({0, 3}, 1) )::rank_v == 1 , "!");
	BOOST_REQUIRE( size(m.sliced(0, 3)) == 3 );

	BOOST_REQUIRE( m.range({0, 3}).rotated()[1].unrotated().size() == 3 );

	BOOST_REQUIRE( m({0, 3}, {0, 3})[1][1] == 99. );

	static_assert(not std::is_assignable<decltype(m(1, {0, 3})[1]), double>{}, "!");
//  none of these lines should compile because m is read-only
//  m(1, {0, 3})[1] = 88.;
//  m({0, 3}, 1)[1] = 77.;
//  m({0, 3}, {0, 3})[1][1] = 66.;
}

#if 1
//BOOST_AUTO_TEST_CASE(multi_test_non_constness_reference) {
//	multi::array<double, 2> m({10, 10}, 99.);

//	BOOST_REQUIRE( size( m(1, {0, 3}) ) == 3 );
//	static_assert(std::is_assignable<decltype(m(1, {0, 3})[1]), double>{}, "!");

//	static_assert( decltype( m({0, 3}, 1) )::rank_v == 1 , "!");
//	BOOST_REQUIRE( m(1, {0, 3})[1] == 99. );
//	BOOST_REQUIRE( size(m.sliced(0, 3)) == 3 );

//	BOOST_REQUIRE( size(m.range({0, 3}).rotated()(1L).unrotated()) == 3 );
//	BOOST_REQUIRE( size(m(multi::index_range{0, 3}, 1)) == 3 );

//	BOOST_REQUIRE( m({0, 3}, {0, 3})[1][1] == 99. );

//	m(1, {0, 3})[1] = 88.;
//	BOOST_REQUIRE( m(1, {0, 3})[1] == 88. );
//}

BOOST_AUTO_TEST_CASE(multi_test_stencil) {
	multi::array<std::string, 2> A =
		{{"a", "b", "c", "d", "e"},
		 {"f", "g", "h", "f", "g"},
		 {"h", "i", "j", "k", "l"}}
	;

	BOOST_REQUIRE(      size(A) == 3                                            );
	BOOST_REQUIRE(           A.num_elements() == 3*5L                           );
	BOOST_REQUIRE(           A[1][2] == "h"                                     );

	BOOST_REQUIRE(      size(A          ({1, 3}, {2, 5})) == 2                  );
	BOOST_REQUIRE( extension(A          ({1, 3}, {2, 5})).start() == 0          );
	BOOST_REQUIRE(           A          ({1, 3}, {2, 5}).num_elements() == 2*3L );
	BOOST_REQUIRE(           A          ({1, 3}, {2, 5}).num_elements() == 2*3L );
	BOOST_REQUIRE(           A          ({1, 3}, {2, 5})[0][0] == "h"           );
	BOOST_REQUIRE(          &A          ({1, 3}, {2, 5})[0][0] == &A[1][2]      );

	BOOST_REQUIRE(      size(A.stenciled({1, 3}, {2, 5})) == 2                  );
	BOOST_REQUIRE( extension(A.stenciled({1, 3}, {2, 5})).start() == 1          );
	BOOST_REQUIRE(           A.stenciled({1, 3}, {2, 5}).num_elements() == 2*3L );
	BOOST_REQUIRE(           A.stenciled({1, 3}, {2, 5}) [1][2] == "h"          );
	BOOST_REQUIRE(          &A.stenciled({1, 3}, {2, 5}) [1][2] == &A[1][2]     );

	BOOST_REQUIRE(  A().elements().size() == A.num_elements() );

	BOOST_REQUIRE( &A({1, 3}, {2, 5}).elements()[0] == &A(1, 2) );
	BOOST_REQUIRE( &A({1, 3}, {2, 5}).elements()[A({1, 3}, {2, 5}).elements().size() - 1] == &A(2, 4) );

	BOOST_REQUIRE( &A({1, 3}, {2, 5}).elements().front() == &A(1, 2) );
	BOOST_REQUIRE( &A({1, 3}, {2, 5}).elements().back()  == &A(2, 4) );
}

BOOST_AUTO_TEST_CASE(multi_test_elements_1D) {
	multi::array<double, 1> A = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
	BOOST_REQUIRE( A.size() == 10 );

	BOOST_REQUIRE(  A.elements().size() == 10 );
	BOOST_REQUIRE( &A.elements()[0] == &A[0] );
	BOOST_REQUIRE( &A.elements()[9] == &A[9] );

	BOOST_REQUIRE(      A.elements().begin() <  A.elements().end()     );
	BOOST_REQUIRE(      A.elements().end()   >  A.elements().begin()   );
	BOOST_REQUIRE(      A.elements().begin() != A.elements().end()     );
	BOOST_REQUIRE( not( A.elements().begin() == A.elements().end()   ) );

	BOOST_REQUIRE(  A().elements().begin() <  A().elements().end() );
	BOOST_REQUIRE(  A().elements().begin() == A().elements().begin() );

	BOOST_REQUIRE( A().elements().begin() <  A().elements().end() or A().elements().begin() == A().elements().end() );
	BOOST_REQUIRE( A().elements().begin() <= A().elements().end() );

	BOOST_REQUIRE(  A().elements().end()  >  A().elements().begin() );
	BOOST_REQUIRE(  A().elements().end()  >= A().elements().begin() );

	A.elements() = {9., 8., 7., 6., 5., 4., 3., 2., 1., 0.};
	BOOST_REQUIRE( A[2] == 7. );
	BOOST_REQUIRE( A.elements()[2] == 7. );
}

BOOST_AUTO_TEST_CASE(multi_test_elements_1D_as_range) {
	multi::array<double, 1> A = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.};
	BOOST_REQUIRE( A.size() == 10 );

	A().elements() = {9., 8., 7., 6., 5., 4., 3., 2., 1., 0.};
	BOOST_REQUIRE( A[2] == 7. );
	BOOST_REQUIRE( A.elements()[2] == 7. );
}


//BOOST_AUTO_TEST_CASE(multi_extension_intersection) {
//	multi::array<double, 1> A = {{2., 2., 2.}};
//	multi::array<double, 1> B = {{3., 3., 3., 3.}};

//	BOOST_REQUIRE( intersection( extension(A), extension(B) ).size() == 3 );
//	for(auto i : intersection( extension(A), extension(B) ) ) {
//		B[i] += A[i];
//	}

//	BOOST_REQUIRE( B[2] == 5. );
//	BOOST_REQUIRE( B[3] == 3. );
//}

//BOOST_AUTO_TEST_CASE(multi_extensions_intersection) {
//	multi::array<double, 2> A = {{2., 2., 2.}, {2., 2., 2.}};
//	multi::array<double, 2> B = {{3., 3., 3., 3.}, {3., 3., 3., 3.}, {3., 3., 3., 3.}};

//	BOOST_REQUIRE( extensions(A).num_elements() ==  6 );
//	BOOST_REQUIRE( extensions(B).num_elements() == 12 );

//	auto is = intersection( extensions(A), extensions(B) );
//	BOOST_REQUIRE( is.num_elements() == 6 );

//	multi::array<double, 2> C(is);
//	C(std::get<0>(is), std::get<1>(is)) = A;
//	BOOST_REQUIRE( C == A );
//}

//BOOST_AUTO_TEST_CASE(multi_extensions_intersection_2) {
//	multi::array<double, 2> A({80, 20}, 4.);
//	multi::array<double, 2> B({30, 70}, 8.);

//	BOOST_REQUIRE( extensions(A).num_elements() == 80*20L );
//	BOOST_REQUIRE( extensions(B).num_elements() == 30*70L );

//	auto is = intersection( extensions(A), extensions(B) );
//	BOOST_REQUIRE( is.num_elements() == 30*20L );

//	multi::array<double, 2> C(is);
//	C(std::get<0>(is), std::get<1>(is)) = A(std::get<0>(is), std::get<1>(is));
//	BOOST_REQUIRE( C[16][17] == 4. );

//	C(std::get<0>(is), std::get<1>(is)) = B(std::get<0>(is), std::get<1>(is));
//	BOOST_REQUIRE( C[16][17] == 8. );
//}

//BOOST_AUTO_TEST_CASE(multi_elements_iterator) {
//	multi::array<double, 2> A({3, 4});
//	std::iota(A.data_elements(), A.data_elements() + A.num_elements(), 0.);

//	BOOST_REQUIRE( A().elements()[7] == 7. );
//	BOOST_REQUIRE( A().elements().begin()[7] == 7. );
//}

//BOOST_AUTO_TEST_CASE(multi_elements_iterator_1D) {
//	multi::array<double, 1> A({100}, 0.);
//	std::iota(A.data_elements(), A.data_elements() + A.num_elements(), 0.);

//	BOOST_REQUIRE( A().elements()[7] == 7. );
//	BOOST_REQUIRE( A().elements().begin()[7] == 7. );
//}

//BOOST_AUTO_TEST_CASE(multi_range_rotated) {
//	multi::array<double, 5> A({3, 5, 7, 11, 13});

//	using multi::_;

//	BOOST_REQUIRE( &A( _,  _, {3, 5}) == &A.rotated().rotated().range({3, 5}).unrotated().unrotated() );
//	BOOST_REQUIRE( &A( _,  _, {3, 5}) == &A( _,  _, {3, 5}, _) );

//	BOOST_REQUIRE( &A(*_, *_, {3, 5}) == &A.rotated().rotated().range({3, 5}).unrotated().unrotated() );
//	BOOST_REQUIRE( &A(*_, *_, {3, 5}) == &A(*_, *_, {3, 5}, *_) );

//	BOOST_REQUIRE( &A(multi::ALL, multi::ALL, {3, 5}) == &A.rotated().rotated().range({3, 5}).unrotated().unrotated() );
//	BOOST_REQUIRE( &A(multi::ALL, multi::ALL, {3, 5}) == &A(multi::ALL, multi::ALL, {3, 5}, multi::ALL) );
//}

//BOOST_AUTO_TEST_CASE(multi_home_2d) {
//	multi::array<double, 2> A({3, 5});

//	auto h = A.home();
//	BOOST_REQUIRE( & h[1][2] == & A[1][2] );
//	BOOST_REQUIRE( & h[0][0] == & A[0][0] );
//	BOOST_REQUIRE(   h[1][2] ==   A[1][2] );

//	BOOST_REQUIRE( & *h == & A[0][0] );
//	BOOST_REQUIRE(   h[0][0] ==   A[0][0] );

//	BOOST_REQUIRE(   *h ==   A[0][0] );

//	BOOST_REQUIRE(    h(0, 0) ==   A[0][0] );
//	BOOST_REQUIRE(    h(1, 0) ==   A[1][0] );
//	h += multi::detail::tuple<multi::size_t, multi::size_t>{1L, 0L};
////	BOOST_REQUIRE(    *(h + multi::make_tuple(1, 0)) == A[1][0] );

//}

//BOOST_AUTO_TEST_CASE(multi_home_5d) {
//	multi::array<double, 5> A({3, 5, 7, 11, 13});

//	auto const h = A.home();
//	BOOST_REQUIRE( & h[1][2][3][4][5] == & A[1][2][3][4][5] );
//}

BOOST_AUTO_TEST_CASE(elements_from_init_list_2D) {
	multi::array<double, 2> A({3, 2});
	A().elements() = {1., 2., 3., 4., 5., 6.};
	BOOST_REQUIRE(A[1][0] == 3.);

	A.elements() = {10., 20., 30., 40., 50., 60.};
	BOOST_REQUIRE(A[1][0] == 30.);
}

BOOST_AUTO_TEST_CASE(front_back_2D) {
	multi::array<double, 2> A({3, 4});
	std::iota(A.data_elements(), A.data_elements() + A.num_elements(), 0.);

	BOOST_REQUIRE(  A.front()[2] ==  A[0][2] );
	BOOST_REQUIRE( &A.front()[2] == &A[0][2] );

	BOOST_REQUIRE(  A.back ()[2] ==  A[2][2] );
	BOOST_REQUIRE( &A.back ()[2] == &A[2][2] );
}

BOOST_AUTO_TEST_CASE(front_back_1D) {
	multi::array<double, 1> A({30}, double{});
	std::iota(A.data_elements(), A.data_elements() + A.num_elements(), 0.);

	BOOST_REQUIRE(  A.front() ==  A[ 0] );
	BOOST_REQUIRE( &A.front() == &A[ 0] );

	BOOST_REQUIRE(  A.back () ==  A[29] );
	BOOST_REQUIRE( &A.back () == &A[29] );
}

BOOST_AUTO_TEST_CASE(elements_rvalues) {
	using movable_type = std::vector<double>;
	movable_type movable_value(5., 99.);

	multi::array<movable_type, 1> A = {movable_value, movable_value, movable_value};
	BOOST_REQUIRE( A.size() == 3 );

	movable_type front = std::move(A)[0];

	BOOST_REQUIRE( front == movable_value );
	BOOST_REQUIRE( A[0].empty()           );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing purposes
	BOOST_REQUIRE( A[1] == movable_value  );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing purposes

	std::move(A)[1] = movable_value;
}

template<class Array1D>
void assign_elements_from_to(Array1D&& arr, std::deque<std::vector<double>>& dest) {
	std::copy(std::forward<Array1D>(arr).begin(), std::forward<Array1D>(arr).end(), std::back_inserter(dest));
}

BOOST_AUTO_TEST_CASE(elements_rvalues_nomove) {
	using movable_type = std::vector<double>;
	movable_type movable_value(5., 99.);

	multi::array<movable_type, 1> A = {movable_value, movable_value, movable_value};
	BOOST_REQUIRE( A.size() == 3 );

	std::deque<std::vector<double>> q1;

	assign_elements_from_to(A, q1);

	BOOST_REQUIRE( A[0] == movable_value );

	std::deque<std::vector<double>> q2;

	assign_elements_from_to(std::move(A), q2);

	BOOST_REQUIRE( A[0].empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing purposes

	BOOST_REQUIRE( q1 == q2 );
}

BOOST_AUTO_TEST_CASE(elements_rvalues_assignment) {
	std::vector<double> v = {1., 2., 3.};
	std::move(v) = std::vector<double>{3., 4., 5.};
	std::move(v)[1] = 99.;  // it compiles  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved) for testing purposes
//  std::move(v[1]) = 99.;  // does not compile

//  double a = 5.;
//	std::move(a) = 9.;  // does not compile
//  BOOST_REQUIRE( a == 9. );

	multi::array<double, 1> A = {1., 2., 3.};
	multi::array<double, 1> B = {1., 2., 3.};
	std::move(A) = B;  // this compiles TODO(correaa) should it?

//  std::move(A)[0] = 10.;  // does not compile
}

#endif
