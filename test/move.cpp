// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2020-2022 Alfredo A. Correa

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi move"
#include<boost/test/unit_test.hpp>

#include <boost/multi_array.hpp>

#include "multi/array.hpp"

#include <algorithm>  // for std::move
#include <memory>
#include <vector>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(move_unique_ptr_1D) {
	{
		multi::array<std::unique_ptr<int>, 1> A(multi::extensions_t<1>{10});
		multi::array<std::unique_ptr<int>, 1> B(multi::extensions_t<1>{10});
		std::move(A.begin(), A.end(), B.begin());
	}
	{
		multi::array<std::unique_ptr<int>, 1> A(multi::extensions_t<1>{10});
		multi::array<std::unique_ptr<int>, 1> B = std::move(A);
	}
	{
		multi::array<std::unique_ptr<int>, 1> A(multi::extensions_t<1>{10});
		multi::array<std::unique_ptr<int>, 1> B(multi::extensions_t<1>{10});
		B = std::move(A);
	}
	{
		multi::array<std::unique_ptr<int>, 1> A(multi::extensions_t<1>{10});
		A[1] = std::make_unique<int>(42);

		multi::array<std::unique_ptr<int>, 1> B(multi::extensions_t<1>{10});
	//  B() = A();  // fails to compile, elements are not copy assignable
		B() = A().moved();
		BOOST_REQUIRE( !A[1] );
		BOOST_REQUIRE(  B[1] );
		BOOST_REQUIRE( *B[1] == 42 );
	}
}

BOOST_AUTO_TEST_CASE(multi_swap) {
	multi::array<double, 2> A({3,  5}, 99.);
	multi::array<double, 2> B({7, 11}, 88.);
	swap(A, B);
	BOOST_REQUIRE( size(A) == 7 );
	BOOST_REQUIRE( A[1][2] == 88. );
	BOOST_REQUIRE( B[1][2] == 99. );
}

BOOST_AUTO_TEST_CASE(multi_std_swap) {
	multi::array<double, 2> A({3,  5}, 99.);
	multi::array<double, 2> B({7, 11}, 88.);
	using std::swap;
	swap(A, B);
	BOOST_REQUIRE( size(A) == 7 );
	BOOST_REQUIRE( A[1][2] == 88. );
	BOOST_REQUIRE( B[1][2] == 99. );
}

BOOST_AUTO_TEST_CASE(multi_array_clear) {
	multi::array<double, 2> A({10, 10}, 99.);
	A.clear();
	BOOST_REQUIRE(A.is_empty());
	A.reextent({20, 20}, 99.);
	BOOST_REQUIRE(not A.is_empty());
	clear(A).reextent({30, 30}, 88.);
	BOOST_REQUIRE(A[15][15] == 88.);
}

BOOST_AUTO_TEST_CASE(multi_array_move) {
	std::vector<multi::array<double, 2> > Av(10, multi::array<double, 2>({4, 5}, 99.));
	multi::array<double, 2> B(std::move(Av[0]), std::allocator<double>{});

	BOOST_REQUIRE( is_empty(Av[0]) );
	BOOST_REQUIRE( size(B) == 4 );
	BOOST_REQUIRE( B[1][2] == 99. );
}

BOOST_AUTO_TEST_CASE(multi_array_move_into_vector) {
	std::vector<multi::array<double, 2> > Av(10, multi::array<double, 2>({4, 5}, 99.));
	std::vector<multi::array<double, 2> > Bv; Bv.reserve(Av.size());

	std::move( begin(Av), end(Av), std::back_inserter(Bv) );

	BOOST_REQUIRE( size(Bv) == size(Av) );
	BOOST_REQUIRE( is_empty(Av[4]) );
	BOOST_REQUIRE( size(Bv[5]) == 4 );
	BOOST_REQUIRE( Bv[5][1][2] == 99. );
}

BOOST_AUTO_TEST_CASE(multi_array_move_into_vector_reserve) {
	std::vector<multi::array<double, 2> > Av(10, multi::array<double, 2>({4, 5}, 99.));
	std::vector<multi::array<double, 2> > Bv; Bv.reserve(Av.size());

//	for(auto& v: Av) Bv.emplace_back(std::move(v), std::allocator<double>{}); // segfaults nvcc 11.0 but not nvcc 11.1
	std::move(begin(Av), end(Av), std::back_inserter(Bv));

	BOOST_REQUIRE( size(Bv) == size(Av) );
	BOOST_REQUIRE( is_empty(Av[4]) );
	BOOST_REQUIRE( size(Bv[5]) == 4 );
	BOOST_REQUIRE( Bv[5][1][2] == 99. );
}

BOOST_AUTO_TEST_CASE(multi_array_move_into_vector_move) {
	std::vector<multi::array<double, 2> > Av(10, multi::array<double, 2>({4, 5}, 99.));
	std::vector<multi::array<double, 2> > Bv = std::move(Av);

	Av.clear();
	BOOST_REQUIRE( size(Av) == 0 );
	BOOST_REQUIRE( size(Bv) == 10 );
	BOOST_REQUIRE( size(Bv[5]) == 4 );
	BOOST_REQUIRE( Bv[5][1][2] == 99. );
}

BOOST_AUTO_TEST_CASE(multi_array_move_array) {
	multi::array<std::vector<double>, 2> A({10, 10}, std::vector<double>(5) );
	auto B = std::move(A); (void)B;
	BOOST_REQUIRE( A.   empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved,clang-analyzer-cplusplus.Move) test deterministic moved from state
	BOOST_REQUIRE( A.is_empty() );  // NOLINT(bugprone-use-after-move,hicpp-invalid-access-moved,clang-analyzer-cplusplus.Move) test deterministic moved from state
}

BOOST_AUTO_TEST_CASE(multi_array_move_elements) {
	multi::array<std::vector<double>, 1> A({10}, std::vector<double>(5) );

	std::vector<std::vector<double>> sink(5);

	auto* ptr1 = A[1].data();

	std::copy( A({0, 5}).moved().begin(), A({0, 5}).moved().end(), sink.begin() );
	BOOST_REQUIRE(     A[1].empty() );
	BOOST_REQUIRE( not A[5].empty() );

	BOOST_REQUIRE( sink[1].data() == ptr1 );
}

BOOST_AUTO_TEST_CASE(multi_array_move_elements_range) {
	multi::array<std::vector<double>, 1> A({10}, std::vector<double>(5) );

	std::vector<std::vector<double>> sink(5);

	auto* ptr1 = A[1].data();

	std::copy( A({0, 5}).moved().elements().begin(), A({0, 5}).moved().elements().end(), sink.begin() );
	BOOST_REQUIRE(     A[1].empty() );
	BOOST_REQUIRE( not A[5].empty() );

	BOOST_REQUIRE( sink[1].data() == ptr1 );
}

BOOST_AUTO_TEST_CASE(multi_array_move_elements_to_array) {
	multi::array<std::vector<double>, 1> A({10}, std::vector<double>(5, 99.) );
	BOOST_REQUIRE( A.size() == 10 );
	multi::array<std::vector<double>, 1> B({ 5}, {}, {});

	auto* ptr1 = A[1].data();

	B().elements() = A({0, 5}).moved().elements();

	BOOST_REQUIRE( B[1].size() == 5 );
	BOOST_REQUIRE( B[1][4] == 99. );

	BOOST_REQUIRE(     A[1].empty() );
	BOOST_REQUIRE( not A[5].empty() );

	BOOST_REQUIRE( B[1].data() == ptr1 );
}

BOOST_AUTO_TEST_CASE(move_range_vector_1D) {
	std::vector<std::vector<double>> A(10, std::vector<double>{1., 2., 3.});
	std::vector<std::vector<double>> B(10);
	std::move(A.begin(), A.end(), B.begin());

	BOOST_REQUIRE( B[0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0].empty() );
	BOOST_REQUIRE( A[1].empty() );
}

BOOST_AUTO_TEST_CASE(copy_range_1D) {
	multi::array<std::vector<double>, 1> A({3}, std::vector<double>{1., 2., 3.});
	BOOST_REQUIRE( A.size() == 3 );
	multi::array<std::vector<double>, 1> B({3}, std::vector<double>{});
	std::copy(A.begin(), A.end(), B.begin());

	BOOST_REQUIRE( B[0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( A[1] == std::vector<double>({1., 2., 3.}) );
}

BOOST_AUTO_TEST_CASE(move_range_1D) {
	multi::array<std::vector<double>, 1> A({3}, std::vector<double>{1., 2., 3.});
	BOOST_REQUIRE( A.size() == 3 );
	multi::array<std::vector<double>, 1> B({3}, std::vector<double>{});
	std::move(A.begin(), A.end(), B.begin());

	BOOST_REQUIRE( B[0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0].empty() );
	BOOST_REQUIRE( A[1].empty() );
}

BOOST_AUTO_TEST_CASE(move_range_1D_moved_begin) {
	multi::array<std::vector<double>, 1> A({3}, std::vector<double>{1., 2., 3.});
	BOOST_REQUIRE( A.size() == 3 );
	multi::array<std::vector<double>, 1> B({3}, std::vector<double>{});
	std::copy(A.mbegin(), A.mend(), B.begin());

	BOOST_REQUIRE( B[0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0].empty() );
	BOOST_REQUIRE( A[1].empty() );
}

template<class... Ts> void what(Ts&&...) = delete;

BOOST_AUTO_TEST_CASE(copy_move_range) {
	multi::array<std::vector<double>, 2> A({10, 20}, std::vector<double>{1., 2., 3.});
	multi::array<std::vector<double>, 2> B({10, 20}, std::vector<double>{}          );

	std::copy(A.mbegin(), A.mend(), B.begin());

	BOOST_REQUIRE( B[0][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[0][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( B[1][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0][0].empty() );
	BOOST_REQUIRE( A[0][1].empty() );

	BOOST_REQUIRE( A[1][0].empty() );
	BOOST_REQUIRE( A[1][1].empty() );
}

BOOST_AUTO_TEST_CASE(copy_move_range_moved_begin) {
	multi::array<std::vector<double>, 2> A({10, 20}, std::vector<double>{1., 2., 3.});
	multi::array<std::vector<double>, 2> B({10, 20}, std::vector<double>{}          );

	std::copy(A.moved().begin(), A.moved().end(), B.begin());

	BOOST_REQUIRE( B[0][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[0][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( B[1][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0][0].empty() );
	BOOST_REQUIRE( A[0][1].empty() );

	BOOST_REQUIRE( A[1][0].empty() );
	BOOST_REQUIRE( A[1][1].empty() );
}

BOOST_AUTO_TEST_CASE(copy_move_range_moved_begin_block) {
	multi::array<std::vector<double>, 2> A({10, 20}, std::vector<double>{1., 2., 3.});
	multi::array<std::vector<double>, 2> B({ 3,  5}, std::vector<double>{}          );

	std::copy(A({5, 8}, {10, 15}).moved().begin(), A({5, 8}, {10, 15}).moved().end(), B.begin());

	BOOST_REQUIRE( B[0][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[0][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( B[1][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[5][10].empty() );
	BOOST_REQUIRE( A[5][11].empty() );

	BOOST_REQUIRE( A[6][10].empty() );
	BOOST_REQUIRE( A[6][11].empty() );
}


BOOST_AUTO_TEST_CASE(move_reference_range) {
	multi::array<std::vector<double>, 2> A({10, 20}, std::vector<double>{1., 2., 3.});
	multi::array<std::vector<double>, 2> B({10, 20}, std::vector<double>{}          );

//  B() = A().moved();
	std::copy(A().moved().begin(), A().moved().end(), B().begin());

	BOOST_REQUIRE( B[0][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[0][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( B[1][0] == std::vector<double>({1., 2., 3.}) );
	BOOST_REQUIRE( B[1][1] == std::vector<double>({1., 2., 3.}) );

	BOOST_REQUIRE( A[0][0].empty() );
	BOOST_REQUIRE( A[0][1].empty() );

	BOOST_REQUIRE( A[1][0].empty() );
	BOOST_REQUIRE( A[1][1].empty() );
}

//BOOST_AUTO_TEST_CASE(move_move_range) {
//	multi::array<std::vector<double>, 2> A({10, 20}, std::vector<double>{1., 2., 3.});
//	multi::array<std::vector<double>, 2> B({10, 20}, std::vector<double>{}          );

//	*B.begin() = std::move(*A.begin());
//	*std::next(B.begin()) = std::move(*std::next(A.begin()));

//	BOOST_REQUIRE( B[0][0] == std::vector<double>({1., 2., 3.}) );
//	BOOST_REQUIRE( B[0][1] == std::vector<double>({1., 2., 3.}) );

//	BOOST_REQUIRE( B[1][0] == std::vector<double>({1., 2., 3.}) );
//	BOOST_REQUIRE( B[1][1] == std::vector<double>({1., 2., 3.}) );

//	BOOST_REQUIRE( A[0][0].empty() );
//	BOOST_REQUIRE( A[0][1].empty() );

//	BOOST_REQUIRE( A[1][0].empty() );
//	BOOST_REQUIRE( A[1][1].empty() );
//}

//BOOST_AUTO_TEST_CASE(move_move_algo_range) {
//	multi::array<std::vector<double>, 2> A({10, 20}, std::vector<double>{1., 2., 3.});
//	multi::array<std::vector<double>, 2> B({10, 20}, std::vector<double>{}          );

////  std::move(A.begin(), A.end(), B.begin());
//	auto a = A.begin();
//	auto b = B.begin();
//	while( a != A.end() ) {*b++ = (a++)->moved();}

//	BOOST_REQUIRE( B[0][0] == std::vector<double>({1., 2., 3.}) );
//	BOOST_REQUIRE( B[0][1] == std::vector<double>({1., 2., 3.}) );

//	BOOST_REQUIRE( B[1][0] == std::vector<double>({1., 2., 3.}) );
//	BOOST_REQUIRE( B[1][1] == std::vector<double>({1., 2., 3.}) );

//	BOOST_REQUIRE( A[0][0].empty() );
//	BOOST_REQUIRE( A[0][1].empty() );

//	BOOST_REQUIRE( A[1][0].empty() );
//	BOOST_REQUIRE( A[1][1].empty() );
//}
