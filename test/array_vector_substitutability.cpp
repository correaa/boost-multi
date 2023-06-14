// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2019-2023 Alfredo A. Correa

#include<boost/test/unit_test.hpp>

#include <multi/array.hpp>

#include<complex>

namespace multi = boost::multi;

template<class DynamicArray>  // e.g. std::vector or multi::array
void resize_copy_1(std::vector<double> const& source, DynamicArray& darr) {
	darr = DynamicArray(source);
}

template<class DynamicArray>  // e.g. std::vector or multi::array
void resize_copy_2(std::vector<double> const& source, DynamicArray& darr) {
	darr = DynamicArray(source.begin(), source.end());  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
}

template<class DynamicArray>  // e.g. std::vector or multi::array
void resize_copy_3(std::vector<double> const& source, DynamicArray& darr) {
	darr = std::decay_t<decltype(darr)>(source.begin(), source.end());  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
}

template<class It, class DynamicArray>   // e.g. std::vector or multi::array
void resize_copy_4(It first, It last, DynamicArray& darr) {
	darr = DynamicArray(first, last);  // or std::decay_t<decltype(da)>(source.begin(), source.end())  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
}

template<class It, class DynamicArray>  // e.g. std::vector or multi::array
void resize_copy_5(It first, It last, DynamicArray& darr) {
	darr.assign(first, last);  // or std::decay_t<decltype(da)>(source.begin(), source.end())
}

// void resize_copy_6   ----> see below test_resize_copy_6

BOOST_AUTO_TEST_CASE(test_resize_copy_1) {
	std::vector<double> const source = {0.0, 1.0, 2.0, 3.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)

	std::vector<double>     dest_v = {99.0, 99.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
	multi::array<double, 1> dest_a = {88.0, 88.0};

	BOOST_REQUIRE( dest_v.size() == 2 );
	BOOST_REQUIRE( dest_a.size() == 2 );

	resize_copy_1(source, dest_v);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );

	resize_copy_1(source, dest_a);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );
}

BOOST_AUTO_TEST_CASE(test_resize_copy_2) {
	std::vector<double> const source = {0.0, 1.0, 2.0, 3.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)

	std::vector<double>     dest_v = {99.0, 99.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
	multi::array<double, 1> dest_a = {88.0, 88.0};

	BOOST_REQUIRE( dest_v.size() == 2 );
	BOOST_REQUIRE( dest_a.size() == 2 );

	resize_copy_2(source, dest_v);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3. );

	resize_copy_2(source, dest_a);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3. );
}

BOOST_AUTO_TEST_CASE(test_resize_copy_3) {
	std::vector<double> const source = {0.0, 1.0, 2.0, 3.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)

	std::vector<double>     dest_v = {99.0, 99.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
	multi::array<double, 1> dest_a = {88.0, 88.0};

	BOOST_REQUIRE( dest_v.size() == 2 );
	BOOST_REQUIRE( dest_a.size() == 2 );

	resize_copy_3(source, dest_v);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3. );

	resize_copy_3(source, dest_a);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3. );
}

BOOST_AUTO_TEST_CASE(test_resize_copy_4) {
	std::vector<double> const source = {0.0, 1.0, 2.0, 3.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)

	std::vector<double>     dest_v = {99.0, 99.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
	multi::array<double, 1> dest_a = {88.0, 88.0};

	BOOST_REQUIRE( dest_v.size() == 2 );
	BOOST_REQUIRE( dest_a.size() == 2 );

	resize_copy_4(source.begin(), source.end(), dest_v);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );

	resize_copy_4(source.begin(), source.end(), dest_a);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );
}

BOOST_AUTO_TEST_CASE(test_resize_copy_5) {
	std::vector<double> const source = {0.0, 1.0, 2.0, 3.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)

	std::vector<double>     dest_v = {99.0, 99.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
	multi::array<double, 1> dest_a = {88.0, 88.0};

	BOOST_REQUIRE( dest_v.size() == 2 );
	BOOST_REQUIRE( dest_a.size() == 2 );

	resize_copy_5(source.begin(), source.end(), dest_v);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );

	resize_copy_5(source.begin(), source.end(), dest_a);

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );
}

BOOST_AUTO_TEST_CASE(test_resize_copy_6) {
	std::vector<double> const source = {0.0, 1.0, 2.0, 3.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)

	std::vector<double>     dest_v = {99.0, 99.0};  // testing std::vector vs multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
	multi::array<double, 1> dest_a = {88.0, 88.0};

	BOOST_REQUIRE( dest_v.size() == 2 );
	BOOST_REQUIRE( dest_a.size() == 2 );

	{  // look same code as below
		dest_v = decltype(dest_v)(source);
	}

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );

	{  // look same code as above
		dest_a = decltype(dest_a)(source);
	}

	BOOST_REQUIRE( dest_v.size() == 4 );
	BOOST_REQUIRE( dest_v[3] == 3.0 );
}

template<class T> void what(T&&) = delete;

BOOST_AUTO_TEST_CASE(array2D_as_nested_vector) {
	multi::array<double, 2> VV = {
	 {1.0, 2.0, 3.0},
	 {4.0, 5.0, 6.0},
	};

	{
		std::vector<std::vector<double>> vv(2, {}, {});
		vv[0] = std::vector<double>(VV[0].begin(), VV[0].end(), {});
		vv[1] = std::vector<double>(VV[1].begin(), VV[1].end(), {});
	}
	// {  // vvv gives an error in GCC 12 error (GCC 13 ok): ‘void* __builtin_memcpy(void*, const void*, long unsigned int)’ writing 1 or more bytes into a region of size 0 overflows the destination [-Werror=stringop-overflow=]
	//  std::vector<std::vector<double>> vv(2, {}, {});
	//  vv[0].insert(vv[0].begin(), VV[0].begin(), VV[0].end());
	//  vv[1].insert(vv[1].begin(), VV[1].begin(), VV[1].end());
	// }
	{
		std::vector<std::vector<double>> vv(2, {}, {});
		vv[0].assign(VV[0].begin(), VV[0].end());
		vv[1].assign(VV[1].begin(), VV[1].end());
	}
	{
		std::vector<std::vector<double>> vv( static_cast<std::size_t>(VV.size()), std::vector<double>(static_cast<std::size_t>((~VV).size()), {}, {}), {});
		std::copy(VV[0].begin(), VV[0].end(), vv[0].begin());
		std::copy(VV[1].begin(), VV[1].end(), vv[1].begin());
	}
	{
		std::vector<double> const VV0(VV[0].begin(), VV[0].end(), {});
		BOOST_REQUIRE( VV0[1] == VV[0][1] );
	}
	{
		std::vector<double> const VV0(VV[0]);
		BOOST_REQUIRE( VV0[1] == VV[0][1] );
	}
	{
		// #ifndef __circle_build__
		multi::array<double, 1> const V1D = {1.0, 2.0, 3.0};
		std::vector<double> const vec(V1D);
		BOOST_REQUIRE( vec[1] == 2.0 );
		// #endif
	}
	{
		// #ifndef __circle_build__
		multi::array<double, 2> const V2D = {
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
		};
		std::vector<std::vector<double>> const vec(V2D);  // = V2D doesn't work because conversion is explicit
		BOOST_REQUIRE( vec[1][2] == 6.0 );

		{
			multi::array<double, 2> A2D({2, 3});
			BOOST_REQUIRE( A2D.size() == static_cast<multi::size_t>(vec.size()) );
			A2D[0].assign(vec[0].begin());
			A2D[1].assign(vec[1].begin());
			BOOST_REQUIRE( V2D == A2D );
		}
		{
			multi::array<double, 2> A2D({2, 3});
			BOOST_REQUIRE( A2D.size() == static_cast<multi::size_t>(vec.size()) );
			A2D[0] = vec[0];
			A2D[1] = vec[1];
			BOOST_REQUIRE( V2D == A2D );
		}
	}
	{
		using array_int = typename multi::array<double, 2>::rebind<int>;
		static_assert( std::is_same_v<array_int, multi::array<int, 2>> );
	}
}
