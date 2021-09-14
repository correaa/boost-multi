// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2019-2021

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi array reference"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"
#include "../array_ref.hpp"

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(array_ref_from_carray) {
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): test
	double a[4][5] {
		{ 0.,  1.,  2.,  3.,  4.},
		{ 5.,  6.,  7.,  8.,  9.},
		{10., 11., 12., 13., 14.},
		{15., 16., 17., 18., 19.}
	};

	multi::array_ptr<double, 2> map = &a;
	BOOST_REQUIRE( &map->operator[](1)[1] == &a[1][1] );
	BOOST_REQUIRE( (*&a)[1][1] == 6. );

	multi::array_ref<double, 2>&& mar = *map;

	BOOST_REQUIRE( &mar[1][1] == &a[1][1] );

	mar[1][1] = 9.;
	BOOST_REQUIRE( &mar[1][1] == &a[1][1] );

	auto const& a_const = a;
//	double const(&a_const)[4][5] = a;
	BOOST_REQUIRE( &a_const[1][1] == &a[1][1] );

	static_assert( decltype(mar(2, {1, 3}))::rank_v == 1 , "!");

	BOOST_REQUIRE( size(mar(2, {1, 3})) == 2 );
	BOOST_REQUIRE( &mar(2, {1, 3})[1] == &a[2][2] );
}

BOOST_AUTO_TEST_CASE(array_ref_1D_reindexed) {
	std::array<std::string, 5> a{ {"a", "b", "c", "d", "e"} };

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): test type
	multi::Array<std::string(&)[1]> mar = *multi::Array<std::string(*)[1]>(&a);
	BOOST_REQUIRE( &mar[1] == &a[1] );
	BOOST_REQUIRE( sizes(mar.reindexed(1)) == sizes(mar) );
	auto diff = &(mar.reindexed(1)[1]) - &mar[0];
	BOOST_REQUIRE( diff == 0 );

	BOOST_REQUIRE( &mar.blocked(2, 4)[2] == &mar[2] );
	for(auto i : extension(mar.stenciled({2, 4}))) {
		BOOST_REQUIRE( &mar.stenciled({2, 4})[i] == &mar[i] );
	}

	multi::array<std::string, 1> arr({{2, 7}}, std::string{"xx"});
	BOOST_REQUIRE( size(arr) == 5 );
	BOOST_REQUIRE( extension(arr) == multi::iextension(2, 7) );
	arr[2] = "a";
	arr[3] = "b";
	arr[4] = "c";
	arr[5] = "d";
	arr[6] = "e";
	BOOST_REQUIRE( std::equal(arr.begin(), arr.end(), mar.begin()) );

	auto arrB = multi::array<std::string, 1>({"a", "b", "c", "d", "e"}).reindex(2);
	BOOST_REQUIRE( size(arrB) == 5 );
	BOOST_REQUIRE( arrB[2] == "a" );
	BOOST_REQUIRE( arrB[6] == "e" );
}

BOOST_AUTO_TEST_CASE(array_ref_of_nested_std_array_reindexed) {
	std::array<std::array<double, 5>, 4> a = {
		{
			{ 0.,  1.,  2.,  3.,  4.},
			{ 5.,  6.,  7.,  8.,  9.},
			{10., 11., 12., 13., 14.},
			{15., 16., 17., 18., 19.}
		}
	};

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): test type
	multi::Array<double(&)[2]> mar = *multi::Array<double(*)[2]>(&a);

	BOOST_REQUIRE( &mar[1][1] == &a[1][1] );
}

BOOST_AUTO_TEST_CASE(array_ref_reindexed) {
	double (&&a)[4][5] = { // NOLINT(hicpp-avoid-c-arrays, modernize-avoid-c-arrays, cppcoreguidelines-avoid-c-arrays): test
		{ 0,  1,  2,  3,  4},
		{ 5,  6,  7,  8,  9},
		{10, 11, 12, 13, 14},
		{15, 16, 17, 18, 19}
	};

	// NOLINTNEXTLINE(hicpp-avoid-c-arrays, modernize-avoid-c-arrays, cppcoreguidelines-avoid-c-arrays): special type
	multi::Array<double(&)[2]> mar = *multi::Array<double(*)[2]>(&a);

	BOOST_REQUIRE( &mar[1][1] == &a[1][1] );
	BOOST_REQUIRE( size(mar   .reindexed(1)) == size(mar) );
	BOOST_REQUIRE( size(mar[0].reindexed(1)) == size(mar[0]) );

	BOOST_REQUIRE( sizes(mar.reindexed(1)) == sizes(mar) );
	BOOST_REQUIRE( &mar.reindexed(1)[1][0] == &mar[0][0] );

	BOOST_REQUIRE( sizes(mar[0].reindexed(1)) == sizes(mar[0]) );
	BOOST_REQUIRE( mar[0].reindexed(1).extension().start () == mar[0].extension().start () + 1 );
	BOOST_REQUIRE( mar[0].reindexed(1).extension().finish() == mar[0].extension().finish() + 1 );

	auto diff = &mar[0].reindexed(1)[1] - &mar[0][0];
	BOOST_REQUIRE( diff == 0 );

//	BOOST_REQUIRE( &(((mar<<1).reindexed(2)>>1).reindexed(1))[1][2] == &mar[0][0] );
	BOOST_REQUIRE( &mar.reindexed(1, 2)[1][2] == &mar[0][0] );

	BOOST_REQUIRE( &mar.reindexed(1)({1, 5})[1][0] == &mar[0][0] );

	BOOST_REQUIRE( sizes(mar.stenciled({2, 4})) == std::make_tuple(2, 5) );
	BOOST_REQUIRE( &mar.stenciled({2, 4})[2][0] == &mar[2][0] );
	BOOST_REQUIRE( &mar.stenciled({2, 4}, {1, 3})[2][1] == &mar[2][1] );

//	BOOST_REQUIRE( &mar[0][0] == mar.origin() ); // origin changed meaning in on 2020/Dec/16
//	BOOST_REQUIRE( mar.base() == mar.origin() );

//	BOOST_REQUIRE( mar.stenciled({2, 4}).origin() == mar.origin() ); // origin changed meaning in on 2020/Dec/16
	BOOST_REQUIRE( mar.stenciled({2, 4}).base()   != mar.base()   );

	BOOST_REQUIRE( &mar.stenciled({2, 4})[2][0] == mar.stenciled({2, 4}).base() );

	{
		multi::array<std::string, 2> arrB = {
			{"a", "b", "c", "d", "e"},
			{"f", "g", "h", "f", "g"},
			{"h", "i", "j", "k", "l"}
		};
		arrB.reindex(2);
		BOOST_REQUIRE( size(arrB) == 3 );
		BOOST_REQUIRE( arrB[2][0] == "a" );
	}
	{
		multi::array<std::string, 2> arrB = {
			{"a", "b", "c", "d", "e"},
			{"f", "g", "h", "f", "g"},
			{"h", "i", "j", "k", "l"}
		};
		arrB.reindex(2, 1);
		BOOST_REQUIRE( size(arrB) == 3 );
		BOOST_REQUIRE( arrB[2][1] == "a" );
	}
	 {
		multi::array<std::string, 2> arrB = (multi::array<std::string, 2>
			{{"a", "b", "c", "d", "e"},
			 {"f", "g", "h", "f", "g"},
			 {"h", "i", "j", "k", "l"}})//.reindex(2, 1);
		;
		BOOST_REQUIRE( arrB.reindex(2).extension() == multi::iextension(2, 5) );
		auto x = arrB.reindexed(2).extensions();

		multi::array<std::string, 2> arrC(x);
		BOOST_REQUIRE( size(arrC) == 3 );
		BOOST_REQUIRE( size(arrC) == size(arrB) );

		BOOST_REQUIRE( arrC.extension().start()  == 2 );
		BOOST_REQUIRE( arrC.extension().finish() == 5 );

		std::cout<< arrC.extension().start()<<"..."<< arrC.extension().finish() <<std::endl;
	//	BOOST_REQUIRE( arrC[2][1] == "a" );
	}
}

BOOST_AUTO_TEST_CASE(array_ref_with_stencil) {
	std::array<std::array<double, 5>, 4> a = {
		{
			{ 0,  1,  2,  3,  4},
			{ 5,  6,  7,  8,  9},
			{10, 11, 12, 13, 14},
			{15, 16, 17, 18, 19}
		}
	};
	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): test type
	auto const& mar = *multi::Array<double(*)[2]>(&a);

	// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays): test type
	multi::Array<double[2]> s = {
		{ 0, +1,  0},
		{+1, -4, +1},
		{ 0, +1,  0}
	};
	auto const& st = s.reindexed(-1, -1);

	BOOST_REQUIRE( &st[-1][-1] == st.base() );

	multi::array<double, 2> gy(extensions(mar), 0.);

	 {
		auto xs = extensions(mar);
		for(auto i = std::get<0>(xs).start()+1; i != std::get<0>(xs).finish()-1; ++i) {
			for(auto j = std::get<1>(xs).start()+1; j != std::get<1>(xs).finish()-1; ++j) {
				auto xt = extensions(st);
				for(auto b : std::get<0>(xt)) {
					for(auto d : std::get<1>(xt)) {
						gy[i][j] += st[b][d]*mar[i + b][j + d];
					}
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(array_ref_1D_from_vector) {
	std::vector<double> v = {1, 2, 3};
	multi::array_ref<double, 1> aref({{1, 3}}, v.data());
	BOOST_REQUIRE( aref.extension() == multi::iextension(1, 3) );
	BOOST_REQUIRE( &aref[1] == &v[0] );
}

BOOST_AUTO_TEST_CASE(array_ref_2D_from_vector) {
	std::vector<double> v = {1, 2, 3, 4, 5, 6};
	multi::array_ref<double, 2> aref({2, 3}, v.data());
	BOOST_REQUIRE( &aref[1][0] == &v[3] );
}

BOOST_AUTO_TEST_CASE(array_ref_2D_from_vector_with_offset) {
	std::vector<double> v = {
		1, 2, 3,
		4, 5, 6
	};
	multi::array_ref<double, 2> aref({multi::iextension(1, 3), multi::iextension(1, 4)}, v.data());

	auto x = aref.extensions();
	BOOST_REQUIRE( std::get<0>(x) == multi::iextension(1, 3) );
	BOOST_REQUIRE( std::get<1>(x).start()  == 1 );
	BOOST_REQUIRE( std::get<1>(x).finish() == 4 );
	BOOST_REQUIRE( std::get<1>(x) == multi::iextension(1, 4) );
	BOOST_REQUIRE( x == decltype(x)(multi::iextension(1, 3), multi::iextension(1, 4)) );

	BOOST_REQUIRE( &aref[1][1] == &v[0] );
}

BOOST_AUTO_TEST_CASE(array_2D_with_offset) {
	multi::array<double, 2> a({multi::iextension(1, 3), multi::iextension(2, 5)}, 1.2);

	BOOST_REQUIRE( a.extension().start() == 1 );
	BOOST_REQUIRE( a.extension().finish() == 3 );
}

BOOST_AUTO_TEST_CASE(array_ref_1D) {
	std::array<std::string, 5> a = {"a", "b", "c", "d", "e"};

	multi::array_ref<std::string, 1>&& mar = *multi::array_ptr<std::string, 1>{&a};
//	multi::Array<std::string(&)[1]> mar = *multi::Array<std::string(*)[1]>(&a);

	BOOST_REQUIRE(  extension(mar).first() == 0 );
	BOOST_REQUIRE(  extension(mar).last()  == 5 );

	auto&& mar1 = mar.reindexed(1);

	BOOST_REQUIRE( extension(mar1).size() == extension(mar).size() );

	BOOST_REQUIRE( mar1.extension() == extension(mar1) );
	BOOST_REQUIRE(  extension(mar1).first() == 1 );
	BOOST_REQUIRE(  mar1.extension().first() == 1 );
	BOOST_REQUIRE(  mar1.extension().last()  == 6 );
	BOOST_REQUIRE( *extension(mar1).begin() == 1 );

	BOOST_REQUIRE( size(mar1) == size(mar) );
	BOOST_REQUIRE( mar1.layout().extension().start() == 1 );
	BOOST_REQUIRE( extension(mar1).start() == 1 );
	BOOST_REQUIRE( &mar1[1] == &a[0] );
	BOOST_REQUIRE( mar1.base() == &a[0] );
}

