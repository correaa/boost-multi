// Copyright 2019-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// #if defined(__clang__)
//  #pragma clang diagnostic push
//  #pragma clang diagnostic ignored "-Wunknown-warning-option"
//  #pragma clang diagnostic ignored "-Wconversion"
//  #pragma clang diagnostic ignored "-Wextra-semi-stmt"
//  #pragma clang diagnostic ignored "-Wold-style-cast"
//  #pragma clang diagnostic ignored "-Wundef"
//  #pragma clang diagnostic ignored "-Wsign-conversion"
//  #pragma clang diagnostic ignored "-Wswitch-default"
// #elif defined(__GNUC__)
//  #pragma GCC diagnostic push
//  #if (__GNUC__ > 7)
//      #pragma GCC diagnostic ignored "-Wcast-function-type"
//  #endif
//  #pragma GCC diagnostic ignored "-Wconversion"
//  #pragma GCC diagnostic ignored "-Wold-style-cast"
//  #pragma GCC diagnostic ignored "-Wsign-conversion"
//  #pragma GCC diagnostic ignored "-Wundef"
// #endif

// #ifndef BOOST_TEST_MODULE
//  #define BOOST_TEST_MAIN
// #endif

// #include <boost/test/included/unit_test.hpp>

// #if defined(__clang__)
//  #pragma clang diagnostic pop
// #elif defined(__GNUC__)
//  #pragma GCC diagnostic pop
// #endif

#include <boost/multi/array.hpp>

#include <algorithm>    // for max  // IWYU pragma: keep  // for GNU stdlib
#include <complex>      // for complex  // IWYU pragma: keep  // bug in iwyu 18.1.7
#include <cstddef>      // for size_t
#include <iterator>     // for size
#include <string>       // for string
#include <type_traits>  // for is_copy_assignable_v, is_copy_co...
#include <utility>      // for move
#include <vector>       // for vector

namespace multi = boost::multi;

struct multiplies_bind1st {
	using complex = std::complex<double>;
	explicit multiplies_bind1st(multi::array<complex, 2>&& marr) : m_(std::move(marr)) {}  // this produces a bug in nvcc11.0
 private:
	multi::array<complex, 2> m_;
};

#include <boost/core/lightweight_test.hpp>
#define BOOST_AUTO_TEST_CASE(CasenamE) [[maybe_unused]] void* CasenamE;

int main() {
BOOST_AUTO_TEST_CASE(multi_construct_1d) {
	multi::static_array<int, 1> arr(multi::extensions_t<1>{ multi::iextension{ 10 } }, 10);
	//  multi::static_array<int, 1> arr(multi::array<int, 1>::extensions_type{10}, 10);
	BOOST_TEST( size(arr) == 10 );
	BOOST_TEST( arr[1] == 10 );
}

BOOST_AUTO_TEST_CASE(multi_constructors_inqnvcc_bug) {
	using complex = std::complex<double>;

	multi::array<complex, 2> marr({ 10, 10 });
	multiplies_bind1st(std::move(marr));
}

BOOST_AUTO_TEST_CASE(multi_constructors_1d) {
	{
		multi::array<double, 1> const arr(multi::extensions_t<1>{ multi::iextension{ 10 } });
		BOOST_TEST( size(arr) == 10 );
	}
	{
		multi::array<int, 1> arr(multi::extensions_t<1>{ multi::iextension{ 10 } }, int{});
		BOOST_TEST( size(arr) == 10 );
		BOOST_TEST( arr[5] == int{} );
	}
	{
		multi::array<int, 1> arr(multi::extensions_t<1>{ multi::iextension{ 10 } }, int{});
		BOOST_TEST( size(arr) == 10 );
		BOOST_TEST( arr[5] == int{} );
	}
#if defined(__cpp_deduction_guides) && !defined(__NVCC__)
	{
		multi::array arr(multi::extensions_t<1>({ 0, 10 }), int{});
		BOOST_TEST( size(arr) == 10 );
		BOOST_TEST( arr[5] == int{} );
	}
	{
		// clang-format off
		multi::array arr({{0, 10}}, int{});
		// clang-format on
		BOOST_TEST( size(arr) == 10 );
		BOOST_TEST( arr[5] == int{} );
	}
	{
		multi::array arr({ 10 }, int{});
		BOOST_TEST( size(arr) == 10 );
		BOOST_TEST( arr[5] == int{} );
	}
	{
		multi::array arr(10, int{});
		BOOST_TEST( size(arr) == 10 );
		BOOST_TEST( arr[5] == int{} );
	}
#endif
}

BOOST_AUTO_TEST_CASE(multi_constructors_2d_ctad) {
#if defined(__cpp_deduction_guides) && !defined(__NVCC__)
	multi::array arr({ 10, 20 }, int{});
	BOOST_TEST( size(arr) == 10 );
	BOOST_TEST( arr[5][6] == int{} );
#endif
}

BOOST_AUTO_TEST_CASE(multi_constructors) {
	{
		// multi::array<double, 1> arr({10}); assert(size(A)==1); // warning in clang
	} {
		// multi::array<double, 1> arr({10}, double{}); assert(size(arr)==10); // warning in clang
	} {
		// multi::array<double, 1> arr({10}, double{}); assert(size(arr)==10); // warning in clang
	} {
		// multi::array<double, 1> arr({10}, 0.); assert(size(arr)==10); // warning in clang
	} {
		// multi::array<double, 1> arr({10}, {}); assert(size(arr)==10); // error ambiguous
	} {
		multi::array<int, 1> arr = { 10 };
		BOOST_TEST( size(arr) == 1 );
		BOOST_TEST( arr[0] == 10 );
	}
	{
		multi::array<std::size_t, 1> arr = { 10 };
		BOOST_TEST( size(arr) == 1 );
		BOOST_TEST( arr[0] == 10 );
	}
	{
		multi::array<int, 1> arr = { 10 };
		BOOST_TEST( size(arr) == 1 );
		BOOST_TEST( arr[0] == 10 );
	}
	{
		multi::array<int, 1> arr({ 10 });
		BOOST_TEST( size(arr) == 1 );
		BOOST_TEST( arr[0] == 10 );
	}
	{
		multi::array<std::size_t, 1> arr({ 10 });
		BOOST_TEST( size(arr) == 1 );
		BOOST_TEST( arr[0] == 10 );
	}
	{
		multi::array<int, 1> arr({ 10 });
		BOOST_TEST( size(arr) == 1 );
		BOOST_TEST( arr[0] == 10 );
		//}{ multi::array<std::size_t, 1> arr({{10}})   ; assert( size(arr)==1 and arr[0]==10 );  // clang warns about double bracked
		//}{ multi::array<int        , 1> arr({{10}})   ; assert( size(arr)==1 and arr[0]==10 );  // clang warns about double bracked
		//}{ multi::array<double     , 1> arr({{10}})   ; assert( size(arr)==1 and arr[0]==10 );  // clang warns about double bracked
	}
	{
		multi::array<std::size_t, 1> const arr({ 0, 10 });
		BOOST_TEST( size(arr) == 2 );
	}
	{
		multi::array<int, 1> const arr({ 0, 10 });
		BOOST_TEST( size(arr) == 2 );
	}
	{
		multi::array<double, 1> const arr({ 0, 10 });
		BOOST_TEST( size(arr) == 2 );
	}
	{
		using T = multi::array<std::string, 3>;

		static_assert(std::is_nothrow_destructible_v<T>);
		static_assert(std::is_default_constructible_v<T>);
		static_assert(std::is_nothrow_default_constructible_v<T>);

		static_assert(std::is_copy_constructible_v<T>);
		static_assert(std::is_copy_assignable_v<T>);

		// static_assert( std::is_nothrow_copy_constructible_v<T> );
		// static_assert( std::is_nothrow_copy_assignable_v<T> );

		static_assert(std::is_move_constructible_v<T>);
		static_assert(std::is_move_assignable_v<T>);

		static_assert(std::is_nothrow_move_constructible_v<T>);
		static_assert(std::is_nothrow_move_assignable_v<T>);
	}
}

BOOST_AUTO_TEST_CASE(views_are_not_allocable) {
	// multi::array<double, 2> const AA = {{1.0, 2.0}, {3.0, 4.0}};
	// [[maybe_unused]] decltype(AA[0])* pp = new decltype(AA[0]){AA[0]};
	// delete pp;
}

BOOST_AUTO_TEST_CASE(views_are_not_placeable) {
	// multi::array<double, 2> const AA = {{1.0, 2.0}, {3.0, 4.0}};
	// auto&& A0 = AA[0];
	// new(std::addressof(A0)) decltype(AA[0]){AA[1]};
}

BOOST_AUTO_TEST_CASE(views_cannot_be_elements) {
	multi::array<double, 2> const AA = {
		{1.0, 2.0},
		{3.0, 4.0},
	};
	std::vector<decltype(AA[0])> vv;
	vv.emplace_back(AA[0]);
	vv.push_back(AA[0]);
	// auto&& A0 = AA[0];
	// vv.push_back(A0);
}

BOOST_AUTO_TEST_CASE(views_cannot_be_elements2) {
	// multi::array<double, 2> const AA = {{1.0, 2.0}, {3.0, 4.0}};
	// std::vector<decltype(AA[0])> vv(3, AA[0]);
}

// vvv this test gives an error with Windows' GCC
// BOOST_AUTO_TEST_CASE(submultis_are_allocable) {
//  multi::array<double, 2> const AA = {
//    {1.0, 2.0},
//    {3.0, 4.0},
//  };
// [[maybe_unused]] auto pp = std::unique_ptr<multi::array<double, 1>>(new multi::array<double, 1>{AA[0]});  // NOLINT(modernize-make-unique) testing new
// BOOST_TEST(pp);
// }

//  vvv this test gives an error with Windows' GCC
// BOOST_AUTO_TEST_CASE(submultis_are_placeable) {
//  multi::array<double, 2> const AA = {
//    {1.0, 2.0},
//    {3.0, 4.0},
//  };

//  using D1 = multi::array<double, 1>;

//  void* buf = ::operator new(sizeof(D1));
//  D1*   pd1 = new (buf) D1{AA[0]};
//  pd1->~D1();  // NOSONAR(cpp:S3432) testing placement new
//  ::operator delete(buf);
// }

return boost::report_errors();}
