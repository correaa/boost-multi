// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2019-2022 Alfredo A. Correa

// #define BOOST_TEST_MODULE "C++ Unit Tests for Multi allocators"  // NOLINT(cppcoreguidelines-macro-usage) title
#include<boost/test/unit_test.hpp>

#include "multi/array.hpp"

#include <scoped_allocator>
#include <vector>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(std_vector_of_arrays) {
	std::vector<multi::array<double, 2>> va;
	std::transform(
		begin(multi::iextension(3)), end(multi::iextension(3)),
		std::back_inserter(va),
		[](auto idx){return multi::array<double, 2>({idx, idx}, static_cast<double>(idx));}
	);

	BOOST_REQUIRE( size(va[0]) == 0 );
	BOOST_REQUIRE( size(va[1]) == 1 );
	BOOST_REQUIRE( size(va[2]) == 2 );
	BOOST_REQUIRE( va[1] [0][0] == 1 );
	BOOST_REQUIRE( va[2] [0][0] == 2 );

	std::vector<multi::array<double, 2>> const wa = {  // testing std::vector of multi:array NOLINT(fuchsia-default-arguments-calls,-warnings-as-errors)
		multi::array<double, 2>({0, 0}, 0.0),
		multi::array<double, 2>({1, 1}, 1.0),
		multi::array<double, 2>({2, 2}, 2.0),
	};
	BOOST_REQUIRE( size(va) == size(wa) );
	BOOST_REQUIRE( va == wa );

	std::vector<multi::array<double, 2>> ua(3, std::allocator<multi::array<double, 2>>{});
	auto iex = multi::iextension(static_cast<multi::size_type>(ua.size()));
	std::transform(
		begin(iex), end(iex),
		begin(ua),
		[](auto idx) {return multi::array<double, 2>({idx, idx}, static_cast<double>(idx));}
	);
	BOOST_REQUIRE( ua == va );
}

BOOST_AUTO_TEST_CASE(array1d_of_arrays2d) {
	multi::array<multi::array<double, 2>, 1> arr(multi::extensions_t<1>(multi::iextension{10}), multi::array<double, 2>{});
	BOOST_REQUIRE( size(arr) == 10 );

	std::transform(
		begin(extension(arr)), end(extension(arr)), begin(arr),
		[](auto idx) {return multi::array<double, 2>({idx, idx}, static_cast<double>(idx));}
	);

	BOOST_REQUIRE( size(arr[0]) == 0 );
	BOOST_REQUIRE( size(arr[1]) == 1 );
	BOOST_REQUIRE( size(arr[8]) == 8 );
	BOOST_REQUIRE( arr[8][4][4] == 8.0 );
}

BOOST_AUTO_TEST_CASE(array_3d_of_array_2d)  {
	multi::array<multi::array<double, 3>, 2> AA({10, 20}, multi::array<double, 3>{});
	std::transform(extension(AA).begin(), extension(AA).end(), AA.begin(), AA.begin(), [](auto idx, auto&& row) -> decltype(row) {
		std::transform(extension(row).begin(), extension(row).end(), row.begin(), [idx](auto jdx) {
			return multi::array<double, 3>({idx + jdx, idx + jdx, idx + jdx}, 99.0);
		});
		return std::forward<decltype(row)>(row);
	});

	BOOST_REQUIRE( size(AA[9][19]) == 9 + 19 );
	BOOST_REQUIRE( AA[9][19][1][1][1] == 99. );
}

BOOST_AUTO_TEST_CASE(array_3d_of_array_2d_no_init)  {
	multi::array<multi::array<double, 3>, 2> AA({10, 20});
	std::transform(extension(AA).begin(), extension(AA).end(), AA.begin(), AA.begin(), [](auto idx, auto&& row) -> decltype(row) {
		std::transform(extension(row).begin(), extension(row).end(), row.begin(), [idx](auto jdx) {
			return multi::array<double, 3>({idx + jdx, idx + jdx, idx + jdx}, 99.0);
		});
		return std::forward<decltype(row)>(row);
	});

	BOOST_REQUIRE( size(AA[9][19]) == 9 + 19 );
	BOOST_REQUIRE( AA[9][19][1][1][1] == 99. );
}


BOOST_AUTO_TEST_CASE(const_elements) {
	auto ptr = std::make_unique<double const>(2.0);
//  *ptr = 3.0;  // ok, can't assign
	BOOST_REQUIRE( *ptr == 2.0 );

//  multi::array<double const, 2, std::allocator<double>> arr({10, 10}, 99.0);
//
//  BOOST_REQUIRE( arr[1][2] == 99.0 );
}

struct base1 {
	inline static std::size_t heap_size_ = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
	static auto heap_size() {return heap_size_;}
};

template <class T>
class allocator1 : base1 {
 public:
    using value_type    = T;

    allocator1() noexcept = default;
    template <class U> allocator1(allocator1<U> const& /*other*/) noexcept {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

    auto   allocate(std::size_t n) { base1::heap_size_ += 1; return static_cast<value_type*>(::operator new (n*sizeof(value_type)));}
    void deallocate(value_type* ptr, std::size_t /*n*/) noexcept {::operator delete(ptr);}
};

template <class T, class U>
auto operator==(allocator1<T> const& /*x*/, allocator1<U> const& /*y*/) noexcept { return true; }

template <class T, class U>
auto operator!=(allocator1<T> const& /*x*/, allocator1<U> const& /*y*/) noexcept { return false; }

struct base2 {
	inline static std::size_t heap_size_ = 0;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
	static auto heap_size() {return heap_size_;}
};

template<class T>
class allocator2 : base2 {
 public:
    using value_type = T;

    allocator2() noexcept = default;
    template<class U> allocator2(allocator2<U> const& /*other*/)  noexcept {}  // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)

    auto allocate(std::size_t n) { base2::heap_size_ += 1; return static_cast<value_type*>(::operator new(n * sizeof(value_type))); }

    void deallocate(value_type* ptr, std::size_t /*n*/) noexcept { ::operator delete(ptr);}
};

template<class T, class U>
auto operator==(allocator2<T> const& /*x*/, allocator2<U> const& /*y*/) noexcept { return true; }

template<class T, class U>
auto operator!=(allocator2<T> const& /*x*/, allocator2<U> const& /*y*/) noexcept { return false; }

// BOOST_AUTO_TEST_CASE(scoped_allocator_vector) {
//  using InnerCont = std::vector<int, allocator2<int>>;
//  using OuterCont = std::vector<InnerCont, std::scoped_allocator_adaptor<allocator1<InnerCont>>>;

//  OuterCont cont;
//  cont.resize(2);

//  cont.back().resize(10);
//  cont.back().resize(100);
//  cont.back().resize(200);

//  BOOST_TEST( base1::heap_size() == 1 );
//  BOOST_TEST( base2::heap_size() == 3 );
// }

BOOST_AUTO_TEST_CASE(scoped_allocator_array_vector) {
	using InnerCont = std::vector<int, allocator2<int>>;
	using OuterCont = multi::array<InnerCont, 2, std::scoped_allocator_adaptor<allocator1<InnerCont>>>;

	OuterCont cont({3, 4});

	cont[1][2].resize(10);
	cont[1][2].resize(100);
	cont[1][2].resize(200);

	BOOST_TEST( base1::heap_size() == 1 );
	BOOST_TEST( base2::heap_size() == 3 );
}
