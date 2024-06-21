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
	#pragma clang diagnostic ignored "-Wfloat-equal"
#elif defined(__GNUC__)
	#pragma GCC diagnostic push
	#pragma GCC diagnostic ignored "-Wold-style-cast"
	#pragma GCC diagnostic ignored "-Wundef"
	#pragma GCC diagnostic ignored "-Wconversion"
	#pragma GCC diagnostic ignored "-Wsign-conversion"
	#pragma GCC diagnostic ignored "-Wfloat-equal"
#elif defined(_MSC_VER)
	#pragma warning(push)
	#pragma warning(disable : 4244)
#endif

#ifndef BOOST_TEST_MODULE
	#define BOOST_TEST_MAIN
#endif

#include <boost/test/tools/fpc_tolerance.hpp>  // for tolerance
#include <boost/test/unit_test.hpp>

#if defined(__clang__)
	#pragma clang diagnostic pop
#elif defined(__GNUC__)
	#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
	#pragma warning(pop)
#endif

#include <boost/multi/array.hpp>  // for array, apply, operator==

#include <algorithm>    // for fill, all_of, transform
#include <cstddef>      // for ptrdiff_t
#include <cstdint>      // for uint64_t
#include <functional>   // for plus
#include <iterator>     // for begin, end, size, next
#include <limits>       // for numeric_limits
#include <numeric>      // for accumulate
#include <random>       // for uniform_int_distribution
#include <type_traits>  // for enable_if_t, is_same_v
#include <utility>      // for forward

namespace {

using fnv1a_size = std::uint64_t;

// from Howard Hinnart hash
auto fnv1a(unsigned char const* first, std::ptrdiff_t len, fnv1a_size hash) noexcept {  // NOLINT(bugprone-easily-swappable-parameters)
	return std::accumulate(
		first, std::next(first, len), hash,
		[prime = 1099511628211U](auto acc, auto elem) { return (acc ^ elem) * prime; }
	);
}
}  // namespace

class fnv1a_t {
	fnv1a_size h_ = 14695981039346656037U;  // offset

 public:
	using result_type = fnv1a_size;
	static constexpr auto min() { return std::numeric_limits<result_type>::min(); }
	static constexpr auto max() { return std::numeric_limits<result_type>::max(); }
	void                  operator()(unsigned char const* key, std::ptrdiff_t len) noexcept { h_ = fnv1a(key, len, h_); }
	template<class T, std::enable_if_t<std::is_fundamental_v<T>, int> = 0>
	auto operator()(T const& value) noexcept -> decltype(auto) {
		operator()(&value, sizeof(value));
		return *this;
	}
	//  result_type operator()() && noexcept{return h;}
	auto operator()() const& noexcept { return h_; }
	//  explicit operator result_type() && noexcept {return h;}
	explicit operator result_type() const& noexcept { return h_; }
};

BOOST_AUTO_TEST_CASE(fill_1d_a) {
	namespace multi = boost::multi;

	multi::array<multi::index, 1> d1D(multi::extensions_t<1>{ multi::iextension{ 10 } });
	static_assert(std::is_same_v<std::iterator_traits<decltype(begin(d1D))>::value_type, multi::index>, "!");

	using std::copy;
	copy(begin(extension(d1D)), end(extension(d1D)), begin(d1D));
	BOOST_REQUIRE( d1D[0] == 0 );
	BOOST_REQUIRE( d1D[1] == 1 );
	BOOST_REQUIRE( d1D[9] == 9 );

	d1D.assign(extension(d1D));
	BOOST_REQUIRE( d1D[0] == 0 );
	BOOST_REQUIRE( d1D[1] == 1 );
	BOOST_REQUIRE( d1D[9] == 9 );
}

BOOST_AUTO_TEST_CASE(fill_1d_b) {
	namespace multi = boost::multi;

	multi::array<multi::index, 1> d1D(begin(multi::index_extension(10)), end(multi::index_extension(10)));
	BOOST_REQUIRE( size(d1D) == 10 );
	BOOST_REQUIRE( d1D[0] == 0 );
	BOOST_REQUIRE( d1D[1] == 1 );
	BOOST_REQUIRE( d1D[9] == 9 );
}

BOOST_AUTO_TEST_CASE(fill_1d_c) {
	namespace multi = boost::multi;

	multi::array<multi::index, 1> d1D(multi::extensions_t<1>{ multi::iextension{ 10 } });
	BOOST_REQUIRE( size(d1D) == 10 );

	d1D.assign(begin(extension(d1D)), end(extension(d1D)));
	BOOST_REQUIRE( d1D[0] == 0 );
	BOOST_REQUIRE( d1D[1] == 1 );
	BOOST_REQUIRE( d1D[9] == 9 );
}

BOOST_AUTO_TEST_CASE(fill_1d_d) {
	namespace multi = boost::multi;

	multi::array<multi::index, 1> d1D(multi::extensions_t<1>{ multi::iextension{ 10 } });
	d1D.assign(extension(d1D));
	BOOST_REQUIRE( d1D[0] == 0 );
	BOOST_REQUIRE( d1D[1] == 1 );
	BOOST_REQUIRE( d1D[9] == 9 );
}

BOOST_AUTO_TEST_CASE(fill_member) {
	namespace multi = boost::multi;

	multi::array<int, 1> d1D = { 10, 20, 30, 40 };
	d1D.fill(420);

	multi::array<int, 2> d2D = {
		{1500, 160, 170, 180, 190},
		{  50,  50,  50,  50,  50},
		{1000, 110, 120, 130, 140},
		{ 500,  60,  70,  80,  90},
	};

	BOOST_REQUIRE(   d2D.elements().size()  == d2D.num_elements()  );
	BOOST_REQUIRE(   d2D.elements().base()  == d2D.base()          );
	BOOST_REQUIRE(   d2D.elements()[3]      == 180                 );
	BOOST_REQUIRE( &*d2D.elements().begin() == d2D.data_elements() );
	BOOST_REQUIRE( &*d2D.elements().end()   == d2D.data_elements() + d2D.num_elements() );

	std::fill(d2D.elements().begin(), d2D.elements().end(), 990);

	BOOST_REQUIRE( d2D[1][1] == 990 );
}

BOOST_AUTO_TEST_CASE(fill) {
	std::random_device randdev;

	namespace multi = boost::multi;

	multi::array<int, 2> d2D = {
		{1500, 160, 170, 180, 190},
		{  50,  50,  50,  50,  50},
		{1000, 110, 120, 130, 140},
		{ 500,  60,  70,  80,  90},
	};
	using std::all_of;
	BOOST_REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto const& elem) { return elem == 50;}) );

	using std::fill;
	fill(d2D[1].begin(), d2D[1].end(), 80);

	BOOST_REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto const& elem) { return elem == 80;}) );

	fill(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), 80);
	BOOST_REQUIRE( all_of(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), [](auto&& elem) { return elem == 80;}) );

	fill(begin((d2D.rotated())[1]), end((d2D.rotated())[1]), 80);
	BOOST_REQUIRE( all_of(begin((d2D.rotated())[1]), end((d2D.rotated())[1]), [](auto&& elem) { return elem == 80;}) );

	auto rand = [gauss = std::uniform_int_distribution<>(0, 10), gen = std::mt19937_64(randdev())]() mutable { return gauss(gen); };  // NOSONAR

	multi::array<int, 2> r2D({ 5, 5 });
	std::for_each(begin(r2D), end(r2D), [&](decltype(r2D)::reference elem) { std::generate(begin(elem), end(elem), rand); });
}

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(fill_1D) {
	multi::array<double, 1> const arr = { 1.0, 2.0, 3.0 };

	multi::array<double, 2> arr2({ 10, 3 });

	std::fill(begin(arr2), end(arr2), arr);

	BOOST_REQUIRE( arr2[0] == arr );
	BOOST_REQUIRE( arr2[1] == arr );

	BOOST_REQUIRE( arr2[9] == arr );
}

template<class BinaryOp, class Column, class Array, class Out>
auto broadcast(BinaryOp op, Column const& col, Array const& in, Out&& out) -> Out&& {  // NOLINT(readability-identifier-length) clang-tidy 14 bug
	std::transform(
		begin(~in), end(~in), begin(~out), begin(~out),
		[acol = (~col)[0], &op](auto const& Acol, auto&& Bcol) {
			std::transform(begin(Acol), end(Acol), begin(acol), begin(Bcol), op);
			return std::forward<decltype(Bcol)>(Bcol);
		}
	);

	return std::forward<Out>(out);
}

BOOST_AUTO_TEST_CASE(julia_broadcast, *boost::unit_test::tolerance(0.00001)) {
	multi::array<double, 2> const col = {
		{ 0.1 },
		{ 0.2 },
	};
	multi::array<double, 2> arr = {
		{1.10813, 1.72068, 1.15387},
		{1.36851, 1.66401, 1.47846},
	};

	// "broadcast"
	multi::array<double, 2> arr2(extensions(arr));
	broadcast(std::plus<>{}, col, arr, arr2);

	BOOST_REQUIRE_CLOSE( arr2[0][0], 1.20813, 1E-6 );
	BOOST_REQUIRE_CLOSE( arr2[0][1], 1.82068, 1E-6 );
	BOOST_REQUIRE_CLOSE( arr2[0][2], 1.25387, 1E-6 );
	BOOST_REQUIRE_CLOSE( arr2[1][0], 1.56851, 1E-6 );
	BOOST_REQUIRE_CLOSE( arr2[1][1], 1.86401, 1E-6 );
	BOOST_REQUIRE_CLOSE( arr2[1][2], 1.67846, 1E-6 );

	// inefficient: replicate the vector before summing elementwise
	multi::array<double, 2> ax3({ 2, 3 });

	std::fill(begin(~ax3), end(~ax3), (~col)[0]);
	BOOST_REQUIRE_CLOSE( ax3[0][0], 0.1, 1E-6 );
	BOOST_REQUIRE_CLOSE( ax3[0][1], 0.1, 1E-6 );
	BOOST_REQUIRE_CLOSE( ax3[0][2], 0.1, 1E-6 );
	BOOST_REQUIRE_CLOSE( ax3[1][0], 0.2, 1E-6 );
	BOOST_REQUIRE_CLOSE( ax3[1][1], 0.2, 1E-6 );
	BOOST_REQUIRE_CLOSE( ax3[1][2], 0.2, 1E-6 );

	multi::array<double, 2> Ap(extensions(arr));
	std::transform(begin(arr.elements()), end(arr.elements()), begin(ax3.elements()), begin(Ap.elements()), std::plus<>{});

	BOOST_REQUIRE_CLOSE( Ap[0][0], 1.20813, 1E-6 );
	BOOST_REQUIRE_CLOSE( Ap[0][1], 1.82068, 1E-6 );
	BOOST_REQUIRE_CLOSE( Ap[0][2], 1.25387, 1E-6 );
	BOOST_REQUIRE_CLOSE( Ap[1][0], 1.56851, 1E-6 );
	BOOST_REQUIRE_CLOSE( Ap[1][1], 1.86401, 1E-6 );
	BOOST_REQUIRE_CLOSE( Ap[1][2], 1.67846, 1E-6 );
}
