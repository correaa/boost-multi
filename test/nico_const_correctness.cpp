// Copyright 2022-2026 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for subarray, array, range, operator!=

#include <boost/core/lightweight_test.hpp>

#if __cplusplus >= 202002L
#include <atomic>   // for atomic_ref
#include <version>  // IWYU pragma: keep  // for _GLIBCXX_RELEASE
#endif

#include <algorithm>  // for fill, copy, for_each
#include <array>      // for array
#include <iostream>   // for operator<<, basic_ostream::opera...
#include <iterator>   // for begin, end, ostream_iterator

#if __cplusplus >= 202002L
#include <thread>  // for thread
#include <tuple>   // for get  // NOLINT(misc-include-cleaner) // IWYU pragma: keep
#endif

#include <type_traits>  // for decay_t
#include <utility>      // for forward
#include <vector>       // for vector  // IWYU pragma: keep

namespace multi = boost::multi;

namespace {
template<class Array1D>
void print(Array1D const& coll) {
	// *coll.begin() = 99;  // doesn't compile "assignment of read-only location"

	std::copy(std::begin(coll), std::end(coll), std::ostream_iterator<typename Array1D::value_type>(std::cout, ", "));
	std::cout << '\n';
}

template<class Array1D>
auto fill_99(Array1D&& col) -> Array1D&& {
	std::fill(std::begin(col), std::end(col), 99);
	return std::forward<Array1D>(col);
}

template<class Array2D>
void print_2d(Array2D const& coll) {
	// *(coll.begin()->begin()) = 99;  // doesn't compile "assignment of read-only location"

	std::for_each(std::begin(coll), std::end(coll), [](auto const& row) {
		std::copy(std::begin(row), std::end(row), std::ostream_iterator<typename Array2D::element>(std::cout, ", "));
		std::cout << '\n';
	});
}

template<class Array1D>
auto fill_2d_99(Array1D&& coll) -> Array1D&& {
	// for(auto const& row : coll) {  // does not work because it would make it const
	std::for_each(std::begin(coll), std::end(coll), [](typename std::decay_t<Array1D>::reference row) {
		std::fill(std::begin(row), std::end(row), 99);
	});
	// std::transform(coll.begin(), coll.end(), coll.begin(), [](auto&& row) {
	//  std::fill(row.begin(), row.end(), 99);
	//  return std::forward<decltype(row)>(row);
	// });
	return std::forward<Array1D>(coll);
}
}  // end unnamed namespace

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// BOOST_AUTO_TEST_CASE(const_views)
	{
		multi::array<int, 1> coll1 = {0, 8, 15, 47, 11, 42};
		print(coll1);  // prints "0, 8, 15, 47, 11, 42"

		print(coll1({0, 3}));  // similar to coll1 | take(3) // prints "0, 8, 15"

		auto&& coll1_take3 = coll1({0, 3});
		print(coll1_take3);  // prints "0, 8, 15"
	}

	// BOOST_AUTO_TEST_CASE(mutating_views)
	{
		multi::array<int, 1> coll1 = {0, 8, 15, 47, 11, 42};

		fill_99(coll1);
		fill_99(coll1({0, 3}));

		auto&& coll1_take3 = coll1({0, 3});
		fill_99(coll1_take3);

		auto const& coll2 = coll1;
		// fill_99( coll2 );  // doesn't compile because coll2 is const ("assignment of read-only" inside fill_99)
		// fill_99( coll2({0, 3}) );  // similar to coll2 | take(3) doesn't compile ("assignment of read-only")

		auto const& coll1_take3_const = coll1({0, 3});
		// fill_99( coll1_take3_const );  // doesn't compile because coll1_take3_const is const ("assignment of read-only")

		(void)coll2, (void)coll1_take3_const, (void)coll1_take3;
	}

	// BOOST_AUTO_TEST_CASE(const_views_2d)
	{
		multi::array<int, 2> coll1 = {
			{0, 8, 15, 47, 11, 42},
			{0, 8, 15, 47, 11, 42},
		};

		print_2d(coll1);  // prints "0, 8, 15, 47, 11, 42"

		print_2d(coll1({0, 2}, {0, 3}));  // similar to coll1 | take(3) // prints "0, 8, 15"

		auto&& coll1_take3 = coll1({0, 2}, {0, 3});
		print_2d(coll1_take3);  // prints "0, 8, 15"
	}

	// BOOST_AUTO_TEST_CASE(mutating_views_2d)
	{
		multi::array<int, 2> coll1 = {
			{0, 8, 15, 47, 11, 42},
			{0, 8, 15, 47, 11, 42},
		};

		fill_2d_99(coll1);
		fill_2d_99(coll1({0, 2}, {0, 3}));

		auto&& coll1_take3 = coll1({0, 2}, {0, 3});
		fill_2d_99(coll1_take3);

		auto const& coll2 = coll1;
		// fill_99( coll2 );  // doesn't compile because coll2 is const ("assignment of read-only" inside fill_99)
		// fill_99( coll2({0, 3}) );  // similar to coll2 | take(3) doesn't compile ("assignment of read-only")

		auto const& coll1_take3_const = coll1({0, 2}, {0, 3});
		// fill_99( coll1_take3_const );  // doesn't compile because coll1_take3_const is const ("assignment of read-only")

		(void)coll2, (void)coll1_take3_const, (void)coll1_take3;
	}

	{
		multi::array<int, 1> arr1d = {1, 2, 3};

		// multi::array<int, 1>::const_iterator cfirst = arr1d.cbegin();
		// *cfirst.base() = 5;
		// *cfirst = 5;  // correctly fails to compile
		// cfirst[0] = 5;  // correctly fails to compile

		BOOST_TEST( arr1d[0] == 1 );
	}
	{
		multi::array<int, 1> const arr1d = {1, 2, 3};

		// multi::array<int, 1>::iterator cfirst = arr1d.begin();  // correctly fails to compile

		BOOST_TEST( arr1d[0] == 1 );
	}

	{
		std::array<int, 12> arr{
			{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		};

		{
			auto&& mds1 = multi::array_ref(arr.data(), {3, 4});

			BOOST_TEST( mds1.empty() == false );
			BOOST_TEST( mds1.size() == 3 );  // not 12
			BOOST_TEST( mds1.num_elements() == 12 );
			BOOST_TEST( mds1.dimensionality == 2 );
			BOOST_TEST( mds1.extent().size() == 3 );
			BOOST_TEST( mds1.extent().last() == 3 );

			using std::get;
			BOOST_TEST( get<0>(mds1.extents()).size() == 3 );
			BOOST_TEST( get<0>(mds1.extents()).last() == 3 );

			BOOST_TEST( get<1>(mds1.extents()).size() == 4 );
			BOOST_TEST( get<1>(mds1.extents()).last() == 4 );

			auto const [is, js] = mds1.extents();
			for(auto const i : is) {      // NOLINT(altera-unroll-loops) test range iteration
				for(auto const j : js) {  // NOLINT(altera-unroll-loops) test range iteration
#if defined(cpp_multidimensional_subscript) && (cpp_multidimensional_subscript >= 202110L)
					BOOST_TEST(( mds1[i, j] != 0 ));
#else
					BOOST_TEST(( mds1[i][j] != 0 ));
#endif
				}
			}
		}
		{
			auto&& mds2 = multi::array_ref(arr.data(), {2, 3, 2});

			BOOST_TEST( mds2.empty() == false );
			BOOST_TEST( mds2.size() == 2 );  // not 12
			BOOST_TEST( mds2.num_elements() == 12 );
			BOOST_TEST( mds2.dimensionality == 3 );
			BOOST_TEST( mds2.extent().size() == 2 );
			BOOST_TEST( mds2.extent().last() == 2 );

			using std::get;
			BOOST_TEST( get<0>(mds2.extents()).size() == 2 );
			BOOST_TEST( get<0>(mds2.extents()).last() == 2 );

			BOOST_TEST( get<1>(mds2.extents()).size() == 3 );
			BOOST_TEST( get<1>(mds2.extents()).last() == 3 );

			BOOST_TEST( get<2>(mds2.extents()).size() == 2 );
			BOOST_TEST( get<2>(mds2.extents()).last() == 2 );

			auto const [is, js, ks] = mds2.extents();
			for(auto const i : is) {          // NOLINT(altera-unroll-loops) test range iteration
				for(auto const j : js) {      // NOLINT(altera-unroll-loops) test range iteration
					for(auto const k : ks) {  // NOLINT(altera-unroll-loops) test range iteration
#if defined(cpp_multidimensional_subscript) && (cpp_multidimensional_subscript >= 202110L)
						BOOST_TEST(( mds2[i, j, k] != 0 ));
#else
						BOOST_TEST(( mds2[i][j][k] != 0 ));
#endif
					}
				}
			}
		}
		{
			auto coll = std::vector{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
			{
				auto&& mds = multi::array_ref(coll.data(), {static_cast<multi::index>(coll.size() / 3), 3});

				BOOST_TEST( mds[0][1] == 2 );

				auto&& mds1 = mds.transposed();

				BOOST_TEST( mds1[1][0] == 2 );
			}
			{
				auto&& mds = multi::array_ref(coll.data(), {3, 4});

				BOOST_TEST( mds.strided(1, 2).size() == 6 );

				auto const& mds2 = ~mds.strided(1, 2).taked(5);

				BOOST_TEST( mds2.size() == 4 );

				using std::get;
				for(auto const i : get<0>(mds2.extents())) {      // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
					for(auto const j : get<1>(mds2.extents())) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
						std::cout << mds2[i][j] << ' ';
					}
					std::cout << '\n';
				}
			}
			{
				auto&& mds = multi::array_ref(coll.data(), {4, 3});

				auto const& mds2 = ~mds;

				using std::get;

				BOOST_TEST( get<0>(mds2.sizes()) == 3 );
				BOOST_TEST( get<1>(mds2.sizes()) == 4 );

				for(int i = 0; i != get<0>(mds2.sizes()); ++i) {      // NOLINT(altera-id-dependent-backward-branch)
					for(int j = 0; j != get<1>(mds2.sizes()); ++j) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
						std::cout << mds2[i][j] << ' ';
					}
					std::cout << '\n';
				}
			}
			{
				auto&& mds = multi::array_ref(coll.data(), {4, 3});

				auto const& mds2 = ~mds;

				using std::get;

				BOOST_TEST( get<0>(mds2.sizes()) == 3 );
				BOOST_TEST( get<1>(mds2.sizes()) == 4 );

				for(int i = 0; i != get<0>(mds2.sizes()); ++i) {      // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
					for(int j = 0; j != get<1>(mds2.sizes()); ++j) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
						std::cout << mds2[i][j] << ' ';
					}
					std::cout << '\n';
				}
			}
			{
				auto&& mds = multi::array_ref(coll.data(), {4, 3});

				auto const& mds2 = ~(~mds).sliced(2, -1, -1);

				using std::get;

				BOOST_TEST( get<0>(mds2.sizes()) == 4 );
				BOOST_TEST( get<1>(mds2.sizes()) == 3 );

				for(int i = 0; i != get<0>(mds2.sizes()); ++i) {      // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
					for(int j = 0; j != get<1>(mds2.sizes()); ++j) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
						std::cout << mds2[i][j] << ' ';
					}
					std::cout << '\n';
				}
			}
		}
	}
// gcc-10's libstdc++ (32-bit multilib at least) advertises __cpp_lib_jthread without actually defining std::jthread
#if defined(__cpp_lib_jthread) && (__cpp_lib_jthread >= 201911L) && (!defined(__GLIBCXX__) || (defined(_GLIBCXX_RELEASE) && (_GLIBCXX_RELEASE >= 11)))
#if defined(__cpp_lib_atomic_ref) && (__cpp_lib_atomic_ref >= 201806L)
	{
		multi::array<int, 2> data({4, 4}, 0);

		auto counters = data.element_transformed(
			[](int& elem) noexcept { return std::atomic_ref<int>(elem); }
		);

		constexpr int nthreads = 8;

		{
			std::vector<std::jthread> pool;

			pool.reserve(nthreads);

			for(int ti = 0; ti != nthreads; ++ti) {
				pool.emplace_back(
					[&counters] {
						for(auto&& row : counters) {
							for(auto&& elem : row) {  // NOLINT(altera-unroll-loops)
								elem.fetch_add(1, std::memory_order_relaxed);
							}
						}
					}
				);
			}
		}

		BOOST_TEST( data[3][3] == 8 );
	}
#endif
#endif

	return boost::report_errors();
}
