// Copyright 2018-2025 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#if defined(__GNUC__)
#   pragma GCC diagnostic ignored "-Wdouble-promotion"
#elif defined(_MSC_VER)
#   pragma warning(disable : 4244)  // warning C4244: 'initializing': conversion from '_Ty' to '_Ty', possible loss of data
#endif

#include <boost/multi/adaptors/blas.hpp>  // IWYU pragma: keep
#include <boost/multi/array.hpp>          // for array, implicit_cast, explicit_cast

#include <boost/core/lightweight_test.hpp>

#include <algorithm>  // IWYU pragma: keep
#include <chrono>     // NOLINT(build/c++11)
#include <cmath>      // IWYU pragma: keep
#include <iostream>
#include <numeric>  // IWYU pragma: keep
#include <string>
#include <string_view>
// ssssIsWsYsUs pragma: no_include <stdlib.h>                         // for abs
#include <utility>  // for move  // IWYU pragma: keep  // NOLINT(misc-include-cleaner) bug in clang-tidy 19

// IWYU pragma: no_include <pstl/glue_numeric_impl.h>         // for reduce, transform_reduce
// IWYU pragma: no_include <cstdlib>                          // for abs
// IWYU pragma: no_include <new>                              // for bad_alloc

#ifndef __NVCC__
#   if defined(__has_include) && __has_include(<execution>) && (!defined(__INTEL_LLVM_COMPILER) || (__INTEL_LLVM_COMPILER > 20240000))
#       if !(defined(__clang__) && defined(__CUDA__))
#           include <execution>  // IWYU pragma: keep
#       endif
#   endif
#endif

namespace multi = boost::multi;

class watch {
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ = std::chrono::high_resolution_clock::now();

	std::string msg_;

 public:
	explicit watch(std::string_view msg) : msg_(msg) {}  // NOLINT(fuchsia-default-arguments-calls)
	~watch() {
		std::cerr << msg_ << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_).count() << " ms\n";
	}
	watch(watch const&)          = delete;
	watch(watch&&)               = delete;
	auto operator=(watch const&) = delete;
	auto operator=(watch&&)      = delete;
	//  non-default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator
};

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	// multi::array<double, 2>::size_type const maxsize = 39062;  // 390625;
	// multi::array<double, 2>::size_type const nmax    = 1000;   // 10000;

	// auto pp = [] /*__host__ __device__*/ (long ix, long iy) -> double { return double(ix) * double(iy); };

	auto const nx = 40000;  // nmax;     // for(long nx = 1; nx <= nmax; nx *= 10)
	auto const ny = 2000;   // maxsize;  // for(long ny = 1; ny <= maxsize; ny *= 5)

	// auto const nx = 4000;  // nmax;     // for(long nx = 1; nx <= nmax; nx *= 10)
	// auto const ny = 200;   // maxsize;  // for(long ny = 1; ny <= maxsize; ny *= 5)

	// auto total = nx*ny;

	// nx = 2;
	// ny = total / nx;

	multi::array<double, 2> K2D({nx, ny});

	for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
		for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
			K2D[ix][iy] = static_cast<double>(ix) * static_cast<double>(iy);
		}
	}

	{
		auto const accumulator = [&]() {
			multi::array<double, 1> ret({nx}, 0.0);
			for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
				for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
					ret[ix] += K2D[ix][iy];
				}
			}
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - (static_cast<double>(ix) * ny * (ny - 1.0) / 2.0) ) < 1.0e-8);
		}
	}

#if defined(NDEBUG) && !defined(RUNNING_ON_VALGRIND)

	{
		auto const accumulator = [&](watch = watch("raw loop")) {  // NOLINT(fuchsia-default-arguments-declarations)
			multi::array<double, 1> ret({nx}, 0.0);
			for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
				for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
					ret[ix] += K2D[ix][iy];
				}
			}
			return ret;
		}();  // NOLINT(fuchsia-default-arguments-calls)

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("accumulate for");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				multi::array<double, 1>(multi::array<double, 1>::extensions_type{K2D.extension()}, 0.0),
				[](auto const& acc, auto const& col) {
					multi::array<double, 1> res(acc.extensions());
					for(auto const i : col.extension()) {  // NOLINT(altera-unroll-loops)
						res[i] = acc[i] + col[i];
					}
					return res;
				}
			);
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&](auto init) {
			watch const _("accumulate move");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(), std::move(init), [](auto&& acc, auto const& col) {
					multi::array<double, 1> ret(std::forward<decltype(acc)>(acc));
					for(auto const i : col.extension()) {  // NOLINT(altera-unroll-loops)
						ret[i] += col[i];
					}
					return ret;
				}
			);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&](auto init) {
			watch const _("accumulate forward");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(), std::move(init), [](auto&& acc, auto const& col) -> decltype(acc) {
					for(auto const i : col.extension()) {  // NOLINT(altera-unroll-loops)
						acc[i] += col[i];
					}
					return std::forward<decltype(acc)>(acc);
				}
			);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&](auto init, watch = watch("accumulate transform forward")) {  // NOLINT(fuchsia-default-arguments-declarations)
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(), std::move(init), [](auto&& acc, auto const& col) -> decltype(acc) {
					std::transform(col.begin(), col.end(), acc.begin(), acc.begin(), [](auto const& cole, auto&& acce) { return std::forward<decltype(acce)>(acce) + cole; });
					return std::forward<decltype(acc)>(acc);
				}
			);
		}(multi::array<double, 1>(K2D.extension(), 0.0));  // NOLINT(fuchsia-default-arguments-calls)

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

#   if (!defined(__GLIBCXX__) || (__GLIBCXX__ >= 20190502))
	{
		auto const accumulator = [&] {
			watch const _("reduce transform forward");
			return std::reduce(
				(~K2D).begin(), (~K2D).end(), multi::array<double, 1>(K2D.extension(), 0.0), [](auto acc, auto const& col) {
					multi::array<double, 1> ret(std::move(acc));
					std::transform(col.begin(), col.end(), ret.begin(), ret.begin(), [](auto const& cole, auto&& acce) { return std::forward<decltype(acce)>(acce) + cole; });
					return ret;
				}
			);
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}
#   endif

	{
		auto const accumulator = [&] {
			watch const _("transform accumulate element zero");

			multi::array<double, 1> ret(K2D.extension());
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(), [](auto const& row) { return std::accumulate(row.begin(), row.end(), 0.0); }
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

#   if (!defined(__GLIBCXX__) || (__GLIBCXX__ >= 20190502))
	{
		auto const accumulator = [&] {
			watch const _("transform reduce element zero");

			multi::array<double, 1> ret(K2D.extension());
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(), [](auto const& row) { return std::reduce(row.begin(), row.end(), 0.0); }
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}
#   endif

	{
		auto const accumulator = [&](auto&& init) {
			watch const _("transform accumulate");
			std::transform(
				K2D.begin(), K2D.end(), init.begin(), init.begin(), [](auto const& row, auto rete) { return std::accumulate(row.begin(), row.end(), std::move(rete)); }
			);
			return std::forward<decltype(init)>(init);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

#   if (!defined(__GLIBCXX__) || (__GLIBCXX__ >= 20200000))
	{
		auto const accumulator = [&](auto&& init) {
			watch const _("> transform reduce");
			std::transform(
				K2D.begin(), K2D.end(), init.begin(), init.begin(), [](auto const& row, auto rete) { return std::reduce(row.begin(), row.end(), std::move(rete)); }
			);
			return std::forward<decltype(init)>(init);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}
#   endif

#   if (__cplusplus >= 202002L)
#       if (defined(__has_include) && __has_include(<execution>))
#           if !defined(__NVCC__) && !defined(__NVCOMPILER) && !(defined(__clang__) && defined(__CUDA__)) && (!defined(__clang_major__) || (__clang_major__ > 7))
#               if (!defined(__GLIBCXX__) || (__GLIBCXX__ >= 20220000)) && !defined(_LIBCPP_VERSION)
#                   if !defined(__apple_build_version__) && (!defined(__INTEL_LLVM_COMPILER) || (__INTEL_LLVM_COMPILER > 20240000))
	{
		auto const accumulator = [&](watch = watch("transform reduce[unseq]")) {  // NOLINT(fuchsia-default-arguments-declarations)
			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				K2D.begin(), K2D.end(),
				ret.begin(),
				ret.begin(),
				[](auto const& row, auto rete) { return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(rete)); }
			);
			return ret;
		}();  // NOLINT(fuchsia-default-arguments-calls)

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("transform reduce[par]");

			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				K2D.begin(), K2D.end(),
				ret.begin(),
				ret.begin(),
				[](auto const& row, auto rete) { return std::reduce(std::execution::par, row.begin(), row.end(), std::move(rete)); }
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("transform reduce[par_unseq]");

			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				K2D.begin(), K2D.end(),
				ret.begin(),
				ret.begin(),
				[](auto const& row, auto rete) { return std::reduce(std::execution::par_unseq, row.begin(), row.end(), std::move(rete)); }
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&]() {
			watch const _("transform[par] reduce");

			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				std::execution::par,
				K2D.begin(), K2D.end(),
				ret.begin(),
				ret.begin(),
				[](auto const& row, auto rete) { return std::reduce(row.begin(), row.end(), std::move(rete)); }
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&](auto ret) {
			watch const _("* transform[par] reduce[unseq]");
			std::transform(
				std::execution::par,
				K2D.begin(), K2D.end(),
				ret.begin(),
				ret.begin(),
				[](auto const& row, auto rete) { return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(rete)); }
			);
			return ret;
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		multi::array<double, 1> accumulator(K2D.extension(), 0.0);
		[&](auto acc_begin) {
			watch const _("transform[par] reduce[unseq] iterator");
			return std::transform(
				std::execution::par,
				K2D.begin(), K2D.end(),
				acc_begin, acc_begin,
				[](auto const& row, auto rete) { return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(rete)); }
			);
		}(accumulator.begin());

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&](auto zero_elem, watch = watch("transform[par] reduce[unseq] element zero")) {  // NOLINT(fuchsia-default-arguments-declarations)
			multi::array<double, 1> ret(K2D.extension());
			std::transform(
				std::execution::par,
				K2D.begin(), K2D.end(),
				ret.begin(),
				[zz = std::move(zero_elem)](auto const& row) { return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(zz)); }
			);
			return ret;
		}(0.0);  // NOLINT(fuchsia-default-arguments-calls)

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-6);
		}
	}
#                   endif
#               endif
#           endif
#       endif
#   endif  // __NVCC__
#endif

	// {
	//  auto const accumulator = [&](auto&& init) {
	//      watch const             _("blas gemv");
	//      multi::array<double, 1> ones({init.extension()}, 1.0);
	//      multi::blas::gemv_n(1.0, K2D.begin(), K2D.size(), ones.begin(), 0.0, init.begin());
	//      return std::forward<decltype(init)>(init);
	//  }(multi::array<double, 1>(K2D.extension(), 0.0));

	//  for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
	//      BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
	//  }
	// }

	// {
	//  auto const accumulator = [&](auto&& init) {
	//      watch const _("blas gemv smart");
	//      multi::blas::gemv_n(1.0, K2D.begin(), K2D.size(), init[0].begin(), 0.0, init[1].begin());
	//      return +init[1];
	//  }(multi::array<double, 2>({2, K2D.extension()}, 1.0));

	//  for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {  // NOLINT(altera-unroll-loops)
	//      BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
	//  }
	// }

	// Chris
	{
		int const em  = 1200;
		int const en  = 1000;
		int const ell = 800;

		multi::array<double, 3> a3d({em, en, ell});
		multi::array<double, 2> b2d({en, ell});

		std::iota(a3d.elements().begin(), a3d.elements().end(), 20.0);
		std::iota(b2d.elements().begin(), b2d.elements().end(), 30.0);

		multi::array<double, 1> c_gold(em, 0.0);
		{
			for(int k = 0; k != em; ++k) {
				for(int j = 0; j != en; ++j) {       // NOLINT(altera-unroll-loops)
					for(int i = 0; i != ell; ++i) {  // NOLINT(altera-unroll-loops)
						c_gold[k] += a3d[k][j][i] * b2d[j][i];
					}
				}
			}
			c_gold = multi::array<double, 1>(em, 0.0);

			watch const _("chris raw 3-loop");

			for(int k = 0; k != em; ++k) {
				for(int j = 0; j != en; ++j) {       // NOLINT(altera-unroll-loops)
					for(int i = 0; i != ell; ++i) {  // NOLINT(altera-unroll-loops)
						c_gold[k] += a3d[k][j][i] * b2d[j][i];
					}
				}
			}
		}

		{
			multi::array<double, 1> c_flat(em, 0.0);
			{
				watch const _("chris raw 2-loop flat");

				for(auto const k : c_flat.extension()) {
					for(auto const ji : b2d.elements().extension()) {  // NOLINT(altera-unroll-loops)
						c_flat[k] += a3d[k].elements()[ji] * b2d.elements()[ji];
					}
				}
			}

			BOOST_TEST( c_gold == c_flat );
		}
		{
			multi::array<double, 1> c_flat(em, 0.0);
			{
				watch const _("chris raw 2-loop flat reversed");

				for(auto const ji : b2d.elements().extension()) {  // NOLINT(altera-unroll-loops)
					for(auto const k : c_flat.extension()) {
						c_flat[k] += a3d[k].elements()[ji] * b2d.elements()[ji];
					}
				}
			}

			BOOST_TEST( c_gold == c_flat );
		}

		{
			multi::array<double, 1> c_flat(em, 0.0);
			{
				watch const _("chris raw 1-loop flat reversed transform");

				for(auto const ji : b2d.elements().extension()) {  // NOLINT(altera-unroll-loops)
					std::transform(c_flat.begin(), c_flat.end(), a3d.begin(), c_flat.begin(), [&](auto&& c_flat_elem, auto const& a3d_row) {
						return std::forward<decltype(c_flat_elem)>(c_flat_elem) + a3d_row.elements()[ji] * b2d.elements()[ji];
					});
				}
			}

			BOOST_TEST( c_gold == c_flat );
		}

		{
			multi::array<double, 1> c_flat(em, 0.0);
			{
				watch const _("chris raw 1-loop transform_reduce");

				for(auto const k : c_flat.extension()) {
					c_flat[k] = std::transform_reduce(
						a3d[k].elements().begin(), a3d[k].elements().end(),
						b2d.elements().begin(), c_flat[k]
					);
				}
			}

			BOOST_TEST( c_gold == c_flat );
		}

		{
			multi::array<double, 1> c_flat(em, 0.0);
			{
				watch const _("chris transform transform_reduce");

				std::transform(
					a3d.begin(), a3d.end(), c_flat.begin(), c_flat.begin(),
					[&](auto const& a3d_row, auto&& c_flat_elem) {
						return std::transform_reduce(
							a3d_row.elements().begin(), a3d_row.elements().end(),
							b2d.elements().begin(), std::forward<decltype(c_flat_elem)>(c_flat_elem)
						);
					}
				);
			}

			BOOST_TEST( c_gold == c_flat );
		}
		{
			multi::array<double, 1> c_flat(em, 0.0);
			{
				watch const _("chris accumulate");

				c_flat = std::accumulate(
					b2d.elements().extension().begin(), b2d.elements().extension().end(),
					multi::array<double, 1>(em, 0.0),
					[&](auto const& acc, auto const& ij) {
						multi::array<double, 1> ret = acc;
						for(auto const k : ret.extension()) {
							ret[k] += a3d[k].elements()[ij] * b2d.elements()[ij];
						}
						return ret;
					}
				);
			}

			BOOST_TEST( c_gold == c_flat );
		}

		{
			watch const _("chris accumulate move");

			auto const c_flat = std::accumulate(
				b2d.elements().extension().begin(), b2d.elements().extension().end(),
				multi::array<double, 1>(em, 0.0),
				[&](auto&& acc, auto const& ij) {
					std::transform(
						acc.begin(), acc.end(), a3d.begin(), acc.begin(),
						[&](auto&& acce, auto const& a3de) { return std::move(acce) + a3de.elements()[ij] * b2d.elements()[ij]; }
					);
					return std::forward<decltype(acc)>(acc);
				}
			);
			BOOST_TEST( c_gold == c_flat );
		}

		{
			auto const c_flat = [&] {
				watch const _("chris transform reduce move transforms");

				return std::transform_reduce(
					b2d.elements().extension().begin(), b2d.elements().extension().end(),
					multi::array<double, 1>(em, 0.0),
					[&](auto&& acc, auto const& rhs) {
						std::transform(
							acc.begin(), acc.end(), rhs.begin(), acc.begin(),
							[&](auto&& acce, auto const& rhse) { return std::move(acce) + rhse; }
						);
						return std::forward<decltype(acc)>(acc);
					},
					[&](auto const& ij) {
						multi::array<double, 1> ret(em);
						std::transform(
							a3d.begin(), a3d.end(), ret.begin(),
							[&](auto const& a3de) { return a3de.elements()[ij] * b2d.elements()[ij]; }
						);
						return ret;
					}
				);
			}();
			BOOST_TEST( c_gold == c_flat );
		}
	}

	return boost::report_errors();
}
