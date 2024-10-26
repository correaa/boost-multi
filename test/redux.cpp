// Copyright 2018-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, implicit_cast, explicit_cast

#include <boost/core/lightweight_test.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>

#if defined __has_include && __has_include(<execution>)
#    include <execution>
#endif

namespace multi = boost::multi;

class watch {
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ = std::chrono::high_resolution_clock::now();
	std::string                                                 msg_;

 public:
	explicit watch(std::string msg) : msg_(std::move(msg)) {}
	~watch() {
		std::cerr << msg_ << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_).count() << " ms\n";
	};
	watch(watch const&) = delete;
	auto& operator=(watch const&) = delete;
	//	non-default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator
};

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	// multi::array<double, 2>::size_type const maxsize = 39062;  // 390625;
	// multi::array<double, 2>::size_type const nmax    = 1000;   // 10000;

	// auto pp = [] /*__host__ __device__*/ (long ix, long iy) -> double { return double(ix) * double(iy); };

	auto nx = 80000;  // nmax;     // for(long nx = 1; nx <= nmax; nx *= 10)
	auto ny = 2000;  // maxsize;  // for(long ny = 1; ny <= maxsize; ny *= 5)

	// auto total = nx*ny;

	// nx = 2;
	// ny = total / nx;

	multi::array<double, 2> K2D({nx, ny});

	for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
		for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
			K2D[ix][iy] = static_cast<double>(ix) * static_cast<double>(iy);
		}
	}

#ifndef RUNNING_ON_VALGRIND
#define RUNNING_ON_VALGRIND 0
#endif

	if(RUNNING_ON_VALGRIND) return boost::report_errors();

	{
		auto const accumulator = [&] () {
			watch _("raw loop");
			multi::array<double, 1> ret({nx}, 0.0);

			for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
				for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
					ret[ix] += K2D[ix][iy];
				}
			}
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch _("accumulate for");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				multi::array<double, 1>(K2D.extension(), 0.0),
				[](auto const& acc, auto const& col) {
					multi::array<double, 1> res(acc.extensions());
					for(auto const i : col.extension()) {
						res[i] = acc[i] + col[i];
					}
					return res;
				}
			);
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto init) {
			watch _("accumulate move");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				std::move(init),
				[](auto&& acc, auto const& col) {
					multi::array<double, 1> ret(std::forward<decltype(acc)>(acc));
					for(auto const i : col.extension()) {
						ret[i] += col[i];
					}
					return ret;
				}
			);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto init) {
			watch _("accumulate forward");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				std::move(init),
				[](auto&& acc, auto const& col) -> decltype(acc) {
					for(auto const i : col.extension()) {
						acc[i] += col[i];
					}
					return std::forward<decltype(acc)>(acc);
				}
			);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto init) {
			watch _("accumulate transform forward");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				std::move(init),
				[](auto&& acc, auto const& col) -> decltype(acc) {
					std::transform(col.begin(), col.end(), acc.begin(), acc.begin(), [](auto const& cole, auto&& acce) { return std::forward<decltype(acce)>(acce) + cole; });
					return std::forward<decltype(acc)>(acc);
				}
			);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch _("reduce transform forward");
			return std::reduce(
				(~K2D).begin(), (~K2D).end(),
				multi::array<double, 1>(K2D.extension(), 0.0),
				[](auto acc, auto const& col) {
					multi::array<double, 1> ret(std::move(acc));
					// multi::array<double, 1> ret(acc.extensions());
					std::transform(col.begin(), col.end(), ret.begin(), ret.begin(), [](auto const& cole, auto&& acce) { return std::move(acce) + cole; });
					return ret;
				}
			);
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("transform accumulate element zero");
			multi::array<double, 1> ret(K2D.extension());
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(),
				[](auto const& row) {return std::accumulate(row.begin(), row.end(), 0.0);}
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("transform reduce element zero");
			multi::array<double, 1> ret(K2D.extension());
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(),
				[](auto const& row) {return std::reduce(row.begin(), row.end(), 0.0);}
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto&& init) {
			watch const _("transform accumulate");
			std::transform(
				K2D.begin(), K2D.end(), init.begin(), init.begin(),
				[](auto const& row, auto rete) {return std::accumulate(row.begin(), row.end(), std::move(rete));}
			);
			return std::forward<decltype(init)>(init);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto&& init) {
			watch const _("> transform reduce");
			std::transform(
				K2D.begin(), K2D.end(), init.begin(), init.begin(),
				[](auto const& row, auto rete) {return std::reduce(row.begin(), row.end(), std::move(rete));}
			);
			return std::forward<decltype(init)>(init);
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

#if !defined(__NVCC__) || (__GNUC__>7)
	{
		auto const accumulator = [&] {
			watch const _("transform reduce[unseq]");
			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(), ret.begin(),
				[](auto const& row, auto rete) {return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(rete));}
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("transform reduce[par]");
			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(), ret.begin(),
				[](auto const& row, auto rete) {return std::reduce(std::execution::par, row.begin(), row.end(), std::move(rete));}
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch const _("transform reduce[par_unseq]");
			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(
				K2D.begin(), K2D.end(), ret.begin(), ret.begin(),
				[](auto const& row, auto rete) {return std::reduce(std::execution::par_unseq, row.begin(), row.end(), std::move(rete));}
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] () {
			watch const _("transform[par] reduce");
			multi::array<double, 1> ret(K2D.extension(), 0.0);
			std::transform(std::execution::par,
				K2D.begin(), K2D.end(), ret.begin(), ret.begin(),
				[](auto const& row, auto rete) {return std::reduce(row.begin(), row.end(), std::move(rete));}
			);
			return ret;
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto ret) {
			watch _("* transform[par] reduce[unseq]");

			std::transform(std::execution::par,
				K2D.begin(), K2D.end(), ret.begin(), ret.begin(),
				[](auto const& row, auto rete) {return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(rete));}
			);
			return ret;
		}(multi::array<double, 1>(K2D.extension(), 0.0));

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		multi::array<double, 1> accumulator(K2D.extension(), 0.0);
		[&] (auto acc_begin) {
			watch _("transform[par] reduce[unseq] iterator");
			return std::transform(std::execution::par,
				K2D.begin(), K2D.end(), acc_begin, acc_begin,
				[](auto const& row, auto rete) {return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(rete));}
			);
		}(accumulator.begin());

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] (auto zero_elem) {
			watch _("transform[par] reduce[unseq] element zero");
			multi::array<double, 1> ret(K2D.extension());
			std::transform(std::execution::par,
				K2D.begin(), K2D.end(), ret.begin(),
				[zz = std::move(zero_elem)](auto const& row) {return std::reduce(std::execution::unseq, row.begin(), row.end(), std::move(zz));}
			);
			return ret;
		}(0.0);

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}
#endif  // __NVCC__

	return boost::report_errors();
}
