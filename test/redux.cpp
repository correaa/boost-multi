// Copyright 2018-2024 Alfredo A. Correa
// Copyright 2024 Matt Borland
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/array.hpp>  // for array, implicit_cast, explicit_cast

#include <boost/core/lightweight_test.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>

namespace multi = boost::multi;

class watch {
	std::chrono::time_point<std::chrono::high_resolution_clock> start_ = std::chrono::high_resolution_clock::now();
	std::string                                                 msg_;

 public:
	watch(std::string msg) : msg_(std::move(msg)) {}
	~watch() {
		std::cerr << msg_ << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_).count() << " ms\n";
	};
	watch(watch const&) = delete;
	//	auto operator=(watch const&) = delete;
	//	non-default destructor but does not define a copy constructor, a copy assignment operator, a move constructor or a move assignment operator
};

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	// multi::array<double, 2>::size_type const maxsize = 39062;  // 390625;
	// multi::array<double, 2>::size_type const nmax    = 1000;   // 10000;

	// auto pp = [] /*__host__ __device__*/ (long ix, long iy) -> double { return double(ix) * double(iy); };

	auto const nx = 80000;  // nmax;     // for(long nx = 1; nx <= nmax; nx *= 10)
	auto const ny = 2000;  // maxsize;  // for(long ny = 1; ny <= maxsize; ny *= 5)

	multi::array<double, 2> K2D({nx, ny});

	for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
		for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
			K2D[ix][iy] = static_cast<double>(ix) * static_cast<double>(iy);
		}
	}

	{
		auto const              start = std::chrono::high_resolution_clock::now();
		multi::array<double, 1> accumulator({nx}, 0.0);

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {      // NOLINT(altera-id-dependent-backward-branch)
			for(multi::array<double, 2>::index iy = 0; iy != ny; ++iy) {  // NOLINT(altera-id-dependent-backward-branch,altera-unroll-loops)
				accumulator[ix] += K2D[ix][iy];
			}
		}
		std::cout << "raw loop " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << " ms\n";

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
		auto const accumulator = [&] {
			watch _("accumulate move");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				multi::array<double, 1>(K2D.extension(), 0.0),
				[](auto&& acc, auto const& col) {
					multi::array<double, 1> ret(std::forward<decltype(acc)>(acc));
					for(auto const i : col.extension()) {
						ret[i] += col[i];
					}
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
			watch _("accumulate forward");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				multi::array<double, 1>(K2D.extension(), 0.0),
				[](auto&& acc, auto const& col) -> decltype(acc) {
					for(auto const i : col.extension()) {
						acc[i] += col[i];
					}
					return std::forward<decltype(acc)>(acc);
				}
			);
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch _("accumulate transform forward");
			return std::accumulate(
				(~K2D).begin(), (~K2D).end(),
				multi::array<double, 1>(K2D.extension(), 0.0),
				[](auto&& acc, auto const& col) -> decltype(acc) {
					std::transform(col.begin(), col.end(), acc.begin(), acc.begin(), [](auto const& cole, auto const& acce) { return std::move(acce) + cole; });
					return std::forward<decltype(acc)>(acc);
				}
			);
		}();

		for(multi::array<double, 2>::index ix = 0; ix != nx; ++ix) {
			BOOST_TEST( std::abs( accumulator[ix] - static_cast<double>(ix) * ny * (ny - 1.0) / 2.0 ) < 1.0e-8);
		}
	}

	{
		auto const accumulator = [&] {
			watch _("transform accumulate");
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


	return boost::report_errors();
}
