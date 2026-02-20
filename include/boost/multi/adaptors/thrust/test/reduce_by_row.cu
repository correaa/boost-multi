// Copyright 2021-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <multi/array.hpp>
#include <multi/restriction.hpp>
#include <multi/io.hpp>
#include <multi/adaptors/thrust.hpp>

#include <thrust/device_allocator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/execution_policy.h>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

template<class In2D, typename OutputIt>
auto reduce_by_row(In2D const& in, OutputIt d_first) -> OutputIt {
	using index = typename In2D::index;
	auto const& row_id = multi::restricted(
		[] (auto i, auto /*j*/) constexpr { return i; },
		in.extensions()
	);

	auto [discard, ret] = ::thrust::reduce_by_key(
		thrust::device,
		row_id.elements().begin(), row_id.elements().end(),
		in.elements().begin(),
		::thrust::make_discard_iterator(),
		d_first  //, ::thrust::equal_to<multi::index>{}  //, thrust::plus<>{}
	);

	return ret;
};

int main() {
	multi::size_t N = 20;
	multi::size_t M = 10;

	using index = multi::index;
	{
		multi::array<double, 2, thrust::device_allocator<double>> A = 
			[] (auto i, auto j) constexpr { return i + j; }
			^ multi::extensions_t<2>({N, M})
		;

		multi::array<double, 1, thrust::device_allocator<double>> sums(multi::extensions_t<1>{N});

		auto end = reduce_by_row(A, sums.begin());
		BOOST_TEST( end == sums.end() );

		multi::array<double, 1> sums_host = sums;

		std::cout << sums_host << '\n';
	}
	{
		auto const& hankel = 
			[] (auto i, auto j) constexpr { return i + j; }
			^ multi::extensions_t<2>({N, M})
		;

		multi::array<double, 1, thrust::device_allocator<double>> sums(multi::extensions_t<1>{N});

		auto end = reduce_by_row(hankel, sums.begin());
		BOOST_TEST( end == sums.end() );

		multi::array<double, 1> sums_host = sums;

		BOOST_TEST( thrust::equal(
			sums_host.begin(), sums_host.end(),
			([ATs = (~hankel).size()] (multi::index i) { return ATs*i + ATs*(ATs - 1)/2; }
			^ sums.extensions()).begin()
		));

		std::cout << sums << '\n';
	}
}
