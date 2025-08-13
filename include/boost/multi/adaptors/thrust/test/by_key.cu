// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <thrust/iterator/discard_iterator.h>

namespace multi = boost::multi;

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)

	multi::thrust::universal_array<float, 2> M = {
		{15.0,  6.0, 18.0, 15.0},
		{29.0, 26.0, 24.0, 10.0},
		{ 1.0,  4.0, 12.0,  8.0},
		{ 8.0, 15.0,  1.0, 14.0},
		{26.0,  5.0, 12.0, 25.0},
		{13.0, 14.0, 23.0, 25.0},
		{20.0, 23.0, 19.0, 18.0},
		{11.0, 14.0,  3.0, 12.0}
	};

	BOOST_TEST( M.size() == 8 );
	BOOST_TEST( (~M).size() == 4 );


	using std::get;
	std::cout
        << get<0>(M.extensions()[1][1]) << ' '
        << get<1>(M.extensions()[1][1]) << '\n'
    ;

	BOOST_TEST(true);
	// M.extensions().elements();

	auto row_ids_begin =
	    thrust::make_transform_iterator(
			M.extensions().elements().begin(),
	        [] __host__ __device__ (decltype(M)::indexes const& e) { using std::get; return get<0>(e); }
	    )
	;
	auto row_ids_end =
		thrust::make_transform_iterator(
			M.extensions().elements().end(),
			[] __host__ __device__ (decltype(M)::indexes const& e) { using std::get; return get<0>(e); }
		)
	;

	multi::thrust::universal_array<float, 1> sums(M.size());

	thrust::reduce_by_key(thrust::cuda::par,
		row_ids_begin, row_ids_end,
		M.elements().begin(),
		thrust::make_discard_iterator(),
		sums.begin()
	);

	// thrust::reduce_by_key(
	//     thrust::make_counting_iterator(0),
	//     thrust::make_counting_iterator(M.size()),
	//     M.data(),
	//     thrust::make_discard_iterator(),
	//     sums.data()
	// );

	return boost::report_errors();
}
