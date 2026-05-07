// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

#include <thrust/complex.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/random.h>

namespace multi = boost::multi;

auto main() -> int {
	namespace multi = boost::multi;

	multi::thrust::universal_array<thrust::complex<double>, 3> olap({5, 5, 3}, thrust::complex<double>{1.0, 2.0});

	// ::thrust::tabulate(olap.elements().begin(), olap.elements().end(), [seed = 42] __device__ (multi::index i) {
	// 	thrust::default_random_engine rng(seed);
	// 	rng.discard(i);
	// 	thrust::uniform_real_distribution<double> dist(-1.0, 1.0);
	// 	return thrust::complex<double>(dist(rng), dist(rng));
	// });

	multi::thrust::universal_array<double, 2> vel_gold;

	{
		multi::array<double, 2> vel({5, 5});

		std::transform(
			olap.flatted().begin(), olap.flatted().end(),
			vel.flatted().begin(),
			[](auto const& e) { return norm(e[0]) + norm(e[1]) + norm(e[2]); }
		);

		vel_gold = vel;

		BOOST_TEST( std::abs(vel[2][3] - vel_gold[2][3]) < 1e-12  );
	}
	{
		multi::thrust::universal_array<double, 2> vel({5, 5});

		// thrust::transform cannot iterate over multi subarrays (proxy types), so
		// we use counting_iterator and index into olap.base() directly.
		auto const* olap_base = thrust::raw_pointer_cast(olap.base());
		using std::get;
		auto const  inner     = get<2>(olap.sizes());  // == 3

		BOOST_TEST( inner == 3 );

		thrust::transform(
			thrust::make_counting_iterator(0),
			thrust::make_counting_iterator(static_cast<int>(vel.num_elements())),
			vel.elements().begin(),
			[olap_base, inner] __device__ (int mm) {
				double result = 0.0;
				for(auto kk = 0; kk < inner; ++kk) {
					result += norm(olap_base[mm * inner + kk]);
				}
				return result;
			}
		);

		BOOST_TEST( std::abs(vel[2][3] - vel_gold[2][3]) < 1e-12  );
	}
	// {
	// 	multi::thrust::universal_array<double, 2> vel({5, 5});

	// 	auto const* olap_base = thrust::raw_pointer_cast(olap.base());
	// 	auto const  inner     = olap.size(2);

	// 	thrust::transform(
	// 		thrust::cuda::par,
	// 		thrust::make_counting_iterator(0),
	// 		thrust::make_counting_iterator(static_cast<int>(vel.num_elements())),
	// 		vel.elements().begin(),
	// 		[olap_base, inner] __device__ (int mm) {
	// 			double result = 0.0;
	// 			for(auto kk = 0; kk < inner; ++kk) {
	// 				result += norm(olap_base[mm * inner + kk]);
	// 			}
	// 			return result;
	// 		}
	// 	);

	// 	BOOST_TEST( std::abs(vel[2][3] - vel_gold[2][3]) < 1e-12  );
	// }

	return boost::report_errors();
}
