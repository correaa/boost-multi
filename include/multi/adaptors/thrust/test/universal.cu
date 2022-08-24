#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA thrust universal"
#include<boost/test/unit_test.hpp>

#include <multi/array.hpp>

#include <multi/adaptors/thrust.hpp>

#include <thrust/system/cuda/memory.h>

#include <boost/mpl/list.hpp>

#include <memory_resource>
#include <numeric>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(thrust_universal_ptr) {

	multi::array<double, 2                                           > Host({1024, 1024});
	std::iota(Host.data_elements(), Host.data_elements() + Host.num_elements(), 12.);

	multi::array<double, 2, thrust::cuda::universal_allocator<double>> Univ({1024, 1024});

	Univ({0, 10}, {0, 20}) = Host({0, 10}, {0, 20});

	multi::array<double, 2                                           > Hos2({1024, 1024});
	Hos2({0, 10}, {0, 20}) = Univ({0, 10}, {0, 20});

	BOOST_REQUIRE( Hos2[0][0] == 12. );

}

