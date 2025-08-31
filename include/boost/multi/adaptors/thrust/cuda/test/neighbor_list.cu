#include <multi/adaptors/thrust.hpp>
#include <multi/array.hpp>
#include <thrust/device_allocator.h>
#include <thrust/execution_policy.h>  // Include for execution policies
#include <thrust/transform_reduce.h>  // for thrust::transform_reduce

#include <execution>  // for std::execution::par
#include <iostream>
#include <numeric>  // for std::transform_reduce

namespace multi = boost::multi;

struct v2d {
	double x;
	double y;

	friend __host__ __device__ auto x(v2d const& self) { return self.x; }
	friend __host__ __device__ auto y(v2d const& self) { return self.y; }
};

__host__ __device__ auto v(double dist) -> double {
	return dist * dist;
}

// declare universal array type
// pros: can be used in CPU, and GPU
// cons: members cannot be used directly on array elements (ref wrapped), careless use might cause a lot of page faults (CPU<->GPU page swapping)
template<class T, multi::dimensionality_type D>
using array = multi::thrust::universal_array<T, D>;

// pros: familiarity
// cons: bug prone, lots of state and variables, for init is ugly
auto energy_raw_loops(auto const& positions, auto const& neighbors) {
	double ret = 0;
	for(multi::index i = 0; i != positions.size(); ++i) {
		auto const& neighbors_i = neighbors[i];
		for(multi::index j = 0; j != neighbors_i.size(); ++j) {
			if(neighbors_i[j] == -1) {
				continue;
			}  // or add zero
			auto dist = std::abs(x(positions[i]) - x(positions[neighbors_i[j]]));
			ret += v(dist);
		}
	}
	return ret;
}

// pros: familiar, more compact
// cons: bug prone
auto energy_range_loops(auto const& positions, auto const& neighbors) {
	double ret = 0;
	for(auto const i : positions.extension()) {
		auto const positions_i = positions[i];
		for(auto const& nbidx : neighbors[i]) {
			if(nbidx == -1) {
				continue;
			}
			auto dist = std::abs(x(positions_i) - x(positions[nbidx]));
			ret += v(dist);
		}
	}
	return ret;
}

// pros: familiar, less state
// const: too nested
auto energy_reduce_in_loop(auto const& positions, auto const& neighbors) {
	double ret = 0.0;
	for(auto const i : positions.extension()) {
		ret += std::transform_reduce(neighbors[i].begin(), neighbors[i].end(), 0.0, std::plus<>{}, [positions_i = positions[i], &positions](auto nbidx) { return nbidx == -1 ? 0.0 : v(std::abs(x(positions_i) - x(positions[nbidx]))); });
	}
	return ret;
}

// pros: single ouput, no state, ready for some parallelism, ready for CUDA thrust
// cons: unfamiliar, too many nested structures, return type buried
auto energy_nested_reduce(auto const& positions, auto const& neighbors) {
	return std::transform_reduce(
		positions.extension().begin(), positions.extension().end(),
		0.0, std::plus<>{},
		[&positions, &neighbors](auto i) {
			return std::transform_reduce(
				neighbors[i].begin(), neighbors[i].end(),
				0.0, std::plus<>{},
				[positions_i = positions[i], &positions](auto nbidx) {
					return nbidx == -1 ? 0.0 : v(std::abs(x(positions_i) - x(positions[nbidx])));
				}
			);
		}
	);
}

// pros: single ouput, no state, parallel
// cons: unfamiliar, too many nested structures, parallelization is partial and nested
auto energy_nested_par_reduce(auto const& positions, auto const& neighbors) {
	return std::transform_reduce(
		std::execution::par,
		positions.extension().begin(), positions.extension().end(),
		0.0, std::plus<>{},
		[&positions, &neighbors](auto i) {
			return std::transform_reduce(
				std::execution::unseq,
				neighbors[i].begin(), neighbors[i].end(),
				0.0, std::plus<>{},
				[positions_i = positions[i], &positions](auto nbidx) {
					return nbidx == -1 ? 0.0 : v(std::abs(x(positions_i) - x(positions[nbidx])));
				}
			);
		}
	);
}

// pros: correct parallelization, no state
// cons: unfamiliar, needs coordinate decomposition, use special Multi features
auto energy_flatten_par_reduce(auto const& positions, auto const& neighbors) {
	return std::transform_reduce(
		std::execution::par,
		neighbors.extensions().elements().begin(), neighbors.extensions().elements().end(),
		0.0,
		std::plus<>{},
		[&positions, &neighbors](multi::array<int, 2>::indexes c) {
			auto [i, j] = c;
			return neighbors[i][j] == -1 ? 0.0
										 : v(std::abs(x(positions[i]) - x(positions[neighbors[i][j]])));
		}
	);
}

// pros: GPU optimized, runs completely on GPU
// const: verbose, enclosing function cannot deduce return, or auto parameters, lamba captures need special care, order of arguments is different from STL, may need a complete diffeent algorithm to extract different information: e.g. thrust::reduce_by_key
template<class Arr1D, class Arr2D>
double energy_flatten_gpu_reduce(Arr1D const& positions, Arr2D const& neighbors) {
	return thrust::transform_reduce(
		thrust::cuda::par,
		neighbors.extensions().elements().begin(), neighbors.extensions().elements().end(),
		[positions = positions.begin(), neighbors = neighbors.begin()] __device__(array<int, 2>::indexes c) -> double {
			auto [i, j] = c;
			return neighbors[i][j] == -1 ? 0.0
										 : v(std::abs(x(positions[i]) - x(positions[neighbors[i][j]])));
		},
		0.0,
		std::plus<>{}
	);
}

int main() {
	array<v2d, 1> positions = {
		{1.0, 0.0},
		{2.0, 0.0},
		{3.0, 0.0},
		{4.0, 0.0},
		{5.0, 0.0}
	};

	array<int, 2> neighbors = {
		{1, 2, -1, -1}, /* of at 0*/
		{0, 2,  3, -1}, /* of at 1*/
		{0, 1,  3,  4}, /* of at 2*/
		{1, 2,  4, -1}, /* of at 3*/
		{2, 3, -1, -1}  /* of at 4*/
	};

	{
		auto en = energy_raw_loops(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
	{
		auto en = energy_range_loops(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
	{
		auto en = energy_reduce_in_loop(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
	{
		auto en = energy_nested_reduce(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
	{
		auto en = energy_nested_par_reduce(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
	{
		auto en = energy_flatten_par_reduce(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
	{
		auto en = energy_flatten_gpu_reduce(positions, neighbors);
		BOOST_TEST(en == 32.0 );
	}
}
