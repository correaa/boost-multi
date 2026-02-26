// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>
#include <boost/multi/restriction.hpp>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

struct fun_ce {
	int            m_base;
	constexpr auto operator()(multi::index i) const -> multi::index { return i + m_base; }
};

struct fun1_hd {
	__host__ __device__ auto operator()(multi::index i) const -> multi::index { return i + 1; }
};

struct fun_hd {
	int                      m_base;
	__host__ __device__ auto operator()(multi::index i) const -> multi::index { return i + m_base; }
};

int main() {
	// ok, all happens in the host
	{
		multi::thrust::host_array<int, 1> arr = {1, 2, 3, 4};

		auto iota = [](multi::index i) { return i + 1; } ^ arr.extensions();

		BOOST_TEST(
			thrust::equal(
				arr.begin(), arr.end(),
				iota.begin()
			)
		);
	}
	// ok, all happens in the host, argument can be auto
	{
		multi::thrust::host_array<int, 1> arr = {1, 2, 3, 4};

		auto iota = [](auto i) { return i + 1; } ^ arr.extensions();

		BOOST_TEST(
			thrust::equal(
				arr.begin(), arr.end(),
				iota.begin()
			)
		);
	}

	thrust::host_vector<int> hv(4);
	hv[0] = 1;
	hv[1] = 2;
	hv[2] = 3;
	hv[3] = 4;
	{
		thrust::device_vector<int> arr = hv;

		auto first = thrust::make_counting_iterator(1);
		auto last  = thrust::make_counting_iterator(hv.size() + 1);

		BOOST_TEST( thrust::equal(
			thrust::device,
			arr.begin(), arr.end(),
			first
		));
	}
	{
		thrust::device_vector<int> arr = hv;

		auto iota1 = [] __host__ __device__ (multi::index i) { return i + 1; } ^ multi::extensions_t<1>(hv.size());

		auto first = iota1.begin();
		auto last  = iota1.end();

		BOOST_TEST( thrust::equal(
			thrust::device,
			arr.begin(), arr.end(),
			first
		) );
	}
	{
		thrust::device_vector<int> arr = hv;

		auto iota1 = [] __host__ __device__ (multi::index i) { return i + 1; } ^ multi::extensions_t<1>(hv.size());

		auto first = iota1.begin();
		auto last  = iota1.end();

		bool result = thrust::equal(
			arr.begin(), arr.end(),
			first
		);

		BOOST_TEST( result );
	}
	// {
	// 	thrust::device_vector<int> arr = hv;

	// 	auto iota1 = multi::thrust::device_function([] __host__ __device__ (multi::index i) { return i + 1; }) ^ multi::extensions_t<1>(hv.size());

	// 	auto first = iota1.begin();
	// 	auto last  = iota1.end();

	// 	bool result = thrust::equal(
	// 		arr.begin(), arr.end(),
	// 		first
	// 	);

	// 	BOOST_TEST( result );
	// }
	// {
	// 	thrust::device_vector<int> arr = hv;

	// 	auto iota1 = fun1_hd{} ^ multi::extensions_t<1>(hv.size());

	// 	auto first = iota1.begin();
	// 	auto last  = iota1.end();

	// 	BOOST_TEST( thrust::equal(
	// 		arr.begin(), arr.end(),
	// 		first
	// 	));
	// }
	// {
	// 	thrust::device_vector<int> arr = hv;

	// 	int base = 1;
	// 	auto iota1 = fun_hd{base} ^ multi::extensions_t<1>(hv.size());

	// 	auto first = iota1.begin();
	// 	auto last  = iota1.end();

	// 	bool result = thrust::equal(
	// 		arr.begin(), arr.end(),
	// 		first
	// 	);

	// 	BOOST_TEST( result );
	// }
	// {
	// 	thrust::device_vector<int> arr = hv;

	// 	int base = 1;
	// 	auto iota1 = multi::thrust::device_function([=] __host__ __device__ (multi::index i) { return i + base; }) ^ multi::extensions_t<1>(hv.size());

	// 	auto first = iota1.begin();
	// 	auto last  = iota1.end();

	// 	bool result = thrust::equal(
	// 		arr.begin(), arr.end(),
	// 		first
	// 	);

	// 	BOOST_TEST( result );
	// }
	// multi::thrust::host_array<int, 1> arr_host = {1, 2, 3, 4};
	// {
	// 	multi::thrust::device_array<int, 1> arr = arr_host;

	// 	auto first = thrust::make_counting_iterator(1);
	// 	auto last  = thrust::make_counting_iterator(hv.size() + 1);

	// 	bool result = thrust::equal(
	// 		thrust::device,
	// 		arr.begin(), arr.end(),
	// 		first
	// 	);

	// 	BOOST_TEST( result );
	// }
	// {
	// 	multi::thrust::device_array<int, 1> arr = arr_host;

	// 	auto first = thrust::make_counting_iterator(1);
	// 	auto last  = thrust::make_counting_iterator(hv.size() + 1);

	// 	bool result = thrust::equal(
	// 		arr.begin(), arr.end(),
	// 		first
	// 	);

	// 	BOOST_TEST( result );
	// }
	// {
	// 	multi::thrust::device_array<int, 1> arr = arr_host;

	// 	auto iota1 = [](multi::index i) { return i + 1; } ^ multi::extensions_t<1>(arr.size());

	// 	auto first = iota1.begin();
	// 	auto last  = iota1.end();

	// 	bool result = thrust::equal(
	// 		arr.begin(), arr.end(),
	// 		first
	// 	);

	// 	BOOST_TEST( result );
	// }
	// {  // compilation error
	// 	thrust::device_vector<int> arr  = hv;
	// 	auto                       iota = [](multi::index i) { return i + 1; } ^ multi::extensions_t<1>(hv.size());
	// 	BOOST_TEST(
	// 		thrust::equal(
	// 			arr.begin(), arr.end(),
	// 			iota.begin()
	// 		)
	// 	);
	// }
	// {  // compilation error
	// 	multi::thrust::device_array<int, 1> arr = {1, 2, 3, 4};
	// 	auto iota = [] (multi::index i) { return i + 1; } ^ arr.extensions();
	// 	BOOST_TEST(
	// 		thrust::equal(
	// 			arr.begin(), arr.end(),
	// 			iota.begin()
	// 		)
	// 	);
	// }
	// {
	// 	multi::thrust::device_array<int, 1> arr = {1, 2, 3, 4};
	// 	auto iota = [] __host__ __device__ (multi::index i) { return i + 1; } ^ arr.extensions();
	// 	BOOST_TEST(
	// 		thrust::equal(
	// 			arr.begin(), arr.end(),
	// 			iota.begin()
	// 		)
	// 	);
	// }
	// {
	// 	multi::thrust::device_array<int, 1> arr = {1, 2, 3, 4};
	// 	BOOST_TEST(
	// 		thrust::equal(
	// 			arr.elements().begin(), arr.elements().end(),
	// 			([] __host__ __device__ (multi::index i) { return i + 1; } ^ arr.extensions()).elements().begin()
	// 		)
	// 	);
	// }
	// {
	// 	multi::thrust::device_array<int, 1> arr = {1, 2, 3, 4};
	// 	BOOST_TEST(
	// 		thrust::equal(
	// 			arr.elements().begin(), arr.elements().end(),
	// 			([base = 0] __host__ __device__ (multi::index i) { return i; } ^ arr.extensions()).elements().begin()
	// 		)
	// 	);
	// }

	return boost::report_errors();
}
