// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <boost/core/lightweight_test.hpp>

namespace multi = boost::multi;

auto main() -> int {
	namespace multi = boost::multi;

	multi::thrust::universal_array<thrust::complex<double>, 3> const olap({5, 5, 3}, thrust::complex<double>{1.0, 3.0});

	multi::thrust::universal_array<double, 2> vel_gold;

	{
		multi::array<double, 2> vel({5, 5});

		// using std::norm; using std::transform
		std::transform(
			olap.flatted().begin(), olap.flatted().end(),
			vel.flatted().begin(),
			[](auto const& e) { return norm(e[0]) + norm(e[1]) + norm(e[2]); }
		);

		vel_gold = vel;
	}
	{
		multi::array<double, 2> vel({5, 5});

		thrust::transform(
			olap.flatted().begin(), olap.flatted().end(),
			vel.flatted().begin(),
			[](auto const& e) { return norm(e[0]) + norm(e[1]) + norm(e[2]); }
		);

		BOOST_TEST( std::abs(vel[2][3] - vel_gold[2][3]) < 1e-12  );
	}
	{
		multi::array<double, 2> vel({5, 5});

		transform(
			olap.flatted().begin(), olap.flatted().end(),
			vel.flatted().begin(),
			[](auto const& e) { return norm(e[0]) + norm(e[1]) + norm(e[2]); }
		);

		BOOST_TEST( std::abs(vel[2][3] - vel_gold[2][3]) < 1e-12  );
	}

	return boost::report_errors();
}
