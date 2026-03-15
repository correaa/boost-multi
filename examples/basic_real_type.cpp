#ifdef COMPILATION  // -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 - std = c++ 17 - o $0x - lboost_timer `pkg - config-- libs tbb` && $0x && rm $0x;
exit
#endif

// Copyright 2026 Amlal El Mahrouss

#include <boost/multi/array.hpp>
#include <boost/multi/detail/real_type.hpp>

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/timer/timer.hpp>

#include <cmath>
#include <complex>
#include <iostream>

namespace multi = boost::multi;
using boost::multiprecision::cpp_bin_float_double;

int main() {
	boost::timer::auto_cpu_timer t;
	{
		boost::multi::double_type left{pow(M_E, M_PI)};
		boost::multi::double_type right{1.0};

		cpp_bin_float_double f = left.get() + right.get();
		std::cout.precision(250);
		std::cout << f << std::endl;
	}
	{
		boost::multi::double_type left{pow(M_E, M_PI)};
		boost::multi::double_type right{1.0};

		cpp_bin_float_double f = left.get() + right.get();
		std::cout.precision(500);
		std::cout << f << std::endl;
	}
	return 0;
}
