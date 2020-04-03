#ifdef COMPILATION_INSTRUCTIONS//-*-indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4;-*-
$CXX -Wall -Wextra -Wpedantic -Wfatal-errors $0 -o $0x -lcudart -lboost_timer -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2020
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuda adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>
namespace utf = boost::unit_test;
#include <boost/timer/timer.hpp>

#include "../../../adaptors/cuda.hpp"
#include "../../../complex.hpp"

#include<complex>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(cudart_double, *utf::tolerance(0.00001)*utf::timeout(2)){

	auto const in = []{
		multi::array<double, 4> r({32, 90, 98, 96});
		std::generate(data_elements(r), data_elements(r)+num_elements(r), &std::rand);
		return r;
	}();
	std::cout<<"memory size "<< in.num_elements()*sizeof(decltype(in)::element)/1e6 <<" MB\n";

	{
		boost::timer::auto_cpu_timer t{"%ws wall, CPU (%p%)\n"};
		multi::cuda::array<double, 4> const in_gpu = in;
	}
	{
		boost::timer::auto_cpu_timer t{"%ws wall, CPU (%p%)\n"};
		multi::cuda::array<double, 4> const in_gpu = in;
	}


}

BOOST_AUTO_TEST_CASE(cudart_complex, *utf::tolerance(0.00001)*utf::timeout(2)){

	using complex = std::complex<double>;

	auto const in = []{
		multi::array<complex, 4> r({32, 90, 98, 96});
		std::generate(data_elements(r), data_elements(r)+num_elements(r), &std::rand);
		return r;
	}();
	std::cout<<"memory size "<< in.num_elements()*sizeof(decltype(in)::element)/1e6 <<" MB\n";

	{
		boost::timer::auto_cpu_timer t{"%ws wall, CPU (%p%)\n"};
		multi::cuda::array<complex, 4> const in_gpu = in;
	}
	{
		boost::timer::auto_cpu_timer t{"%ws wall, CPU (%p%)\n"};
		multi::cuda::array<complex, 4> const in_gpu = in;
	}

}
