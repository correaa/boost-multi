#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
$CXX $0 -o $0x -lboost_unit_test_framework -lcudart -lboost_timer&&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA adaptor real to complex"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>
#include <boost/timer/timer.hpp>

#include "../../../../array.hpp"
#include "../../../../adaptors/cuda.hpp"

namespace multi = boost::multi;
namespace cuda = multi::cuda;

BOOST_AUTO_TEST_CASE(cuda_adaptor_real_to_complex_copy){

	cuda::array<double              , 1> R(1<<26, 1.);
	cuda::array<std::complex<double>, 1> C(1<<26, 0.);

	std::cout << "elements (M) " << size(R)/1e6 << " total memory " << (size(R)*sizeof(double) + size(C)*sizeof(std::complex<double>))/1e6 << "MB" << std::endl;
	
CUDA_SLOW(
	R[13] = 3.;
)

	{
		boost::timer::auto_cpu_timer t;
		C = R;
		BOOST_REQUIRE( static_cast<double>(R[13]) == 3. );
		BOOST_REQUIRE( static_cast<std::complex<double>>(C[13]) == 3. );
	}
	{
		boost::timer::auto_cpu_timer t;
		C() = R();
		BOOST_REQUIRE( static_cast<double>(R[13]) == 3. );
		BOOST_REQUIRE( static_cast<std::complex<double>>(C[13]) == 3. );
	}


	
}


