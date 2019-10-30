#ifdef COMPILATION_INSTRUCTIONS
nvcc -x cu --expt-relaxed-constexpr -O3 -ccbin=c++ -std=c++14 $0 -o $0x -DBOOST_TEST_DYN_LINK -lboost_unit_test_framework -D_DISABLE_CUDA_SLOW -lcudart &&$0x&&rm $0x; exit
#endif

#include "../../cuda/allocator.hpp"
#include "../../../../array.hpp"

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA allocators"
#include<boost/test/unit_test.hpp>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

BOOST_AUTO_TEST_CASE(cuda_allocators){
	multi::array<double, 2> A2({32, 64}, double{}); A2[2][4] = 8.;
	multi::array<double, 2, cuda::allocator<double>> A2_gpu = A2;
	multi::array<double, 2, cuda::managed::allocator<double>> A2_mgpu = A2;
	BOOST_REQUIRE( A2[2][4] == 8. );
	BOOST_REQUIRE( A2_gpu[2][4] == 8. );
	BOOST_REQUIRE( A2_mgpu[2][4] == 8. );
}

