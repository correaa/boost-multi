#ifdef COMPILATION_INSTRUCTIONS
nvcc -x cu --expt-relaxed-constexpr -O3 -ccbin=cuda-c++ -std=c++14 `pkg-config cudart --cflags --libs` $0 -o $0x -D_DISABLE_CUDA_SLOW &&$0x&&rm $0x; exit
#endif

#include "../../cuda/allocator.hpp"
#include "../../../../../multi/array.hpp"

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

int main(){
	multi::array<double, 2> A2({32, 64}, double{}); 
	A2[2][4] = 8.;
	multi::array<double, 2, cuda::allocator<double>> A2_gpu = A2;
	assert( A2_gpu[2][4] == 8. );
}

