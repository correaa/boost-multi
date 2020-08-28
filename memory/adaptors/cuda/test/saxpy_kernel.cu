#ifdef COMPILATION
/usr/local/cuda/bin/nvcc -g -pg -O3 --expt-relaxed-constexpr --extended-lambda $0 -o $0x&&time $0x&&rm $0x;exit
#/usr/local/cuda/bin/nvcc -g -pg -O3 --expt-relaxed-constexpr --extended-lambda --expt-extended-lambda --Werror=cross-execution-space-call --compiler-options -Ofast,-g,-pg,-Wall,-Wfatal-errors -g $0 - $0x &&$0x&&rm $0x; exit
#endif

#include "../../cuda/managed/allocator.hpp"
#include "../../../../array.hpp"

#include <cuda.h>

#include <stdio.h>

template<class Kernel>
__global__ void cuda_run_kernel_1(unsigned n, Kernel k){
	auto i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < n) k(i);
}

template<class Kernel>
void run(std::size_t n, Kernel k){
	constexpr int CUDA_BLOCK_SIZE = 256;
	
	unsigned nblock = (n + CUDA_BLOCK_SIZE - 1)/CUDA_BLOCK_SIZE;
	
	cuda_run_kernel_1<<<nblock, CUDA_BLOCK_SIZE>>>(n, k);
	
	auto code = cudaGetLastError();
	if(code != cudaSuccess){
		fprintf(stderr,"GPU assert: %s %s %d\n", cudaGetErrorString(code), __FILE__, __LINE__);
		exit(-1);
	}
	
	cudaDeviceSynchronize();
}

namespace multi = boost::multi;

#include "../../../../adaptors/cuda.hpp"
#include "../../../../array.hpp"

int main(){

	int N = 1<<29; // 1<<20 == 1048576
	std::cout<< N <<std::endl;

	multi::array<float, 1> x(N, 1.f);
	multi::array<float, 1> y(N, 2.f);

	multi::cuda::managed::array<float, 1> Dx = x;
	multi::cuda::managed::array<float, 1> Dy = y;

	assert( Dx.size() == Dy.size() );

	float a = 2.f;

//	for(int i = 0; i < Dx.size(); ++i)
//		Dy[i] = a*Dx[i] + Dy[i];

	run(
		Dx.size(), 
		[a, DyP = &Dy(), DxP = &Dx()] __device__ (int i){
			(*DyP)[i] = a*(*DxP)[i] + (*DyP)[i];
		}
	);

	y = Dy;

	float maxError = 0.0f;
	for (int i = 0; i < N; i++)
		maxError = max(maxError, abs(y[i]-4.0f));

	std::cout<<"Max error:"<< maxError << std::endl;
	assert( maxError < 1e-10 );

}

