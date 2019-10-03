#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& `#nvcc -ccbin=cuda-`c++ -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_MALLOC `pkg-config cudart --cflags --libs` $0.cpp -o $0x &&$0x&&rm $0x; exit
#endif

#ifndef MULTI_MEMORY_ADAPTORS_CUDA_MALLOC
#define MULTI_MEMORY_ADAPTORS_CUDA_MALLOC

#include "../../adaptors/cuda/clib.hpp"
#include "../../adaptors/cuda/ptr.hpp"

namespace boost{namespace multi{
namespace memory{namespace cuda{
	using size_t = Cuda::size_t;
	ptr<void> malloc(size_t bytes){return ptr<void>{Cuda::malloc(bytes)};}
	void free(ptr<void> p){Cuda::free(static_cast<void*>(p));}
}}
}}

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_MALLOC

int main(){
}

#endif
#endif
