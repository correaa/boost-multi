#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& `#nvcc -ccbin=cuda-`c++ -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_MALLOC $0.cpp -o $0x -lcudart &&$0x&&rm $0x; exit
#endif

#ifndef MULTI_MEMORY_ADAPTORS_CUDA_MALLOC
#define MULTI_MEMORY_ADAPTORS_CUDA_MALLOC

#include "../../../adaptors/cuda/managed/clib.hpp"
#include "../../../adaptors/cuda/managed/ptr.hpp"

namespace boost{namespace multi{
namespace memory{

namespace cuda{

namespace managed{
	[[nodiscard]]
	managed::ptr<void> malloc(size_t bytes){return managed::ptr<void>{Cuda::Managed::malloc(bytes)};}
	void free(managed::ptr<void> p){Cuda::Managed::free(static_cast<void*>(p));}
}

}

}
}}

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_MALLOC

int main(){
}

#endif
#endif
