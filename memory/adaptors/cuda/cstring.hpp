#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ -std=c++14 -Wfatal-errors -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_CSTRING -D_DISABLE_CUDA_SLOW `pkg-config cudart --cflags --libs` $0.cpp -o$0x -lboost_timer&&$0x&&rm $0x $0.cpp; exit
#endif
// © 2019 Alfredo A. Correa 
#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_CSTRING_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_CSTRING_HPP

#include<cuda_runtime.h> // cudaMemcpy/cudaMemset
#include "../../adaptors/cuda/ptr.hpp"

#include<iostream>

namespace boost{
namespace multi{
namespace memory{
namespace cuda{

namespace memcpy_{
//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b
	enum class kind{ // 
		host_to_host=cudaMemcpyHostToHost, host_to_device=cudaMemcpyHostToDevice,
		device_to_host=cudaMemcpyDeviceToHost, device_to_device=cudaMemcpyDeviceToDevice
	};
	kind type(void*    , void const*    ){return kind::host_to_host    ;}
	kind type(ptr<void>, void const*    ){return kind::host_to_device  ;}
	kind type(void*    , ptr<void const>){return kind::device_to_host  ;}
	kind type(ptr<void>, ptr<void const>){return kind::device_to_device;}
}

template<typename Dest, typename Src, typename = decltype(memcpy_::type(Dest{}, Src{}))>
Dest memcpy(Dest dest, Src src, std::size_t byte_count){
//	assert( byte_count > 1000 );
	cudaError_t const s = cudaMemcpy(
		static_cast<void*>(dest), static_cast<void const*>(src), 
		byte_count, static_cast<cudaMemcpyKind>(memcpy_::type(dest, src))
	); assert(s == cudaSuccess);
	return dest;
}

ptr<void> memset(ptr<void> dest, int ch, std::size_t byte_count){
	[[maybe_unused]] cudaError_t s = cudaMemset(static_cast<void*>(dest), ch, byte_count); assert(s == cudaSuccess);
//	if(s == cudaErrorInvalidDevicePointer) throw std::runtime_error{"cudaErrorInvalidDevicePointer"};
//	if(s == cudaErrorInvalidValue)         throw std::runtime_error{"cudaErrorInvalidValue"};//, probably could not allocate fuzzy memory"};
	return dest;
}

}}}}

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_CSTRING

#include<boost/timer/timer.hpp>
#include<numeric>
#include "../cuda/allocator.hpp"

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

int main(){
	std::size_t const n = 2e9/sizeof(double);
	cuda::ptr<double> p = cuda::allocator<double>{}.allocate(n);
	{
		boost::timer::auto_cpu_timer t;
		memset(p, 0, n*sizeof(double));
	}
	assert( p[n/2]==0 );
	p[n/2] = 99.;
	cuda::ptr<double> q = cuda::allocator<double>{}.allocate(n);
	{
		boost::timer::auto_cpu_timer t;
		memcpy(q, p, n*sizeof(double));
	}
	assert( p[n/2] == 99. );
	assert( q[n/2] == 99. );
}
#endif
#endif


