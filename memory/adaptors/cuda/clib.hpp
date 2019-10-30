#ifdef COMPILATION_INSTRUCTIONS
(echo '#include "'$0'"'>$0.cpp)&&c++ -std=c++11 -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_MEMORY_ADAPTOR_CUDA_DETAIL_MALLOC $0.cpp -lcudart -o $0x &&$0x&& rm $0x $0.cpp; exit
#endif

#ifndef MULTI_MEMORY_ADAPTOR_CUDA_LIB_HPP
#define MULTI_MEMORY_ADAPTOR_CUDA_LIB_HPP

#include<cuda_runtime.h> // cudaMalloc

#include "../../adaptors/cuda/error.hpp"

namespace Cuda{
	using size_t = ::size_t;
	error Malloc(void** p, size_t bytes){return static_cast<error>(cudaMalloc(p, bytes));}
	void* malloc(size_t bytes){
		void* ret;
		switch(auto e = Malloc(&ret, bytes)){
			case success           : return ret;
			case memory_allocation : return nullptr;
			default                : 
				throw std::system_error{e, "cannot allocate "+std::to_string(bytes)+" bytes in '"+__PRETTY_FUNCTION__+"'"};
		}
	}
	error Free(void* p){return static_cast<error>(cudaFree(p));}
	void free(void* p){
		auto e = Free(p);
		if(e != Cuda::success) throw std::system_error{e, std::string{"cannot free "}+__PRETTY_FUNCTION__};
	}
	namespace pointer{
		using attributes_t = cudaPointerAttributes;
		error GetAttributes(attributes_t* ret, void* p){return static_cast<error>(cudaPointerGetAttributes(ret, p));}
		attributes_t attributes(void* p){
			attributes_t ret;
			auto e = GetAttributes(&ret, p); 
			if(e!=success) throw std::system_error{e, std::string{"cannot "}+ __PRETTY_FUNCTION__};
			return ret;
		}
		bool is_device(void* p){return attributes(p).devicePointer or p==nullptr;}
	}
	namespace Managed{
		error Malloc(void** p, size_t bytes){return static_cast<error>(cudaMallocManaged(p, bytes/*cudaMemAttachGlobal*/));}
		void* malloc(size_t bytes){
			void* ret;
			switch(auto e = Malloc(&ret, bytes)){
				case success           : return ret;
				case memory_allocation : return nullptr;
				default                : 
					throw std::system_error{e, "cannot allocate "+std::to_string(bytes)+" bytes in '"+__PRETTY_FUNCTION__+"'"};
			}
		}
	}
}


#ifdef _TEST_MULTI_MEMORY_ADAPTOR_CUDA_DETAIL_MALLOC

#include "../cuda/ptr.hpp"

#include<iostream>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

using std::cout;

int main(){
	void* p = Cuda::malloc(100);
	Cuda::free(p);
}
#endif
#endif

