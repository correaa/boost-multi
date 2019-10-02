#ifdef COMPILATION_INSTRUCTIONS
(echo '#include "'$0'"'>$0.cpp)&& c++ -std=c++14 -Wall -Wextra -Wpedantic -Wfatal-errors -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_DETAIL_ERROR $0.cpp -o $0x -lcudart && $0x && rm $0x $0.cpp; exit
#endif

#ifndef MULTI_MEMORY_ADAPTOR_CUDA_DETAIL_ERROR_HPP
#define MULTI_MEMORY_ADAPTOR_CUDA_DETAIL_ERROR_HPP

#include<cuda/driver_types.h> // cudaError_t
#include<cuda/cuda_runtime_api.h> // cudaGetErrorString

#include<system_error>
#include<type_traits> // underlying_type

namespace Cuda{

enum /*class*/ error : typename std::underlying_type<cudaError_t>::type{
	success = cudaSuccess, // = 0 The API call returned with no errors. In the case of query calls, this also means that the operation being queried is complete (see cudaEventQuery() and cudaStreamQuery()). 
	invalid_value /*invalid_argument*/ = cudaErrorInvalidValue, // = 1, This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values. 
	memory_allocation = cudaErrorMemoryAllocation // = 2 // The API call failed because it was unable to allocate enough memory to perform the requested operation. 
};

inline std::string string(enum error e){return {cudaGetErrorString(static_cast<cudaError_t>(e))};}

struct error_category : std::error_category{
	char const* name() const noexcept override{return "cuda wrapper";}
	std::string message(int e) const override{return string(static_cast</*enum*/ error>(e));}
	static error_category& instance(){
		static error_category instance;
		return instance;
	}
};

inline std::error_code make_error_code(error err) noexcept{
	return std::error_code(int(err), error_category::instance());
}

}

namespace std{
	template<> struct is_error_code_enum<Cuda::error> : true_type{};
}

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_DETAIL_ERROR

#include<cassert>
#include<iostream>

using std::cout;

int main(){

	std::error_code ec = Cuda::error::memory_allocation;
	assert( ec == Cuda::error::memory_allocation);

	try{
		auto e = Cuda::error::memory_allocation; // return from a cudaFunction
		throw std::system_error{e, "cannot do something"};
	}catch(std::system_error const& e){
		cout
			<<"code: "   << e.code()           <<'\n'
			<<"message: "<< e.code().message() <<'\n'
			<<"what: "   << e.what()           <<'\n'
		;
	}

}
#endif
#endif

