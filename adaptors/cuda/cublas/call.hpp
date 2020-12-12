#ifndef MULTI_ADAPTORS_CUDA_CUBLAS_CALL_HPP
#define MULTI_ADAPTORS_CUDA_CUBLAS_CALL_HPP

#include "../cublas/error.hpp"

namespace boost{
namespace multi::cuda::cublas{

template<auto F, class... Args> // needs C++17
void call(Args... args){
	auto e = static_cast<enum cublas::error>(F(args...));
	if(e != cublas::error::success) throw std::system_error{e, "cannot call function " + std::string{__PRETTY_FUNCTION__}};
	cudaDeviceSynchronize();
}

#define CUBLAS_(F) call<F>

}
}
#endif
