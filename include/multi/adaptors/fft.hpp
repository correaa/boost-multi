// Copyright 2020-2024 Alfredo A. Correa

#ifndef MULTI_ADAPTORS_FFT_HPP
#define MULTI_ADAPTORS_FFT_HPP

#include "../adaptors/fftw.hpp"

#if defined(__CUDA__) || defined(__NVCC__)
#include "../adaptors/cufft.hpp"
#elif defined(__HIPCC__)
#include "../adaptors/hipfft.hpp"
#endif

namespace boost {
namespace multi {
namespace fft {

	static constexpr int forward = fftw::forward;  // FFTW_FORWARD;
	static constexpr int none = 0;
	static constexpr int backward = fftw::backward;  // FFTW_BACKWARD;

	static_assert( forward != none and none != backward and backward != forward, "!");

	template<std::size_t I> struct priority : std::conditional_t<I==0, std::true_type, struct priority<I-1>>{};

	template<class... Args> auto dft_aux_(priority<0>, Args&&... args) DECLRETURN(  fftw::dft_backward(std::forward<Args>(args)...))
	template<class... Args> auto dft_aux_(priority<1>, Args&&... args) DECLRETURN(cufft ::dft_backward(std::forward<Args>(args)...))
	template<class... Args> auto dft(Args&&... args) DECLRETURN(dft_backward_aux_(priority<1>{}, std::forward<Args>(args)...))
	template<class In, class... Args> auto dft(std::array<bool, std::decay_t<In>::dimensionality> which, In const& in, Args&&... args) -> decltype(auto) {return dft_aux_(priority<1>{}, which, in, std::forward<Args>(args)...);}

	template<class... Args> auto dft_forward_aux_(priority<0>, Args&&... args) DECLRETURN(  fftw::dft_forward(std::forward<Args>(args)...))
	template<class... Args> auto dft_forward_aux_(priority<1>, Args&&... args) DECLRETURN(cufft ::dft_forward(std::forward<Args>(args)...))
	template<class In, class... Args> auto dft_forward(std::array<bool, In::dimensionality> which, In const& in, Args&&... args) -> decltype(auto) {return dft_forward_aux_(priority<1>{}, which, in, std::forward<Args>(args)...);}

	template<class... Args> auto dft_backward_aux_(priority<0>, Args&&... args) DECLRETURN(  fftw::dft_backward(std::forward<Args>(args)...))
	template<class... Args> auto dft_backward_aux_(priority<1>, Args&&... args) DECLRETURN(cufft ::dft_backward(std::forward<Args>(args)...))
	template<class In, class... Args> auto dft_backward(std::array<bool, In::dimensionality> which, In const& in, Args&&... args) -> decltype(auto) {return dft_backward_aux_(priority<1>{}, which, in, std::forward<Args>(args)...);}

}}}

#endif
