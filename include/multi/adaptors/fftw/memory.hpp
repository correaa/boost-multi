// Copyright 2020-2023 Alfredo A. Correa

#ifndef MULTI_ADAPTOR_FFTW_MEMORY_HPP
#define MULTI_ADAPTOR_FFTW_MEMORY_HPP

#include <fftw3.h>

#include<cstddef>  // for std:::size_t

namespace boost::multi {
namespace fftw{

template<class T = void>
struct allocator {
	using value_type = T;
	using size_type  = std::size_t;

	auto allocate(size_type n) -> T* {
		if(n == 0) {return nullptr;}
		if (n > max_size()) {throw std::length_error("multi::fftw::allocator<T>::allocate() overflow.");}
		void* ptr = fftw_malloc(sizeof(T) * n);
		if(ptr == nullptr) {throw std::bad_alloc{};}
		return static_cast<T*>(ptr);
	}
	void deallocate(T* ptr, size_type n) {if(n != 0) {fftw_free(ptr);}}

	constexpr auto operator==(allocator const& /*other*/) const -> bool {return true ;}
	constexpr auto operator!=(allocator const& /*other*/) const -> bool {return false;}

 private:
    static constexpr auto max_size() {return (static_cast<size_type>(0) - static_cast<size_type>(1)) / sizeof(T);}
};

template <class T, class U>
bool operator==(allocator<T> const&, allocator<U> const&) noexcept{return true;}

template <class T, class U>
bool operator!=(allocator<T> const& x, allocator<U> const& y) noexcept{
	return !(x == y);
}

}
}

#if not __INCLUDE_LEVEL__

#include "../../array.hpp"

#include<vector>

namespace multi = boost::multi;

int main(){
	{
		std::vector<double, multi::fftw::allocator<double>> v(100);
		multi::array<double, 2> arr({10, 20});
	}
	{
		std::vector<std::complex<double>, multi::fftw::allocator<std::complex<double>>> v(100);
		multi::array<std::complex<double>, 2> arr({10, 20});
	}
}
#endif
#endif
