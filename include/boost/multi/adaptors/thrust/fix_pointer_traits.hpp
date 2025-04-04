// Copyright 2021-2023 Alfredo A. Correa

#ifndef BOOST_MULTI_ADAPTORS_THRUST_FIX_POINTER_TRAITS_HPP_
#define BOOST_MULTI_ADAPTORS_THRUST_FIX_POINTER_TRAITS_HPP_
#pragma once

#include <thrust/memory.h>

#include <memory>   // for std::pointer_traits

#include <thrust/detail/type_traits/pointer_traits.h>

// #if(__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__ < 120500)
// begin of nvcc thrust 11.5 workaround : https://github.com/NVIDIA/thrust/issues/1629
namespace thrust {

// template<typename Element, typename Tag, typename Reference, typename Derived> class pointer;
// template<class T> struct pointer_traits;

}  // end namespace thrust

template<class... As>
struct std::pointer_traits<::thrust::pointer<As...>>  // NOLINT(cert-dcl58-cpp) normal way to specialize pointer_traits
: ::thrust::detail::pointer_traits<thrust::pointer<As...>> {
	template<class T>
	using rebind = typename ::thrust::detail::pointer_traits<::thrust::pointer<As...>>::template rebind<T>::other;
};
// end of nvcc thrust 11.5 workaround
// #endif

#endif
