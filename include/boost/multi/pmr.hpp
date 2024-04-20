// Copyright 2018-2024 Alfredo A. Correa

#ifndef MULTI_PMR_HPP_
#define MULTI_PMR_HPP_

#include <multi/array.hpp>

#if(!defined(__GLIBCXX__) || (__GLIBCXX__ >= 20210601)) && (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION > 14000))
#include <memory_resource>
#endif

#if(defined(__cpp_lib_memory_resource) && (__cpp_lib_memory_resource >= 201603))
namespace boost::multi::pmr {

template<class T, boost::multi::dimensionality_type D>
using array = boost::multi::array<T, D, std::pmr::polymorphic_allocator<T>>;

}  // end namespace boost::multi::pmr
#else
namespace boost::multi::pmr {
template<class T, boost::multi::dimensionality_type D>
struct [[deprecated("no PMR allocator")]] array;  // your version of C++ doesn't provide polymorphic_allocators
}  // end namespace boost::multi::pmr
#endif

#endif
