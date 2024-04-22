// Copyright 2018-2024 Alfredo A. Correa

#ifndef MULTI_PMR_HPP_
#define MULTI_PMR_HPP_

#include <boost/multi/array.hpp>

#if(!defined(__GLIBCXX__) || (__GLIBCXX__ >= 20210601)) && (!defined(_LIBCPP_VERSION) || (_LIBCPP_VERSION > 14000))
#include <memory_resource>
#endif

namespace boost::multi::pmr {

#if(defined(__cpp_lib_memory_resource) && (__cpp_lib_memory_resource >= 201603))
template<class T, boost::multi::dimensionality_type D>
using array = boost::multi::array<T, D, std::pmr::polymorphic_allocator<T>>;
#else
template<class T, boost::multi::dimensionality_type D>
struct [[deprecated("no PMR allocator")]] array;  // your version of C++ doesn't provide polymorphic_allocators
#endif

}  // end namespace boost::multi::pmr

#endif
