// Copyright 2018-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_PMR_HPP_
#define BOOST_MULTI_PMR_HPP_

#include <boost/multi/array.hpp>

#if __has_include(<memory_resource>)
#  include <memory_resource>
// Apple clang provides the header but not the compiled library prior to version 16
#  if (defined(__cpp_lib_memory_resource) && (__cpp_lib_memory_resource >= 201603)) && !(defined(__APPLE__) && defined(__clang_major__) && __clang_major__ <= 15)
#    define BOOST_MULTI_HAS_MEMORY_RESOURCE
#  endif
#endif

namespace boost::multi::pmr {

#ifdef BOOST_MULTI_HAS_MEMORY_RESOURCE
template<class T, boost::multi::dimensionality_type D>
using array = boost::multi::array<T, D, std::pmr::polymorphic_allocator<T>>;
#else
template<class T, boost::multi::dimensionality_type D>
struct [[deprecated("no PMR allocator")]] array;  // your version of C++ doesn't provide polymorphic_allocators
#endif

}  // end namespace boost::multi::pmr

#endif
