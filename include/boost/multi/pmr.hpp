// Copyright 2018-2024 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_PMR_HPP_
#define BOOST_MULTI_PMR_HPP_

#include <boost/multi/array.hpp>

#if __has_include(<memory_resource>)
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
