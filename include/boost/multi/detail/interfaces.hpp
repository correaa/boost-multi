// Copyright 2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_DETAIL_INTERFACES_HPP
#define BOOST_MULTI_DETAIL_INTERFACES_HPP
// #pragma once

#include <cstddef>      // for ptrdiff_t
#include <iterator>     // for random_access_iterator_tag
#include <type_traits>  // for enable_if_t, is_base_of
#include <utility>      // for forward

#ifdef __NVCC__
	#define BOOST_MULTI_HD __host__ __device__
#else
	#define BOOST_MULTI_HD
#endif

namespace boost::multi::detail {

template<class Self> 
class equality_comparable {
	using self_type = Self;
	friend bool operator!=(self_type const& self, self_type const& other) {
		return !(self == other);
	}

 protected:
	equality_comparable() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
};

template<class Self>
class regular : equality_comparable<Self> {
 protected:
	regular() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
};

template<class Self>
class weakly_incrementable_facade {
	using self_type = Self;
 protected:
	weakly_incrementable_facade() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	using difference_type = typename self_type::difference_type;

 public:
	friend auto operator++(self_type& self, int) {
		auto ret{self}; ++self; return ret;
	}
};

template<class Self>
class incrementable_facade : regular<Self> {
	using self_type = Self;
 protected:
	incrementable_facade() = default;  // NOLINT(bugprone-crtp-constructor-accessibility)
	
 public:
	friend auto operator++(self_type& self, int) {
		auto ret{self}; ++self; return ret;
	}
};

}  // end namespace boost::multi::detail

#undef BOOST_MULTI_HD

#endif  // BOOST_MULTI_DETAIL_INTERFACES_HPP
