// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2022 Alfredo A. Correa

#pragma once

#include "../detail/type_traits.hpp"

#include "../detail/fix_complex_traits.hpp"

#include<complex>

namespace boost {
namespace multi {

#pragma message "By including this header, the behavior of initialization of std::complex<T> in multi::array's changes. std::complex<T>  elements will not be initialized."

template<class T> struct is_trivially_default_constructible<std::complex<T>> : std::is_trivially_default_constructible<T> {};
template<class T> struct is_trivial<std::complex<T>> : std::is_trivial<T> {};

}
}

static_assert(not std::is_trivially_default_constructible<::std::complex<double>>::value);
static_assert(not std::is_trivially_default_constructible<::std::complex<float >>::value);

static_assert(boost::multi::is_trivially_default_constructible<::std::complex<double>>::value);
static_assert(boost::multi::is_trivially_default_constructible<::std::complex<float >>::value);

static_assert(boost::multi::is_trivial<::std::complex<double>>::value);
static_assert(boost::multi::is_trivial<::std::complex<float >>::value);

static_assert(std::is_trivially_assignable<::std::complex<double>&, ::std::complex<double>>::value);
static_assert(std::is_trivially_assignable<::std::complex<float >&, ::std::complex<float >>::value);

static_assert(std::is_trivially_copyable<::std::complex<double>>::value);
static_assert(std::is_trivially_copyable<::std::complex<float >>::value);


