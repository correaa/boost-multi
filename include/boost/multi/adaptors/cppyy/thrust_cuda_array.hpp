// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Plain C-style interface: no templates, no thrust, no boost-multi visible here.
// This is ALL that cppyy/Cling ever parses. The real boost::multi::thrust::cuda::array<T,D>
// (see ../thrust.hpp) lives only in thrust_cuda_array.cu, compiled ahead of time by real nvcc.
//
// Cling cannot instantiate thrust::cuda::allocator<T> itself (confirmed: it hits unrelated
// CCCL SFINAE-detection failures deep inside thrust's allocator_traits/pointer_traits, and
// separately Clang-CUDA mode in this cppyy-cling release only partially supports CUDA <= 11.8
// and cannot parse a modern GCC libstdc++'s __float128 usage on the device side). So unlike
// boost_multi.py (genuine cppyy/Cling reflection of CPU-side multi::array), the Python module
// boost_multi_thrust_cuda.py that wraps this header cannot do genuine template reflection
// either: it dispatches to a small, explicitly precompiled set of (T, D) combinations declared
// in thrust_cuda_array.cu. See doc/modules/ROOT/pages/interop.adoc, "Python (cppyy) / GPU".
#pragma once

// Real/imaginary pair, used uniformly for both real (im always 0) and complex element types
// so set/get do not need a family of overloads per type category. Whether a given type_id is
// complex is tracked Python-side in _TYPE_IDS, this struct itself is type-agnostic.
struct thrust_cuda_array_value {
	double re;
	double im;
};

extern "C" {

void* thrust_cuda_array_create1(int type_id, long n0);
void* thrust_cuda_array_create2(int type_id, long n0, long n1);
void* thrust_cuda_array_create3(int type_id, long n0, long n1, long n2);
void* thrust_cuda_array_create4(int type_id, long n0, long n1, long n2, long n3);

void thrust_cuda_array_destroy1(int type_id, void* h);
void thrust_cuda_array_destroy2(int type_id, void* h);
void thrust_cuda_array_destroy3(int type_id, void* h);
void thrust_cuda_array_destroy4(int type_id, void* h);

long thrust_cuda_array_size1(int type_id, void* h, int dim);
long thrust_cuda_array_size2(int type_id, void* h, int dim);
long thrust_cuda_array_size3(int type_id, void* h, int dim);
long thrust_cuda_array_size4(int type_id, void* h, int dim);

void thrust_cuda_array_set1(int type_id, void* h, long i0, thrust_cuda_array_value v);
void thrust_cuda_array_set2(int type_id, void* h, long i0, long i1, thrust_cuda_array_value v);
void thrust_cuda_array_set3(int type_id, void* h, long i0, long i1, long i2, thrust_cuda_array_value v);
void thrust_cuda_array_set4(int type_id, void* h, long i0, long i1, long i2, long i3, thrust_cuda_array_value v);

thrust_cuda_array_value thrust_cuda_array_get1(int type_id, void* h, long i0);
thrust_cuda_array_value thrust_cuda_array_get2(int type_id, void* h, long i0, long i1);
thrust_cuda_array_value thrust_cuda_array_get3(int type_id, void* h, long i0, long i1, long i2);
thrust_cuda_array_value thrust_cuda_array_get4(int type_id, void* h, long i0, long i1, long i2, long i3);

int thrust_cuda_array_memory_type1(int type_id, void* h);
int thrust_cuda_array_memory_type2(int type_id, void* h);
int thrust_cuda_array_memory_type3(int type_id, void* h);
int thrust_cuda_array_memory_type4(int type_id, void* h);

}  // extern "C"
