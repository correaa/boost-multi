// Copyright 2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// Real boost::multi::thrust::cuda::array<T,D> instances, compiled ahead of time by real nvcc
// (never by Cling). (type_id, D) dispatch is the only thing standing in for a Python-selectable
// template parameter: the actual C++ template is instantiated here, at ordinary compile time,
// once per (T,D) pair we choose to support.
//
// Adding a new element type is a one-line addition to THRUST_CUDA_ARRAY_TYPES below (and the
// matching entry in boost_multi_thrust_cuda.py's _TYPE_IDS), not a redesign. Adding a new
// dimension means adding one more create/destroy/size/set/get/memory_type extern "C" overload
// (declared in thrust_cuda_array.hpp) that forwards to the existing <T, D> impl templates,
// also not a redesign of the dispatch machinery itself. What this file cannot do is accept a
// (T, D) nobody listed here at Python call time; that would need an on-demand
// nvcc-subprocess-compile approach instead of this precompiled-set approach.
#include "thrust_cuda_array.hpp"

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/array.hpp>

#include <thrust/complex.h>
#include <thrust/system/cuda/memory.h>

#include <complex>
#include <cstdlib>

namespace {

namespace multi = boost::multi;

template<class T, multi::dimensionality_type D>
using GpuArray = multi::thrust::cuda::array<T, D>;

// --- value <-> T conversion, uniform across real and complex T ---------------------------
template<class T>
inline constexpr bool is_complex_v = false;
template<class U>
inline constexpr bool is_complex_v<std::complex<U>> = true;
template<class U>
inline constexpr bool is_complex_v<::thrust::complex<U>> = true;

template<class T>
T value_to_T(thrust_cuda_array_value v) {
	if constexpr(is_complex_v<T>) {
		using U = typename T::value_type;
		return T(static_cast<U>(v.re), static_cast<U>(v.im));
	} else {
		return static_cast<T>(v.re);
	}
}

template<class T>
thrust_cuda_array_value T_to_value(T x) {
	if constexpr(is_complex_v<T>) {
		return {static_cast<double>(x.real()), static_cast<double>(x.imag())};
	} else {
		return {static_cast<double>(x), 0.0};
	}
}

// --- (T, D) implementation, D handled via variadic templates instead of per-D duplication -
template<class T, multi::dimensionality_type D, class... Ns>
void* create_impl(Ns... ns) {
	static_assert(sizeof...(Ns) == D);
	return new GpuArray<T, D>(multi::extensions_t<D>(ns...));
}

template<class T, multi::dimensionality_type D>
void destroy_impl(void* h) { delete static_cast<GpuArray<T, D>*>(h); }

template<class T, multi::dimensionality_type D>
long size_impl(void* h, int dim) {
	auto* a = static_cast<GpuArray<T, D>*>(h);
	if(dim == 0) { return a->size(); }
	if constexpr(D >= 2) {
		if(dim == 1) { return (*a)[0].size(); }
	}
	if constexpr(D >= 3) {
		if(dim == 2) { return (*a)[0][0].size(); }
	}
	if constexpr(D >= 4) {
		if(dim == 3) { return (*a)[0][0][0].size(); }
	}
	std::abort();
}

// the value comes before the indices so the index pack can stay in trailing (deducible)
// position; extern "C" wrappers below reorder arguments back to the natural i0,...,iN,v shape.
template<class T, multi::dimensionality_type D, class... Is>
void set_impl(void* h, thrust_cuda_array_value v, Is... is) {
	static_assert(sizeof...(Is) == D);
	(*static_cast<GpuArray<T, D>*>(h))(is...) = value_to_T<T>(v);
}

template<class T, multi::dimensionality_type D, class... Is>
thrust_cuda_array_value get_impl(void* h, Is... is) {
	static_assert(sizeof...(Is) == D);
	return T_to_value<T>((*static_cast<GpuArray<T, D>*>(h))(is...));
}

template<class T, multi::dimensionality_type D>
int memtype_impl(void* h) {
	auto* raw = thrust::raw_pointer_cast(static_cast<GpuArray<T, D>*>(h)->data_elements());
	cudaPointerAttributes attrs{};
	cudaPointerGetAttributes(&attrs, raw);
	return static_cast<int>(attrs.type);
}

// --- type_id <-> T dispatch --------------------------------------------------------------
// Single source of truth for which element types are precompiled. type_id must match
// _TYPE_IDS in boost_multi_thrust_cuda.py.
#define THRUST_CUDA_ARRAY_TYPES(X)         \
	X(0, int)                              \
	X(1, unsigned)                         \
	X(2, float)                            \
	X(3, double)                           \
	X(4, std::complex<float>)              \
	X(5, std::complex<double>)             \
	X(6, ::thrust::complex<float>)         \
	X(7, ::thrust::complex<double>)

template<class T>
struct Tag {
	using type = T;
};

template<multi::dimensionality_type D, class F>
auto dispatch(int type_id, F&& f) {
	switch(type_id) {
#define X(id, T) \
	case id: return f(Tag<T>{});
		THRUST_CUDA_ARRAY_TYPES(X)
#undef X
	default: std::abort();  // Python side never sends an id outside _TYPE_IDS
	}
}

}  // namespace

extern "C" {

void* thrust_cuda_array_create1(int type_id, long n0) {
	return dispatch<1>(type_id, [&](auto tag) -> void* {
		using T = typename decltype(tag)::type;
		return create_impl<T, 1>(n0);
	});
}
void* thrust_cuda_array_create2(int type_id, long n0, long n1) {
	return dispatch<2>(type_id, [&](auto tag) -> void* {
		using T = typename decltype(tag)::type;
		return create_impl<T, 2>(n0, n1);
	});
}
void* thrust_cuda_array_create3(int type_id, long n0, long n1, long n2) {
	return dispatch<3>(type_id, [&](auto tag) -> void* {
		using T = typename decltype(tag)::type;
		return create_impl<T, 3>(n0, n1, n2);
	});
}
void* thrust_cuda_array_create4(int type_id, long n0, long n1, long n2, long n3) {
	return dispatch<4>(type_id, [&](auto tag) -> void* {
		using T = typename decltype(tag)::type;
		return create_impl<T, 4>(n0, n1, n2, n3);
	});
}

void thrust_cuda_array_destroy1(int type_id, void* h) {
	dispatch<1>(type_id, [&](auto tag) { destroy_impl<typename decltype(tag)::type, 1>(h); });
}
void thrust_cuda_array_destroy2(int type_id, void* h) {
	dispatch<2>(type_id, [&](auto tag) { destroy_impl<typename decltype(tag)::type, 2>(h); });
}
void thrust_cuda_array_destroy3(int type_id, void* h) {
	dispatch<3>(type_id, [&](auto tag) { destroy_impl<typename decltype(tag)::type, 3>(h); });
}
void thrust_cuda_array_destroy4(int type_id, void* h) {
	dispatch<4>(type_id, [&](auto tag) { destroy_impl<typename decltype(tag)::type, 4>(h); });
}

long thrust_cuda_array_size1(int type_id, void* h, int dim) {
	return dispatch<1>(type_id, [&](auto tag) -> long { return size_impl<typename decltype(tag)::type, 1>(h, dim); });
}
long thrust_cuda_array_size2(int type_id, void* h, int dim) {
	return dispatch<2>(type_id, [&](auto tag) -> long { return size_impl<typename decltype(tag)::type, 2>(h, dim); });
}
long thrust_cuda_array_size3(int type_id, void* h, int dim) {
	return dispatch<3>(type_id, [&](auto tag) -> long { return size_impl<typename decltype(tag)::type, 3>(h, dim); });
}
long thrust_cuda_array_size4(int type_id, void* h, int dim) {
	return dispatch<4>(type_id, [&](auto tag) -> long { return size_impl<typename decltype(tag)::type, 4>(h, dim); });
}

void thrust_cuda_array_set1(int type_id, void* h, long i0, thrust_cuda_array_value v) {
	dispatch<1>(type_id, [&](auto tag) { set_impl<typename decltype(tag)::type, 1>(h, v, i0); });
}
void thrust_cuda_array_set2(int type_id, void* h, long i0, long i1, thrust_cuda_array_value v) {
	dispatch<2>(type_id, [&](auto tag) { set_impl<typename decltype(tag)::type, 2>(h, v, i0, i1); });
}
void thrust_cuda_array_set3(int type_id, void* h, long i0, long i1, long i2, thrust_cuda_array_value v) {
	dispatch<3>(type_id, [&](auto tag) { set_impl<typename decltype(tag)::type, 3>(h, v, i0, i1, i2); });
}
void thrust_cuda_array_set4(int type_id, void* h, long i0, long i1, long i2, long i3, thrust_cuda_array_value v) {
	dispatch<4>(type_id, [&](auto tag) { set_impl<typename decltype(tag)::type, 4>(h, v, i0, i1, i2, i3); });
}

thrust_cuda_array_value thrust_cuda_array_get1(int type_id, void* h, long i0) {
	return dispatch<1>(type_id, [&](auto tag) { return get_impl<typename decltype(tag)::type, 1>(h, i0); });
}
thrust_cuda_array_value thrust_cuda_array_get2(int type_id, void* h, long i0, long i1) {
	return dispatch<2>(type_id, [&](auto tag) { return get_impl<typename decltype(tag)::type, 2>(h, i0, i1); });
}
thrust_cuda_array_value thrust_cuda_array_get3(int type_id, void* h, long i0, long i1, long i2) {
	return dispatch<3>(type_id, [&](auto tag) { return get_impl<typename decltype(tag)::type, 3>(h, i0, i1, i2); });
}
thrust_cuda_array_value thrust_cuda_array_get4(int type_id, void* h, long i0, long i1, long i2, long i3) {
	return dispatch<4>(type_id, [&](auto tag) { return get_impl<typename decltype(tag)::type, 4>(h, i0, i1, i2, i3); });
}

int thrust_cuda_array_memory_type1(int type_id, void* h) {
	return dispatch<1>(type_id, [&](auto tag) -> int { return memtype_impl<typename decltype(tag)::type, 1>(h); });
}
int thrust_cuda_array_memory_type2(int type_id, void* h) {
	return dispatch<2>(type_id, [&](auto tag) -> int { return memtype_impl<typename decltype(tag)::type, 2>(h); });
}
int thrust_cuda_array_memory_type3(int type_id, void* h) {
	return dispatch<3>(type_id, [&](auto tag) -> int { return memtype_impl<typename decltype(tag)::type, 3>(h); });
}
int thrust_cuda_array_memory_type4(int type_id, void* h) {
	return dispatch<4>(type_id, [&](auto tag) -> int { return memtype_impl<typename decltype(tag)::type, 4>(h); });
}

}  // extern "C"
