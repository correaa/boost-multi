# Copyright 2025 Alfredo A. Correa
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

"""Simulated `multi.thrust.cuda.array["T", D](shape...)` for cppyy.

Mirrors the real C++ alias ``boost::multi::thrust::cuda::array<T,D>`` (see
``../thrust.hpp``) over a *precompiled* set of (T, D) combinations::

    import boost_multi_thrust_cuda as bmc
    bmc.enable("/path/to/libthrust_cuda_array.so")

    from boost_multi_thrust_cuda import multi
    a = multi.thrust.cuda.array["double", 2](3, 3)
    a[1, 2] = 42.0
    c = multi.thrust.cuda.array["std::complex<double>", 1](4)
    c[0] = 1 + 2j

Unlike ``boost_multi.py`` (genuine cppyy/Cling reflection of CPU-side ``multi::array``),
this module is NOT genuine template reflection: Cling cannot instantiate
``thrust::cuda::allocator<T>`` itself (see ``doc/modules/ROOT/pages/interop.adoc``,
"Python (cppyy) / GPU", for the full investigation). Instead, ``thrust_cuda_array.cu``
explicitly instantiates ``multi::thrust::cuda::array<T,D>`` for a known set of (T, D)
pairs at ordinary compile time (real nvcc, never Cling), exposes a plain C dispatch API
declared in ``thrust_cuda_array.hpp``, and this module wires that up so the Python call
site still reads like the real alias. Requesting a (T, D) outside the precompiled set
raises ``KeyError``; it never silently falls back or compiles anything on the fly.

Adding a new element type is a one-line addition to ``THRUST_CUDA_ARRAY_TYPES`` in
``thrust_cuda_array.cu`` plus the matching entry in ``_TYPE_INFO`` below, it is not a
redesign. Adding a new dimension means one more create/destroy/size/set/get/memory_type
overload in thrust_cuda_array.hpp/.cu plus the matching entry in ``_PRECOMPILED_DIMS``,
also not a redesign of the dispatch machinery. A (T, D) nobody listed still will not
work; that would need an on-demand nvcc-subprocess-compile approach instead of this
precompiled-set approach.
"""

import os

import cppyy

__all__ = ["enable", "multi"]

# Must match THRUST_CUDA_ARRAY_TYPES in thrust_cuda_array.cu: name -> (type_id, is_complex).
_TYPE_INFO = {
    "int": (0, False),
    "unsigned": (1, False),
    "float": (2, False),
    "double": (3, False),
    "std::complex<float>": (4, True),
    "std::complex<double>": (5, True),
    "thrust::complex<float>": (6, True),
    "thrust::complex<double>": (7, True),
}
_PRECOMPILED_DIMS = {1, 2, 3, 4}

_enabled = False
_lib_path_used = None


def enable(lib_path):
    """Register the plain-C interface and load the precompiled GPU library.

    lib_path must point at the built libthrust_cuda_array.so (out-of-source CMake
    builds place it in the build directory, not next to this .py file, so it cannot
    be defaulted the way boost_multi.py's pure-Python pythonizations can).
    """
    global _enabled, _lib_path_used  # noqa: PLW0603
    if _enabled:
        if lib_path != _lib_path_used:
            raise RuntimeError(
                "boost_multi_thrust_cuda already enabled with lib_path=%r, cannot "
                "re-enable with %r" % (_lib_path_used, lib_path)
            )
        return
    include_dir = os.path.dirname(os.path.abspath(__file__))
    cppyy.add_include_path(include_dir)
    cppyy.include("thrust_cuda_array.hpp")
    cppyy.load_library(lib_path)
    _enabled = True
    _lib_path_used = lib_path


class _GpuArray:
    def __init__(self, type_id, is_complex, dim, handle):
        self._type_id = type_id
        self._is_complex = is_complex
        self._dim = dim
        self._h = handle

    def _fn(self, name):
        return getattr(cppyy.gbl, "thrust_cuda_array_%s%d" % (name, self._dim))

    def _idx_args(self, idx):
        return (idx,) if self._dim == 1 else tuple(idx)

    def __del__(self):
        self._fn("destroy")(self._type_id, self._h)

    def sizes(self):
        """Tuple of all dimension sizes, matching boost::multi's own .sizes() (as opposed
        to .size(), which is dim-0-only)."""
        size_fn = self._fn("size")
        return tuple(size_fn(self._type_id, self._h, k) for k in range(self._dim))

    def memory_type(self):
        """cudaMemoryType of the underlying pointer (cudaMemoryTypeDevice == 2)."""
        return self._fn("memory_type")(self._type_id, self._h)

    def __getitem__(self, idx):
        v = self._fn("get")(self._type_id, self._h, *self._idx_args(idx))
        return complex(v.re, v.im) if self._is_complex else v.re

    def __setitem__(self, idx, value):
        if self._is_complex:
            c = complex(value)
            v = cppyy.gbl.thrust_cuda_array_value(c.real, c.imag)
        else:
            v = cppyy.gbl.thrust_cuda_array_value(float(value), 0.0)
        self._fn("set")(self._type_id, self._h, *self._idx_args(idx), v)


class _ArrayFactory:
    """Stands in for boost::multi::thrust::cuda::array["T", D] over the precompiled set."""

    def __getitem__(self, key):
        t, d = key
        if t not in _TYPE_INFO:
            raise KeyError(
                "no multi::thrust::cuda::array precompiled for element type %r; available: %s"
                % (t, sorted(_TYPE_INFO))
            )
        if d not in _PRECOMPILED_DIMS:
            raise KeyError(
                "no multi::thrust::cuda::array precompiled for dimension %r; available: %s"
                % (d, sorted(_PRECOMPILED_DIMS))
            )
        type_id, is_complex = _TYPE_INFO[t]

        def make(*shape):
            if len(shape) != d:
                raise ValueError("array['%s', %d] needs %d shape arguments, got %d" % (t, d, d, len(shape)))
            create_fn = getattr(cppyy.gbl, "thrust_cuda_array_create%d" % d)
            h = create_fn(type_id, *shape)
            return _GpuArray(type_id, is_complex, d, h)

        return make


class _Namespace:
    """Trivial attribute container, mirrors C++ namespace nesting in Python
    (multi.thrust.cuda.array), the shape cppyy.gbl.boost.multi.thrust.cuda.array would
    have if Cling could reflect the real alias directly."""


array = _ArrayFactory()

cuda = _Namespace()
cuda.array = array

thrust = _Namespace()
thrust.cuda = cuda

multi = _Namespace()
multi.thrust = thrust
