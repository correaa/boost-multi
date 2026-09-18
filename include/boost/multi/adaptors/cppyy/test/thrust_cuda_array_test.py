# Copyright 2025 Alfredo A. Correa
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

# Test for the multi.thrust.cuda.array["T", D] simulation over a precompiled set of GPU
# (T, D) combinations (see ../boost_multi_thrust_cuda.py and
# doc/modules/ROOT/pages/interop.adoc, section "Python (cppyy) / GPU").
#
# Run standalone with:
#   PYTHONPATH=<venv-site-packages> python3 thrust_cuda_array_test.py <path-to-libthrust_cuda_array.so>

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boost_multi_thrust_cuda as bmc  # noqa: E402

lib_path = sys.argv[1] if len(sys.argv) > 1 else "./libthrust_cuda_array.so"
bmc.enable(lib_path)

from boost_multi_thrust_cuda import multi  # noqa: E402

array = multi.thrust.cuda.array  # mirrors cppyy.gbl.boost.multi.thrust.cuda.array

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


DEVICE = 2  # cudaMemoryTypeDevice

# --- double, 2D ---
a = array["double", 2](3, 3)
check("double 2d sizes", a.sizes() == (3, 3))
a[1, 2] = 42.5
check("double 2d set/get", a[1, 2] == 42.5)
check("double 2d is device memory", a.memory_type() == DEVICE)

# --- float, 1D ---
b = array["float", 1](5)
b[3] = 7.5
check("float 1d set/get", b[3] == 7.5 and b.sizes() == (5,))

# --- int, unsigned, 1D ---
i = array["int", 1](3)
i[0] = -5
check("int 1d signed value", i[0] == -5)

u = array["unsigned", 1](3)
u[0] = 5
check("unsigned 1d set/get", u[0] == 5)

# --- complex types, all four flavors, 1D ---
for tname in ["std::complex<float>", "std::complex<double>", "thrust::complex<float>", "thrust::complex<double>"]:
    ca = array[tname, 1](4)
    ca[0] = 1 + 2j
    ca[1] = complex(-3.5, 4.25)
    check("%s set/get element 0" % tname, ca[0] == complex(1, 2))
    check("%s set/get element 1" % tname, ca[1] == complex(-3.5, 4.25))
    check("%s is device memory" % tname, ca.memory_type() == DEVICE)
    del ca

# --- dimensions 1..4, double ---
d1 = array["double", 1](4)
d1[2] = 9.0
check("dim1 set/get", d1[2] == 9.0)

d2 = array["double", 2](2, 3)
d2[1, 2] = 8.0
check("dim2 set/get + sizes", d2[1, 2] == 8.0 and d2.sizes() == (2, 3))

d3 = array["double", 3](2, 3, 4)
d3[1, 2, 3] = 7.0
check("dim3 set/get + sizes", d3[1, 2, 3] == 7.0 and d3.sizes() == (2, 3, 4))

d4 = array["double", 4](2, 2, 2, 2)
d4[1, 0, 1, 0] = 6.0
check("dim4 set/get + sizes", d4[1, 0, 1, 0] == 6.0 and d4.sizes() == (2, 2, 2, 2))
check("dim4 is device memory", d4.memory_type() == DEVICE)

del a, b, i, u, d1, d2, d3, d4

# --- requesting a non-precompiled combination fails clearly, not silently ---
try:
    array["long double", 2](3, 3)
    check("unlisted type correctly rejected", False)
except KeyError as e:
    check("unlisted type correctly rejected", "long double" in str(e))

try:
    array["double", 5](2, 2, 2, 2, 2)
    check("dim 5 correctly rejected", False)
except KeyError as e:
    check("dim 5 correctly rejected", "5" in str(e))


if failures:
    print("\n{} failure(s): {}".format(len(failures), ", ".join(failures)))
    sys.exit(1)
print("\nall multi.thrust.cuda.array['T', D] simulation checks passed")
