# Copyright 2025 Alfredo A. Correa
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

# Test for the optional `boost_multi` cppyy pythonization module (see
# ../boost_multi.py and doc/modules/ROOT/pages/interop.adoc, section "Python (cppyy)").
#
# Run standalone with:
#   PYTHONPATH=<venv-site-packages> python3 pythonize_test.py <path-to-boost-multi/include>

import os
import sys

import cppyy

include_path = sys.argv[1] if len(sys.argv) > 1 else "./include"

cppyy.add_include_path(include_path)
cppyy.include("boost/multi/array.hpp")

# make `import boost_multi` resolve to ../boost_multi.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import boost_multi  # noqa: E402  (registers the pythonizations on import)

boost_multi.enable()  # idempotent, exercised here on purpose

multi = cppyy.gbl.boost.multi

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


# --- shape as a Python sequence (the whole point of the module) -----------
a2d = multi.array["int", 2]((3, 3), 7)
check("2d array from (tuple, value)", a2d.num_elements() == 9 and a2d[2][2] == 7)

a2d_list = multi.array["int", 2]([3, 3], 7)
check("2d array from [list, value]", a2d_list.num_elements() == 9 and a2d_list[1][1] == 7)

a1d = multi.array["double", 1]((4,), 1.5)
check("1d array from (tuple, value)", list(a1d) == [1.5, 1.5, 1.5, 1.5])

a3d = multi.array["int", 3]((2, 3, 4), 0)
check("3d array from (tuple, value)", a3d.num_elements() == 24)

# --- the explicit extensions spelling still works ------------------------
a2d_ext = multi.array["int", 2](multi.extensions_t[2](3, 3), 7)
check("2d array from extensions_t still works", a2d_ext[2][2] == 7)

# --- a lone sequence is NOT reinterpreted as a shape --------------------
a1d_elems = multi.array["double", 1]([1.0, 2.0, 3.0, 4.0])
check("1d element list is untouched", list(a1d_elems) == [1.0, 2.0, 3.0, 4.0])

a1d_int_elems = multi.array["int", 1]([3, 3, 3])
check("1d int element list is untouched", list(a1d_int_elems) == [3, 3, 3])

# --- array_ref over NumPy memory using a sequence shape ----------------
try:
    import numpy as np

    npa = np.zeros((2, 3))
    ref = multi.array_ref["double", 2]((2, 3), npa)
    ref[1][2] = 5.0
    check("array_ref from (tuple, numpy) shares memory", npa[1][2] == 5.0)
except ImportError:
    print("SKIP array_ref / numpy check (numpy not available)")


if failures:
    print("\n{} failure(s): {}".format(len(failures), ", ".join(failures)))
    sys.exit(1)
print("\nall boost_multi pythonization checks passed")
