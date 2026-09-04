# Copyright 2025 Alfredo A. Correa
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

# Self-contained cppyy test for Boost.Multi.
#
# There is NO cppyy adaptor header and NO Python glue/pythonization code:
# this exercises what works purely out-of-the-box when cppyy parses the
# public Boost.Multi headers.  See doc/modules/ROOT/pages/interop.adoc,
# section "Python (cppyy)".
#
# Verified with cppyy 3.5.0 and NumPy 2.4 (see also cppyy 3.5.0 / NumPy 2.5
# in the documentation).
#
# Run standalone with:
#   PYTHONPATH=<venv-site-packages> python3 cppyy_test.py <path-to-boost-multi/include>

import sys
import cppyy

include_path = sys.argv[1] if len(sys.argv) > 1 else "./include"

cppyy.add_include_path(include_path)
cppyy.include("boost/multi/array.hpp")  # only the public header, nothing else

multi = cppyy.gbl.boost.multi

failures = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


# --- 1D array: construction ------------------------------------------------
a1d = multi.array["double", 1]([1.0, 2.0, 3.0, 4.0])
check("1d construct from list", list(a1d) == [1.0, 2.0, 3.0, 4.0])
check("1d size()", a1d.size() == 4)
check("1d num_elements()", a1d.num_elements() == 4)

a1d_filled = multi.array["double", 1](4, 0.0)
check("1d construct from (size, value)", list(a1d_filled) == [0.0, 0.0, 0.0, 0.0])

# --- 1D array: element access and mutation --------------------------------
check("1d element read", a1d[2] == 3.0)
a1d[2] = 99.9
check("1d element write", a1d[2] == 99.9)

# --- 2D array: construction from extensions -------------------------------
a2d = multi.array["double", 2](multi.extensions_t[2](2, 2), 0.0)
a2d[0][0] = 1.0
a2d[0][1] = 2.0
a2d[1][0] = 3.0
a2d[1][1] = 4.0
check("2d size() (number of rows)", a2d.size() == 2)
check("2d num_elements()", a2d.num_elements() == 4)
check("2d nested element access", a2d[1][0] == 3.0)
check("2d sizes().get[0]/get[1]", (a2d.sizes().get[0](), a2d.sizes().get[1]()) == (2, 2))

# --- rows / subarrays are references into the parent ---------------------
arow = a2d[0]
check("2d row view content", list(arow) == [1.0, 2.0])
arow[0] = 66.6
check("2d row is a reference (write-through)", a2d[0][0] == 66.6)

# a decayed copy (unary plus) is independent of the parent
arow_copy = +a2d[0]
arow_copy[0] = 11111.1
check("2d unary-plus makes an independent copy", a2d[0][0] == 66.6)

# --- streaming (operator<< via cppyy) ------------------------------------
check("1d printable via str()", str(multi.array["double", 1]([1.0, 2.0])).strip().startswith("{"))

# --- integer and complex element types ---------------------------------
ai = multi.array["int", 2](multi.extensions_t[2](3, 3), 7)
check("2d int array fill", ai.num_elements() == 9 and ai[2][2] == 7)

cdbl = cppyy.gbl.std.complex["double"]
ac = multi.array["std::complex<double>", 1](3, cdbl(1.0, 2.0))
check("1d std::complex fill", ac[0] == complex(1.0, 2.0))

# --- NumPy interoperability (no copies) --------------------------------
try:
    import numpy as np

    # view a contiguous Multi array as a NumPy array over the same memory
    npv = np.frombuffer(
        a2d.data_elements(), dtype=np.float64, count=a2d.num_elements()
    ).reshape(a2d.sizes().get[0](), a2d.sizes().get[1]())
    check("numpy view over multi memory", npv.tolist() == [[66.6, 2.0], [3.0, 4.0]])

    npv[1][1] = 999.9
    check("write through numpy view reaches multi", a2d[1][1] == 999.9)

    # the other way around: a multi array_ref over NumPy-owned memory
    npa = np.zeros((2, 3))
    ref = multi.array_ref["double", 2](multi.extensions_t[2](2, 3), npa)
    ref[1][2] = 5.0
    check("multi array_ref over numpy memory", npa[1][2] == 5.0)
except ImportError:
    print("SKIP numpy interoperability (numpy not available)")


if failures:
    print("\n{} failure(s): {}".format(len(failures), ", ".join(failures)))
    sys.exit(1)
print("\nall cppyy out-of-the-box checks passed")
