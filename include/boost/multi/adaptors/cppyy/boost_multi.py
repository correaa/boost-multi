# Copyright 2025 Alfredo A. Correa
# Distributed under the Boost Software License, Version 1.0.
# https://www.boost.org/LICENSE_1_0.txt

"""Optional cppyy pythonizations for Boost.Multi.

Boost.Multi already works out-of-the-box under cppyy (see ``cppyy_test.py``).
Importing this module additionally lets you pass a plain Python sequence as the
*shape* of an array::

    import cppyy
    cppyy.add_include_path("path-to/boost-multi/include")
    cppyy.include("boost/multi/array.hpp")

    import boost_multi  # noqa: F401  (registers the pythonizations on import)

    multi = cppyy.gbl.boost.multi
    a = multi.array["int", 2]((3, 3), 7)      # (3, 3) is the shape, 7 the fill value
    b = multi.array["double", 1]((4,), 0.0)
    r = multi.array_ref["double", 2]((2, 3), numpy_array)

Without this module, ``multi.array["T", D](shape_sequence, value)`` sends
cppyy/Cling into an unbounded constructor-overload resolution and hangs the
interpreter; the shape then has to be spelled explicitly as
``multi.extensions_t[D](n0, n1, ...)``.

The rewrite is deliberately conservative: it only fires when the first argument
is a non-empty sequence of ints *and* at least one more argument follows (the
fill value or the target buffer).  A lone sequence is left untouched so that
1-D element lists such as ``multi.array["double", 1]([1.0, 2.0, 3.0])`` keep
their meaning.
"""

import cppyy

__all__ = ["enable"]

_enabled = False


def _looks_like_shape(args):
    return (
        len(args) >= 2
        and isinstance(args[0], (tuple, list))
        and len(args[0]) > 0
        and all(isinstance(n, int) for n in args[0])
    )


def _pythonizor(klass, name):
    if not (
        name.startswith("array<")
        or name.startswith("array_ref<")
        or name.startswith("static_array<")
    ):
        return

    real_init = klass.__init__

    def __init__(self, *args):
        if _looks_like_shape(args):
            dim = len(args[0])
            extents = cppyy.gbl.boost.multi.extents_t[dim](*args[0])
            args = (extents,) + args[1:]
        real_init(self, *args)

    klass.__init__ = __init__


def enable():
    """Register the Boost.Multi pythonizations with cppyy (idempotent).

    Called automatically when this module is imported; exposed so it can also be
    invoked explicitly.
    """
    global _enabled
    if _enabled:
        return
    cppyy.py.add_pythonization(_pythonizor, "boost::multi")
    _enabled = True


enable()
