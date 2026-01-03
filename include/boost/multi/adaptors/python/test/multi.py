import cppyy
import ctypes
import numpy as np

#cppyy.set_debug(True)

cppyy.add_include_path('/home/correaa/boost-multi/include')
cppyy.include("boost/multi/array.hpp")
cppyy.include("boost/multi/io.hpp")

multi = cppyy.gbl.boost.multi
std = cppyy.gbl.std

a2d = multi.array['double', 2]();
print(type(a2d))
a2d.operator=([[1.0, 2.0],[3.0, 4.0]])
print(type(a2d))

a2d = multi.array['double', 2]([[1.0, 2.0],[4.0, 5.0]])
print(a2d)

print(a2d.data_elements())

npa2d = np.frombuffer(a2d.base(), dtype=np.float64, count=a2d.num_elements()).reshape(a2d.sizes().get[0](), a2d.sizes().get[1]())
print(npa2d)

a2d_ref = multi.array_ref['double', 2](multi.extensions_t[2](*npa2d.shape), npa2d)
print(a2d_ref)

a2d_ref[1][1] = 999.9

print(npa2d)
print(a2d)
