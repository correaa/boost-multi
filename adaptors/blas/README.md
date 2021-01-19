<!--
(pandoc `#--from gfm` --to html --standalone --metadata title=" " $0 > $0.html) && firefox --new-window $0.html; sleep 5; rm $0.html; exit
-->
# [Boost.]Multi BLAS Adaptor

(not an official Boost library)

_© Alfredo A. Correa, 2018-2021_

The BLAS Adaptor provides an interface for BLAS-like libraries.

## Contents
[[_TOC_]]

## Numeric Arrays, Conjugation Real and Imaginary

```cpp
	using complex = std::complex<double>; 
	complex const I{0, 1};
	multi::array<complex, 2> B = {
		{1. - 3.*I, 6. + 2.*I},
		{8. + 2.*I, 2. + 4.*I},
		{2. - 1.*I, 1. + 1.*I}
	};

	namespace blas = multi::blas;

	assert( blas::conj(B)[2][1] == std::conj(B[2][1]) );
	assert( blas::hermitized(B)[2][1] == blas::conj(B)[1][2] );
	assert( blas::transposed(B)[1][2] == B[2][1] );
	assert( blas::real(B)[2][1] == std::real(B[2][1]) );
	assert( blas::imag(B)[2][1] == std::imag(B[2][1]) );
	
	multi::array<double, 2> B_real_doubled = {
		{ 1., -3., 6., 2.},
		{ 8.,  2., 2., 4.},
		{ 2., -1., 1., 1.}
	};
	assert( blas::real_doubled(B) == B_real_doubled );
```

## Installation and Tests

`Multi` doesn't require instalation, single file `#include<multi/array.hpp>` is enough to use the full core library.
`Multi`'s _only_ dependecy is the standard C++ library.

It is important to compile programs that use the library with a decent level of optimization (e.g. `-O2`) to avoid slowdown if indiviudual element-access is intensively used.
For example, when testing speed, please make sure that you are compiling in release mode (`-DNDEBUG`) and with optimizations (`-O3`), 
if your test involves mathematical operations add arithmetic optimizations (`-Ofast`) to compare with Fortran code.

