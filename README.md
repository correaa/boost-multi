<!--
(pandoc `#--from gfm` --to html --standalone --metadata title=" " $0 > $0.html) && firefox --new-window $0.html; sleep 5; rm $0.html; exit
-->

**[Boost.] Multi**

> **Disclosure: This is not an official or accepted Boost library and is unrelated to the std::mdspan proposal. It is in the process of being proposed for inclusion in [Boost](https://www.boost.org/) and it doesn't depend on Boost libraries.**

_© Alfredo A. Correa, 2018-2025_

_Multi_ is a modern C++ library that provides manipulation and access of data in multidimensional arrays for both CPU and GPU memory.

# [Introduction](doc/multi/intro.adoc)

**Contents:**

[[_TOC_]]

# [Installation and tests](doc/multi/install.adoc)

# [Primer (basic usage)](doc/multi/primer.adoc)

# [Tutorial (advanced usage)](doc/multi/tutorial.adoc)

# Type Requirements

The library design tries to impose the minimum possible requirements over the types that parameterize the arrays.
Array operations assume that the contained type (element type) are regular (i.e. different element represent disjoint entities that behave like values).
Pointer-like random access types can be used as substitutes of built-in pointers.
(Therefore pointers to special memory and fancy-pointers are supported.)

## Linear Sequences: Pointers

An `array_ref` can reference an arbitrary random access linear sequence (e.g. memory block defined by pointer and size).
This way, any linear sequence (e.g. `raw memory`, `std::vector`, `std::queue`) can be efficiently arranged as a multidimensional array.

```cpp
std::vector<double> buffer(100);
multi::array_ref<double, 2> A({10, 10}, buffer.data());
A[1][1] = 9.0;

assert( buffer[11] == 9.0 );  // the target memory is affected
```
Since `array_ref` does not manage the memory associated with it, the reference can be simply dangle if the `buffer` memory is reallocated (e.g. by vector-`resize` in this case).

## Special Memory: Pointers and Views

`array`s manage their memory behind the scenes through allocators, which can be specified at construction.
It can handle special memory, as long as the underlying types behave coherently, these include [fancy pointers](https://en.cppreference.com/w/cpp/named_req/Allocator#Fancy_pointers) (and fancy references).
Associated fancy pointers and fancy reference (if any) are deduced from the allocator types.

### Allocators and Fancy Pointers

Specific uses of fancy memory are file-mapped memory or interprocess shared memory.
This example illustrates memory persistency by combining with Boost.Interprocess library. 
The arrays support their allocators and fancy pointers (`boost::interprocess::offset_ptr`).

```cpp
#include <boost/interprocess/managed_mapped_file.hpp>
using namespace boost::interprocess;
using manager = managed_mapped_file;
template<class T> using mallocator = allocator<T, manager::segment_manager>;
decltype(auto) get_allocator(manager& m) {return m.get_segment_manager();}

template<class T, auto D> using marray = multi::array<T, D, mallocator<T>>;

int main() {
{
	manager m{create_only, "mapped_file.bin", 1 << 25};
	auto&& arr2d = *m.construct<marray<double, 2>>("arr2d")(marray<double, 2>::extensions_type{1000, 1000}, 0.0, get_allocator(m));
	arr2d[4][5] = 45.001;
}
// imagine execution restarts here, the file "mapped_file.bin" persists
{
	manager m{open_only, "mapped_file.bin"};
	auto&& arr2d = *m.find<marray<double, 2>>("arr2d").first;
	assert( arr2d[7][8] == 0. );
	assert( arr2d[4][5] == 45.001 );
	m.destroy<marray<double, 2>>("arr2d");
}
}
```
([live](https://godbolt.org/z/oeTss3s35))

(See also, examples of interactions with the CUDA Thrust library to see more uses of special pointer types to handle special memory.)

### Transformed views

Another kind of use of the internal pointer-like type is to transform underlying values.
These are useful to create "projections" or "views" of data elements.
In the following example a "transforming pointer" is used to create a conjugated view of the elements.
In combination with a transposed view, it can create a hermitic (transposed-conjugate) view of the matrix (without copying elements).
We can adapt the library type `boost::transform_iterator` to save coding, but other libraries can be used also.
The hermitized view is read-only, but with additional work, a read-write view can be created (see `multi::::hermitized` in multi-adaptors).

```cpp
constexpr auto conj = [](auto const& c) {return std::conj(c);};

template<class T> struct conjr : boost::transform_iterator<decltype(conj), T*> {
	template<class... As> conjr(As const&... as) : boost::transform_iterator<decltype(conj), T*>{as...} {}
};

template<class Array2D, class Complex = typename Array2D::element_type>
auto hermitized(Array2D const& arr) {
	return arr
		.transposed() // lazily tranposes the array
		.template static_array_cast<Complex, conjr<Complex>>(conj)  // lazy conjugate elements
	;
}

int main() {
	using namespace std::complex_literals;
	multi::array A = {
		{ 1.0 + 2.0i,  3.0 +  4.0i},
		{ 8.0 + 9.0i, 10.0 + 11.0i}
	};

	auto const& Ah = hermitized(A);

	assert( Ah[1][0] == std::conj(A[0][1]) );
}
```

To simplify this boilerplate, the library provides the `.element_transformed(F)` method that will apply a transformation `F` to each element of the array.
In this example, the original array is transformed into a transposed array with duplicated elements.

```cpp
	multi::array<double, 2> A = {
		{1.0, 2.0},
		{3.0, 4.0},
	};

	auto const scale = [](auto x) { return x * 2.0; };

	auto B = + A.transposed().element_transformed(scale);
	assert( B[1][0] == A[0][1] * 2 );
```

([live](https://godbolt.org/z/TYavYEG1T))

Since `element_transformed` is a reference-like object (transformed view) to the original data, it is important to understand the semantics of evaluation and possible allocations incurred.
As mentioned in other sections using `auto` and/or `+` appropriately can lead to simple and efficient expressions.

| Construction    | Allocation of `T`s | Initialization (of `T`s) | Evaluation (of `fun`) | Notes |
| -------- | ------- | ------- | ------- | ------- |
| `multi::array<T, D> const B = A.element_transformed(fun);` | Yes        | No  | Yes | Implicit conversion to `T` if result is different, dimensions must match. B can be mutable.   |
| `multi::array<T, D> const B = + A.element_transformed(fun);` | Yes (and move, or might allocate twice if types don't match)  | No  | Yes | Not recommended | 
| `multi::array<T, D> const B{A.element_transformed(fun)};` | Yes        | No  | Yes | Explicit conversion to `T` if result is different, dimensions must match   |
| `auto const B = + A.elements_transformed(fun);`           | Yes         | No  | Yes | Types and dimension are deduced, result is contiguous, preferred |
| `auto const B = A.element_transformed(fun);`               | No         | No  | No (delayed) | Result is effective a reference, may dangle with `A`, usually `const`, not recommended   |
| `auto const& B = A.elements_transformed(fun);`           | No         | No  | No (delayed) | Result is effective a reference, may dangle with `A`. Preferred way.  |
| `multi::array<T, D> B(A.extensions()); B = A.element_transformed(fun);`           | Yes         | Yes (during construction)  | Yes | "Two-step" construction. `B` is mutable. Not recommended  |

| Assigment    | Allocation of `T`s | Initialization (of `T`s) | Evaluation (of `fun`) | Notes |
| -------- | ------- | ------- | ------- | ------- |
| `B = A.elements_transformed(fun);`           | No, if sizes match | Possibly (when `B` was initialized)  | Yes | `B` can't be declared `const`, it can be a writable subarray, preferred  |
| `B = + A.elements_transformed(fun);`           | Yes | Possibly (when `B` was initialized)  | Yes | Not recommended. |

# [Reference](doc/multi/reference.adoc)

# [Interoperability](doc/multi/interop.adoc)

# Technical points

## Indexing (square brackets vs. parenthesis?)

The chained bracket notation (`A[i][j][k]`) allows you to refer to elements and lower-dimensional subarrays consistently and generically, and it is the recommended way to access array objects.
It is a frequently raised question whether the chained bracket notation is beneficial for performance, as each use of the bracket leads to the creation of temporary objects, which in turn generates a partial copy of the layout.
Moreover, this goes against [historical recommendations](https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op).

It turns out that modern compilers with a fair level of optimization (`-O2`) can elide these temporary objects so that `A[i][j][k]` generates identical machine code as `A.base() + i*stride1 + j*stride2 + k*stride3` (+offsets not shown).
In a subsequent optimization, constant indices can have their "partial stride" computation removed from loops. 
As a result, these two loops lead to the [same machine code](https://godbolt.org/z/ncqrjnMvo):

```cpp
	// given the values of i and k and accumulating variable acc ...
    for(long j = 0; j != M; ++j) {acc += A[i][j][k];}
```
```cpp
    auto* base = A.base() + i*std::get<0>(A.strides()) + k*std::get<2>(A.strides());
    for(long j = 0; j != M; ++j) {acc += *(base + j*std::get<1>(A.strides()));}
```

Incidentally, the library also supports parenthesis notation with multiple indices `A(i, j, k)` for element or partial access;
it does so as part of a more general syntax to generate sub-blocks.
In any case, `A(i, j, k)` is expanded to `A[i][j][k]` internally in the library when `i`, `j`, and `k` are normal integer indices.
For this reason, `A(i, j, k)`, `A(i, j)(k)`, `A(i)(j)(k)`, `A[i](j)[k]` are examples of equivalent expressions.

Sub-block notation, when at least one argument is an index range, e.g., `A({i0, i1}, j, k)` has no equivalent square-bracket notation.
Note also that `A({i0, i1}, j, k)` is not equivalent to `A({i0, i1})(j, k)`; their resulting sublocks have different dimensionality.

Additionally, array coordinates can be directly stored in tuple-like data structures, allowing this functional syntax:

```cpp
std::array<int, 3> p = {2, 3, 4};
std::apply(A, p) = 234;  // same as assignment A(2, 3, 4) = 234; and same as A[2][3][4] = 234;
```

## Iteration past-end in the abstract machine

It's crucial to grasp that pointers are limited to referencing valid memory in the strict C abstract machine, such as allocated memory.
This understanding is key to avoiding undefined behavior in your code.
Since the library iteration is pointer-based, the iterators replicate these restrictions.

There are three cases to consider; the first two can be illustrated with one-dimensional arrays, and one is intrinsic to multiple dimensions.

The first case is that of strided views (e.g. `A.strided(n)`) whose stride value are not divisors of original array size.
The second case is that or negative strides in general.
The third case is that of iterators of transposed array.

In all these cases, the `.end()` iterator may point to invalid memory. 
It's important to note that the act of constructing certain iterators, even if the element is never dereferenced, is undefined in the abstract machine.
This underscores the need for caution when using such operations in your code.

A thorough description of the cases and workaround is beyond the scope of this section.

# Appendix: Comparison to other array libraries (mdspan, Boost.MultiArray, etc)

The C++23 standard provides `std::mdspan`, a non-owning _multidimensional_ array.
So here is an appropriate point to compare the two libraries.
Although the goals are similar, the two libraries differ in their generality and approach.

The Multi library concentrates on _well-defined value- and reference-semantics of arbitrary memory types with regularly arranged elements_ (distributions described by strides and offsets) and _extreme compatibility with STL algorithms_ (via iterators) and other fundamental libraries.
While `mdspan` concentrates on _arbitrary layouts_ for non-owning memory of a single type (CPU raw pointers).
Due to the priority of arbitrary layouts, the `mdspan` research team didn't find efficient ways to introduce iterators into the library. 
Therefore, its compatibility with the rest of the STL is lacking.
[Preliminarily](https://godbolt.org/z/aWW3vzfPj), Multi array can be converted (viewed as) `mdspan`.

[Boost.MultiArray](https://www.boost.org/doc/libs/1_82_0/libs/multi_array/doc/user.html) is the original multidimensional array library shipped with Boost.
This library can replace Boost.MultiArray in most contexts, it even fulfillis the concepts of `boost::multi_array_concepts::ConstMultiArrayConcept` and `...::MutableMultiArrayConcept`.
Boost.MultiArray has technical and semantic limitations that are overcome in this library, regarding layouts and references;
it doesn't support value-semantics, iterator support is limited and it has other technical problems.

[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is a very popular matrix linear algebra framework library, and as such, it only handles the special 2D (and 1D) array case.
Instead, the Multi library is dimension-generic and doesn't make any algebraic assumptions for arrays or contained elements (but still can be used to _implement_, or in combination, with dense linear algebra algorithms.)

Other frameworks includes the OpenCV (Open Computing Vision) framework, which is too specialized to make a comparison here.

Here is a table comparing with `mdspan`, R. Garcia's [Boost.MultiArray](https://www.boost.org/doc/libs/1_82_0/libs/multi_array/doc/user.html) and Eigen. 
[(online)](https://godbolt.org/z/555893MqW).


|                             | Multi                                                           | mdspan/mdarray                                                                          | Boost.MultiArray (R. Garcia)                                                                         | Inria's Eigen                                                                           |
|---                          | ---                                                             | ---                                                                             | ---                                                                                                  | ---                                                                                     |
| No external Deps            | **yes** (only Standard Library C++17)                           | **yes** (only Standard Library C++17/C++26)                                                 | **yes** (only Boost)                                                                                 | **yes**                                                                                 |
| Arbritary number of dims    | **yes**, via positive dimension (compile-time) parameter `D`    | **yes**                                                                         | **yes**                                                                                              | no  (only 1D and 2D)                                                                    |
| Non-owning view of data     | **yes**, via `multi::array_ref<T, D>(ptr, {n1, n2, ..., nD})`   | **yes**, via `mdspan m{T*, extents{n1, n2, ..., nD}};`                          | **yes**, via `boost::multi_array_ref<T, D>(T*, boost::extents[n1][n2]...[nD])` | **yes**, via `Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>(ptr, n1, n2)` |
| Compile-time dim size       | no                                                              | **yes**, via template paramaters `mdspan{T*, extent<16, dynamic_extents>{32} }` | no                                                                             | **yes**, via `Eigen::Array<T, N1, N2>` |
| Array values (owning data)  | **yes**, via `multi::array<T, D>({n1, n2, ..., nD})`            | yes for `mdarray`                                               | **yes**, via `boost::multi_array<T, D>(boost::extents[n1][n2]...[nD])` | **yes**, via `Eigen::Array<T>(n1, n2)` |
| Value semantic (Regular)    | **yes**, via cctor, mctor, assign, massign, auto decay of views | yes (?) for `mdarray` planned                                   | partial, assigment on equal extensions  | **yes** (?) |
| Move semantic               | **yes**, via mctor and massign                                  | yes (?) for `mdarray` (depends on adapted container)            | no (C++98 library)                      | **yes** (?) |
| const-propagation semantics | **yes**, via `const` or `const&`                                | no, const mdspan elements are assignable!                       | no, inconsistent                        | (?) |
| Element initialization      | **yes**, via nested init-list                                   | no (?)                                                          | no                                      | no, only delayed init via `A << v1, v2, ...;` |
| References w/no-rebinding   | **yes**, assignment is deep                                     | no, assignment of mdspan rebinds!                               | **yes**                                 | **yes** (?) |
| Element access              | **yes**, via `A(i, j, ...)` or `A[i][j]...`                     | **yes**, via `A[i, j, ...]`                                     | **yes**, via `A[i][j]...`               | **yes**, via `A(i, j)` (2D only) |
| Partial element access      | **yes**, via `A[i]` or `A(i, multi::all)`                       | no, only via `submdspan(A, i, full_extent)`                     | **yes**, via `A[i]`                     | **yes**, via `A.row(i)` |
| Subarray views              | **yes**, via `A({0, 2}, {1, 3})` or `A(1, {1, 3})`              | **yes**, via `submdspan(A, std::tuple{0, 2}, std::tuple{1, 3})` | **yes**, via `A[indices[range(0, 2)][range(1, 3)]]` | **yes**, via `A.block(i, j, di, dj)` |
| Subarray with lower dim     | **yes**, via `A(1, {1, 3})`                                     | **yes**, via `submdspan(A, 1, std::tuple{1, 3})`                | **yes**, via `A[1][indices[range(1, 3)]]`                    | **yes**, via `A(1, Eigen::placeholders::all)` |
| Subarray w/well def layout  | **yes** (strided layout)                                        | no                                                              | **yes** (strided layout)                      | **yes** (strided) |
| Recursive subarray          | **yes** (layout is stack-based and owned by the view)           | **yes** (?)                                                     | no (subarray may dangle layout, design bug?)  | **yes** (?) (1D only) |
| Custom Alloctors            | **yes**, via `multi::array<T, D, Alloc>`                        | yes(?) through `mdarray`'s adapted container                    | **yes** (stateless?)                          | no | 
| PMR Alloctors               | **yes**, via `multi::pmr::array<T, D>`                          | yes(?) through `mdarray`'s adapted container                    |   no                          | no |
| Fancy pointers / references | **yes**, via `multi::array<T, D, FancyAlloc>` or views          | no                                                              |   no                          | no |
| Stride-based Layout         | **yes**                                                         | **yes**                                                   |  **yes**                      | **yes** |
| Fortran-ordering            | **yes**, only for views, e.g. resulted from transposed views    | **yes**                                                   |  **yes**.                     | **yes** |
| Zig-zag / Hilbert ordering  | no                                                              | **yes**, via arbitrary layouts (no inverse or flattening) | no                            | no |
| Arbitrary layout            | no                                                              | **yes**, possibly inneficient, no efficient slicing       | no                            | no |
| Flattening of elements      | **yes**, via `A.elements()` range (efficient representation)    | **yes**, but via indices roundtrip (inefficient)          | no, only for allocated arrays | no, not for subblocks (?) |
| Iterators                   | **yes**, standard compliant, random-access-iterator             | no                                                        | **yes**, limited | no |
| Multidimensional iterators (cursors) | **yes** (experimental)                                 | no                                                        | no               | no |         
| STL algorithms or Ranges    | **yes**                                                         | no, limited via `std::cartesian_product`                  | **yes**, some do not work | no |
| Compatibility with Boost    | **yes**, serialization, interprocess  (see below)               | no                                                        | no | no |
| Compatibility with Thrust or GPUs | **yes**, via flatten views (loop fusion), thrust-pointers/-refs | no                                                  | no          | no |
| Used in production          | [QMCPACK](https://qmcpack.org/), [INQ](https://gitlab.com/npneq/inq)  | (?) , experience from Kokkos incarnation            | **yes** (?) | [**yes**](https://eigen.tuxfamily.org/index.php?title=Main_Page#Projects_using_Eigen) |

# Appendix: Multi for FORTRAN programmers

This section summarizes simple cases translated from FORTRAN syntax to C++ using the library.
The library strives to give a familiar feeling to those who use multidimensional arrays in FORTRAN.
Arrays can be indexed using square brackets or parenthesis, which would be more familiar to FORTRAN syntax.
The most significant differences are that array indices in FORTRAN start at `1`, and that index ranges are specified as closed intervals, while in Multi, they start by default at `0`, and ranges are half-open, following C++ conventions.
Like in FORTRAN, arrays are not initialized automatically for simple types (e.g., numeric); such initialization needs to be explicit.

|                             | FORTRAN                                          | C++ Multi                                            |
|---                          | ---                                              | ---                                                  |
| Declaration/Construction 1D | `real, dimension(2) :: numbers` (at top)         | `multi::array<double, 1> numbers(2);` (at scope)     |
| Initialization (2 elements) | `real, dimension(2) :: numbers = [ 1.0, 2.0 ]`   | `multi::array<double, 1> numbers = { 1.0, 2.0 };`    |
| Element assignment          | `numbers(2) = 99.0`                              | `numbers(1) = 99.0;` (or `numbers[1]`)               |
| Element access (print 2nd)  | `Print *, numbers(2)`                            | `std::cout << numbers(1) << '\n';`                   |
| Initialization              | `DATA numbers / 10.0 20.0 /`                     | `numbers = {10.0, 20.0};`                            |

In the more general case for the dimensionality, we have the following correspondance:

|                              | FORTRAN                                          | C++ Multi                                            |
|---                           | ---                                              | ---                                                  |
| Construction 2D (3 by 3)     | `real*8 :: A2D(3,3)` (at top)                    | `multi::array<double, 2> A2D({3, 3});` (at scope)    |
| Construction 2D (2 by 2)     | `real*8 :: B2D(2,2)` (at top)                    | `multi::array<double, 2> B2D({2, 2});` (at scope)    |
| Construction 1D (3 elements) | `real*8 :: v1D(3)`   (at top)                    | `multi::array<double, 2> v1D({3});` (at scope)       |
| Assign the 1st column of A2D | `v1D(:) = A2D(:,1)`                              | `v1( _ ) = A2D( _ , 0 );`                            | 
| Assign the 1st row of A2D    | `v1D(:) = A2D(1,:)`                              | `v1( _ ) = A2D( 0 , _ );`                            |
| Assign upper part of A2D     | `B2D(:,:) = A2D(1:2,1:2)`                        | `B2D( _::_ , _ ) = A2D({0, 2}, {0, 2});`                |

Note that these correspondences are notationally logical;
internal representation (memory ordering) can still be different, affecting operations that interpret 2D arrays as contiguous elements in memory.

Range notation such as `1:2` is replaced by `{0, 2}`, which considers both the difference in the start index and the half-open interval notation in the C++ conventions.
Stride notation such as `1:10:2` (i.e., from first to tenth included, every two elements) is replaced by `{0, 10, 2}`.
Complete range interval (single `:` notation) is replaced by `multi::_`, which can be used simply as `_` after the declaration `using multi::_;`.
These rules extend to higher dimensionality.

Unlike FORTRAN, Multi doesn't provide algebraic operators, using algorithms is encouraged instead.
For example, a FORTRAN statement like `A = A + B` is translated as this in the one-dimensional case:

```cpp
std::transform(A.begin(), A.end(), B.begin(), A.begin(), std::plus{});  // valid for 1D arrays only
```

In the general dimensionality case we can write:

```cpp
auto&&      Aelems = A.elements();
auto const& Belems = B.elements();
std::transform(Aelems.begin(), A.elems.end(), Belems.begin(), Aelems.begin(), std::plus<>{});  // valid for arbitrary dimension
```

or
```
std::ranges::transform(A.elements(), B.elements(), A.elements().begin(), std::plus<>{});  // alternative using C++20 ranges
```

A FORTRAN statement like `C = 2.0*C` is rewritten as `std::ranges::transform(C.elements(), C.elements().begin(), [](auto const& e) { return 2.0*e; });`.

It is possible to use C++ operator overloading for functions such as `operartor+=` (`A += B;`) or `operator*=` (`C *= 2.0;`);
however, this possibility can become unwindenly complicated beyond simple cases (also it can become inefficient if implemented naively).

Simple loops can be mapped as well, taking into account indexing differences:
```fortran
do i = 1, 5         ! for(int i = 0; i != 5; ++i) {
  do j = 1, 5       !   for(int j = 0; j != 5; ++j) {
    D2D(i, j) = 0   !     D2D(i, j) = 0;
  end do            !   }
end do              ! }
```
[(live)](https://godbolt.org/z/77onne46W)

However, algorithms like `transform`, `reduce`, `transform_reduce`and `for_each`, and offer a higher degree of control over operations, including memory allocations if needed, and even enable parallelization, providing a higher level of flexibility.
In this case, `std::fill(D2D.elements().begin(), D2D.elements().end(), 0);` will do.

> **Thanks** to Joaquín López Muñoz and Andrzej Krzemienski for the critical reading of the documentation and to Matt Borland for his help integrating Boost practices in the testing code.
ray
arow_copy = + a2d[0]
arow_copy = 11111.1
print(a2d[0])
```
> ```python
> { 66.600000, 2.0000000 }
> ```

## Legacy libraries (C-APIs)

Multi-dimensional array data structures exist in all languages, whether implicitly defined by its strides structure or explicitly at the language level.
Functions written in C tend to receive arrays by pointer arguments (e.g., to the "first" element) and memory layout (sizes and strides).

A C-function taking a 2D array with a concrete type might look like this in the general case:
```c
void fun(double* data, int size1, int size2, int stride1, int stride2);
```
such a function can be called from C++ on Multi array (`arr`), by extracting the size and layout information,
```cpp
fun(arr.base(), std::get<0>(arr.sizes()), std::get<1>(arr.sizes()), std::get<0>(arr.strides()), std::get<1>(arr.strides());
```
or
```cpp
auto const [size1, size2] = arr.sizes();
auto const [stride1, stride2] = arr.strides();

fun(arr.base(), size1, size2, stride1, stride2);
```

Although the recipe can be applied straightforwardly, different libraries make various assumptions about memory layouts (e.g.,  2D arrays assume that the second stride is 1), and some might take stride information in a different way (e.g., FFTW doesn't use strides but stride products).
Furthermore, some arguments may need to be permuted if the function expects arrays in column-major (Fortran) ordering.

For these reasons, the library is accompanied by a series of adaptor libraries to popular C-based libraries, which can be found in the `include/multi/adaptors/` subdirectory:

- ##### [BLAS/cuBLAS Adator 🔗](include/boost/multi/adaptors/blas/README.md)

Interface for BLAS-like linear algebra libraries, such as openblas, Apple's Accelerate, MKL and hipBLAS/cuBLAS (GPUs).
Simply `#include "multi/adaptors/blas.hpp"` (and link your program with `-lblas` for example).

- ##### Lapack

Interface for Lapack linear solver libraries.
Simply `#include "multi/adaptors/lapack.hpp"` (and link your program with `-llapack` for example).

- ##### FFTW/cuFFT

Interface for FFTW libraries, including FFTW 3, MKL, cuFFT/hipFFT (for GPU).
Simply `#include "multi/adaptors/fftw.hpp"` (and link your program with `-lfftw3` for example).

- ##### [MPI Adaptor 🔗](include/boost/multi/adaptors/mpi/README.md)

Use arrays (and subarrays) as messages for distributed interprocess communication (GPU and CPU) that can be passed to MPI functions through datatypes.
Simply `#include "multi/adaptors/mpi.hpp"`.

- ##### TotalView: visual debugger (commercial)

Popular in HPC environments, can display arrays in human-readable form (for simple types, like `double` or `std::complex`).
Simply `#include "multi/adaptors/totalview.hpp"` and link to the TotalView libraries, compile and run the code with the TotalView debugger.

# Technical points

## Indexing (square brackets vs. parenthesis?)

The chained bracket notation (`A[i][j][k]`) allows you to refer to elements and lower-dimensional subarrays consistently and generically, and it is the recommended way to access array objects.
It is a frequently raised question whether the chained bracket notation is beneficial for performance, as each use of the bracket leads to the creation of temporary objects, which in turn generates a partial copy of the layout.
Moreover, this goes against [historical recommendations](https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op).

It turns out that modern compilers with a fair level of optimization (`-O2`) can elide these temporary objects so that `A[i][j][k]` generates identical machine code as `A.base() + i*stride1 + j*stride2 + k*stride3` (+offsets not shown).
In a subsequent optimization, constant indices can have their "partial stride" computation removed from loops. 
As a result, these two loops lead to the [same machine code](https://godbolt.org/z/ncqrjnMvo):

```cpp
	// given the values of i and k and accumulating variable acc ...
    for(long j = 0; j != M; ++j) {acc += A[i][j][k];}
```
```cpp
    auto* base = A.base() + i*std::get<0>(A.strides()) + k*std::get<2>(A.strides());
    for(long j = 0; j != M; ++j) {acc += *(base + j*std::get<1>(A.strides()));}
```

Incidentally, the library also supports parenthesis notation with multiple indices `A(i, j, k)` for element or partial access;
it does so as part of a more general syntax to generate sub-blocks.
In any case, `A(i, j, k)` is expanded to `A[i][j][k]` internally in the library when `i`, `j`, and `k` are normal integer indices.
For this reason, `A(i, j, k)`, `A(i, j)(k)`, `A(i)(j)(k)`, `A[i](j)[k]` are examples of equivalent expressions.

Sub-block notation, when at least one argument is an index range, e.g., `A({i0, i1}, j, k)` has no equivalent square-bracket notation.
Note also that `A({i0, i1}, j, k)` is not equivalent to `A({i0, i1})(j, k)`; their resulting sublocks have different dimensionality.

Additionally, array coordinates can be directly stored in tuple-like data structures, allowing this functional syntax:

```cpp
std::array<int, 3> p = {2, 3, 4};
std::apply(A, p) = 234;  // same as assignment A(2, 3, 4) = 234; and same as A[2][3][4] = 234;
```

## Iteration past-end in the abstract machine

It's crucial to grasp that pointers are limited to referencing valid memory in the strict C abstract machine, such as allocated memory.
This understanding is key to avoiding undefined behavior in your code.
Since the library iteration is pointer-based, the iterators replicate these restrictions.

There are three cases to consider; the first two can be illustrated with one-dimensional arrays, and one is intrinsic to multiple dimensions.

The first case is that of strided views (e.g. `A.strided(n)`) whose stride value are not divisors of original array size.
The second case is that or negative strides in general.
The third case is that of iterators of transposed array.

In all these cases, the `.end()` iterator may point to invalid memory. 
It's important to note that the act of constructing certain iterators, even if the element is never dereferenced, is undefined in the abstract machine.
This underscores the need for caution when using such operations in your code.

A thorough description of the cases and workaround is beyond the scope of this section.

# Appendix: Comparison to other array libraries (mdspan, Boost.MultiArray, etc)

The C++23 standard provides `std::mdspan`, a non-owning _multidimensional_ array.
So here is an appropriate point to compare the two libraries.
Although the goals are similar, the two libraries differ in their generality and approach.

The Multi library concentrates on _well-defined value- and reference-semantics of arbitrary memory types with regularly arranged elements_ (distributions described by strides and offsets) and _extreme compatibility with STL algorithms_ (via iterators) and other fundamental libraries.
While `mdspan` concentrates on _arbitrary layouts_ for non-owning memory of a single type (CPU raw pointers).
Due to the priority of arbitrary layouts, the `mdspan` research team didn't find efficient ways to introduce iterators into the library. 
Therefore, its compatibility with the rest of the STL is lacking.
[Preliminarily](https://godbolt.org/z/aWW3vzfPj), Multi array can be converted (viewed as) `mdspan`.

[Boost.MultiArray](https://www.boost.org/doc/libs/1_82_0/libs/multi_array/doc/user.html) is the original multidimensional array library shipped with Boost.
This library can replace Boost.MultiArray in most contexts, it even fulfillis the concepts of `boost::multi_array_concepts::ConstMultiArrayConcept` and `...::MutableMultiArrayConcept`.
Boost.MultiArray has technical and semantic limitations that are overcome in this library, regarding layouts and references;
it doesn't support value-semantics, iterator support is limited and it has other technical problems.

[Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) is a very popular matrix linear algebra framework library, and as such, it only handles the special 2D (and 1D) array case.
Instead, the Multi library is dimension-generic and doesn't make any algebraic assumptions for arrays or contained elements (but still can be used to _implement_, or in combination, with dense linear algebra algorithms.)

Other frameworks includes the OpenCV (Open Computing Vision) framework, which is too specialized to make a comparison here.

Here is a table comparing with `mdspan`, R. Garcia's [Boost.MultiArray](https://www.boost.org/doc/libs/1_82_0/libs/multi_array/doc/user.html) and Eigen. 
[(online)](https://godbolt.org/z/555893MqW).


|                             | Multi                                                           | mdspan/mdarray                                                                          | Boost.MultiArray (R. Garcia)                                                                         | Inria's Eigen                                                                           |
|---                          | ---                                                             | ---                                                                             | ---                                                                                                  | ---                                                                                     |
| No external Deps            | **yes** (only Standard Library C++17)                           | **yes** (only Standard Library C++17/C++26)                                                 | **yes** (only Boost)                                                                                 | **yes**                                                                                 |
| Arbritary number of dims    | **yes**, via positive dimension (compile-time) parameter `D`    | **yes**                                                                         | **yes**                                                                                              | no  (only 1D and 2D)                                                                    |
| Non-owning view of data     | **yes**, via `multi::array_ref<T, D>(ptr, {n1, n2, ..., nD})`   | **yes**, via `mdspan m{T*, extents{n1, n2, ..., nD}};`                          | **yes**, via `boost::multi_array_ref<T, D>(T*, boost::extents[n1][n2]...[nD])` | **yes**, via `Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>(ptr, n1, n2)` |
| Compile-time dim size       | no                                                              | **yes**, via template paramaters `mdspan{T*, extent<16, dynamic_extents>{32} }` | no                                                                             | **yes**, via `Eigen::Array<T, N1, N2>` |
| Array values (owning data)  | **yes**, via `multi::array<T, D>({n1, n2, ..., nD})`            | yes for `mdarray`                                               | **yes**, via `boost::multi_array<T, D>(boost::extents[n1][n2]...[nD])` | **yes**, via `Eigen::Array<T>(n1, n2)` |
| Value semantic (Regular)    | **yes**, via cctor, mctor, assign, massign, auto decay of views | yes (?) for `mdarray` planned                                   | partial, assigment on equal extensions  | **yes** (?) |
| Move semantic               | **yes**, via mctor and massign                                  | yes (?) for `mdarray` (depends on adapted container)            | no (C++98 library)                      | **yes** (?) |
| const-propagation semantics | **yes**, via `const` or `const&`                                | no, const mdspan elements are assignable!                       | no, inconsistent                        | (?) |
| Element initialization      | **yes**, via nested init-list                                   | no (?)                                                          | no                                      | no, only delayed init via `A << v1, v2, ...;` |
| References w/no-rebinding   | **yes**, assignment is deep                                     | no, assignment of mdspan rebinds!                               | **yes**                                 | **yes** (?) |
| Element access              | **yes**, via `A(i, j, ...)` or `A[i][j]...`                     | **yes**, via `A[i, j, ...]`                                     | **yes**, via `A[i][j]...`               | **yes**, via `A(i, j)` (2D only) |
| Partial element access      | **yes**, via `A[i]` or `A(i, multi::all)`                       | no, only via `submdspan(A, i, full_extent)`                     | **yes**, via `A[i]`                     | **yes**, via `A.row(i)` |
| Subarray views              | **yes**, via `A({0, 2}, {1, 3})` or `A(1, {1, 3})`              | **yes**, via `submdspan(A, std::tuple{0, 2}, std::tuple{1, 3})` | **yes**, via `A[indices[range(0, 2)][range(1, 3)]]` | **yes**, via `A.block(i, j, di, dj)` |
| Subarray with lower dim     | **yes**, via `A(1, {1, 3})`                                     | **yes**, via `submdspan(A, 1, std::tuple{1, 3})`                | **yes**, via `A[1][indices[range(1, 3)]]`                    | **yes**, via `A(1, Eigen::placeholders::all)` |
| Subarray w/well def layout  | **yes** (strided layout)                                        | no                                                              | **yes** (strided layout)                      | **yes** (strided) |
| Recursive subarray          | **yes** (layout is stack-based and owned by the view)           | **yes** (?)                                                     | no (subarray may dangle layout, design bug?)  | **yes** (?) (1D only) |
| Custom Alloctors            | **yes**, via `multi::array<T, D, Alloc>`                        | yes(?) through `mdarray`'s adapted container                    | **yes** (stateless?)                          | no | 
| PMR Alloctors               | **yes**, via `multi::pmr::array<T, D>`                          | yes(?) through `mdarray`'s adapted container                    |   no                          | no |
| Fancy pointers / references | **yes**, via `multi::array<T, D, FancyAlloc>` or views          | no                                                              |   no                          | no |
| Stride-based Layout         | **yes**                                                         | **yes**                                                   |  **yes**                      | **yes** |
| Fortran-ordering            | **yes**, only for views, e.g. resulted from transposed views    | **yes**                                                   |  **yes**.                     | **yes** |
| Zig-zag / Hilbert ordering  | no                                                              | **yes**, via arbitrary layouts (no inverse or flattening) | no                            | no |
| Arbitrary layout            | no                                                              | **yes**, possibly inneficient, no efficient slicing       | no                            | no |
| Flattening of elements      | **yes**, via `A.elements()` range (efficient representation)    | **yes**, but via indices roundtrip (inefficient)          | no, only for allocated arrays | no, not for subblocks (?) |
| Iterators                   | **yes**, standard compliant, random-access-iterator             | no                                                        | **yes**, limited | no |
| Multidimensional iterators (cursors) | **yes** (experimental)                                 | no                                                        | no               | no |         
| STL algorithms or Ranges    | **yes**                                                         | no, limited via `std::cartesian_product`                  | **yes**, some do not work | no |
| Compatibility with Boost    | **yes**, serialization, interprocess  (see below)               | no                                                        | no | no |
| Compatibility with Thrust or GPUs | **yes**, via flatten views (loop fusion), thrust-pointers/-refs | no                                                  | no          | no |
| Used in production          | [QMCPACK](https://qmcpack.org/), [INQ](https://gitlab.com/npneq/inq)  | (?) , experience from Kokkos incarnation            | **yes** (?) | [**yes**](https://eigen.tuxfamily.org/index.php?title=Main_Page#Projects_using_Eigen) |

# Appendix: Multi for FORTRAN programmers

This section summarizes simple cases translated from FORTRAN syntax to C++ using the library.
The library strives to give a familiar feeling to those who use multidimensional arrays in FORTRAN.
Arrays can be indexed using square brackets or parenthesis, which would be more familiar to FORTRAN syntax.
The most significant differences are that array indices in FORTRAN start at `1`, and that index ranges are specified as closed intervals, while in Multi, they start by default at `0`, and ranges are half-open, following C++ conventions.
Like in FORTRAN, arrays are not initialized automatically for simple types (e.g., numeric); such initialization needs to be explicit.

|                             | FORTRAN                                          | C++ Multi                                            |
|---                          | ---                                              | ---                                                  |
| Declaration/Construction 1D | `real, dimension(2) :: numbers` (at top)         | `multi::array<double, 1> numbers(2);` (at scope)     |
| Initialization (2 elements) | `real, dimension(2) :: numbers = [ 1.0, 2.0 ]`   | `multi::array<double, 1> numbers = { 1.0, 2.0 };`    |
| Element assignment          | `numbers(2) = 99.0`                              | `numbers(1) = 99.0;` (or `numbers[1]`)               |
| Element access (print 2nd)  | `Print *, numbers(2)`                            | `std::cout << numbers(1) << '\n';`                   |
| Initialization              | `DATA numbers / 10.0 20.0 /`                     | `numbers = {10.0, 20.0};`                            |

In the more general case for the dimensionality, we have the following correspondance:

|                              | FORTRAN                                          | C++ Multi                                            |
|---                           | ---                                              | ---                                                  |
| Construction 2D (3 by 3)     | `real*8 :: A2D(3,3)` (at top)                    | `multi::array<double, 2> A2D({3, 3});` (at scope)    |
| Construction 2D (2 by 2)     | `real*8 :: B2D(2,2)` (at top)                    | `multi::array<double, 2> B2D({2, 2});` (at scope)    |
| Construction 1D (3 elements) | `real*8 :: v1D(3)`   (at top)                    | `multi::array<double, 2> v1D({3});` (at scope)       |
| Assign the 1st column of A2D | `v1D(:) = A2D(:,1)`                              | `v1( _ ) = A2D( _ , 0 );`                            | 
| Assign the 1st row of A2D    | `v1D(:) = A2D(1,:)`                              | `v1( _ ) = A2D( 0 , _ );`                            |
| Assign upper part of A2D     | `B2D(:,:) = A2D(1:2,1:2)`                        | `B2D( _::_ , _ ) = A2D({0, 2}, {0, 2});`                |

Note that these correspondences are notationally logical;
internal representation (memory ordering) can still be different, affecting operations that interpret 2D arrays as contiguous elements in memory.

Range notation such as `1:2` is replaced by `{0, 2}`, which considers both the difference in the start index and the half-open interval notation in the C++ conventions.
Stride notation such as `1:10:2` (i.e., from first to tenth included, every two elements) is replaced by `{0, 10, 2}`.
Complete range interval (single `:` notation) is replaced by `multi::_`, which can be used simply as `_` after the declaration `using multi::_;`.
These rules extend to higher dimensionality.

Unlike FORTRAN, Multi doesn't provide algebraic operators, using algorithms is encouraged instead.
For example, a FORTRAN statement like `A = A + B` is translated as this in the one-dimensional case:

```cpp
std::transform(A.begin(), A.end(), B.begin(), A.begin(), std::plus{});  // valid for 1D arrays only
```

In the general dimensionality case we can write:

```cpp
auto&&      Aelems = A.elements();
auto const& Belems = B.elements();
std::transform(Aelems.begin(), A.elems.end(), Belems.begin(), Aelems.begin(), std::plus<>{});  // valid for arbitrary dimension
```

or
```
std::ranges::transform(A.elements(), B.elements(), A.elements().begin(), std::plus<>{});  // alternative using C++20 ranges
```

A FORTRAN statement like `C = 2.0*C` is rewritten as `std::ranges::transform(C.elements(), C.elements().begin(), [](auto const& e) { return 2.0*e; });`.

It is possible to use C++ operator overloading for functions such as `operartor+=` (`A += B;`) or `operator*=` (`C *= 2.0;`);
however, this possibility can become unwindenly complicated beyond simple cases (also it can become inefficient if implemented naively).

Simple loops can be mapped as well, taking into account indexing differences:
```fortran
do i = 1, 5         ! for(int i = 0; i != 5; ++i) {
  do j = 1, 5       !   for(int j = 0; j != 5; ++j) {
    D2D(i, j) = 0   !     D2D(i, j) = 0;
  end do            !   }
end do              ! }
```
[(live)](https://godbolt.org/z/77onne46W)

However, algorithms like `transform`, `reduce`, `transform_reduce`and `for_each`, and offer a higher degree of control over operations, including memory allocations if needed, and even enable parallelization, providing a higher level of flexibility.
In this case, `std::fill(D2D.elements().begin(), D2D.elements().end(), 0);` will do.

> **Thanks** to Joaquín López Muñoz and Andrzej Krzemienski for the critical reading of the documentation and to Matt Borland for his help integrating Boost practices in the testing code.
