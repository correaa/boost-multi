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

# Fundamental types and concepts

The library interface presents several closely related C++ types (classes) representing arrays.
The fundamental types represent multidimensional containers (called `array`), references that can refer to subsets of these containers (called `subarray`), and iterators.
In addition, there are other classes for advanced uses, such as multidimensional views of existing buffers (called `array_ref`) and non-resizable owning containers (called `static_array`).

When using the library, it is simpler to start from `array`, and other types are rarely explicitly used, especially if using `auto`;
however, it is convenient for documentation to present the classes in a different order since the classes `subarray`, `array_ref`, `static_array`, and `array` have an *is-a* relationship (from left to right). 
For example, `array_ref` has all the methods available to `subarray`, and `array` has all the operations of `array_ref`.
Furthermore, the *is-a* relationship is implemented through C++ public inheritance, so, for example, a reference of type `subarray<T, D>&` can refer to a variable of type `array<T, D>`.

<details><summary>class <code>multi::subarray&lt;T, D, P = T* &gt;</code></summary>

A subarray-reference is part (or a whole) of another larger array.
It is important to understand that `subarray`s have referential semantics, their elements are not independent of the values of the larger arrays they are part of.
An instance of this class represents a subarray with elements of type `T` and dimensionality `D`, stored in memory described by the pointer type `P`.
(`T`, `D`, and `P` initials are used in this sense across the documentation.)

Instances of this class have reference semantics and behave like "language references" as much as possible.
As references, they cannot be rebinded or resized; assignments are always "deep".
They are characterized by a size that does not change in the lifetime of the reference.
They are usually the result of indexing over other `multi::subarray`s and `multi::array`s objects, typically of higher dimensions;
therefore, the library doesn't expose constructors for this class.
The whole object can be invalidated if the original array is destroyed.

<details><summary>Member types</summary>

| `subarray::`      | Description               |
|---                |---                        |
| `value_type`      | `multi::array<T, D - 1, P >` or, for `D == 1`, `iterator_traits<P>::value_type` (usually `T`)   
| `reference`       | `multi::subarray<T, D-1, P >` or, for `D == 1`, `pointer_traits<P>::reference` (usually `T&`) 
| `const_reference` | `multi::const_subarray<T, D-1, P >` or, for `D == 1`, `pointer_traits<P>::rebind<T const>::reference` (usually `T const&`)
| `index`           | indexing type in the leading dimension (usually `std::diffptr_t`)
| `size_type`       | describe size (number of subarrays) in the leading dimension (signed version of pointer size type, usually `std::diffptr_t`)
| `index_range`     | describe ranges of indices, constructible from braced indices types or from an `extension_type`. Can be continuous (e.g. `{2, 14}`) or strided (e.g. `{2, 14, /*every*/ 3}`)
| `extesion_type`   | describe a contiguous range of indices, constructible from braced index (e.g. `{0, 10}`) or from a single integer size (e.g. 10, equivalent to `{0, 10}`). 
| `difference_type` | describe index differences in leading dimension (signed version of pointer size type, usually `std::diffptr_t`)
| `pointer`         | `multi::subarray_ptr<T, D-1, P > or, for `D == 1, `P` (usually `T*`)
| `const_pointer`   | `multi::const_subarray_ptr<T, D-1, P >` or, for `D == 1, `pointer_traits<P>::rebind<T const>` (usually `T const*`)
| `iterator`        | `multi::array_iterator_t<T, D-1, P >`
| `const_iterator`  | `multi::const_array_iterator_t<T, D-1, P >`

</details>

<details><summary>Special member functions</summary>

| `subarray::`      | Description |
|---                |--- |
| (constructors)    | not exposed; copy constructor is not available since the instances are not copyable; destructors are trivial since it doesn't own the elements |
| `operator=`       | assigns the elements from the source; the sizes must match |

It is important to note that assignments in this library are always "deep," and reference-like types cannot be rebound after construction.
(Reference-like types have corresponding pointer-like types that provide an extra level of indirection and can be rebound (just like language pointers);
these types are `multi::array_ptr` and `multi::subarray_ptr` corresponding to `multi::array_ref` and `multi::subarray` respectively.)

</details>

<details><summary>Relational functions</summary>

| Relational fuctions       |    |
|---                        |--- |
| `operator==`/`operator!=` | Tells if elements of two `subarray` are equal (and if extensions of the subarrays are the same)
| `operator<`/`operator<=`  | Less-than/less-or-equal      lexicographical comparison (requires elements to be comparable)
| `operator>`/`operator>=`  | Greater-than/grater-or-equal lexicographical comparison (requires elements to be comparable)

It is important to note that, in this library, comparisons are always "deep".
Lexicographical order is defined recursively, starting from the first dimension index and from left to right.
For example, `A < B` if `A[0] < B[0]`, or `A[0] == B[0]` and `A[1] < B[1]`, or ..., etc.
Lexicographical order applies naturally if the extensions of `A` and `B` are different; however, their dimensionalities must match.
(See sort examples).

</details>

<details><summary>Shape access</summary>

| Shape             |    |
|---                |--- |
| `sizes`           | returns a tuple with the sizes in each dimension
| `extensions`      | returns a tuple with the extensions in each dimension
| `size`            | returns the number of subarrays contained in the first dimension |
| `extension`       | returns a contiguous index range describing the set of valid indices
| `num_elements`    | returns the total number of elements

</details>

<details><summary>Element access</summary>

| Element access    |    |
|---                |--- |
|`operator[]`       | access specified element by index (single argument), returns a `reference` (see above), for `D > 1` it can be used recursively |
|`front`            | access first element (undefined result if array is empty). Takes no argument.
|`back`             | access last element  (undefined result if array is empty). Takes no argument.
|`operator()`       | When used with zero arguments, it returns a `subarray` representing the whole array. When used with one argument, access a specified element by index (return a `reference`) or by range (return a `subarray` of equal dimension). For more than one, arguments are positional and reproduce expected array access syntax from Fortran or Matlab: |

- `subarray::operator()(i, j, k, ...)`, as in `S(i, j, k)` for indices `i`, `j`, `k` is a synonym for `A[i][j][k]`, the number of indices can be lower than the total dimension (e.g., `S` can be 4D).
Each index argument lowers the dimension by one.
- `subarray::operator()(ii, jj, kk)`, the arguments can be indices or ranges of indices (`index_range` member type).
This function allows positional-aware ranges.
Each index argument lowers the rank by one.
A special range is given by `multi::_`, which means "the whole range" (also spelled `multi::all`).
For example, if `S` is a 3D of sizes 10-by-10-by-10, `S(3, {2, 8}, {3, 5})` gives a reference to a 2D array where the first index is fixed at 3, with sizes 6-by-2 referring the subblock in the second and third dimension.
Note that `S(3, {2, 8}, {3, 5})` (6-by-2) is not equivalent to `S[3]({2, 8})({3, 5})` (2-by-10).
- `operator()()` (no arguments) gives the same array but always as a subarray type (for consistency), `S()` is equivalent to `S(S.extension())` and, in turn to `S(multi::_)` or `S(multi::all)`.

</details>

<details><summary>Structure access</summary>

| Structure access  | (Generally used for interfacing with C-libraries)   |
|---                |--- |
| `base`            | direct access to underlying memory pointer (`S[i][j]... == S.base() + std::get<0>(S.strides())*i + std::get<1>(S.strides())*j + ...`)
| `stride`          | return the stride value of the leading dimension, e.g `(&A[1][0][0]... - &A[0][0]...)`
| `strides`         | returns a tuple with the strides defining the internal layout
| `layout`          | returns a single layout object with stride and size information |

</details>

<details><summary>Iterators</summary>

| Iterators         |    |
|---                |--- |
| `begin/cbegin`    | returns (const) iterator to the beginning
| `end/cend`        | returns (const) iterator to the end

</details>

<details><summary>Subarray/array generators</summary>

| Subarray generators   | (these operations do not copy elements or allocate)    |
|---                    |---  |
| `broadcasted`         | returns a view of dimensionality `D + 1` obtained by infinite repetition of the original array. (This returns a special kind of subarray with a degenerate layout and no size operation. Takes no argument.)
| `dropped`             | (takes one integer argument `n`) returns a subarray with the first n-elements (in the first dimension) dropped from the original subarray. This doesn't remove or destroy elements or resize the original array 
| `element_transformed` | creates a view of the array, where each element is transformed according to a function (first and only argument) |
| `elements`            | a flatted view of all the elements rearranged canonically. `A.elements()[0] -> A[0][0]`, `A.elements()[1] -> A[0][1]`, etc. The type of the result is not a subarray but a special kind of range. Takes no argument.
| `rotated/unrotated`   | a view (`subarray`) of the original array with indices (un)rotated from right to left (left to right), for `D = 1` returns the same `subarray`. For given `i`, `j`, `k`, `A[i][j][k]` gives the same element as `A.rotated()[j][k][i]` and, in turn the same as `A.unrotated()[k][i][j])`. Preserves dimension. The function is cyclic; `D` applications will give the original view. Takes no argument. |
| `transposed` (same as `operator~`) | a view (`subarray`) of the original array with the first two indices exchanged, only available for `D > 1`; for `D = 2`, `rotated`, `unrotated` and `transposed` give same view. Takes no argument.  |
| `sliced`              | (takes two index arguments `a` and `b`) returns a subarray with elements from index `a` to index `b` (non-inclusive) `{S[a], ... S[b-1]}`. Preserves the dimension.
| `strided`             | (takes one integer argument `s`) returns a subarray skipping `s` elements. Preserves the dimension.

| Creating views by pointer manipulation     |     |
|---                                         |---  |
| `static_array_cast<T2, P2 = T2*>(args...)` | produces a view where the underlying pointer constructed by `P2{A.base(), args...}`. Usually, `args...` is empty. Non-empty arguments are useful for stateful fancy pointers, such as transformer iterators.
| `reinterpret_cast_array<T2>`               | underlying elements are reinterpreted as type T2, element sizes (`sizeof`) have to be equal; `reinterpret_cast_array<T2>(n)` produces a view where the underlying elements are interpreted as an array of `n` elements of type `T2`.

| Creating arrays                     |     |
|---                                  |---  |
| `decay` (same as prefix unary `operator+`) | creates a concrete independent `array` with the same dimension and elements as the view. Usually used to force a value type (and forcing a copy of the elements) and avoid the propagation of a reference type in combination with `auto` (e.g., `auto A2_copy = + A[2];`).

A reference `subarray` can be invalidated when its origin array is invalidated or destroyed.
For example, if the `array` from which it originates is destroyed or resized.

</details>
</details>

<details><summary>class <code>multi::array_ref&lt;T, D, P = T* &gt;</code></summary>

A _D_-dimensional view of the contiguous pre-existing memory buffer.
This class doesn't manage the elements it contains, and it has reference semantics (it can't be rebound, assignments are deep, and have the same size restrictions as `subarray`)

Since `array_ref` is-a `subarray`, it inherits all the class methods and types described before and, in addition, it defines these members below.

| Member types      | same as for `subarray` |
|---                |---                        |

| Member functions  | same as for `subarray` plus ... |
|---                |--- |
| (constructors)    | `array_ref::array_ref({e1, e2, ...}, p)` constructs a D-dimensional view of the contiguous range starting at p and ending at least after the size size of the multidimensional array (product of sizes). The default constructor and copy constructor are not exposed. Destructor is trivial since elements are not owned or managed. |

| Element access    | same as for `subarray` |
|---                |--- |

| Structure access  | same as for `subarray` |
|---                |--- |

| Iterators         | same as for `subarray`   |
|---                |--- |

| Capacity          | same as for `subarray`   |
|---                |--- |

| Creating views    | same as for `subarray`  |
|---                |---  |

| Creating arrays   | same as for `subarray`  |
|---                |---  |

| Relational functions   |  same as for `subarray`  |
|---                |--- |

An `array_ref` can be invalidated if the original buffer is deallocated.

</details>

<details><summary>class <code>multi::static_array&lt;T, D, Alloc = std::allocator<T> &gt;</code></summary>

A _D_-dimensional array that manages an internal memory buffer.
This class owns the elements it contains; it has restricted value semantics because assignments are restricted to sources with equal sizes.
Memory is requested by an allocator of type Alloc (standard allocator by default).
It supports stateful and polymorphic allocators, which are the default for the special type `multi::pmr::static_array`.

The main feature of this class is that its iterators, subarrays, and pointers do not get invalidated unless the whole object is destroyed.
In this sense, it is semantically similar to a C-array, except that elements are allocated from the heap.
It can be useful for scoped uses of arrays and multi-threaded programming and to ensure that assignments do not incur allocations.
The C++ coreguiles proposed a similar (albeith one-dimensional) class, called [`gsl::dyn_array`](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#gslowner-ownership-pointers).

For most uses, a `multi::array` should be preferred instead.

| Member types      | same as for `array_ref` |
|---                |---                        |

| Member fuctions   | same as for `array_ref` plus ... |
|---                |--- |
| (constructors)    | `static_array::static_array({e1, e2, ...}, T val = {}, Alloc = {})` constructs a D-dimensional array by allocating elements. `static_array::static_array(std::initializer_list<...>` constructs the array with elements initialized from a nested list.
| (destructor)      | Destructor deallocates memory and destroy the elements |
| `operator=`       | assigns the elements from the source, sizes must match.

| Element access    | same as for `array_ref` |
|---                |--- |

| Structure access  | same as for `array_ref` |
|---                |--- |

| Iterators         | same as for `array_ref`   |
|---                |--- |

| Capacity          | same as for `array_ref`   |
|---                |--- |

| Creating views    | same as for `array_ref`  |
|---                |---  |

| Creating arrays   | same as for `array_ref`  |
|---                |---  |

| Relational fuctions   |  same as for `array_ref`  |
|---                |--- |

</details>

<details><summary>class <code>multi::array&lt;T, D, Alloc = std::allocator<T> &gt;</code></summary>

An array of integer positive dimension D has value semantics if element type T has value semantics.
It supports stateful and polymorphic allocators, which is implied for the special type `multi::pmr::array<T, D>`.

| Member types      | same as for `static_array` (see above) |
|---                |---                         |

| Member fuctions   |    |
|---                |--- |
| (constructors)    | `array::array({e1, e2, ...}, T val = {}, Alloc = {})` constructs a D-dimensional array by allocating elements;`array::array(It first, It last)` and `array::array(Range const& rng)`, same for a range of subarrays. `static_array::static_array(std::initializer_list<...>, Alloc = {})` constructs the array with elements initialized from a nested list.
| (destructor)      | Destructor deallocates memory and destroy the elements |
| `operator=`       | assigns for a source `subarray`, or from another `array`. `array`s can be moved |

| Element access    | same as for `static_array` |
|---                |--- |

| Structure access  | same as for `static_array` |
|---                |--- |

| Iterators         | same as for `static_array`   |
|---                |--- |

| Capacity          | same as for `static_array`  |
|---                |--- |

| Creating views    | same as for `static_array`  |
|---                |---  |

| Creating arrays   | same as for `static_array`  |
|---                |---  |

| Relational fuctions   |  same as for `static_array`  |
|---                |--- |

| Manipulation      |     |
|---                |---  |
| `clear`           | Erases all elements from the container. The array is resized to zero size. |
| `reextent`        | Changes the size of the array to new extensions. `reextent({e1, e2, ...})` elements are preserved when possible. New elements are initialized with a default value `v` with a second argument `reextent({e1, e2, ...}, v)`. The first argument is of `extensions_type`, and the second is optional for element types with a default constructor. 

</details>

<details><summary>class <code>multi::[sub]array&lt;T, D, P &gt;::(const_)iterator</code></summary>

A random-access iterator to subarrays of dimension `D - 1`, that is generally used to interact with or implement algorithms.
They can be default constructed but do not expose other constructors since they are generally created from `begin` or `end`, manipulated arithmetically, `operator--`, `operator++` (pre and postfix), or random jumps `operator+`/`operator-` and `operator+=`/`operator-=`.
They can be dereferenced by `operator*` and index access `operator[]`, returning objects of lower dimension `subarray<T, D, ... >::reference` (see above).
Note that this is the same type for all related arrays, for example, `multi::array<T, D, P >::(const_)iterator`.

`iterator` can be invalidated when its original array is invalidated, destroyed or resized.
An `iterator` that stems from `static_array` becomes invalid only if the original array was destroyed or out-of-scope.
</details>

# Interoperability

## STL (Standard Template Library)

The fundamental goal of the library is that the arrays and iterators can be used with STL algorithms out-of-the-box with a reasonable efficiency.
The most dramatic example of this is that `std::sort` works with array as it is shown in a previous example.

Along with STL itself, the library tries to interact with other existing quality C++ libraries listed below.

### Ranges (C++20)

[Standard ranges](https://en.cppreference.com/w/cpp/ranges) extend standard algorithms, reducing the need for iterators, in favor of more composability and a less error-prone syntax.

In this example, we replace the values of the first row for which the sum of the elements is odd:

```cpp
	static constexpr auto accumulate = [](auto const& R) {return std::ranges::fold_left(R, 0, std::plus<>{});};

	auto arr = multi::array<int, 2>{
		{2, 0, 2, 2},
		{2, 7, 0, 2},  // this row adds to an odd number
		{2, 2, 0, 4},
	};

	auto const row = std::ranges::find_if(arr, [](auto const& r) { return accumulate(r) % 2 == 1; });
	if(row != arr.end()) std::ranges::fill(*row, 9);

	assert(arr[1][0] == 9 );
```
[(live)](https://godbolt.org/z/cT9WGffM3)

Together with the array constructors, the ranges library enables a more functional programming style;
this allows us to work with immutable variables in many cases.

```cpp
	multi::array<double, 2> const A = {{...}};
	multi::array<double, 1> const V = {...};

	multi::array<double, 1> const R = std::views::zip_transform(std::plus<>{}, A[0], V);

	// Alternative imperative mutating code:
	// multi::array<double, 1> R(V.size());  // R is created here...
	// for(auto i : R.extension()) {R[i] = A[0][i] + V[i];}  // ...and then mutated here
```
[(live)](https://godbolt.org/z/M84arKMnT)


The "pipe" (`|`) notation of standard ranges allows one-line expressions.
In this example, the expression will yield the maximum value of the rows sums:
[`std::ranges::max(arr | std::views::transform(accumulate))`](https://godbolt.org/z/hvqnsf4xb)

Like in classic STL, standard range algorithms acting on sequences operate in the first dimension by default,
for example, lexicographical sorting on rows can be performed with the `std::ranges::sort` algorithm.

```cpp
	auto A = multi::array<char, 2>{
		{'S', 'e', 'a', 'n', ' ', ' '},
		{'A', 'l', 'e', 'x', ' ', ' '},
		{'B', 'j', 'a', 'r', 'n', 'e'},
	};
	assert(!std::ranges::is_sorted(A));

	std::ranges::sort(A);  // will sort on rows

	assert( std::ranges::is_sorted(A));

	assert(
		A == multi::array<char, 2>{
			{'A', 'l', 'e', 'x', ' ', ' '},
			{'B', 'j', 'a', 'r', 'n', 'e'},
			{'S', 'e', 'a', 'n', ' ', ' '},
		}
	);
```

To operate on the second dimension (sort by columns), use `std::ranges::sort(~A)` (or `std::ranges::sort(A.transposed())`).

### Execution policies (parallel algorithms)

Multi's iterators can exploit parallel algorithms by specifying execution policies.
This code takes every row of a two-dimensional array and sums its elements, putting the results in a one-dimensional array of compatible size.
The execution policy (`par`) selected is passed as the first argument.

```cpp
    multi::array<double, 2> const A = ...;
    multi::array<double, 1> v(size(A));

    std::transform(std::execution::par, arr.begin(), arr.end(), vec.begin(), [](auto const& row) {return std::reduce(row.begin(), row.end());} );
```
[(live)](https://godbolt.org/z/63jEdY7zP)

For an array of 10000x10000 elements, the execution time decreases to 0.0288 sec, compared to 0.0526 sec for the non-parallel version (i.e. without the `par` argument).

Note that parallelization is, in this context, inherently one-dimensional.
For example, parallelization happens for the transformation operation, but not to the summation.

The optimal way to parallelize specific operations strongly depends on the array's size and shape.
Generally, straightforward parallelization without exploiting the n-dimensional structure of the data has a limited pay-off;
and nesting parallelization policies usually don't help either.

Flattening the n-dimensional structure for certain algorithms might help, but such techniques are beyond the scope of this documentation.

Some member functions internally perform algorithms and that can benefit from execution policies;
in turn, some of these functions have the option to pass a policy.
For example, this copy construction can initialize elements in parallel from the source:

```cpp
    multi::array<double, 2> const A = ...;
    multi::array<double, 1> const B(std::execution::par, A);  // copies A into B, in parallel, same effect as multi::array<double, 1> const B(A); or ... B = A;
```

Execution policies are not limited to STL;
Thrust and oneAPI also offer execution policies that can be used with the corresponding algorithms.

Execution policies and ranges can be mixed (`x` and `y` can be 1D dimensional arrays, of any arithmetic element type)
```cpp
template <class X1D, class Y1D>
auto dot_product(X1D const& x, Y1D const& y) {
	assert(x.size() == y.size());
	auto const& z = std::ranges::views::zip(x, y)
		| std::ranges::views::transform([](auto const& ab) { auto const [a, b] = ab;
			return a * b;
		})
	;
	return std::reduce(std::execution::par_unseq, z.begin(), z.end());
}
```
[(live)](https://godbolt.org/z/cMq87xPvb)

### Polymorphic Memory Resources

In addition to supporting classic allocators (`std::allocator` by default), the library is compatible with C++17's [polymorphic memory resources (PMR)](https://en.cppreference.com/w/cpp/header/memory_resource), which allows using advanced allocation strategies, including preallocated buffers.
This example code uses a buffer as memory for two arrays; 
in it, a predefined buffer will contain the arrays' data (something like `"aaaabbbbbbXX"`).

```cpp
#include <memory_resource>  // for polymorphic memory resource, monotonic buffer

int main() {
	char buffer[13] = "XXXXXXXXXXXX";  // a small buffer on the stack
	std::pmr::monotonic_buffer_resource pool{std::data(buffer), std::size(buffer)};

	multi::pmr::array<char, 2> A({2, 2}, 'a', &pool);
	multi::pmr::array<char, 2> B({3, 2}, 'b', &pool);

	assert( buffer != std::string{"XXXXXXXXXXXX"} );  // overwritten w/elements, implementation-dependent (libstd consumes from left, and libc++, from the right)
}
```

`multi::pmr::array<T, D>` is a synonym for `multi::array<T, D, std::pmr::polymorphic_allocator<T>>`.
In this particular example, the technique can be used to avoid dynamic memory allocations of small local arrays. [(live)](https://godbolt.org/z/fP9P5Ksvb)

The library also supports memory resources from other libraries, including those returning special pointer types (see the [CUDA Thrust](#cuda-thrust) section and the Boost.Interprocess section).

### Substitutability with standard vector and span

The one-dimensional case `multi::array<T, 1>` is special and overlaps functionality with other dynamic array implementations, such as `std::vector`.
Indeed, both types of containers are similar and usually substitutable, with no or minor modifications.
For example, both can be constructed from a list of elements (`C c = {x0, x2, ...};`) or from a size `C c(size);`, where `C` is either type.

Both values are assignable, have the same element access patterns and iterator interface, and implement all (lexical) comparisons.

They differ conceptually in their resizing operations: `multi::array<T, 1>` doesn't insert or push elements and resizing works differently.
The difference is that the library doesn't implement *amortized* allocations; therefore, these operations would be of a higher complexity cost than the `std::vector`.
For this reason, `resize(new_size)` is replaced with `reextent({new_size})` in `multi::array`, whose primary utility is for element preservation when necessary.

In a departure from standard containers, elements are left initialized if they have trivial constructor.
So, while `multi::array<T, 1> A({N}, T{})` is equivalent to `std::vector<T> V(N, T{})`, `multi::array<T, 1> A(N)` will leave elements `T` uninitialized if the type allows this (e.g. built-ins), unlike `std::vector<T> V(N)` which will initialize the values.
RAII types (e.g. `std::string`) do not have trivial default constructor, therefore they are not affected by this rule.

With the appropriate specification of the memory allocator, `multi::array<T, 1, Alloc>` can refer to special memory not supported by `std::vector`.

Finally, an array `A1D` can be copied by `std::vector<T> v(A1D.begin(), A1D.end());` or `v.assign(A1D.begin(), A1D.end());` or vice versa.
Without copying, a reference to the underlying memory can be created `auto&& R1D = multi::array_ref<double, 1>(v.data(), v.size());` or conversely `std::span<T>(A1D.data_elements(), A1D.num_elements());`. 
(See examples [here](https://godbolt.org/z/n4TY998o4).)

The `std::span` (C++20) has not a well defined reference- or pointer-semantics; it doesn't respect `const` correctness in generic code.
This behavior is contrary to the goals of this library;
and for this reason, there is no single substitute for `std::span` for all cases.
Depending on how it is used, either `multi::array_ref<T, 1> [const& | &&]` or `multi::array_ptr<T [const], 1>` may replace the features of `std::span`.
The former typically works when using it as function argument.

Multi-dimensinal arrays can interoperate with C++23's non-owning `mdspan`.
[Preliminarily](https://godbolt.org/z/aWW3vzfPj), Multi's subarrays (arrays) can be converted (viewed as) `mdspan`.

A detailed comparison with other array libraries (mspan, Boost.MultiArray, Eigen) is explained in an Appendix.

## Serialization

The ability to serialize arrays is essential for storing data in a persistent medium (files on disk) and communicating values via streams or networks (e.g., MPI).
Unfortunately, the C++ language does not provide facilities for serialization, and the standard library doesn't either.

However, there are a few libraries that offer a certain common protocol for serialization,
such as [Boost.Serialization](https://www.boost.org/doc/libs/1_76_0/libs/serialization/doc/index.html) and [Cereal](https://uscilab.github.io/cereal/).
The Multi library is compatible with both (and doesn't depend on any of them).
The user can choose one or the other, or none, if serialization is not needed.
The generic protocol is such that variables are (de)serialized using the (`>>`)`<<` operator with the archive; operator `&` can be used to have a single code for both.
Serialization can be binary (efficient) or text-based (human-readable).

Here, it is a small implementation of save and load functions for an array to JSON format with the Cereal library.
The example can be easily adapted to other formats or libraries.
(An alternative for XML with Boost.Serialization is commented on the right.)

```cpp
#include<multi/array.hpp>  // this library

#include<cereal/archives/json.hpp>  // or #include<cereal/archives/xml.hpp>   // #include <boost/archive/xml_iarchive.hpp>
                                                                              // #include <boost/archive/xml_oarchive.hpp>
// for serialization of array elements (in this case strings)
#include<cereal/types/string.hpp>                                             // #include <boost/serialization/string.hpp>

#include<fstream>  // saving to files in example

using input_archive  = cereal::JSONInputArchive ;  // or ::XMLInputArchive ;  // or boost::archive::xml_iarchive;
using output_archive = cereal::JSONOutputArchive;  // or ::XMLOutputArchive;  // or boost::archive::xml_oarchive;

using cereal::make_nvp;                                                       // or boost::serialization::make_nvp;

namespace multi = boost::multi;

template<class Element, multi::dimensionality_type D, class IStream> 
auto array_load(IStream&& is) {
	multi::array<Element, D> value;
	input_archive{is} >> make_nvp("value", value);
	return value;
}

template<class Element, multi::dimensionality_type D, class OStream>
void array_save(OStream&& os, multi::array<Element, D> const& value) {
	output_archive{os} << make_nvp("value", value);
}

int main() {
	multi::array<std::string, 2> const A = {{"w", "x"}, {"y", "z"}};
	array_save(std::ofstream("file.string2D.json"), A);  // use std::cout to print serialization to the screen

	auto const B = array_load<std::string, 2>(std::ifstream("file.string2D.json"));
	assert(A == B);
}
```
[(online)](https://godbolt.org/z/Grr7Mqef5)

These templated functions work for any dimension and element type (as long as the element type is serializable in itself; all basic types are serializable by default).
However, note that the user must ensure that data is serialized and deserialized into the same type;
the underlying serialization libraries only do minimal consistency checks for efficiency reasons and don't try to second-guess file formats or contained types.
Serialization is a relatively low-level feature for which efficiency and economy of bytes are a priority.
Cryptic errors and crashes can occur if serialization libraries, file formats, or C++ types are mixed between writes and reads.
Some formats are human-readable but still not particularly pretty for showing as output (see the section on Formatting on how to print to the screen).

References to subarrays (views) can also be serialized; however, size information is not saved in such cases.
The reasoning is that references to subarrays cannot be resized in their number of elements if there is a size mismatch during deserialization.
Therefore, array views should be deserialized as other array views with matching sizes.

The output JSON file created by Cereal in the previous example looks like this.

```json
{
    "value": {
        "cereal_class_version": 0,
        "extensions": {
            "cereal_class_version": 0,
            "extension": {
                "cereal_class_version": 0,
                "first": 0,
                "last": 2
            },
            "extension": {
                "first": 0,
                "last": 2
            }
        },
        "elements": {
            "cereal_class_version": 0,
            "item": "w",
            "item": "x",
            "item": "y",
            "item": "z"
        }
    }
}
```
(The [Cereal XML](https://godbolt.org/z/de814Ycar) and Boost XML output would have a similar structure.)

Large datasets tend to be serialized slowly for archives with heavy formatting.
Here it is a comparison of speeds when (de)serializing a 134 MB 4-dimensional array of with random `double`s.

| Archive format (Library)     | file size     | speed (read - write)           | time (read - write)   |
| ---------------------------- | ------------- | ------------------------------ |-----------------------|
| JSON (Cereal)                | 684 MB        |    3.9 MB/sec  -   8.4 MB/sec  |  32.1 sec - 15.1  sec |
| XML (Cereal)                 | 612 MB        |    2.0  MB/sec -   4.0 MB/sec  |  56.0 sec - 28.0  sec |
| XML (Boost)                  | 662 MB        |   11.0  MB/sec -  13.0 MB/sec  |  11.0 sec -  9.0  sec |
| YAML ([custom archive)](https://gitlab.com/correaa/boost-archive-yml) | 702 MB        |   10.0  MB/sec -    4.4 MB/sec  |  12.0   sec  - 28.0   sec |
| Portable Binary (Cereal)     | 134 MB        |  130  MB/sec -  121  MB/sec  |  9.7  sec  - 10.6 sec |
| Text (Boost)                 | 411 MB        |   15.0  MB/sec -   16.0  MB/sec  |  8.2  sec  - 7.6  sec |
| Binary (Cereal)              | 134 MB        |  134.4 MB/sec -  126.  MB/sec  |  0.9  sec  -  0.9 sec |
| Binary (Boost)               | 134 MB        | 5200  MB/sec - 1600  MB/sec  |  0.02 sec -   0.1 sec |
| gzip-XML (Cereal)            | 191 MB        |    2.0  MB/sec -    4.0  MB/sec  | 61    sec  - 32   sec |
| gzip-XML (Boost)             | 207 MB        |    8.0  MB/sec -    8.0  MB/sec  | 16.1  sec  - 15.9 sec |

## Range-v3

The library works out of the box with Eric Niebler's Range-v3 library, a precursor to the standard Ranges library (see above).
The library helps removing explicit iterators (e.g. `begin`, `end`) from the code when possible.

Every Multi array object can be regarded as range.
Every subarray references (and array values) are interpreted as range views.

For example for a 2D array `d2D`, `d2D` itself is interpreted as a range of rows.
Each row, in turn, is interpreted as a range of elements.
In this way, `d2D.transposed()` is interpreted as a range of columns (of the original array), and each column a range of elements (arranged vertically in the original array).

```cpp
#include <range/v3/all.hpp>

int main(){

	multi::array<int, 2> const d2D = {
		{ 0,  1,  2,  3}, 
		{ 5,  6,  7,  8}, 
		{10, 11, 12, 13}, 
		{15, 16, 17, 18}
	};
	assert( ranges::inner_product(d2D[0], d2D[1], 0.) == 6+2*7+3*8 );
	assert( ranges::inner_product(d2D[0], rotated(d2D)[0], 0.) == 1*5+2*10+15*3 );

	static_assert(ranges::RandomAccessIterator<multi::array<double, 1>::iterator>{});
	static_assert(ranges::RandomAccessIterator<multi::array<double, 2>::iterator>{});
}
```

In this other [example](https://godbolt.org/z/MTodPEnsr), a 2D Multi array (or subarray) is modified such that each element of a column is subtracted the mean value of such column.

```cpp
#include<multi/array.hpp>
#include<range/v3/all.hpp>

template<class MultiArray2D>
void subtract_mean_columnwise(MultiArray2D&& arr) {
    auto&& tarr = arr.transposed();
    auto const column_mean = 
        tarr
        | ranges::views::transform([](auto const& row) {return ranges::accumulate(row, 0.0)/row.size();})
        | ranges::to<multi::array<double, 1>>
    ;

    ranges::transform(
        arr.elements(),
        column_mean | ranges::views::cycle,
        arr.elements().begin(),
        [](auto const elem, auto const mean) {return elem - mean;}
    );
}
```

## Boost.Interprocess

Using Interprocess allows for shared memory and for persistent mapped memory.

```cpp
#include <boost/interprocess/managed_mapped_file.hpp>
#include "multi/array.hpp"
#include<cassert>

namespace bip = boost::interprocess;
using manager = bip::managed_mapped_file;
template<class T> using mallocator = bip::allocator<T, manager::segment_manager>;
auto get_allocator(manager& m){return m.get_segment_manager();}

namespace multi = boost::multi;
template<class T, int D> using marray = multi::array<T, D, mallocator<T>>;

int main(){
{
	manager m{bip::create_only, "bip_mapped_file.bin", 1 << 25};
	auto&& arr2d = *m.construct<marray<double, 2>>("arr2d")(std::tuple{1000, 1000}, 0., get_allocator(m));
	arr2d[4][5] = 45.001;
	m.flush();
}
{
	manager m{bip::open_only, "bip_mapped_file.bin"};
	auto&& arr2d = *m.find<marray<double, 2>>("arr2d").first;
	assert( arr2d[4][5] == 45.001 );
	m.destroy<marray<double, 2>>("arr2d");//    eliminate<marray<double, 2>>(m, "arr2d");}
}
}
```

(Similarly works with [LLNL's Meta Allocator](https://github.com/llnl/metall))

## CUDA (and HIP, and OMP, and TBB) via Thrust

The library works out-of-the-box in combination with the Thrust library.

```cpp
#include <multi/array.hpp>  // this library

#include <thrust/device_allocator.h>  // from CUDA or ROCm distributions

namespace multi = boost::multi;

int main() {
	multi::array<double, 2, thrust::device_allocator<double>> A({10,10});
	multi::array<double, 2, thrust::device_allocator<double>> B({10,10});
	A[5][0] = 50.0;

	thrust::copy(A.rotated()[0].begin(), A.rotated()[0].end(), B.rotated()[0].begin());  // copy row 0
	assert( B[5][0] == 50.0 );
}
```
[(live)](https://godbolt.org/z/oM4YbPYz8)

which uses the default Thrust device backend (i.e. CUDA when compiling with `nvcc`, HIP/ROCm when compiling with a HIP/ROCm compiler, or OpenMP or TBB in other cases).
Universal memory (accessible from normal CPU code) can be used with `thrust::universal_allocator` (from `<thrust/universal_allocator.h>`) instead.

More specific allocators can be used ensure CUDA backends, for example CUDA managed memory:

```cpp
#include <thrust/system/cuda/memory.h>
...
	multi::array<double, 2, thrust::cuda::universal_allocator<double>> A({10,10});
```

In the same way, to *ensure* HIP backends please replace the `cuda` namespace by the `hip` namespace, and in the directory name `<thrust/system/hip/memory.h>`.
`<thrust/system/hip/memory.h>` is provided by rocThrust in the ROCm distribution (in `/opt/rocm/include/thrust/system/hip/`, and not by the NVIDIA distribution.)

Multi doesn't have a dependency on Thrust (or vice versa);
they just work well together, both in terms of semantics and efficiency.
Certain "patches" (to improve Thrust behavior) can be applied to Thrust to gain extra efficiency and achieve near native speed by adding the `#include<multi/adaptors/thrust.hpp>`.

Multi can be used on existing memory in a non-invasive way via (non-owning) reference arrays:

```cpp
	// assumes raw_pointer was allocated with cudaMalloc or hipMalloc
	using gpu_ptr = thrust::cuda::pointer<double>;  // or thrust::hip::pointer<double> 
	multi::array_ref<double, 2, gpu_ptr> Aref({n, n}, gpu_ptr{raw_pointer});
```

Finally, the element type of the device array has to be device-friendly to work correctly; 
this includes all build in types, and classes with basic device operations, such as construction, destruction, and assigment.
They notably do not include `std::complex<T>`, in which can be replaced by the device-friendly `thrust::complex<T>` can be used as replacement.

### OpenMP via Thrust

In an analogous way, Thrust can also handle OpenMP (omp) allocations and multi-threaded algorithms of arrays.
The OMP backend can be enabled by the compiler flags `-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_BACKEND_OMP` or by using the explicit `omp` system types: 

```cpp
#include <multi/array.hpp>
#include <multi/adaptors/thrust/omp.hpp>

#include <thrust/copy.h>

namespace multi = boost::multi;

int main() {
    auto A = multi::thrust::omp::array<double, 2>({10,10}, 0.0);  // or multi::array<double, 2, thrust::omp::allocator<double>>;
    auto B = multi::thrust::omp::array<double, 2>({10,10});  // or multi::array<double, 2, thrust::omp::allocator<double>>;

	A[5][0] = 50.0;

    // copy row 0
	thrust::copy(
        A.rotated()[0].begin(), A.rotated()[0].end(),
        B.rotated()[0].begin()
    );
	assert( B[5][0] == 50.0 );
	auto C = B;  // uses omp automatically for copying behind the scenes
}
```
https://godbolt.org/z/KW19zMYnE

Compilation might need to link to an omp library, `-fopenmp -lgomp`.

Without Thrust, OpenMP pragmas would also work with this library, however OpenMP memory allocation, would need to be manually managed.

### Thrust memory resources

GPU memory is relative expensive to allocate, therefore any application that allocates and deallocates arrays often will suffer performance issues.
This is where special memory management is important, for example for avoiding real allocations when possible by caching and reusing memory blocks.

Thrust implements both polymorphic and non-polymorphic memory resources via `thrust::mr::allocator<T, MemoryResource>`;
Multi supports both.

```cpp
auto pool = thrust::mr::disjoint_unsynchronized_pool_resource(
	thrust::mr::get_global_resource<thrust::universal_memory_resource>(),
	thrust::mr::get_global_resource<thrust::mr::new_delete_resource>()
);

// memory is handled by pool, not by the system allocator
multi::array<int, 2, thrust::mr::allocator<int, decltype(pool)>> arr({1000, 1000}, &pool);
```

The associated pointer type for the array data is deduced from the _upstream_ resource; in this case, `thrust::universal_ptr<int>`.

As as quick way to improve performance in many cases, here it is a recipe for a `caching_allocator` which uses a global (one per thread) memory pool that can replace the default Thrust allocator.
The requested memory resides in GPU (managed) memory (`thrust::cuda::universal_memory_resource`) while the cache _bookkeeping_ is held in CPU memory (`new_delete_resource`).

```cpp
template<class T, class Base_ = thrust::mr::allocator<T, thrust::mr::memory_resource<thrust::cuda::universal_pointer<void>>>>
struct caching_allocator : Base_ {
	caching_allocator() : 
		Base_{&thrust::mr::tls_disjoint_pool(
			thrust::mr::get_global_resource<thrust::cuda::universal_memory_resource>(),
			thrust::mr::get_global_resource<thrust::mr::new_delete_resource>()
		)} {}
	caching_allocator(caching_allocator const&) : caching_allocator{} {}  // all caching allocators are equal
	template<class U> struct rebind { using other = caching_allocator<U>; };
};
...
int main() {
	...
	using array2D = multi::array<double, 2, caching_allocator<double>>;

	for(int i = 0; i != 10; ++i) { array2D A({100, 100}); /*... use A ...*/ }
}
```
https://godbolt.org/z/rKG8PhsEh

In the example, most of the frequent memory requests are handled by reutilizing the memory pool avoiding expensive system allocations.
More targeted usage patterns may require locally (non-globally) defined memory resources.

## CUDA C++

CUDA is a dialect of C++ that allows writing pieces of code for GPU execution, known as "CUDA kernels".
CUDA code is generally "low level" (less abstracted) but it can be used in combination with CUDA Thrust or the CUDA runtime library, specially to implement custom algorithms.
Although code inside kernels has certain restrictions, most Multi features can be used. 
(Most functions in Multi, except those involving memory allocations, are marked `__device__` to allow this.)

Calling kernels involves a special syntax (`<<< ... >>>`), and they cannot take arguments by reference (or by values that are not trivial).
Since arrays are usually passed by reference (e.g. `multi::array<double, 2>&` or `Array&&`), a different idiom needs to be used.
(Large arrays are not passed by value to avoid copies, but even if a copy would be fine, kernel arguments cannot allocate memory themselves.)
Iterators (e.g. `.begin()/.end()`) and "cursors" (e.g. `.home()`) are "trivial to copy" and can be passed by value and represent a "proxy" to an array, including allowing the normal index syntax and other transformations.

Cursors are a generalization of iterators for multiple dimensions.
They are cheaply copied (like iterators) and they allow indexing.
Also, they have no associated `.size()` or `.extensions()`, but this is generally fine for kernels.
(Since `cursors` have minimal information for indexing, they can save stack/register space in individual kernels.)

Here it is an example implementation for matrix multiplication, in combination with Thrust and Multi,

```cpp
#include <multi/array.hpp>  // from https://gitlab.com/correaa/boost-multi
#include <thrust/system/cuda/memory.h>  // for thrust::cuda::allocator

template<class ACursor, class BCursor, class CCursor>
__global__ void Kernel(ACursor A, BCursor B, CCursor C, int N) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	typename CCursor::element_type value{0.0};
	for (int k = 0; k != N; ++k) { value += A[y][k] * B[k][x]; }
	C[y][x] = value;
}

namespace multi = boost::multi;

int main() {
	int N = 1024;

	// declare 3 square arrays
	multi::array<double, 2, thrust::cuda::allocator<double>> A({N, N}); A[0][0] = ...;
	multi::array<double, 2, thrust::cuda::allocator<double>> B({N, N}); B[0][0] = ...;
	multi::array<double, 2, thrust::cuda::allocator<double>> C({N, N});

	// kernel invocation code
	assert(N % 32 == 0);
	dim3 dimBlock(32, 32);
	dim3 dimGrid(N/32, N/32);
	Kernel<<<dimGrid, dimBlock>>>(A.home(), B.home(), C.home(), N);
	cudaDeviceSynchronize();

    // now C = A x B
}
```
[(live)](https://godbolt.org/z/eKbeosrWa)

Expressions such as `A.begin()` (iterators) can also be passed to kernels, but they could unnecessarely occupy more kernel "stack space" when size information is not needed (e.g. `A.begin()->size()`).

## SYCL

The SYCL library promises the unify CPU, GPU and FPGA code.
At the moment, the array containers can use the Unified Shared Memory (USM) allocator, but no other tests have been investigated.

```cpp
    sycl::queue q;

    sycl::usm_allocator<int, sycl::usm::alloc::shared> q_alloc(q);
    multi::array<int, 1, decltype(q_alloc)> data(N, 1.0, q_alloc);

    //# Offload parallel computation to device
    q.parallel_for(sycl::range<1>(N), [=,ptr = data.base()] (sycl::id<1> i) {
        ptr[i] *= 2;
    }).wait();
```
https://godbolt.org/z/8WG8qaf4s

Algorithms are expected to work with oneAPI execution policies as well (not tested)

```cpp
    auto policy = oneapi::dpl::execution::dpcpp_default;
    sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(policy.queue());
    multi::array<int, 1, decltype(alloc)> vec(n, alloc);

    std::fill(policy, vec.begin(), vec.end(), 42);
```

## Formatting ({fmt} pretty printing)

The library doesn't have a "pretty" printing facility to display arrays.
Although it is not ideal, arrays can be printed and formated by looping over elements and dimension, as shown in other examples (using standard streams).

Fortunatelly, the library automatically works with the external library [{fmt}](https://fmt.dev/latest/index.html), both for arrays and subarrays.
The fmt library is not a dependency of the Multi library; they simply work well together using the "ranges" part of the formatting library.
fmt allows a high degree of confurability.

This example prints a 2-dimensional subblock of a larger array.

```cpp
#include "fmt/ranges.h"
...
    multi::array<double, 2> A2 = {
        {1.0, 2.0,      3.0}, 
        /*-subblock-**/
        {3.0, 4.0, /**/ 5.0},
        {6.0, 7.0, /**/ 8.0},
    };

    fmt::print("A2 subblock = {}", A2({1, 3}, {0, 2}));  // second and third row, first and second column
```
obtaining the "flat" output `A2 subblock = [[3, 4], [6, 7]]`.
(A similar effect can be achieved with [experimental C++23 `std::print` in libc++]( https://godbolt.org/z/4ehd4s5vf).)

For 2 or more dimensions the output can be conveniently structured in different lines using the `fmt::join` facility:

```cpp
    fmt::print("{}\n", fmt::join(A2({1, 3}, {0, 2}), "\n"));  // first dimension rows are printer are in different lines
```
with the output:

> ```
> [3, 4]
> [6, 7]
> ```

In same way, the size of the array can be printed simply by passing the sizes output; `fmt::print("{}", A2(...).sizes() )`, which print `(2, 2)`.

https://godbolt.org/z/WTEfnMG7n

When saving arrays to files, consider using serialization (see section) instead of formatting facilities.

## Python (cppyy)

There is no special code to interoperated with Python;
however the memory layout is compatible with Numpy and pybind11 binding with zero-copy could be written.
Furthermore the library works out-of-the-box via de automatic `cppyy` bindings.

Here it is a complete Python session:

```python
# We import the library
import cppyy
cppyy.add_include_path('path-to/boost-multi/include')
cppyy.include("boost/multi/array.hpp")
from cppyy.gbl.boost import multi

# We can created a one-dimensional array with 4 initialized to 0.0, or from a list of numbers.
# We can print the array and change the element values:
a1d = multi.array['double', 1](4, 0.0)
a1d = multi.array['double', 1]([1.0, 2.0, 3.0, 4.0])
print(a1d)
```
> ```python
> { 1.0000000, 2.0000000, 3.0000000, 4.0000000 }
> ```
```python
# elements and assignable
a1d[2] = 99.9
print(a1d)
```
> ```python
> { 1.0000000, 2.0000000, 99.900000, 4.0000000 }
> ```
```python
# We can also create a 2x2 array, or directly from a bidimensional list:
a2d = multi.array['double', 2](multi.extensions_t[2](2, 2), 0.0)
a2d = multi.array['double', 2]([[1.0, 2.0], [3.0, 4.0]])

# We can retrive information from an individual element, or from a row
print(a2d.transposed()[0])
```
> ```python
> { 1.0000000, 3.0000000 }
> ```
```python
arow = a2d[0]
print(arow)

# rows (or slices in general) are references to the original arrays
arow[0] = 66.6
print(a2d[0])
```
> ```python
> { 66.600000, 2.0000000 }
> ```
```python
r
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
