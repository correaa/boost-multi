<!--
(pandoc `#--from gfm` --to html --standalone --metadata title=" " $0 > $0.html) && firefox --new-window  $0.html; exit
-->
# [Boost.]Multi

(not an official Boost library)

_Alfredo Correa, 2018-2019_

Multi provides multidimensional array access to contiguous or regularly contiguous memory (or ranges).
It shares the goals of [Boost.MultiArray](https://www.boost.org/doc/libs/1_69_0/libs/multi_array/doc/index.html), 
although the code is completely independent and the syntax has slight differences or has been extended.
Multi and Boost.MultiArray types can be used interchangeably for the most part, they differ slightly in the semantics of reference and value types. 

Multi aims to simplify the semantics of Boost.MultiArray and make it more compatible with the Standard (STL) Algorithms and special memory.
It requires C++14. 
The code was developed on `clang` and `gcc` compilers, and tested regularly with Intel compilers.

Before testing speed, please make sure that you are compiling in release mode (`-DNDEBUG`) and with optimizations (`-O3`), 
if your test involves mathematical operations add arithmetic optimizations (`-Ofast`) to compare with Fortran code.

Some features:

* Arbitrary pointer types (minimal requirements)
* Simplified implementation (~600 lines)
* Fast access of subarray (view) types
* Value semantics of multi-dimensional array container
* Better semantics of subarray (view) types

## Types

* `multi::array<T, D, A>`: Array of dimension `D`, it has value semantics if `T` has value semantics. Memory is requested by allocator of type `A`, should support stateful allocators.
* `multi::array_ref<T, D, P = T*>`: Array interpretation of a random access range, usually a memory block. It has reference semantics. Thanks to (non-virtual) inheritance an `array<T, D, A>` is-a `array_ref<T, D, A::pointer>`.
* other derived "unspecified types" fulfil (a still loosely defined) `MultiArrayView` concept, for example by taking partial indices or rotations (transpositions). These reference types cannot be stored except through life-time extensions `auto&&`. Due to language limitations `auto` will not deduce a corresponding value-sematics type; for this reason it is necessary to use a "decay" idiom to obtain value object.
* `MultiArrayView<T,D,P>::(const_)iterator`: Iterator to subarrays of dimension `D-1`. For `D==1` this is an iterator to an element. This types are generated by `begin` and `end` functions.
* `MultiArrayView<T, D, P>::(const_)reference`: Reference to subarrays of dimension `D-1`. For `D>1` this are not true C++-references but types emulate them (with reference semantics), therefore `auto` is not well behaved. For `D==1` this is a true C++ reference to an elements. These types are generated by dereferencing iterators, e.g. `*begin(MA)`.

## Concept Requirements

The design tries to impose the minimum possible requirements over the used referred types.
Pointer-like random access types can be used as substitutes of built-in pointers.

```c++
namespace minimal{
    template<class T> class ptr{ // minimalistic pointer
    	T* impl_;
    	T& operator*() const{return *impl_;}
    	auto operator+(std::ptrdiff_t n) const{return ptr{impl_ + n};}
    //	operator[], operator+=, etc are optional but not necessary
    };
}

int main(){
	double* buffer = new double[100];
	multi::array_ref<double, 2, minimal::ptr<double> > CC(minimal::ptr<double>{buffer}, {10, 10});
	CC[2]; // requires operator+ 
	CC[1][1]; // requires operator*
	CC[1][1] = 9;
	assert(CC[1][1] == 9);
	delete[] buffer;
}
```

### Linear Sequences: Pointers

An `array_ref` can reference to an arbitrary random access iterator sequence.
This way, any linear (random access) sequence (e.g. `raw memory`, `std::vector`, `std::queue`) can be efficiently arranged as a multidimensional array. 

```c++
	std::vector<double> buffer(100);
	multi::array_ref<double, 2, std::vector<double>::iterator> A({10, 10}, buffer.begin());
	A[1][1] = 9;
	assert(A[1][1] == 9);
	assert(buffer[11]==9);
```
Since `array_ref` does not manage the memory associated with it, the reference can be simply dangle if the `buffer` memory is reallocated (e.g. by `resize`).

### Special Memory: Allocators and Fancy Pointers

`array`'s manages its memory through allocators. 
It can handle all sorts of special memory, as long as the underlying types behave coherently, these include fancy pointers and fancy references.
Associated fancy pointers and fancy reference (if any) are deduced from the allocator types.

The behavior regarding memory managament of the fancy pointers can be customized (if necessary) by specializations of some or all of these functions:

```c++
destroy(a, first, last)
destroy_n(a, first, n) -> last
uninitialized_copy_n(a, first, n, dest) -> last;
uninitialized_fill_n(a, first, n, value) -> last
uninitialized_default_construct_n(a, first, n) -> last
uninitialized_value_construct_n(a, first, n) -> last
```

where `a` is the special allocator, `n` is a size (usually the number of elements), `first`, `last` and `dest` are fancy pointers.

Copying underlying memory can be customized by specializing 

```c++
copy_n(first, n, dest)
fill_n(first, n, value)
```

Specific cases of fancy memory are file-mapped memory or interprocess shared memory.
This example illustrates memory persistency by combining with Boost.Interprocess library. 
The arrays support their allocators and fancy pointers (`boost::interprocess::offset_ptr`).

```c++
#include <boost/interprocess/managed_mapped_file.hpp>
using namespace boost::interprocess;
using manager = managed_mapped_file;
template<class T> using mallocator = allocator<T, manager::segment_manager>;
decltype(auto) get_allocator(manager& m){return m.get_segment_manager();}

template<class T, auto D> using marray = multi::array<T, D, mallocator<T>>;

int main(){
{
	manager m{create_only, "mapped_file.bin", 1 << 25};
	auto&& arr2d = *m.construct<marray<double, 2>>("arr2d")(std::tuple{1000, 1000}, 0.0, get_allocator(m));
	arr2d[4][5] = 45.001;
}
// imagine execution restarts here
{
	manager m{open_only, "mapped_file.bin"};
	auto&& arr2d = *m.find<marray<double, 2>>("arr2d").first;
	assert( arr2d[7][8] == 0. );
	assert( arr2d[4][5] == 45.001 );
	m.destroy<marray<double, 2>>("arr2d");
}
}
```

## Usage

We create a static C-array of `double`s, and refer to it via a bidimensional array `multi::array_ref<double, 2>`.

```c++
	#include "../array_ref.hpp"
	#include "../array.hpp"
	
	#include<algorithm> // for sort
	#include<iostream> // for print
	
	namespace multi = boost::multi;
	using std::cout; using std::cerr;
	
	int main(){
		double d2D[4][5] = {
			{150, 16, 17, 18, 19},
			{ 30,  1,  2,  3,  4}, 
			{100, 11, 12, 13, 14}, 
			{ 50,  6,  7,  8,  9} 
		};
		multi::array_ref<double, 2> d2D_ref{&d2D[0][0], {4, 5}};
															...
```

Note that the syntax of creating a reference array involves passing the pointer to a memory block (20 elements here) and the logical dimensions of that memory block (4 by 5 here).

Next we print the elements in a way that corresponds to the logical arrangement:

```c++
		for(auto i : d2D_ref.extension(0)){
			for(auto j : d2D_ref.extension(1))
				cout << d2D_ref[i][j] <<' ';
			cout <<'\n';
		}
```

This will output:

> 150 16 17 18 19  
> 30 1 2 3 4  
> 100 11 12 13 14  
> 50 6 7 8 9

It is sometimes said (by Sean Parent) that the whole of STL algorithms can be seen as intermediate pieces to implement`std::stable_sort`. 
Pressumably if one can sort over a range, one can perform any other standard algorithm.

```c++
		std::stable_sort( begin(d2D_ref), end(d2D_ref) );
```

If we print this we will get

> ```c++
> 30 1 2 3 4  
> 50 6 7 8 9  
> 100 11 12 13 14  
> 150 16 17 18 19
> ```


The array has been changed to be in row-based lexicographical order.
Since the sorted array is a reference to the original data, the original array has changed. 

```c++
		assert( d2D[1][1] == 6 );
```

(Note that `std::*sort` cannot be applied directly to a multidimensional C-array or to Boost.MultiArray types.)

If we want to order the matrix in a per-column basis we need to "view" the matrix as range of columns. This is done in the bidimensional case, by accessing the matrix as a range of columns:

```c++
	    std::stable_sort( d2D_ref.begin(1), d2D_ref.end(1) );
```

Which will transform the matrix into. 

> 1 2 3 4 30  
> 6 7 8 9 50  
> 11 12 13 14 100  
> 16 17 18 19 150 

In other words, a matrix of dimension `D` can be viewed simultaneously as `D` different ranges of different "transpositions" by passing an interger value to `begin` and `end` indicating the preferred dimension.
`begin(0)` is equivalent to `begin()`.

## Initialization

`array_ref` is initialized from a preexisting contiguous memory (or range), the index extensions should compatible with the total number of elements.

```c++
double* dp = new double[12];
multi::array_ref<double, 2> A({3,4}, dp);
multi::array_ref<double, 2> B({2,6}, dp);
...
delete[] dp;
```
`array` is initialized by specifying the index extensions (and optionally a default value) or alternatively from a rectangular list. 

```c++
/*In C++17 the element-type and the dimensionality can be omitted*/
multi::array/*<double, 1>*/ A1 = {1.,2.,3.}; 
                     assert(A1.dimensionality==1 and A1.num_elements()==3);
multi::array/*<double, 2>*/ A2 {
	 {1.,2.,3.},
	 {4.,5.,6.}
};                   assert(A2.dimensionality==2 and A2.num_elements()==2*3);
multi::array/*<double, 3>*/ const A3 = {
    {{ 1.2,  0.}, { 2.4, 1.}},
    {{11.2,  3.}, {34.4, 4.}},
    {{15.2, 99.}, {32.4, 2.}}
};                   assert(A3.dimensionality==3 and A3.num_elements()==3*2*2);
```

## Iteration

Accessing arrays by iterators (`begin`/`end`) enables the use of many iterator based algorithms (see the sort example above).
`begin/end(A)` (or equivalently `A.begin/end()`) gives iterators that linear and random access in the leading dimension. 

`A.begin/end(n)` gives access in non-leading nested dimension number `n`. 

`cbegin/cend(A)` (or equivalently `A.cbegin/cend()`) gives read-only iterators.

For example in three dimensional array,

	(cbegin(A)+1)->operator[](1).begin()[0] = 342.4; //error, read-only
	(begin(A)+1)->operator[](1).begin()[0] = 342.4; // assigns to A[1][1][0]
	assert( (begin(A)+1)->operator[](1).begin()[0] == 342.4 );

As an example, this function allows printing arrays of arbitrary dimension into a linear comma-separated form.

```c++
void print(double const& d){cout<<d;};
template<class MultiArray> 
void print(MultiArray const& ma){
	cout<<"{";
	if(not ma.empty()){
		print(*cbegin(ma));
		std::for_each(cbegin(ma)+1, cend(ma), [](auto&& e){cout<<","; print(e);});
	}
	cout<<"}";
}
...
print(A);
```
> {{{1.2,1.1},{2.4,1}},{{11.2,3},{34.4,4}},{{15.2,99},{32.4,2}}}


Except for those corresponding to the last dimension, derreferencing iterators generally produce proxy-reference objects. 
Therefore this is not allowed:

    auto row = *begin(A); // compile error 

This because `row` doesn't have the expected value semantics, and didn't produce any data copy.
However this express the intention better

    decltype(A)::value_type row = *begin(A); // there is a real copy.

In my experience, however, this produces a more consistent idiom to hold references without copying elements.

    auto const& crow = *cbegin(A); // same as decltype(A)::const_reference crow = *cbegin(A);
    auto&&       row = * begin(A); // same as decltype(A)::      reference  row = * begin(A);

## Indexing

Many algorithms on arrays are oriented to linear algebra, which are ubiquitously implemented in terms of multidimensional index access. 

### Element access and partial access

Index access mimics that of C-fixed sizes arrays, for example a 3-dimensional array will access to an element by `m[1][2][3]`, 
which can be used for write and read operations. 

Partial index arguments `m[1][2]` generate a view 1-dimensional object.
Transpositions are also multi-dimensional arrays views in which the index are *logically* rearranged, for example `m.rotated(1)[2][3][1] == m[1][2][3]`.

As an illustration of an algorithm based on index access (as opposed to iterators), 
this example code implements Gauss Jordan Elimination without pivoting:

```c++
template<class Matrix, class Vector>
auto gj_solve(Matrix&& A, Vector&& y)->decltype(y[0]/=A[0][0], y){
	std::ptrdiff_t Asize = size(A); 
	for(std::ptrdiff_t r = 0; r != Asize; ++r){
		auto&& Ar = A[r];
		auto&& Arr = Ar[r];
		for(std::ptrdiff_t c = r + 1; c != Asize; ++c) Ar[c] /= Arr;
		auto const yr = (y[r] /= Arr);
		for(std::ptrdiff_t r2 = r + 1; r2 != Asize; ++r2){
			auto&& Ar2 = A[r2];
			auto const& Ar2r = Ar2[r]; // auto&& Ar = A[r];
			for(std::ptrdiff_t c = r + 1; c != Asize; ++c) Ar2[c] -= Ar2r*Ar[c];
			y[r2] -= Ar2r*yr;
		}
	}
	for(std::ptrdiff_t r = Asize - 1; r > 0; --r){
		auto const& yr = y[r];
		for(std::ptrdiff_t r2 = r-1; r2 >=0; --r2) y[r2] -= yr*A[r2][r];
	}
	return y;
}
```

This function can be applied to a `multi::array` container:

```c++
		multi::array<double, 2> A = {{-3., 2., -4.},{0., 1., 2.},{2., 4., 5.}};
		multi::array<double, 1> y = {12.,5.,2.}; //(M); assert(y.size() == M); iota(y.begin(), y.end(), 3.1);
		gj_solve(A, y);
```

and also to a combination of `MultiArrayView`-type objects:

```c++
		multi::array<double, 2> A({6000, 7000}); std::iota(A.data(), A.data() + A.num_elements(), 0.1);
		std::vector<double> y(3000); std::iota(y.begin(), y.end(), 0.2);
		gj_solve(A({1000, 4000}, {0, 3000}), y);
```

### Slices and strides

Given an array, a slice in the first dimension can be taken with the `sliced` function. `sliced` takes two arguments, the first index of the slice and the last index (not included) of the slice. For example,

```c++
multi::array<double, 2> d2D({4, 5});
assert( d2D.size(0) == 4 and d2D.size(1) == 5 );

auto&& d2D_sliced = d2D.sliced(1, 3); // {{d2D[1], d2D[2]}}
assert( d2D_sliced.size(0) == 2 and d2D_sliced.size(1) == 5 );
```

The number of rows in the sliced matrix is 2 because we took only two rows, row 1 and row 2 (row 3 is excluded).

In the same way a strided view of the original array can be taken with the `strided` function.

```c++
auto&& d2D_strided = d2D.strided(2); // {{ d2D[0], d2D[1] }};
assert( d2D_strided.size(0) == 2 and d2D_strided.size(1) == 5 );
```

In this case the number of rows is 2 because, out of the 4 original rows we took one every two.

Operations can be combined in a single line:

```c++
auto&& d2D_slicedstrided = d2D.sliced(1, 3).strided(2); // {{ d2D[1] }};
assert( d2D_slicedstrided.size(0) == 1 and d2D_slicedstrided.size(1) == 5 );
```

For convenience, `A.sliced(a, b, c)` is the same as `A.sliced(a, b).strided(c)`.

By combining `rotated`, `sliced` and `strided` one can take sub arrays at any dimension. 
For example in a two dimensional array one can take a subset of columns by defining.

```c++
auto&& subA = A.rotated(1).strided(1, 3).sliced(2).rotated(-1);
```

Other notations are available, but when in doubt the `rotated/strided/sliced/rotated` and combinations of them idioms provides the most control over the subview operations.
(At the moment the `strided` argument has to divide the total size of the slice (or matrix), otherwise the behavior is undefined.)

Blocks (slices) in multidimensions can be obtained but pure index notation using `.operator()`:

```c++
multi::array<double, 2> A({6, 7}); // 6x7 array
A({1, 4}, {2, 4}) // 3x2 array, containing indices 1 to 4 in the first dimension and 2 to 4 in the second dimension.
```

# Interoperability

Along with STL itself, the library tries to interact with other existing C++ libraries.

## Range v3

```c++
#include <range/v3/all.hpp>
int main(){

	multi::array const d2D = {
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


## Boost.Interprocess

Using Interprocess allows for shared memory and for persistent mapped memory.

```c++
#include <boost/interprocess/managed_mapped_file.hpp>
#include "multi/array.hpp"
#include<cassert>

using namespace boost::interprocess;
using manager = managed_mapped_file;
template<class T> using mallocator = allocator<T, manager::segment_manager>;
auto get_allocator(manager& m){return m.get_segment_manager();}
void sync(manager& m){m.flush();}

namespace multi = boost::multi;
template<class T, int D> using marray = multi::array<T, D, mallocator<T>>;

int main(){
{
	manager m{create_only, "mapped_file.bin", 1 << 25};
	auto&& arr2d = *m.construct<marray<double, 2>>("arr2d")(std::tuple{1000, 1000}, 0., get_allocator(m));
	arr2d[4][5] = 45.001;
	sync(m);
}
{
	manager m{open_only, "mapped_file.bin"};
	auto&& arr2d = *m.find<marray<double, 2>>("arr2d").first;
	assert( arr2d[4][5] == 45.001 );
	m.destroy<marray<double, 2>>("arr2d");//	eliminate<marray<double, 2>>(m, "arr2d");}
}
}
```
(Similarly works with [LLNL's Meta Allocator](https://github.com/llnl/metall))

# Technical points

### What's up with the multiple bracket notation? 

The chained bracket notation (`A[i][j][k]`) allows to refer to elements and subarrays lower dimensional subarrays in a consistent and _generic_ manner and it is the recommended way to access the array objects.
It is a frequently raised question whether the chained bracket notation is good for performance, since it appears that each utilization of the bracket leads to the creation of a temporary which in turn generates a partial copy of the layout.
Moreover, this goes against [historical recommendations](https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op).

It turns out that [modern compilers with a fair level of optimization (`-O2`)](https://godbolt.org/z/3fYd5c) can elide these temporary objects, so that `A[i][j][k]` generates identical assembly code as `A.base() + i*stride1 + j*stride2 + k*stride3` (offsets are not shown for simplicity).

In a subsequence optimization, constant indices can have their "partial stride" computation removed from loops. 
As a result, these two loops lead to the [same machine code](https://godbolt.org/z/p_ELwQ):

```c++
    for(int j = 0; j != nj; ++j)
        ++A[i][j][k];
```
```c++
    double* Ai_k = A.base() + i*A_stride1 + k*A_stride3;
    for(int j = 0; j != nj; ++jj)
        ++(*(Ai_k + j*A_stride2));
```

Incidentally, the library also supports parenthesis notation with multiple indices `A(i, j, k)` for element or partial access, but it does so for accidental reasons as part of a more general syntax to generate sub blocks.
In any case `A(i, j, k)` is expanded to `A[i][j][k]` internally in the library when `i, j, k` are integer indices. 
Additionally, array coordinates can be directly stored in tuple-like data structures, allowing this syntax 

```c++
std::array p = {2,3,4};
std::apply(A, p) = 234; // A[2][3][4] = 234;
```

### Customizing recursive operations: SCARY iterators

A custom level of customization can be achieved by intercepting internal recursive algorithms.
Multi iterators are [SCARY](http://www.open-std.org/jtc1/sc22/WG21/docs/papers/2009/n2980.pdf). 
SCARY means that they are independent of any container and can be accessed generically through their dimension and underlying pointer types:

For example, `boost::multi::array_iterator<double, 2, double*> it` is a row (or column) iterator of an array of dimension 2 or higher, whose underlying pointer type is `double*`.
This row (or column) and subsequent ones can be accessed by `*it` and `it[n]` respectively. 
The base pointer, the strides and the size of the arrow can be accessed by `base(it)`, `stride(it)`, `it->size()`.

The template arguments of the iterator can be used to customize operations that are recursive (and possibly inefficient in certain context) in the library:

```c++
namespace boost{namespace multi{
template<class It, class T>  // custom copy 1D (aka strided copy)
void copy(It first, It last, multi::array_iterator<T, 1, fancy::ptr<T> > dest){
	assert( stride(first) == stride(last) );
	std::cerr<<"1D copy(it1D, it1D, it1D) with strides "<< stride(first) <<" "<< stride(dest) <<std::endl;
}

template<class It, class T> // custom copy 2D (aka double strided copy)
void copy(It first, It last, multi::array_iterator<T, 2, fancy::ptr<T> > dest){
	assert( stride(first) == stride(last) );
	std::cerr<<"2D copy(It, It, it2D) with strides "<< stride(first) <<" "<< stride(dest) <<std::endl;
}
}}
```

For example, if your custom pointers refers a memory type in which 2D memory copying (strided copy) is faster than sequencial copying, that kind of instruction can be ejecuted when the library internally calls `copy`.
This customization must be performed (unfortunately) in the `boost::multi` namespace (this is where the Multi iterators are defined) and the customization happens through matching the dimension and the pointer type.

If your custom pointer are not so fancy, it is not necessary to customize these functions in any way and the default behavior will be correct.
