# Boost.Multi

(not an official Boost library)

Boost.Multi provides multidimensional array access to contiguous or regularly contiguous memory.
It shares the goals of Boost.MultiArray, although the code is completely independent and the syntax has slight differences.
Boost.Multi and Boost.MultiArray types can be used interchangeably for the most part, they differ in the semantics of reference and value types. 

Boost.Multi aims to simplify the semantics of Boost.MultiArray and make it more compatible with the Standard (STL) Algorithms and special memory.
It requires C++14. The code was developed on `clang` and `gcc` compilers.

Before testing speed, please make sure that you are compiling in release mode (`-DNDEBUG`) and with optimizations (`-O3`), if your test involves mathematical operations add arithmetic optimizations (`-Ofast`).

Some features:

* Arbitrary pointer types
* Simplified implementation (~600 lines)
* Fast access of subarray (view) types
* Better semantics of subarray (view) types

## Types

* `multi::array<T, D, A>`: Array of dimension `D`, it has value semantics if `T` has value semantics. Memory is requested by allocator of type `A`, should support stateful allocators.
* `multi::array_ref<T, D, P = T*>`: Array interpretation of a contiguos range, usually a memory block. Thanks to inheritance an `array` is-a `array_ref`.
* `multi::array_ref<T, D>::(const_)iterator`: Iterator to subarrays of dimension `D-1`. For `D==1` this is an iterator to elements.
* `multi::array_ref<T, D>::(const_)reference`: Reference to subarrays of dimension `D-1`. For `D>1` this are not true C++-references but types mimic them (with reference semantics). For `D==1` this is a true C++ reference to an elements.
* other derived unspecified types fulfil (a still loosely defined) `ArrayConcept`, for example by taking partial indices or rotations (transpositions).


## Usage

We create a static C-array of `double`s, and refer to it via a bidimensional array `multi::array_ref<double, 2>`.


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


Note that the syntax of creating a reference array involves passing the pointer to a memory block (20 elements here) and the logical dimensions of that memory block (4 by 5 here).

Next we print the elements in a way that corresponds to the logical arrangement:

		for(auto i : d2D_ref.extension(0)){
			for(auto j : d2D_ref.extension(1))
				cout << d2D_ref[i][j] <<' ';
			cout <<'\n';
		}
	
This will output:

> 150 16 17 18 19  
> 30 1 2 3 4  
> 100 11 12 13 14  
> 50 6 7 8 9

It is sometimes said (Sean Parent) that the whole of STL algorithms can be seen as intermediate pieces to implement`std::stable_sort`. 
Pressumably if one can sort over a range, one can do anything.

		std::stable_sort( begin(d2D_ref), end(d2D_ref) );

If we print this we will get

> 30 1 2 3 4  
> 50 6 7 8 9  
> 100 11 12 13 14  
> 150 16 17 18 19


The array has been changed to be in row-based lexicographical order.
Since the sorted array is a reference to the original data, the original array has changed. 

		assert( d2D[1][1] == 6 );

Needless to say that `std::*sort` cannot be applied directly to a multidimensional C-array or to Boost.MultiArray types.

If we want to order the matrix in a per-column basis we need to "view" the matrix as range of columns. This is done in the bidimensional case, by accessing the matrix as a range of columns:

	std::stable_sort( d2D_ref.begin(1), d2D_ref.end(1) );

Which will transform the matrix into. 

> 1 2 3 4 30  
> 6 7 8 9 50  
> 11 12 13 14 100  
> 16 17 18 19 150 

In other words, a matrix of dimension `D` can be viewed simultaneously as `D` different ranges of different "transpositions" by passing an interger value to `begin` and `end` indicating the preferred dimension.
`begin(0)` is equivalent to `begin()`.

## Index Accessing

Many algorithms on arrays are oriented to linear algebra, which are ubiquitously implemented in terms of multidimensional index access. 

Index access is almost equivalent to those of C-fixed sizes arrays, for example a 3-dimensional array will access to an element by `m[1][2][3]`, which can be used for write and read operations. 
Partial index arguments `m[1][2]` generate a view 1-dimensional object.
Transpositions are also multi-dimensional arrays in which the index are logically rearranged, for example `m.rotated(1)[2][3][1] == m[1][2][3]`.

This example implements Gauss Jordan Elimination without pivoting:

```
template<class Matrix, class Vector>
void solve(Matrix& m, Vector& y){
	std::ptrdiff_t msize = size(m); 
	for(auto r = 0; r != msize; ++r){
		auto mr = m[r];
		auto mrr = mr[r];
		for(auto c = r + 1; c != msize; ++c) mr[c] /= mrr;
		auto yr = (y[r] /= mrr);
		for(auto r2 = r + 1; r2 != msize; ++r2){
			auto mr2 = m[r2];
			auto const& mr2r = mr2[r];
			auto const& mr = m[r];
			for(auto c = r + 1; c != msize; ++c) mr2[c] -= mr2r*mr[c];
			y[r2] -= mr2r*yr;
		}
	}
	for(auto r = msize - 1; r > 0; --r){
		auto const& yr = y[r];
		for(auto r2 = r-1; r2 >=0; --r2)
			y[r2] -= yr*m[r2][r];
	}
}
```