# Boost.Multi

(not an official Boost library)

Boost.Multi provides multidimensional array access to contiguous or regularly contiguous memory.
It shares the goals of Boost.MultiArray, although the code is completely independent and the syntax has slight differences.
Boost.Multi and Boost.MultiArray types can be used interchangeably for the most part, they differ in the semantics of reference and value types. 

Boost.Multi aims to simplify the semantics of Boost.MultiArray and make it more compatible with the Standard (STL) Algorithms and special memory.

Some features:

* Arbitrary pointer types
* Simplified implementation (~500 lines)
* Faster access of subarray (view) types
* Better semantics of subarray (view) types

## First Example

We create a static C-array of ``double`s, and refer to it via a bidimensional array `multi::array_ref<double, 2>`.


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

		for(auto i : d2D_ref.extensions()[0]){
			for(auto j : d2D_ref.extensions()[1])
				cout << d2D_ref[i][j] << ' ';
			cout << '\n';
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

Needless to say that ``std::*sort` cannot be applied directly to a multidimensional C-array or to Boost.MultiArray types.

If we want to order the matrix in a per-column basis we need to "view" the matrix as range of columns. This is done in the bidimensional case, by accessing the matrix as a range of columns:

	std::stable_sort( d2D_ref.begin(1), d2D_ref.end(1) );

Which will transform the matrix into. 

> 1 2 3 4 30  
> 6 7 8 9 50  
> 11 12 13 14 100  
> 16 17 18 19 150 

In other words, a matrix of dimension `D` can be viewed simultaneously as `D` different ranges by passing an interger value to `begin` and `end` indicating the preferred dimension. `bgi`
