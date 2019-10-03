#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++17 -Wall -Wextra -Wpedantic $0 -o$0x && $0x && rm $0x; exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

#include<algorithm>
#include<cassert>
#include<iostream>
#include<cmath>
#include<vector>
#include<list>
#include<numeric> //iota

using std::cout; using std::cerr;
namespace multi = boost::multi;

double f(){return 5.;}

int main(){
	{
		double a[4][5] {
			{ 0,  1,  2,  3,  4}, 
			{ 5,  6,  7,  8,  9}, 
			{10, 11, 12, 13, 14}, 
			{15, 16, 17, 18, 19}
		};
		double b[4][5];
		auto&& A = *multi::array_ptr<double, 2>(&a[0][0], {4, 5});
		multi::array_ref<double, 2, double*> B(&b[0][0], {4, 5});
		B = A;
		rotated(A) = rotated(B);
		assert( b[1][2] == a[1][2] );
	}
}

