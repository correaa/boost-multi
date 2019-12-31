#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic $0 -o $0x &&$0x&&rm $0x;exit
#endif

#include "../array.hpp"

#include<cassert>

namespace multi = boost::multi;

int main(){
	{
		double a[4][5] = {
			{ 0,  1,  2,  3,  4}, 
			{ 5,  6,  7,  8,  9}, 
			{10, 11, 12, 13, 14}, 
			{15, 16, 17, 18, 19}
		};
		double b[4][5];
#ifdef __cpp_deduction_guides
		auto&& A = *multi::array_ptr(&a[0][0], {4, 5});
		multi::array_ref B(&b[0][0], {4, 5});
#else
		auto&& A = *multi::array_ptr<double, 2>(&a[0][0], {4, 5});
		multi::array_ref<double, 2, double*> B(&b[0][0], {4, 5});
#endif
		B = A;
		rotated(A) = rotated(B);
		assert( b[1][2] == a[1][2] );
	}
}

