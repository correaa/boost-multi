#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++17 -Wall -Wextra -Wpedantic `#-Wfatal-errors` $0 -o $0.x && $0.x $@ && rm $0.x; exit
#endif

#include<iostream>
#include "../array.hpp"

#include<queue>
#include<vector>
#include<numeric> // iota

namespace multi = boost::multi;
using std::cout;

int main(){
	multi::array<double, 2> A({2, 3}, 0.);
	assert( num_elements(A)==6 );
	A[1][2] = 6.;

//	multi::array<double, 2> B(std::tuple(2, 3)); assert(B.size() == 2);
	multi::array<double, 2> C({2, 3}); assert(C.size() == 2 and C[0].size() == 3);


	A.reextent({5, 4}, 99.); 
	assert( num_elements(A)== 20 );
	assert( A[1][2] == 6. ); // careful, reextent preserves values when it can
	assert( A[4][3] == 99. );

	A = multi::array<double, 2>(extensions(A), 123.); // this is not as inefficient as you might think
	assert( A[1][2] == 123. );

	A.clear();
	assert( num_elements(A) == 0 );
	A.reextent({5, 4}, 66.);
	assert( A[4][3] == 66. );		
}

