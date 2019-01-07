#ifdef COMPILATION_INSTRUCTIONS
c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic $0 -o $0.x && $0.x $@ && rm -f $0.x; exit
#endif

#include "../array.hpp"

namespace multi = boost::multi;

int main(){
{
	multi::array<double, 1> const A = {1.2,3.4,5.6}; 
	assert( size(A) == 3 );
	assert( A[2] == 5.6 );
}
{
	multi::array<double, 2> const A = {
		{ 1.2,  2.4, 3.6},
		{11.2, 34.4, 5.6},
		{15.2, 32.4, 5.6}
	};
	assert( size(A) == 3 );
	assert( size(A) == 3 and size(A[0]) == 3 );
	assert( A[1][1] == 34.4 );
}
{
	multi::array<double, 2> const A = {
		{ 1.2,  2.4},
		{11.2, 34.4},
		{15.2, 32.4}
	};
	assert( size(A) == 3 and size(A[0]) == 2 );
	assert( A[1][1] == 34.4 );
}
{
	multi::array<double, 3> const A = {
		{{ 1.2,  0.}, { 2.4, 1.}},
		{{11.2,  3.}, {34.4, 4.}},
		{{15.2, 99.}, {32.4, 2.}}
	};
	assert( A[1][1][0] == 34.4 and A[1][1][1] == 4.   );
}
#if __cpp_deduction_guides
{
	multi::array const A = {1.2,3.4,5.6}; 
	assert( size(A) == 3 );
	assert( A[2] == 5.6 );
}
{
	multi::array const A = {
		{ 1.2,  2.4, 3.6},
		{11.2, 34.4, 5.6},
		{15.2, 32.4, 5.6}
	};
	assert( size(A) == 3 );
	assert( size(A) == 3 and size(A[0]) == 3 );
	assert( A[1][1] == 34.4 );
}
{
	multi::array const A = {
		{ 1.2,  2.4},
		{11.2, 34.4},
		{15.2, 32.4}
	};
	assert( size(A) == 3 and size(A[0]) == 2 );
	assert( A[1][1] == 34.4 );
}
{
	multi::array const A = {
		{{ 1.2,  0.}, { 2.4, 1.}},
		{{11.2,  3.}, {34.4, 4.}},
		{{15.2, 99.}, {32.4, 2.}}
	};
	assert( A[1][1][0] == 34.4 and A[1][1][1] == 4.   );
}
#endif

}

