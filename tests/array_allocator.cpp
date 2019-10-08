#ifdef COMPILATION_INSTRUCTIONS
nvcc -O3 -std=c++14 --compiler-options -std=c++17,-Wall,-Wextra,-Wpedantic,-Wfatal-errors $0 -o $0x && $0x $@ && rm $0x; exit
#endif

#include "../array.hpp"

//#include "../../multi/detail/cuda/allocator.hpp"
#include "../../multi/memory/stack.hpp"

#include<vector>
#include<complex>
#include<iostream>
#include<scoped_allocator>

namespace multi = boost::multi;
//namespace cuda = multi::detail::memory::cuda;
using std::cout;

int main(){
{
	std::vector<multi::array<double, 2>> VA;
	for(int i = 0; i != 10; ++i)
		VA.emplace_back(multi::index_extensions<2>{i, i}, i);
	assert( size(VA[8]) == 8 );
	assert( VA[2][0][0] == 2 );
}
{
	multi::array<multi::array<double, 3>, 2> AA({10, 20});
	for(int i = 0; i != 10; ++i)
		for(int j = 0; j != 20; ++j)
			AA[i][j] = multi::array<double, 3>({i+j, i+j, i+j}, 99.);

	assert( AA[9][19][1][1][1] == 99. );
}
return 0;
#if 0
{
	using inner_array3 = multi::array<double, 3, cuda::allocator</*double*/>>;
	using outer_array2 = multi::array<inner_array3, 2/*, std::allocator<inner_array3>*/>;
	outer_array2 AA({2, 2});
	for(int i=0; i!=2; ++i) for(int j=0; j!=2; ++j) AA[i][j] = inner_array3({i+j, i+j, i+j}, 99.);
}
{
	multi::stack_buffer<cuda::allocator<char>, 32> buf{10000000};
	using inner_array3 = multi::array<double, 3, multi::stack_allocator<double, cuda::allocator<char>, 32>>;
	using outer_array2 = multi::array<inner_array3, 2/*, std::allocator<inner_array3>*/>;
	inner_array3::allocator_type iaa{&buf};
	inner_array3 ia{&buf};
	outer_array2 AA({2, 2}, inner_array3(&buf));
	for(int i=0; i!=2; ++i) for(int j=0; j!=2; ++j)
		AA[i][j] = multi::array<int, 3>({5, 5, 5}, 12);//inner_array3({i+j, i+j, i+j}, 99., &buf);
	assert( AA[1][1][4][4][4] == 12 );
}
{
	multi::stack_buffer<cuda::allocator<>> buf{10000000};
	using inner_array3 = multi::array<double, 3, multi::stack_allocator<void, cuda::allocator<>>>;
	using scoped_alloc = std::scoped_allocator_adaptor<std::allocator<void>, inner_array3::allocator_type>;
	multi::array<inner_array3, 2, scoped_alloc> AA({2, 2}, {std::allocator<void>{}, &buf});
	for(int i=0; i!=2; ++i) for(int j=0; j!=2; ++j) 
		AA[i][j] = multi::array<int, 3>({5, 5, 5}, 66);
	assert( AA[1][1][4][4][4] == 66 );
}
#endif
}

