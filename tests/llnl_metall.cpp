#ifdef COMPILATION_INSTRUCTIONS
$CXX -std=c++17 -I$HOME/github.com/LLNL/metall.git/include/ $0 -o $0x -lstdc++fs&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2019-2020

#include<cassert>
#include<numeric> // iota
#include<iostream>

#include<metall/metall.hpp>

using namespace metall;

template<class T> using mallocator = manager::allocator_type<T>;

decltype(auto) get_allocator(manager& m){return m.get_allocator();}

void mremove(std::string const& s){for(auto fix:{"bin_directory","chunk_directory","named_object_directory","segment"}) std::filesystem::remove(s +"_"+ fix);}
std::string candidates(manager&){return "";}

#include "../../multi/array.hpp"

namespace multi = boost::multi;

template<class T, auto D> using marray = multi::array<T, D, mallocator<T>>;

using std::tuple;

int main(){

mremove("mapped_file.bin");
{
	manager m{create_only, "mapped_file.bin", 1 << 25};
	auto&& arr1d = 
		*m.construct<marray<int, 1>>("arr1d")(tuple{10}, 99, get_allocator(m));
	auto&& arr2d = 
		*m.construct<marray<double, 2>>("arr2d")(tuple{1000, 1000}, 1.0, get_allocator(m));
	auto&& arr3d = 
		*m.construct<marray<unsigned, 3>>("arr3d")(tuple{10, 10, 10}, 1u, get_allocator(m));
	auto&& arr3d_cpy = 
		*m.construct<marray<unsigned, 3>>("arr3d_cpy")(tuple{0, 0, 0}, get_allocator(m));

	assert( arr1d[3] == 99 );
	assert( arr2d[4][5] == 1.0 );
	assert( arr3d[2][3][4] == 1u );

	arr1d[3] = 33;
	arr2d[4][5] = 45.001;
	std::iota(arr3d[6][7].begin(), arr3d[6][7].end(), 100);

	arr3d_cpy = arr3d;
	assert( arr3d_cpy[6][7][8] == arr3d[6][7][8] );
	m.flush();
}
{
	manager m{open_only, "mapped_file.bin"};

	auto&& arr1d =
		*m.find<marray<int, 1>>("arr1d").first; assert(std::addressof(arr1d));
	auto&& arr2d =
		*m.find<marray<double, 2>>("arr2d").first; assert(std::addressof(arr2d));
	auto&& arr3d =
		*m.find<marray<unsigned, 3>>("arr3d").first; assert(std::addressof(arr3d));
	auto&& arr3d_cpy =
		*m.find<marray<unsigned, 3>>("arr3d_cpy").first; assert(std::addressof(arr3d));

	assert( arr1d[5] == 99 );
	assert( arr1d[3] == 33 );

	assert( arr2d[7][8] == 1.0 );
	assert( arr2d[4][5] == 45.001 );

	assert( arr3d[6][7][3] == 103 );
	assert( arr3d_cpy == arr3d );

	m.destroy<marray<int, 1>>("arr1d");//	eliminate<marray<int, 1>>(m, "arr1d"); 
	m.destroy<marray<double, 2>>("arr2d");//	eliminate<marray<double, 2>>(m, "arr2d");
	m.destroy<marray<unsigned, 3>>("arr3d");//	eliminate<marray<unsigned, 3>>(m, "arr3d");
}
mremove("mapped_file.bin");
}

