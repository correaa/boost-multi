#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++17 -Wall -Wextra -Wpedantic -Werror `#-Wfatal-errors` -I$HOME/include $0 -o $0.x && $0.x $@ && rm -f $0.x; exit
#endif

#include<iostream>

#include "../array_ref.hpp"
#include "../array.hpp"

//#include <range/v3/algorithm/fill.hpp>
//#include <range/v3/algorithm/all_of.hpp>

#include<iostream>
#include<vector>

namespace multi = boost::multi;
using std::cout; using std::cerr;

int main(){

	boost::multi::layout_t<2>::extensions_type_ ieee(
		std::tuple<boost::multi::extension_t<long int>, boost::multi::extension_t<long int> >{
			boost::multi::extension_t<long int>{0, 20}, 
			boost::multi::extension_t<long int>{0, 10} 
		}
	);

	std::tuple<int, int, int> sizes_tuple{4, 5, 3};
	std::array<int, 3> sizes_array{4, 5, 3};
	multi::array<double, 3> a1(sizes_tuple);
//	std::tuple<int, int, int> sss{sizes_array};
	multi::array<double, 3> a2(sizes_tuple);
	multi::array<double, 3>::extensions_type_ et{sizes_array};
//	multi::extensions_t<3> aa{a1};

	multi::array<double, 2> d2D = {
		{150., 16., 17., 18., 19.},
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	assert( std::all_of(begin(d2D[1]), end(d2D[1]), [](auto&& e){return e == 5.;}) );

	std::fill(begin(d2D[1]), end(d2D[1]), 8.);
	assert( std::all_of(begin(d2D[1]), end(d2D[1]), [](auto&& e){return e == 8.;}) );

	std::fill(begin(d2D({0, 4}, 1)), end(d2D({0, 4}, 1)), 9.); //std::fill(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), 9.);
	assert( std::all_of(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), [](auto&& e){return e == 9.;}) );

//	ranges::v3::fill(d2D({0, 4}, 1), 42.);
//	ranges::v3::all_of(d2D({0, 4}, 1), [](auto&& e){return e==42.;});
}

