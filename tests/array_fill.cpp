#ifdef COMPILATION_INSTRUCTIONS
c++ .DCATCH_CONFIG_MAIN -std=c++17 -Wall -Wextra $0 -o$0x &&$0x&&rm $0x; exit
#endif

#include "../array.hpp"
#include<catch.hpp>

namespace multi = boost::multi;

TEST_CASE("Array fill", "[array]"){

	multi::array<double, 2> d2D = {
		{150., 16., 17., 18., 19.},
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	using std::all_of;
	REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto& e){return e==5.;}) );

	using std::fill;

	fill(begin(d2D[1]), end(d2D[1]), 8.);
	REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto& e){return e==8.;}) );

	fill(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), 8.);
	REQUIRE( all_of(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), [](auto&& e){return e==8.;}) );

//	fill( d2D({0, 4}).begin(1), d2D({0, 4}).end(1), 9.);
//	fill(begin(d2D({0, 4}, 1)), end(d2D({0, 4}, 1)), 9.);
}

