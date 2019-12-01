#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra $0 -lboost_unit_test_framework -o$0x&&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi fill"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<random>

#include "../array.hpp"

std::random_device r;


BOOST_AUTO_TEST_CASE(fill){
	namespace multi = boost::multi;

	multi::array<double, 2> d2D = {
		{150., 16., 17., 18., 19.}, 
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	using std::all_of;
	BOOST_REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto& e){return e==5.;}) );

	using std::fill;
	fill(begin(d2D[1]), end(d2D[1]), 8.);
	BOOST_REQUIRE( all_of(begin(d2D[1]), end(d2D[1]), [](auto& e){return e==8.;}) );

	fill(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), 8.);
	BOOST_REQUIRE( all_of(begin(rotated(d2D)[1]), end(rotated(d2D)[1]), [](auto&& e){return e==8.;}) );

	fill(begin((d2D<<1)[1]), end((d2D<<1)[1]), 8.);
	BOOST_REQUIRE( all_of(begin((d2D<<1)[1]), end((d2D<<1)[1]), [](auto&& e){return e==8.;}) );

	std::mt19937 g{r()};
	auto rand = [d=std::normal_distribution<>{}, g = std::mt19937{r()}]() mutable{return d(g);};
	multi::array<double, 2> r2D({5, 5});
	std::for_each(begin(r2D), end(r2D), [&](auto&& e){std::generate(begin(e), end(e), rand);});

}

