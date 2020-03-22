#ifdef COMPILATION_INSTRUCTIONS
$CXX -O2 $0 -o $0x -lboost_unit_test_framework &&$0x&&rm $0x;exit
#endif
// © Alfredo Correa 2018-2020
// this needs O3

#include "../array_ref.hpp"
#include "../array.hpp"

#include <range/v3/all.hpp>

#include<iostream>
#include<complex>

namespace multi = boost::multi;

int main(){

	multi::array<double, 2> d2D = {
		{ 0.,  1.,  2.,  3.}, 
		{ 5.,  6.,  7.,  8.}, 
		{10., 11., 12., 13.}, 
		{15., 16., 17., 18.}
	};
	assert( ranges::inner_product(d2D[0], d2D[1], 0.) == 6+2*7+3*8 );
	assert( ranges::inner_product(d2D[0], rotated(d2D)[0], 0.) == 1*5+2*10+15*3 );

//	assert( ranges::inner_product(d2D, d2D, 0., [](auto&& a, auto&& b){return a + b;}, [](auto&& x, auto&& y){return std::inner_product(x, y);}) );

//	static_assert(ranges::RandomAccessIterator<multi::array<double, 1>::iterator>{});
//	static_assert(ranges::RandomAccessIterator<multi::array<double, 2>::iterator>{});

//	static_assert(ranges::InputRange<multi::array<double, 2>>{});
//	static_assert(ranges::RandomAccessRange<multi::array<double, 2>>{});
//	static_assert(ranges::RandomAccessRange<decltype(d2D[2])>{});

	static_assert( ranges::forward_iterator<std::vector<double>::const_iterator>, "!");
	static_assert( ranges::readable<multi::array<double, 1>::const_iterator>, "!");

	using It = std::vector<bool>::iterator;//multi::array<double, 2>::iterator;
	It i; 
	typename It::value_type val = *i; (void)val;
	typename It::reference ref = *i;
	typename It::value_type val2{ref}; (void)val2;
//	static_assert( ranges::common_reference<typename It::reference&&, typename It::value_type&> );
//	ranges::rvalue_reference_t<It> rr;
	static_assert( ranges::readable<It>, "!");

	{ // this needs O2 to run without segmentation fault
		assert( ranges::inner_product(extension(d2D), extension(d2D), 0) == 0*0 + 1*1 + 2*2 + 3*3 );
	}
	return 0;
#if 0
//	static_assert( ranges::Readable<>{} );


	static_assert( ranges::RandomAccessIterator<multi::array<double, 2>::const_iterator>{} );



	static_assert( ranges::InputRange<multi::array<double, 2> const&>{} );

	multi::array<double, 2>::const_iterator CI;
	multi::array<double, 2>::value_type v{*CI};
	multi::array<double, 2>::const_reference const& r{*CI};

//	static_assert(ranges::Readable<multi::array<double, 2>::const_iterator>{});
	
	multi::array<double, 2>::const_iterator ci;
	auto const& ss = *ci;

	using complex = std::complex<double>;

	std::vector<bool> vb = {0, 1, 0, 0};
	
#if 1
//	using r = ranges::value_t<multi::array<double, 2>::const_iterator>;
//	using r = 	multi::array<double, 2>::const_iterator::iterator_category;
//	using r2 = 	std::iterator_traits<multi::array<double, 2>::const_iterator>::difference_type;


//	multi::array<complex, 1> const c1d = { {8.,9.},  {1., 2.} };
//	auto const& conjc = c1d | ranges::view::transform([](auto&& c){return conj(c);});
//	assert(( conjc[1] == complex{1., -2.} ));

//	auto const& realc = c1d | ranges::view::transform([](auto&& c){return real(c);});
//	assert(( realc[1] == 1. ));


	multi::array<complex, 2> const c2d = {
		{ {8., 1.}, {11.,2.} },
		{ {8., 2.}, {2.,3.} }
	}; assert( size(c2d) == 2 );

	auto const& row = c2d[0];
	auto const& realrow = row | ranges::view::transform([](auto&& e){return real(e);});
	assert( realrow[1] == 11. );

//	auto const& realrow2 = row | ranges::view::transform([](auto&& e){return real(e);});
//	assert( realrow2[1] == 11. );

	namespace view = ranges::view;
#endif

//	auto const& realc2d = 
//		c2d | 
//			view::transform([](auto const& row){
//				return row | view::transform([](auto const& e){return real(e);});
//			});
//	assert( realc2d[1][1] == 2.1 );

#if 0
//	auto const& c2d0 = c2d[0];
	auto realc2d = 
		c2d | 
			view::transform([](auto&& row){
				return row;
			//	auto const& rrr = row;
			//	auto const ret = rrr | view::transform([](auto const& e){return real(e);});
			//	return ret;
			});
#endif
//	assert( realc2d[1] == 11. );
#endif
}

