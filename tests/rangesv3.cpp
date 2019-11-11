#ifdef COMPILATION_INSTRUCTIONS
$CXX -O1 -Wall -Wextra -Wpedantic $0 -o$0x &&$0x&& rm $0x;exit
#endif

#include "../array_ref.hpp"
#include "../array.hpp"

#include <range/v3/all.hpp>

#include<iostream>
#include<complex>

using std::cout; using std::cerr;
namespace multi = boost::multi;

template<class T> T& stick(T&& t){return t;}

#include "../adaptors/rangev3.hpp"

int main(){

	multi::array<double, 2> d2D = {
		{ 0.,  1.,  2.,  3.},
		{ 5.,  6.,  7.,  8.}, 
		{10., 11., 12., 13.}, 
		{15., 16., 17., 18.}
	};
	assert( ranges::inner_product(d2D[0], d2D[1], 0.) == 6+2*7+3*8 );
	assert( ranges::inner_product(d2D[0], rotated(d2D)[0], 0.) == 1*5+2*10+15*3 );

//	ranges::transform(begin(d2D[0]), end(d2D[0]), begin(d2D[1]), begin(d2D[2]), [](auto const& a, auto const& b){return a + b;});

//	auto row0 = d2D[0]; auto row1 = d2D[1];
	using std::get; using std::cout; using std::endl;
//	auto d2D0 = d2D[0];
//	auto d2D1 = d2D[1];
	ranges::for_each(ranges::view::zip(d2D[0](), d2D[1]()),  // produce segmentation fault in O0
		[](auto&& p){cout<< get<0>(p) <<' '<< get<1>(p) <<endl;}
	);

//	assert( ranges::inner_product(d2D, d2D, 0., [](auto&& a, auto&& b){return a + b;}, [](auto&& x, auto&& y){return std::inner_product(x, y);}) );

//	static_assert(ranges::RandomAccessIterator<multi::array<double, 1>::iterator>{});
//	static_assert(ranges::RandomAccessIterator<multi::array<double, 2>::iterator>{});

//	static_assert(ranges::InputRange<multi::array<double, 2>>{});
//	static_assert(ranges::RandomAccessRange<multi::array<double, 2>>{});
//	static_assert(ranges::RandomAccessRange<decltype(d2D[2])>{});

{
	boost::multi::array<double, 1, std::allocator<double> > arr;

	using I = multi::array<double, 3>::const_iterator;
		static_assert(
 			ranges::Same<
				ranges::common_reference_t<
					ranges::reference_t<I>&&, ranges::value_type_t<I>&
				>, ranges::common_reference_t<
					ranges::value_type_t<I>&, 
					ranges::reference_t<I>&&
				>
			>{}, "!"
		);
		static_assert( 
			ranges::ConvertibleTo<ranges::reference_t<I>&&, ranges::common_reference_t<ranges::reference_t<I>&&, ranges::value_type_t<I>&>>{}, "!"
		);
		static_assert( 
			ranges::ConvertibleTo<ranges::value_type_t<I>&, ranges::common_reference_t<ranges::reference_t<I>&&, ranges::value_type_t<I>&>>{}, "!"
		);
		static_assert( ranges::CommonReference<ranges::reference_t<I>&&, ranges::value_type_t<I>&>{} , "!" );
		static_assert( ranges::CommonReference<ranges::reference_t<I>&&, ranges::rvalue_reference_t<I>&&>{}, "!" );
		static_assert( ranges::CommonReference<ranges::rvalue_reference_t<I>&&, const ranges::value_type_t<I>&>{}, "!");

	static_assert( ranges::Readable<I>{}, "!");
	static_assert( ranges::InputIterator<I>{}, "!");
	static_assert( ranges::ForwardIterator<I>{}, "!");
	static_assert( ranges::BidirectionalIterator<I>{}, "!");
	static_assert( ranges::TotallyOrdered<I>{}, "!");
	static_assert( ranges::SizedSentinel<I, I>{}, "!");
	static_assert( ranges::RandomAccessIterator<I>{}, "!");
	static_assert( ranges::Readable<I>{}, "!");
	const ranges::difference_type_t<I> n = {};
	static_assert( std::is_same<decltype(std::declval<I&>() += n), I&>{}, "!");
	static_assert( std::is_convertible<decltype(std::declval<I const&>() + n), I&&>{}, "!");
	static_assert( std::is_convertible<decltype(n + std::declval<I const&>()), I&&>{}, "!");
	static_assert( std::is_same<decltype(std::declval<I&>() -= n), I&>{}, "!");
	static_assert( std::is_convertible<decltype(std::declval<I const&>() - n), I&&>{}, "!");
	static_assert( std::is_same<decltype(std::declval<I const&>()[n]), ranges::reference_t<I>>{}, "!");
}
//	I ii;
//	double d = 6.;
//	what<decltype(d + 7.)>();

	using It = std::vector<bool>::iterator;//multi::array<double, 2>::iterator;
	It i; 
	typename It::value_type val = *i; (void)val;
	typename It::reference ref = *i;
	typename It::value_type val2{ref}; (void)val2;
	static_assert( ranges::CommonReference<typename It::reference&&, typename It::value_type&>{}, "!");
//	ranges::rvalue_reference_t<It> rr;
	static_assert( ranges::Readable<It>::value, "!");
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

