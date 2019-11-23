#ifdef COMPILATION_INSTRUCTIONS
$CXX -O1 -Wall -Wextra -Wpedantic $0 -o$0x -lboost_unit_test_framework&&$0x&& rm $0x;exit
#endif
// © Alfredo A. Correa 2019

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi BLAS scal"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array_ref.hpp"
#include "../array.hpp"

#include <range/v3/all.hpp>

#include<iostream>
#include<complex>

using std::cout; using std::cerr;
namespace multi = boost::multi;

template<class T> T& stick(T&& t){return t;}

#include "../adaptors/rangev3.hpp"

BOOST_AUTO_TEST_CASE(multi_rangev3){

	multi::array<double, 2> d2D = {
		{ 0.,  1.,  2.,  3.},
		{ 5.,  6.,  7.,  8.}, 
		{10., 11., 12., 13.}, 
		{15., 16., 17., 18.}
	};
	{
		BOOST_REQUIRE( ranges::inner_product(d2D[0], d2D[1], 0.) == 6+2*7+3*8 );
		BOOST_REQUIRE( ranges::inner_product(d2D[0], rotated(d2D)[0], 0.) == 1*5+2*10+15*3 );
	}
	{
		ranges::transform(begin(d2D[0]), end(d2D[0]), begin(d2D[1]), begin(d2D[2]), [](auto const& a, auto const& b){return a + b;});
		BOOST_REQUIRE( d2D[2][1] == d2D[0][1] + d2D[1][1] );

		ranges::transform(d2D[0], d2D[1], begin(d2D[3]), std::multiplies<>{});
		BOOST_REQUIRE( d2D[3][1] == d2D[0][1] * d2D[1][1] );
	}
	{
		using std::get; using std::cout; using std::endl;
		ranges::for_each(ranges::view::zip(d2D[0](), d2D[1]()),  // produce segmentation fault in O0
			[](auto&& p){ p = std::make_pair(90, 91);}// get<1>(p) = 91.;}
		);
		BOOST_REQUIRE( d2D[0][2] == 90 );
		BOOST_REQUIRE( d2D[1][2] == 91 );
	}
	{
		std::random_device r;
		auto N = [d=std::normal_distribution<>{}, g = std::mt19937{r()}]() mutable{return d(g);};

		multi::array<double, 2> r2D({5, 5});
		ranges::for_each(r2D, [&](auto&& e){ranges::generate(e, N);});
	}

//	assert( ranges::inner_product(d2D, d2D, 0., [](auto&& a, auto&& b){return a + b;}, [](auto&& x, auto&& y){return std::inner_product(x, y);}) );

//	static_assert(ranges::RandomAccessIterator<multi::array<double, 1>::iterator>{});
//	static_assert(ranges::RandomAccessIterator<multi::array<double, 2>::iterator>{});

//	static_assert(ranges::InputRange<multi::array<double, 2>>{});
//	static_assert(ranges::RandomAccessRange<multi::array<double, 2>>{});
//	static_assert(ranges::RandomAccessRange<decltype(d2D[2])>{});

	{
		using Arr3 = multi::array<double, 3>;
		using I = typename Arr3::const_iterator;
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

		static_assert( ranges::RandomAccessRange<Arr3>{}, "!");
	}
	using complex = std::complex<double>;
	{
		multi::array<double, 1> v = {1.,2.,3.};
		static_assert( ranges::RandomAccessRange<decltype(v)>{}, "!");

		auto const& mv = v | ranges::view::transform([](auto e){return -e;});
//			ranges::end(mv);
//			ranges::begin(mv);
//			static_assert( ranges::Range<decltype(mv)>{}, "!");
//			static_assert( ranges::InputRange<decltype(mv)>{}, "!");
//			static_assert( ranges::ForwardRange<decltype(mv)>{}, "!");
//			static_assert( ranges::BidirectionalRange<decltype(mv)>{}, "!");
//		static_assert( ranges::RandomAccessRange<decltype(mv)>{}, "!");
//		assert( mv[1] == -2. );
		BOOST_REQUIRE( mv.begin()[1] == -2 );
	}
	{
		
		multi::array<complex, 1> const c1d = { {8.,9.},  {1., 2.} };
		static_assert( ranges::RandomAccessRange<multi::array<complex, 1>>{}, "!");

		auto&& conjc = c1d | ranges::view::transform([](auto&& e){return conj(e);});

		//	static_assert( ranges::BidirectionalRange<decltype(conjc)>{}, "!");
	//	static_assert( ranges::RandomAccessRange<decltype(conjc)>{}, "!");
//		static_assert( ranges::RandomAccessRange<>{}, "!");

		BOOST_REQUIRE(( begin(conjc)[1] == complex{1., -2.} ));
	}
	{
		multi::array<complex, 2> const c2d = {
			{ {8., 1.}, {11.,2.} },
			{ {8., 2.}, {2.,3.} }
		};
	// 	auto const& row = c2d[0];
	//	auto const& realrow = row | ranges::view::transform([](auto&& e){return real(e);});
	//	BOOST_REQUIRE( realrow
	}
#if 0	
#if 1



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

