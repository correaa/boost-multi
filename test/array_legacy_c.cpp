#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi legacy adaptor example"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include<iostream>

#include "../array_ref.hpp"
#include "../array.hpp"

#include<complex>
#include<iostream>
#include<tuple>
#include<vector>

namespace multi = boost::multi;
using std::cout; using std::cerr;

namespace fake{
typedef double fftw_complex[2];

using fftw_size_t = int;
void fftw_plan_dft(
	int rank, const fftw_size_t *n, 
	fftw_complex *in, fftw_complex *out, int sign, unsigned flags){
	(void)rank, (void)n, (void)in, (void)out, (void)sign, (void)flags;
}

}

BOOST_AUTO_TEST_CASE(array_legacy_c){
	using std::complex;

	multi::array<complex<double>, 2> const in = {
		{150., 16., 17., 18., 19.},
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	multi::array<std::complex<double>, 2> out(extensions(in));

	static_assert( out.dimensionality() == in.dimensionality(), "!");
	assert( sizes(out) == sizes(in) );

	auto const in_sizes = in.sizes();
	fake::fftw_plan_dft(
		2,
		std::array<fake::fftw_size_t, 2>{fake::fftw_size_t(std::get<0>(in_sizes)), fake::fftw_size_t(std::get<1>(in_sizes))}.data(), 
		(fake::fftw_complex*)in .data_elements(), 
		(fake::fftw_complex*)out.data_elements(), 
		1, 0
	);

struct basic : multi::layout_t<2>{
	double* p;
};

struct ref : basic{
};

struct test{
	multi::layout_t<2> l;
	double* p;
};

	{
		multi::array<double, 2> d2D = 
			{
				{150, 16, 17, 18, 19},
				{ 30,  1,  2,  3,  4}, 
				{100, 11, 12, 13, 14}, 
				{ 50,  6,  7,  8,  9} 
			};

	BOOST_TEST_REQUIRE( sizeof( multi::layout_t<0> ) == 1 );
	#if __has_cpp_attribute(no_unique_address) >=201803 and not defined(__NVCC__)
		BOOST_TEST_REQUIRE( sizeof( multi::layout_t<1> ) == 3*sizeof(std::size_t) );
		BOOST_TEST_REQUIRE( sizeof( multi::layout_t<2> ) == 6*sizeof(std::size_t) );
		BOOST_TEST( sizeof(test) == sizeof(double*)+6*sizeof(std::size_t) );
		BOOST_TEST( sizeof(d2D)==sizeof(double*)+6*sizeof(std::size_t) );
	#endif
		BOOST_REQUIRE(             d2D    .layout().is_compact() );
		BOOST_REQUIRE(     rotated(d2D)   .layout().is_compact() );
		BOOST_REQUIRE(             d2D [3].layout().is_compact() );
		BOOST_REQUIRE( not rotated(d2D)[2].layout().is_compact() );
	}
	{
		using complex = std::complex<double>;
		multi::array<complex, 2> d2D({5, 3});
		BOOST_REQUIRE(             d2D    .layout().is_compact() );
		BOOST_REQUIRE(     rotated(d2D)   .layout().is_compact() );
		BOOST_REQUIRE(             d2D [3].layout().is_compact() );
		BOOST_REQUIRE( not rotated(d2D)[2].layout().is_compact() );
	}

}

