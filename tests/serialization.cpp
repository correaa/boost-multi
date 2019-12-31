#ifdef COMPILATION_INSTRUCTIONS
$CXX -std=c++17 -Wall -Wextra $0 -lboost_unit_test_framework -o$0x -lboost_serialization -lboost_iostreams &&$0x&&rm $0x;exit
#endif

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi fill"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include<boost/archive/xml_oarchive.hpp>
#include<boost/archive/xml_iarchive.hpp>
#include<boost/archive/text_oarchive.hpp>
#include<boost/archive/text_iarchive.hpp>
#include<boost/archive/binary_oarchive.hpp>
#include<boost/archive/binary_iarchive.hpp>

#include<boost/serialization/nvp.hpp>
#include<boost/serialization/complex.hpp>

#include<boost/iostreams/filtering_stream.hpp>
#include<boost/iostreams/filter/gzip.hpp>

#include<fstream>
#include<filesystem>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_serialization){
	using complex = std::complex<float>;

	multi::array<complex, 2> d2D = {
		{150., 16., 17., 18., 19.}, 
		{  5.,  5.,  5.,  5.,  5.}, 
		{100., 11., 12., 13., 14.}, 
		{ 50.,  6.,  7.,  8.,  9.}  
	};
	d2D.reextent({1000, 1000});

	{
		std::ofstream ofs{"serialization.xml"}; assert(ofs);
		boost::archive::xml_oarchive{ofs} << BOOST_SERIALIZATION_NVP(d2D);
	}
	{
		std::ifstream ifs{"serialization.xml"}; assert(ifs);
		multi::array<complex, 2> d2D_copy;//(extensions(d2D), 9999.);
		boost::archive::xml_iarchive{ifs} >> BOOST_SERIALIZATION_NVP(d2D_copy);
		BOOST_REQUIRE( d2D_copy == d2D );
	}

	{
		std::ofstream ofs{"serialization.txt"}; assert(ofs);
		boost::archive::text_oarchive{ofs} << d2D;
	}
	{
		std::ifstream ifs{"serialization.txt"}; assert(ifs);
		multi::array<complex, 2> d2D_copy;//(extensions(d2D), 9999.);
		boost::archive::text_iarchive{ifs} >> d2D_copy;
		BOOST_REQUIRE( d2D_copy == d2D );
	}

	{
		std::ofstream ofs{"serialization.bin"}; assert(ofs);
		boost::archive::binary_oarchive{ofs} << d2D;
	}
	{
		std::ifstream ifs{"serialization.bin"}; assert(ifs);
		multi::array<complex, 2> d2D_copy;//(extensions(d2D), 9999.);
		boost::archive::binary_iarchive{ifs} >> d2D_copy;
		BOOST_REQUIRE( d2D_copy == d2D );
	}
	{
		std::ofstream ofs{"serialization_compressed.bin.gz"};
		{
			boost::iostreams::filtering_stream<boost::iostreams::output> f;
			f.push(boost::iostreams::gzip_compressor());
			f.push(ofs);
			boost::archive::binary_oarchive{f} << d2D;
		}
	}
}

