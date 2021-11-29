// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2019-2021

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi allocators"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/vector.hpp>

#include <fstream>
#include <numeric>
#include <string>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(array_serialization) {
	multi::array<double, 2> arr({10, 10}, 0.);
	std::iota(arr.data_elements(), arr.data_elements() + arr.num_elements(), 1000.);

	std::stringstream ss;
	{
		{
			boost::archive::xml_oarchive xoa{ss};
			xoa<< BOOST_SERIALIZATION_NVP(arr);
		}
		std::ofstream ofs{"serialization.xml"};
		ofs<< ss.str();
	}
	{
		boost::archive::xml_iarchive xia{ss};
		multi::array<double, 2> arr2;
		xia>> BOOST_SERIALIZATION_NVP(arr2);
		BOOST_REQUIRE( extensions(arr2) == extensions(arr) );
		BOOST_REQUIRE( arr2 == arr );
	}
}

BOOST_AUTO_TEST_CASE(array_serialization_string) {
	multi::array<std::string, 2> arr({10, 10});
	auto const x = extensions(arr);
	for(auto i : std::get<0>(x) ) {
		for(auto j : std::get<1>(x) ) {
			arr[i][j] = std::to_string(i) + std::to_string(j);
		}
	}

	std::stringstream ss;
	{
		{
			boost::archive::xml_oarchive xoa{ss};
			xoa<< BOOST_SERIALIZATION_NVP(arr);
		}
		std::ofstream ofs{"serialization_string.xml"};
		ofs<< ss.str();
	}
	{
		boost::archive::xml_iarchive xia{ss};
		multi::array<std::string, 2> arr2;
		xia>> BOOST_SERIALIZATION_NVP(arr2);
		BOOST_REQUIRE( extensions(arr2) == extensions(arr) );
		BOOST_REQUIRE( arr2 == arr );
	}
}

//BOOST_AUTO_TEST_CASE(std_vector_serialization) {
//	std::vector<double> vec(4);
//	std::iota(begin(vec), end(vec), 0);
//	{
//		std::ofstream ofs{"default.xml"};
//		boost::archive::xml_oarchive xoa{ofs, boost::archive::no_header};
//		xoa<< BOOST_NVP(vec);
//	}
//	{
//		std::ofstream ofs{"array.xml"};
//		boost::archive::xml_oarchive xoa{ofs, boost::archive::no_header};
//		auto const size = vec.size();
//		xoa<< BOOST_NVP(size) << boost::serialization::make_nvp("data", boost::serialization::make_array(vec.data(), vec.size()));
//	}
//	{
//		std::ofstream ofs{"binary.xml"};
//		boost::archive::xml_oarchive xoa{ofs, boost::archive::no_header};
//		auto const size = vec.size();
//		xoa<< BOOST_NVP(size) << boost::serialization::make_nvp("binary_data", boost::serialization::make_binary_object(vec.data(), vec.size()*sizeof(double)));
//	}
//}
