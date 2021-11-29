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

#include <boost/multi_array.hpp>

#include <fstream>
#include <numeric>
#include <string>

namespace boost {
namespace serialization {

template<class Archive, class T>//, std::enable_if_t<(boost::multi_array<T, 2>::dimensionality == 2), int*> = 0>
void serialize(Archive& ar, boost::multi_array<T, 2>& arr, unsigned int /*version*/) {
	ar & multi::archive_traits<Archive>::make_nvp("00", arr[0][0]);
}

}  // end namespace serialization
}  // end namespace boost


namespace boost {
namespace multi {

template<class BoostMultiArray, std::size_t... I>
constexpr auto extensions_bma(BoostMultiArray const& arr, std::index_sequence<I...> /*012*/) {
	return boost::multi::extensions_t<BoostMultiArray::dimensionality>( boost::multi::iextension{static_cast<multi::index>(arr.index_bases()[I]), static_cast<multi::index>(arr.index_bases()[I] + arr.shape()[I])} ...);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template<class BoostMultiArray, std::enable_if_t<has_shape<BoostMultiArray>{} and not has_extensions<BoostMultiArray>{}, int> =0>
constexpr auto extensions(BoostMultiArray const& array) {
	return extensions_bma(array, std::make_index_sequence<BoostMultiArray::dimensionality>{});
}

}  // end namespace multi
}  // end namespace boost

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(carray_serialization) {
	double const A[3][3] = {{0., 1., 2.}, {3., 4., 5.}, {6., 7., 8.}};  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) test legacy types
	std::stringstream ss;
	{
		{
			boost::archive::xml_oarchive xoa{ss};
			xoa<< BOOST_SERIALIZATION_NVP(A);
		}
		std::ofstream ofs{"serialization_A.xml"};
		ofs<< ss.str();
	}
	{
		double B[3][3];  // NOLINT(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays) test legacy types
		boost::archive::xml_iarchive xia{ss};
		xia>> BOOST_SERIALIZATION_NVP(B);
		BOOST_REQUIRE( A[1][2] == B[1][2] );
	}
}

BOOST_AUTO_TEST_CASE(boost_multi_array) {
	boost::multi_array<double, 2> arr(boost::extents[10][10]);

//	BOOST_REQUIRE(( boost::multi_array<double, 2>::dimensionality == 2 ));
	BOOST_REQUIRE(( boost::multi::extensions(arr) == boost::multi::extensions_t<2>{10, 10} ));

	std::stringstream ss;
	{
		{
			boost::archive::xml_oarchive xoa{ss};
			xoa<< BOOST_SERIALIZATION_NVP(arr);
		}
		std::ofstream ofs{"serialization_boost_multi_array.xml"};
		ofs<< ss.str();
	}
}

BOOST_AUTO_TEST_CASE(array_serialization) {
	multi::array<double, 2> arr({10, 10}, 0.);

	BOOST_REQUIRE(( arr.extension() == boost::multi::index_range{0, 10} ));

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
