// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2018-2021 Alfredo A. Correa

#ifndef MULTI_DETAIL_SERIALIZATION_HPP

namespace boost {
namespace archive {
namespace detail {

template<class Ar> struct common_iarchive;
template<class Ar> struct common_oarchive;

}  // end namespace detail
}  // end namespace archive

namespace serialization {

template<class T> class  nvp;            // dependency "in name only"
template<class T> class  array_wrapper;  // dependency "in name only"
                  struct binary_object;  // dependency "in name only", if you get an error here, it means that eventually you need to include #include<boost/serialization/binary_object.hpp>

template<typename T> struct version;

class access;

}  // end namespace serialization
}  // end namespace boost

namespace cereal {

template<class ArchiveType, std::uint32_t Flags> struct OutputArchive;
template<class ArchiveType, std::uint32_t Flags> struct InputArchive;

template<class T> class NameValuePair;  // dependency "in name only", if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point

}  // end namespace cereal

namespace boost {
namespace multi {

template<class Ar, class Enable = void>
struct archive_traits {
	template<class T>
	inline static auto make_nvp  (char const* /*n*/, T&& v) noexcept {return std::forward<T>(v);}
};

template<class Ar>
struct archive_traits<Ar, typename std::enable_if<std::is_base_of<boost::archive::detail::common_oarchive<Ar>, Ar>::value or std::is_base_of<boost::archive::detail::common_iarchive<Ar>, Ar>::value>::type> {
	template<class T> using nvp           = boost::serialization::nvp          <T>;
	template<class T> using array_wrapper = boost::serialization::array_wrapper<T>;
	template<class T> struct binary_object_t {using type = boost::serialization::binary_object;};
	template<class T>        inline static auto make_nvp          (char const* n, T& v               ) noexcept -> const nvp          <T> {return nvp          <T>{n, v};}  // NOLINT(readability-const-return-type) : original boost declaration
	template<class T>        inline static auto make_array        (               T* t, std::size_t s) noexcept -> const array_wrapper<T> {return array_wrapper<T>{t, s};}  // NOLINT(readability-const-return-type) : original boost declaration
	template<class T = void> inline static auto make_binary_object(      const void* t, std::size_t s) noexcept -> const typename binary_object_t<T>::type {return typename binary_object_t<T>::type(t, s); }  // if you get an error here you need to eventually `#include<boost/serialization/binary_object.hpp>`// NOLINT(readability-const-return-type,clang-diagnostic-ignored-qualifiers) : original boost declaration
};


template<class Ar>
struct archive_traits<Ar, typename std::enable_if<
		   std::is_base_of<cereal::OutputArchive<Ar, 0>, Ar>::value or std::is_base_of<cereal::OutputArchive<Ar, 1>, Ar>::value
		or std::is_base_of<cereal::InputArchive <Ar, 0>, Ar>::value or std::is_base_of<cereal::InputArchive <Ar, 1>, Ar>::value
	>::type> {
	template<class T>
	inline static auto make_nvp  (char const* n, T&& v) noexcept {return cereal::NameValuePair<T>{n, v};}  // if you get an error here you many need to #include <cereal/archives/xml.hpp> at some point
};

}  // end namespace multi
}  // end namespace boost

namespace boost {

template<class T, std::size_t D, class As>
struct multi_array;

}  // end namespace boost

namespace boost {
namespace serialization {

//template<class Archive, class T, std::size_t D, class A>
//auto serialize(Archive& ar, boost::multi_array<T, D, A>& arr, unsigned int /*version*/)
//{
//	auto x = boost::multi::extensions(arr);
//	ar & multi::archive_traits<Archive>::make_nvp("extensions", x);
//	if( x != boost::multi::extensions(arr) ) {
//		arr.resize( std::array<std::size_t, 2>{} );
//		arr.resize( std::array<std::size_t, 2>{static_cast<std::size_t>(std::get<0>(x).size()), static_cast<std::size_t>(std::get<1>(x).size())} );
//	}
//	ar & multi::archive_traits<Archive>::make_nvp("data_elements", multi::archive_traits<Archive>::make_array(boost::multi::data_elements(arr), static_cast<std::size_t>(boost::multi::num_elements(arr))));
//}

}  // end namespace serialization
}  // end namespace boost

//BOOST_AUTO_TEST_CASE(boost_multi_array) {
//	boost::multi_array<double, 2> arr(boost::extents[10][10]);

////	BOOST_REQUIRE(( boost::multi_array<double, 2>::dimensionality == 2 ));
//	BOOST_REQUIRE(( boost::multi::extensions(arr) == boost::multi::extensions_t<2>{10, 10} ));
//	BOOST_REQUIRE( boost::multi::data_elements(arr) == arr.data() );
//	BOOST_REQUIRE( boost::multi::num_elements(arr) == static_cast<multi::size_t>(arr.num_elements()) );

//	std::stringstream ss;
//	{
//		{
//			boost::archive::xml_oarchive xoa{ss};
//			xoa<< BOOST_SERIALIZATION_NVP(arr);
//		}
//		std::ofstream ofs{"serialization_boost_multi_array.xml"};
//		ofs<< ss.str();
//	}
//}














































#endif
