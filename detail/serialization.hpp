// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// Copyright 2018-2021 Alfredo A. Correa

#ifndef _MULTI_DETAIL_SERIALIZATION_HPP
// #include<boost/serialization/nvp.hpp>  // not needed at this point, *indirectly* can be needed in user code

namespace boost {
namespace serialization {

template<class T> class nvp;            // dependency "in name only"
template<class T> class array_wrapper;  // dependency "in name only"
//template<class T> auto make_nvp(const char* n, T& v) noexcept -> const nvp<T>;  // NOLINT(readability-const-return-type) : original boost declaration, if you get an error here, maybe you need a boost version above 1.71

}  // end namespace serialization
}  // end namespace boost

namespace boost {
namespace multi {

template<class Ar, typename = decltype(typename Ar::template nvp<int>(std::declval<char const*>(), std::declval<int&>()))>
auto has_nvp_aux(Ar const&) -> std::true_type ;
auto has_nvp_aux(...      ) -> std::false_type;

template<class Ar, typename = decltype(typename Ar::template array_wrapper<int>(std::declval<int*>(), std::declval<std::size_t>()))>
auto has_array_wrapper_aux(Ar const&) -> std::true_type ;
auto has_array_wrapper_aux(...      ) -> std::false_type;

template<class Ar,
	typename = decltype(has_nvp_aux          (std::declval<Ar>())),
	typename = decltype(has_array_wrapper_aux(std::declval<Ar>()))
>
struct archive_traits;

template<class Ar>
struct archive_traits<Ar, std::true_type , std::true_type > {
	template<class T> using nvp           = typename Ar::template nvp          <T>;
	template<class T> using array_wrapper = typename Ar::template array_wrapper<T>;
	template<class T> inline static auto make_nvp  (char const* n, T& v) noexcept -> const nvp          <T> {return nvp          <T>{n, v};}  // NOLINT(readability-const-return-type) : original boost declaration
	template<class T> inline static auto make_array(T* t, std::size_t s) noexcept -> const array_wrapper<T> {return array_wrapper<T>{t, s};}  // NOLINT(readability-const-return-type) : original boost declaration
};

template<class Ar>
struct archive_traits<Ar, std::false_type, std::false_type> {
	template<class T> using nvp           = boost::serialization::nvp          <T>;
	template<class T> using array_wrapper = boost::serialization::array_wrapper<T>;
	template<class T> inline static auto make_nvp  (char const* n, T& v) noexcept -> const nvp          <T> {return nvp          <T>{n, v};}  // NOLINT(readability-const-return-type) : original boost declaration
	template<class T> inline static auto make_array(T* t, std::size_t s) noexcept -> const array_wrapper<T> {return array_wrapper<T>{t, s};}  // NOLINT(readability-const-return-type) : original boost declaration
};

}  // end namespace multi
}  // end namespace boost
















































































































#endif
