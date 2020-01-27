#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -std=c++14 -Wall -Wextra -Wpedantic -D_TEST_MULTI_DETAIL_ADL $0.cpp -o$0x -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2020
#ifndef MULTI_DETAIL_ADL_HPP
#define MULTI_DETAIL_ADL_HPP
#include<cstddef> // std::size_t
#include<type_traits> // std::conditional_t
#include<utility>

#include<algorithm> // copy, copy_n, equal
#include<iterator> // begin, end

#define RET(ExpR) decltype(ExpR){return ExpR;}
#define BOOST_MULTI_DEFINE_ADL(FuN) \
namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct FuN##_t;} \
	static constexpr class{ \
		template<std::size_t I> struct priority : std::conditional_t<I==0, std::true_type, priority<I-1>>{}; \
		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete; \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::FuN(std::forward<As>(as)...)) \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(     FuN(std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).FuN(std::forward<As>(as)...))     \
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::FuN##_t<As&&...>::_(std::forward<As>(as)...)) \
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<4>{}, std::forward<As>(as)...)){return _(priority<4>{}, std::forward<As>(as)...);} \
	} FuN; \
} \
}} void* FuN##dummy

BOOST_MULTI_DEFINE_ADL(copy_n);
BOOST_MULTI_DEFINE_ADL(copy);
BOOST_MULTI_DEFINE_ADL(equal);
BOOST_MULTI_DEFINE_ADL(lexicographical_compare);
BOOST_MULTI_DEFINE_ADL(swap_ranges);

BOOST_MULTI_DEFINE_ADL(begin);
BOOST_MULTI_DEFINE_ADL(end);

#ifdef _TEST_MULTI_DETAIL_ADL

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi initializer_list"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE(multi_detail_adl){

	std::vector<double> v{1., 2., 3.};
	std::vector<double> w(3);

	boost::multi::adl::copy_n(v.data(), 3, w.data());
	

}
#endif
#endif

