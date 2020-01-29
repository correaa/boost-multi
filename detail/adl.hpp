#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -Wall -Wextra -Wpedantic -D_TEST_MULTI_DETAIL_ADL $0.cpp -o$0x -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2020
#ifndef MULTI_DETAIL_ADL_HPP
#define MULTI_DETAIL_ADL_HPP
#include<cstddef> // std::size_t
#include<type_traits> // std::conditional_t
#include<utility>

#include<memory> // uninitialized_copy, etc
#include<algorithm> // copy, copy_n, equal
#include<iterator> // begin, end

#include "../detail/memory.hpp"

namespace boost{namespace multi{
	template<std::size_t I> struct priority : std::conditional_t<I==0, std::true_type, struct priority<I-1>>{}; 
}}

#define RET(ExpR) decltype(ExpR){return ExpR;}
#define BOOST_MULTI_DEFINE_ADL(FuN) \
namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct FuN##_t;} \
	static constexpr class FuN##_t{ \
		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete; \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::FuN(std::forward<As>(as)...)) \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(     FuN(std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).FuN(std::forward<As>(as)...))     \
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::FuN##_t<As&&...>::_(std::forward<As>(as)...)) \
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<4>{}, std::forward<As>(as)...)){return _(priority<4>{}, std::forward<As>(as)...);} \
	} FuN; \
} \
}} void* FuN##_dummy

BOOST_MULTI_DEFINE_ADL(copy_n);

#define RETU(ExpR) {return ExpR;}

//BOOST_MULTI_DEFINE_ADL(copy);
namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct copy_t;} \
	static constexpr class copy_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,  As&&... as) const = delete;*/                                                            \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(              std::copy              (std::forward<As>(as)...)) \
/*		template<class T, class... As> auto _(priority<2>, T&& t, As&&... as) const->RET(                   copy<T&&>(std::forward(t), std::forward(as)...))*/ \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(               adl_copy              (std::forward<As>(as)...)) \
		template<class... As>          auto _(priority<3>,        As&&... as) const->RET(                   copy              (std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<4>, T&& t, As&&... as) const->RET(std::decay_t<T>::copy(std::forward<T>(t), std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<5>, T&& t, As&&... as) const->RET(std::forward<T>(t).copy              (std::forward<As>(as)...)) \
/*		template<class... As>          auto _(priority<5>,        As&&... as) const->RET(custom::           copy_t<As&&...>::_(std::forward<As>(as)...))*/ \
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<5>{}, std::forward<As>(as)...)){return _(priority<5>{}, std::forward<As>(as)...);} \
	} copy; \
} \
}} void* copy_dummy;

namespace boost{namespace multi{

template<class Alloc, class InputIt, class ForwardIt>
auto alloc_uninitialized_copy(Alloc& a, InputIt f, InputIt l, ForwardIt d){
	ForwardIt c = d;
	try{
		for(; f != l; ++f, ++d) a.construct(to_address(d), *f);
		return c;
	}catch(...){
		for(; c != d; ++d) a.destroy(to_address(d));
		throw;
	}
}

template<class T, typename = decltype(std::pointer_traits<T>::to_address(std::declval<T const&>()))> 
                  auto use_address_aux(T const& p)->std::true_type;
template<class T> auto use_address_aux(...       )->std::false_type;

template<class T> struct use_address : decltype(use_address_aux<T>()){};

#if 1
template<class T>
constexpr T* to_address(T* p, void* = 0) noexcept
{
    static_assert(!std::is_function<T>{}, "!");
    return p;
}
#endif

template<class T>
auto to_address_aux(const T& p, std::true_type) noexcept{
   return std::pointer_traits<T>::to_address(p);
}

template<class T>
auto to_address_aux(const T& p, std::false_type) noexcept
//->decltype(to_address(p.operator->()))
{
	return to_address(p.operator->());}
 
template<class T>
auto to_address(const T& p) noexcept{
	return to_address_aux(p, use_address<T>{});
//    if constexpr (use_address<T>::value) {
//       return std::pointer_traits<T>::to_address(p);
//    } else {
//       return memory::to_address(p.operator->());
//   }
}

template<class Alloc, class InputIt, class Size, class ForwardIt>//, typename AT = std::allocator_traits<Alloc> >
auto alloc_uninitialized_copy_n(Alloc& a, InputIt f, Size n, ForwardIt d)
//->std::decay_t<decltype(a.construct(to_address(d), *f), d)>
{
	ForwardIt c = d;
	try{
		for(; n > 0; ++f, ++c, --n) a.construct(to_address(c), *f);
		return c;
	}catch(...){
		for(; d != c; ++d) a.destroy(d);
		throw;
	}
}

template<class Alloc, class ForwardIt, class Size, class T>//, typename AT = typename std::allocator_traits<Alloc> >
ForwardIt alloc_uninitialized_fill_n(Alloc& a, ForwardIt first, Size n, const T& v){
	ForwardIt current = first; // using std::to_address;
	try{
		for(; n > 0; ++current, --n) a.construct(to_address(current), v);
		//	allocator_traits<Alloc>::construct(a, to_address(current), v);
		//	a.construct(to_address(current), v); //	AT::construct(a, to_address(current), v); //	AT::construct(a, addressof(*current), v); //a.construct(addressof(*current), v);
		return current;
	}catch(...){
		for(; first != current; ++first) a.destroy(first); 
		throw;
	}
}

}}

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct uninitialized_copy_t;} \
	static constexpr class uninitialized_copy_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(     std::uninitialized_copy(std::forward<As>(as)...))
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(          uninitialized_copy(std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).uninitialized_copy(std::forward<As>(as)...))
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::uninitialized_copy_t<As&&...>::_(std::forward<As>(as)...))
	public:
		template<class... As> auto operator()(As&&... as) const->RET(_(priority<4>{}, std::forward<As>(as)...))
	} uninitialized_copy; \
} \
}}

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_uninitialized_copy_t;} \
	static constexpr class alloc_uninitialized_copy_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
/*		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::copy(std::forward<As>(as)...))*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(     boost::multi::alloc_uninitialized_copy(std::forward<As>(as)...))    \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_copy(std::forward<As>(as)...))    \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_copy(std::forward<As>(as)...))
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::alloc_uninitialized_copy_t<As&&...>::_(std::forward<As>(as)...))
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<4>{}, std::forward<As>(as)...)){return _(priority<4>{}, std::forward<As>(as)...);} \
	} alloc_uninitialized_copy; \
} \
}} void* alloc_uninitialized_copy_dummy;

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_uninitialized_copy_n_t;} \
	static constexpr class alloc_uninitialized_copy_n_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
/*		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::copy(std::forward<As>(as)...))*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(     boost::multi::alloc_uninitialized_copy_n(std::forward<As>(as)...)) \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_copy_n(std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_copy_n(std::forward<As>(as)...))     \
	/*	template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::alloc_uninitialized_copy_n_t<As&&...>::_(std::forward<As>(as)...))*/ \
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<4>{}, std::forward<As>(as)...)){return _(priority<4>{}, std::forward<As>(as)...);} \
	} alloc_uninitialized_copy_n; \
} \
}} void* alloc_uninitialized_copy_n_dummy;

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_uninitialized_fill_n_t;} \
	static constexpr class alloc_uninitialized_fill_n_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
/*		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::copy(std::forward<As>(as)...))*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(     boost::multi::alloc_uninitialized_fill_n(std::forward<As>(as)...)) \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_fill_n(std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_fill_n(std::forward<As>(as)...))     \
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::alloc_uninitialized_fill_n_t<As&&...>::_(std::forward<As>(as)...)) \
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<4>{}, std::forward<As>(as)...)){return _(priority<4>{}, std::forward<As>(as)...);} \
	} alloc_uninitialized_fill_n; \
} \
}} void* alloc_uninitialized_fill_n_dummy;


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

