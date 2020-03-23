#ifdef COMPILATION_INSTRUCTIONS
$CXX -Wall -Wextra -Wpedantic -D_TEST_MULTI_DETAIL_ADL -xc++ $0 -o$0x -lboost_unit_test_framework&&$0x --color-output=no&&rm $0x;exit
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

#include<iostream> // debug

#if defined(__NVCC__) or (defined(__clang__) && defined(__CUDA__))
#include<thrust/copy.h>
#endif

namespace boost{namespace multi{
	template<std::size_t I> struct priority : std::conditional_t<I==0, std::true_type, struct priority<I-1>>{}; 
}}

#define RET(ExpR) decltype(ExpR){return ExpR;}

#define BOOST_MULTI_DEFINE_ADL(FuN) \
namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct FuN##_t;} 	__attribute__((unused))  \
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

//BOOST_MULTI_DEFINE_ADL(copy_n);
//BOOST_MULTI_DEFINE_ADL(fill_n);

#define RETU(ExpR) {return ExpR;}

namespace boost{namespace multi{
namespace adl{
	namespace custom{template<class...> struct copy_n_t;}
	static constexpr class copy_n_t{
		template<class... As>          auto _(priority<0>,        As&&... as) const{return            std::copy_n                (std::forward<As>(as)...);}
#if defined(__NVCC__) or (defined(__clang__) && defined(__CUDA__))
		template<class... As> 		   auto _(priority<1>,        As&&... as) const->RET(          thrust::copy_n                (std::forward<As>(as)...))
#endif
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                 copy_n                (std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<4>, T&& t, As&&... as) const->RET(std::decay_t<T>::copy_n(std::forward<T>(t), std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<5>, T&& t, As&&... as) const->RET(std::forward<T>(t).copy_n              (std::forward<As>(as)...))
		template<class... As>          auto _(priority<6>,        As&&... as) const->RET(custom::           copy_n_t<As&&...>::_(std::forward<As>(as)...)) 
	public:
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<6>{}, std::forward<As>(as)...)){
			return _(priority<6>{}, std::forward<As>(as)...);
		}
	} copy_n;
}}}

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct copy_t;} __attribute__((unused)) 
	static constexpr class copy_t{ \
		template<         class... As> auto _(priority<1>,        As&&... as) const->RET(              std::copy(                    std::forward<As>(as)...))
#if defined(__NVCC__) or (defined(__clang__) && defined(__CUDA__))
		template<class... As> 		   auto _(priority<2>,        As&&... as) const->RET(           thrust::copy(                    std::forward<As>(as)...))
#endif
		template<         class... As> auto _(priority<3>,        As&&... as) const->RET(                   copy(                    std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<4>, T&& t, As&&... as) const->RET(std::forward<T>(t).copy                    (std::forward<As>(as)...)) \
		template<         class... As> auto _(priority<5>,        As&&... as) const->RET(custom::           copy_t<As&&...>::_      (std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<6>, T&& t, As&&... as) const->RET(                   copy(std::forward<T>(t))(std::forward<As>(as)...)) \
	public: \
		template<class... As> auto operator()(As&&... as) const->RET( _(priority<6>{}, std::forward<As>(as)...) ) \
	} copy; \
} \
}}

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct fill_t;}                          __attribute__((unused))
	static constexpr class fill_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,  As&&... as) const = delete;*/                                                            \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(              std::fill              (std::forward<As>(as)...))       \
/*		template<class T, class... As> auto _(priority<2>, T&& t, As&&... as) const->RET(                   copy<T&&>(std::forward(t), std::forward(as)...))*/ \
		template<class... As>          auto _(priority<2>,        As&&... as) const->decltype(               adl_fill              (std::forward<As>(as)...)){
return                adl_fill              (std::forward<As>(as)...);
	} \
		template<class... As>          auto _(priority<3>,        As&&... as) const->RET(                   fill(std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<4>, T&& t, As&&... as) const->RET(std::decay_t<T>::fill(std::forward<T>(t), std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<5>, T&& t, As&&... as) const->RET(std::forward<T>(t).fill              (std::forward<As>(as)...))
/*		template<class... As>          auto _(priority<5>,        As&&... as) const->RET(custom::           fill_t<As&&...>::_(std::forward<As>(as)...))*/ \
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<5>{}, std::forward<As>(as)...)){return _(priority<5>{}, std::forward<As>(as)...);} \
	} fill; \
} \
}}


namespace boost{namespace multi{

template<class Alloc> struct alloc_construct_elem_t{
	Alloc* palloc_;
	template<class T> auto operator()(T&& p) const
	->decltype(std::allocator_traits<Alloc>::construct(*palloc_, std::addressof(p))){
		return std::allocator_traits<Alloc>::construct(*palloc_, std::addressof(p));}
};

namespace xtd{

//template<class T>
//constexpr auto adl_to_address(const T& p) noexcept;

template<class T> // this one goes last!!!
constexpr auto to_address(const T& p) noexcept;

template<class T> 
constexpr auto _to_address(priority<0>, const T& p) noexcept
->decltype(to_address(p.operator->())){
	return to_address(p.operator->());}

template<class T>
constexpr auto _to_address(priority<1>, const T& p) noexcept
->decltype(std::pointer_traits<T>::to_address(p)){
	return std::pointer_traits<T>::to_address(p);}

template<class T, std::enable_if_t<std::is_pointer<T>{}, int> = 0>
constexpr T _to_address(priority<2>, T const& p) noexcept{
    static_assert(!std::is_function<T>{}, "!");
    return p;
}

template<class T> // this one goes last!!!
constexpr auto to_address(const T& p) noexcept
->decltype(_to_address(priority<2>{}, p))
{
	return _to_address(priority<2>{}, p);}


template<class Alloc, class ForwardIt, class Size, typename Value = typename std::iterator_traits<ForwardIt>::value_type>
ForwardIt alloc_uninitialized_value_construct_n(Alloc& alloc, ForwardIt first, Size n)
//->std::decay_t<decltype(std::allocator_traits<Alloc>::construct(alloc, std::addressof(*first), Value()), first)>
{
	ForwardIt current = first;
	try{
		for (; n > 0 ; (void)++current, --n)
			std::allocator_traits<Alloc>::construct(alloc, std::addressof(*current), Value());
		//	::new (static_cast<void*>(std::addressof(*current))) Value();
		return current;
	}catch(...){
		for(; current != first; ++first) std::allocator_traits<Alloc>::destroy(alloc, std::addressof(*first));
		throw;
	}
}
}

template<class Alloc> struct alloc_destroy_elem_t{
	Alloc* palloc_;
	template<class T> auto operator()(T&& p) const
//	->decltype(std::allocator_traits<Alloc>::construct(*palloc_, std::forward<T>(t)...)){
	{	return std::allocator_traits<Alloc>::destroy(*palloc_, std::addressof(p));}
};

template<class Alloc, class BidirIt, class Size>
constexpr BidirIt alloc_destroy_n(Alloc& a, BidirIt first, Size n){
	first += (n-1);
	for (; n > 0; --first, --n)
		std::allocator_traits<Alloc>::destroy(a, std::addressof(*first));
	return first;
}

namespace xtd{


template<class Alloc, class InputIt, class Size, class ForwardIt>//, typename AT = std::allocator_traits<Alloc> >
auto alloc_uninitialized_copy_n(Alloc& a, InputIt f, Size n, ForwardIt d)
->std::decay_t<decltype(a.construct(std::addressof(*d), *f), d)>
{
	ForwardIt c = d;
	try{
		for(; n > 0; ++f, ++c, --n) std::allocator_traits<Alloc>::construct(a, std::addressof(*c), *f);
		return c;
	}catch(...){
		for(; d != c; ++d) std::allocator_traits<Alloc>::destroy(a, std::addressof(*d));
		throw;
	}
}

template<class Alloc, class InputIt, class ForwardIt>//, typename AT = std::allocator_traits<Alloc> >
auto alloc_uninitialized_copy(Alloc& a, InputIt first, InputIt last, ForwardIt d_first)
->std::decay_t<decltype(a.construct(std::addressof(*d_first), *first), d_first)>
{
//    typedef typename std::iterator_traits<ForwardIt>::value_type Value;
    ForwardIt current = d_first;
    try {
        for (; first != last; ++first, (void) ++current) {
			a.construct(std::addressof(*current), *first);
        }
        return current;
    } catch (...) {
        for (; d_first != current; ++d_first) {
			a.destroy(std::addressof(*d_first));
        }
        throw;
    }
}

template<class Alloc, class ForwardIt, class Size, class T>
auto alloc_uninitialized_fill_n(Alloc& a, ForwardIt first, Size n, T const& v)
->std::decay_t<decltype(std::allocator_traits<Alloc>::construct(a, std::addressof(*first), v), first)>
{
	ForwardIt current = first; // using std::to_address;
	try{
		for(; n > 0; ++current, --n) std::allocator_traits<Alloc>::construct(a, std::addressof(*current), v);
		return current;
	}catch(...){
		for(; first != current; ++first) std::allocator_traits<Alloc>::destroy(a, std::addressof(*first)); 
		throw;
	}
}
}

}}

namespace boost{namespace multi{ \

namespace adl{
	namespace custom{template<class...> struct distance_t;} __attribute__((unused)) 
	static constexpr class distance_t{
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(     std::distance(std::forward<As>(as)...))
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(          distance(std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).distance(std::forward<As>(as)...))
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::distance_t<As&&...>::_(std::forward<As>(as)...))
	public:
		template<class... As> auto operator()(As&&... as) const->RET(_(priority<4>{}, std::forward<As>(as)...))
	} distance;
}

namespace adl{ \
	namespace custom{template<class...> struct uninitialized_copy_t;} 	__attribute__((unused)) \
	static constexpr class uninitialized_copy_t { \
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
	namespace custom{template<class...> struct alloc_uninitialized_value_construct_n_t;} 	__attribute__((unused)) 
	static constexpr class alloc_uninitialized_value_construct_n_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const{return xtd::            alloc_uninitialized_value_construct_n              (std::forward<As>(as)...);} // TODO: use boost? 
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_value_construct_n              (std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_value_construct_n              (std::forward<As>(as)...))
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::           alloc_uninitialized_value_construct_n_t<As&&...>::_(std::forward<As>(as)...))
	public:
		template<class... As> auto operator()(As&&... as) const{return (_(priority<4>{}, std::forward<As>(as)...));}
	} alloc_uninitialized_value_construct_n; \
} \
}}

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_destroy_n_t;} 	__attribute__((unused)) \
	static constexpr class alloc_reverse_destroy_n_t { \
		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(multi::            alloc_destroy_n              (std::forward<As>(as)...)) // TODO: use boost?
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_destroy_n              (std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::decay_t<T>::alloc_destroy_n(std::forward<T>(t), std::forward<As>(as)...))
		template<class T, class... As> auto _(priority<4>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_destroy_n              (std::forward<As>(as)...))
		template<class... As>          auto _(priority<5>,        As&&... as) const->RET(custom::           alloc_destroy_n_t<As&&...>::_(std::forward<As>(as)...))
	public:
		template<class... As> auto operator()(As&&... as) const->RET(_(priority<5>{}, std::forward<As>(as)...))
	} alloc_destroy_n; \
} \
}}


namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_uninitialized_copy_t;} 	__attribute__((unused))  \
	static constexpr class alloc_uninitialized_copy_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
/*		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::copy(std::forward<As>(as)...))*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const{return              xtd::alloc_uninitialized_copy(std::forward<As>(as)...);}
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_copy(std::forward<As>(as)...))    \
		template<class... As>          auto _(priority<3>,        As&&... as) const->RET(               adl_alloc_uninitialized_copy(std::forward<As>(as)...))    \
		template<class T, class... As> auto _(priority<4>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_copy(std::forward<As>(as)...))
		template<class... As>          auto _(priority<5>,        As&&... as) const->RET(custom::alloc_uninitialized_copy_t<As&&...>::_(std::forward<As>(as)...))
	public: \
		template<class... As> auto operator()(As&&... as) const->decltype(_(priority<5>{}, std::forward<As>(as)...)){return _(priority<5>{}, std::forward<As>(as)...);} \
	} alloc_uninitialized_copy; \
} \
}} void* alloc_uninitialized_copy_dummy;

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_uninitialized_copy_n_t;} 	__attribute__((unused)) \
	static constexpr class alloc_uninitialized_copy_n_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
/*		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::copy(std::forward<As>(as)...))*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const{return (              xtd::alloc_uninitialized_copy_n(std::forward<As>(as)...));} \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_copy_n(std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_copy_n(std::forward<As>(as)...)) \
	/*	template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::alloc_uninitialized_copy_n_t<As&&...>::_(std::forward<As>(as)...))*/ \
	public: \
		template<class... As> auto operator()(As&&... as) const{return _(priority<4>{}, std::forward<As>(as)...);} \
	} alloc_uninitialized_copy_n; \
} \
}} void* alloc_uninitialized_copy_n_dummy;

namespace boost{namespace multi{ \
namespace adl{ \
	namespace custom{template<class...> struct alloc_uninitialized_fill_n_t;} 	__attribute__((unused)) \
	static constexpr class alloc_uninitialized_fill_n_t{ \
/*		template<class... As> [[deprecated]] auto _(priority<0>,        As&&... as) const = delete;*/ \
/*		template<class... As>          auto _(priority<1>,        As&&... as) const->RET(std::copy(std::forward<As>(as)...))*/ \
		template<class... As>          auto _(priority<1>,        As&&... as) const{return (           xtd::alloc_uninitialized_fill_n(std::forward<As>(as)...));}       \
		template<class... As>          auto _(priority<2>,        As&&... as) const->RET(                   alloc_uninitialized_fill_n(std::forward<As>(as)...)) \
		template<class T, class... As> auto _(priority<3>, T&& t, As&&... as) const->RET(std::forward<T>(t).alloc_uninitialized_fill_n(std::forward<As>(as)...))       \
		template<class... As>          auto _(priority<4>,        As&&... as) const->RET(custom::alloc_uninitialized_fill_n_t<As&&...>::_(std::forward<As>(as)...))    \
	public: \
		template<class T1, class... As> auto operator()(T1&& t1, As&&... as) const->decltype(_(priority<4>{}, std::forward<T1>(t1), std::forward<As>(as)...)){return _(priority<4>{}, t1, std::forward<As>(as)...);} \
	} alloc_uninitialized_fill_n; \
} \
}} void* alloc_uninitialized_fill_n_dummy;

namespace boost{
namespace multi{
template<dimensionality_type N>
struct recursive{
	template<class Alloc, class InputIt, class ForwardIt>
	static auto alloc_uninitialized_copy(Alloc& a, InputIt first, InputIt last, ForwardIt dest){
		using std::begin; using std::end;
		while(first!=last){
			recursive<N-1>::alloc_uninitialized_copy(a, begin(*first), end(*first), begin(*dest));
			++first;
			++dest;
		}
		return dest;
	}
};

template<> struct recursive<1>{
	template<class Alloc, class InputIt, class ForwardIt>
	static auto alloc_uninitialized_copy(Alloc& a, InputIt first, InputIt last, ForwardIt dest){
		return adl::alloc_uninitialized_copy(a, first, last, dest);
	}
};
}}


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

