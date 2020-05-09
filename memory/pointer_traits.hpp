#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXX $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2020

#ifndef MULTI_TRAITS_HPP
#define MULTI_TRAITS_HPP

#include<memory> 

namespace boost{
namespace multi{

#if 0
template<class T, typename = decltype(std::declval<T const&>().default_allocator())>
std::true_type           has_default_allocator_aux(T const&);
std::false_type          has_default_allocator_aux(...);
template<class T> struct has_default_allocator : decltype(has_default_allocator_aux(std::declval<T>())){};

template<class Pointer, class T = void> struct pointer_traits;

template<class P>
struct pointer_traits<P, std::enable_if_t<has_default_allocator<P>{}> > : std::pointer_traits<P>{
	static auto default_allocator_of(typename pointer_traits::pointer const& p){return p.default_allocator();}
	using default_allocator_type = decltype(std::declval<P const&>().default_allocator());
};
template<class P>
struct pointer_traits<P, std::enable_if_t<not has_default_allocator<P>{}> > : std::pointer_traits<P>{
//	using default_allocator_type = std::allocator<std::decay_t<typename std::iterator_traits<P>::value_type> >;
	using default_allocator_type = std::allocator<std::decay_t<typename std::pointer_traits<P>::element_type> >;
	static default_allocator_type default_allocator_of(typename pointer_traits::pointer const&){return {};}
};

template<class P>
typename pointer_traits<P>::default_allocator_type 
default_allocator_of(P const& p){return pointer_traits<P>::default_allocator_of(p);}
#endif

template<
	class Pointer,
	class DefaultAllocator = void//typename Pointer::default_allocator_type
>
struct pointer_traits{
	using default_allocator_type = std::allocator<typename std::iterator_traits<Pointer>::value_type>;
};

template<class Pointer>
struct pointer_traits<Pointer, typename Pointer::default_allocator_type> : std::pointer_traits<Pointer>{
	using default_allocator_type = typename Pointer::default_allocator_type;
};

template<class T> struct pointer_traits<T*> : std::pointer_traits<T*>{
	using default_allocator_type = std::allocator<typename std::iterator_traits<T*>::value_type>;
};

}}

#if not __INCLUDE_LEVEL__ // TEST BELOW

#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi traits"
#include<boost/test/unit_test.hpp>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(multi_pointer_traits){
	static_assert(std::is_same<multi::pointer_traits<double*>::default_allocator_type, std::allocator<double>>{}, "!");
}

#endif
#endif

