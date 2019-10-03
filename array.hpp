#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& c++ -std=c++17 -Wall -Wextra `#-Wfatal-errors` -D_TEST_BOOST_MULTI_ARRAY $0.cpp -o$0x&&$0x&&rm $0x $0.cpp;exit
#endif
//  (C) Copyright Alfredo A. Correa 2018.
#ifndef BOOST_MULTI_ARRAY_HPP 
#define BOOST_MULTI_ARRAY_HPP

//#include<iostream>

#include "./array_ref.hpp"

//#include "../multi/detail/memory.hpp"
#include "./memory/allocator.hpp"
#include "./detail/memory.hpp"

#include "utility.hpp"

#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD 
#endif

namespace boost{
namespace multi{

namespace detail{
    using std::begin;
    template<class C> auto maybestd_begin(C&& c) 
    ->decltype(begin(std::forward<C>(c))){
        return begin(std::forward<C>(c));}
	using std::end;
    template<class C> auto maybestd_end(C&& c) 
    ->decltype(end(std::forward<C>(c))){
        return end(std::forward<C>(c));}
}

template<class C> auto maybestd_begin(C&& c)
->decltype(detail::maybestd_begin(std::forward<C>(c))){
    return detail::maybestd_begin(std::forward<C>(c));}
template<class C> auto maybestd_end(C&& c)
->decltype(detail::maybestd_end(std::forward<C>(c))){
    return detail::maybestd_end(std::forward<C>(c));}


template<class T, dimensionality_type D, class Alloc = std::allocator<T>>
struct static_array : 
	protected std::allocator_traits<Alloc>::template rebind_alloc<T>,
	array_ref<T, D, typename std::allocator_traits<typename std::allocator_traits<Alloc>::template rebind_alloc<T>>::pointer>
{
	using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
	using alloc_traits = typename std::allocator_traits<allocator_type>;
protected:
	using ref = array_ref<T, D, typename std::allocator_traits<typename std::allocator_traits<Alloc>::template rebind_alloc<T>>::pointer>;
	allocator_type& alloc(){return static_cast<allocator_type&>(*this);}
	auto uninitialized_value_construct(){return uninitialized_value_construct_n(alloc(), this->base_, this->num_elements());}
	template<typename It>
	auto uninitialized_copy_(It first){
	//	using boost::multi::uninitialized_copy_n;
		return uninitialized_copy_n(this->alloc(), first, this->num_elements(), this->data());
	}
	auto uninitialized_default_construct(){return uninitialized_default_construct_n(this->alloc(), this->base_, this->num_elements());}
	typename static_array::element_ptr allocate(typename std::allocator_traits<allocator_type>::size_type n){
		return std::allocator_traits<allocator_type>::allocate(this->alloc(), n);
	}
	auto allocate(){return allocate(this->num_elements());}
	void destroy(){
		auto n = this->num_elements();
		while(n){
		//	std::allocator_traits<allocator_type>::destroy(alloc(), to_address(this->data() + n + (-1)));
		//	alloc().destroy(to_address(this->data() + n + (-1)));
			--n;
		}
	}
public:
	using typename ref::value_type;
	using typename ref::size_type;
	using typename ref::difference_type;
	static_array(typename static_array::allocator_type const& a = {}) : static_array::allocator_type{a}{}
protected:
	static_array(static_array&& other, typename static_array::allocator_type const& a)                           //6b
	:	static_array::allocator_type{a},
		ref{std::exchange(other.base_, nullptr), other.extensions()}
	{
		//TODO
		other.ref::layout_t::operator=({});
	}
public:
	using ref::operator==;
//	template<class Array> auto operator==(Array const& other) const{return ref::operator==(other);}
//	auto operator==(static_array const& other) const{return ref::operator==(other);}
	template<class It, typename = typename std::iterator_traits<It>::difference_type>//edecltype(std::distance(std::declval<It>(), std::declval<It>()), *std::declval<It>())>      
	static_array(It first, It last, allocator_type const& a = {}) :        //(4)
		allocator_type{a}, 
		ref{nullptr, index_extension(std::distance(first, last))*multi::extensions(*first)}
	{
	//	auto cat = std::tuple_cat(std::make_tuple(index_extension{std::distance(first, last)}), multi::extensions(*first).base() );
	//	std::cout << std::get<0>(cat) << std::endl;
	//	layout_t<D>::operator=(typename static_array::layout_t{std::tuple_cat(std::make_tuple(index_extension{std::distance(first, last)}), multi::extensions(*first))});
		this->base_ = this->allocate(this->num_elements());
	//	this->base_ = this->allocate(typename static_array::layout_t{std::tuple_cat(std::make_tuple(index_extension{std::distance(first, last)}), multi::extensions(*first))}.num_elements());
		using std::next;
		using std::all_of;
	//	if(first!=last) assert( all_of(next(first), last, [x=multi::extensions(*first)](auto const& e){return extensions(e)==x;}) );
	//	recursive_uninitialized_copy<D>(alloc(), first, last, ref::begin());
		uninitialized_copy(alloc(), first, last, ref::begin());
	}
//	static_array(typename static_array::extensions_type x, typename static_array::allocator_type const& a) : //2
//		allocator_type{a}, ref(allocate(typename static_array::layout_t{x}.num_elements()), x)
//	{
//		uninitialized_fill(e);
//	}
	static_array(typename static_array::extensions_type x, typename static_array::element const& e, typename static_array::allocator_type const& a) : //2
		allocator_type{a}, ref(allocate(typename static_array::layout_t{x}.num_elements()), x)
	{
		uninitialized_fill(e);
	}
	auto uninitialized_fill(typename static_array::element const& e){
		return uninitialized_fill_n(this->alloc(), this->base_, this->num_elements(), e);
	}
	static_array(typename static_array::extensions_type const& x, typename static_array::element const& e)  //2
	:	allocator_type{}, ref(allocate(typename static_array::layout_t{x}.num_elements()), x){
		uninitialized_fill(e);
	}
	explicit static_array(typename ref::size_type n, typename static_array::allocator_type const& a = {})
	: 	static_array(typename static_array::index_extension(n), a){}
	explicit static_array(typename static_array::index n, typename static_array::value_type const& v, typename static_array::allocator_type const& a = {})
	: 	static_array(typename static_array::index_extension(n), v, a){}
	explicit static_array(typename static_array::index_extension const& e, typename static_array::value_type const& v, typename static_array::allocator_type const& a = {}) //3
	: static_array(e*extensions(v), a){
	//	assert(0);
		using std::fill; fill(this->begin(), this->end(), v);
	}
	static_array(typename static_array::extensions_type const& x, allocator_type const& a) //3
	:	allocator_type{a}, ref{allocate(typename static_array::layout_t{x}.num_elements()), x}{
	//	uninitialized_value_construct();
	}
	static_array(typename static_array::extensions_type const& x) //3
	: static_array(x, allocator_type{}){}
//	:	allocator_type{}, ref{allocate(typename static_array::layout_t{x}.num_elements()), x}{
//		uninitialized_value_construct();
//	}
#if 0
	template<
		class Array, 
	//	typename=std::enable_if_t<not std::is_constructible<typename static_array::extensions_type, std::decay_t<Array>>{}>,//, 
		typename=std::enable_if_t<not std::is_same<static_array, Array>{}>,
		typename=std::enable_if_t<multi::rank<std::remove_reference_t<Array>>{}()>=1>//,
	//	typename=decltype(typename static_array::element{std::declval<typename array_traits<Array>::element const&>()})
	//	typename = decltype(ref{typename alloc_traits::allocate(num_elements(std::declval<Array&&>())), extensions(std::declval<Array&&>())}) 
	>
	static_array(Array const& o, allocator_type const& a = {})
	:	allocator_type{a}, ref{allocate(num_elements(o)), extensions(o)}{
		using std::begin; using std::end;
		uninitialized_copy(alloc(), begin(o), end(o), ref::begin());
	}
#endif
	template<class TT, dimensionality_type DD, class... Args>
	static_array(multi::basic_array<TT, DD, Args...> const& other, allocator_type const& a = {})
		: allocator_type{a}, ref{allocate(other.num_elements()), extensions(other)}
	{
//		assert(0);
		using std::copy; 
		copy(other.begin(), other.end(), this->begin());
	}
	template<class TT, class... Args>
	static_array(array_ref<TT, D, Args...> const& other)
	:	allocator_type{}, ref{allocate(other.num_elements()), extensions(other)}{
		uninitialized_copy_(other.data());
	}
	static_array(static_array const& other, allocator_type const& a)                      //5b
	:	allocator_type{a}, ref{allocate(other.num_elements()), extensions(other)}{
		assert(0);
	//	uninitialized_copy_(other.data());
	}
	template<class O, typename = std::enable_if_t<not std::is_base_of<static_array, O>{}>>
	static_array(O const& o)                                          //
	:	allocator_type{}, 
		ref(allocate(num_elements(o)), extensions(o))
	{
		uninitialized_copy_(data_elements(o));
	}
	static_array(static_array const& o)                                     //5b
	:	allocator_type{o.get_allocator()}, 
		ref{allocate(o.num_elements()), o.extensions()}
	{
		uninitialized_copy_(o.data());
	}
	static_array(static_array&& o)                                     //5b
	:	allocator_type{o.get_allocator()}, 
		ref{allocate(o.num_elements()), o.extensions()}
	{
		assert(0);
		uninitialized_copy_(o.data());
	}
#if (not defined(__INTEL_COMPILER)) or (__GNUC >= 6)
	static_array(
		std::initializer_list<typename static_array::value_type> il, 
		typename static_array::allocator_type const& a={}
	) 
	: static_array(il.begin(), il.end(), a)
	{}
#if 0
	template<class TT, std::size_t N>
	explicit static_array(
		const TT(&t)[N], 
		typename static_array::allocator_type const& a={}
	) 
	: static_array(std::begin(t), std::end(t), a)
	{}
#endif
//	template<class TT, typename std::enable_if_t<TT, >>
//	array(std::initializer_list<typename array::size_type> il, allocator_type const& a={}) 
//	:	array(il.begin(), il.end(), a){
//		assert(0);
//	}

//	template<class T>//, typename = std::enable_if_t<std::is_same<T, int>>
//	array(std::sinitializer_list<int> il, allocator_type const& a={}) 
//	:	array(il.size()!=D?il.begin():throw std::runtime_error{"warning"}, il.end(), a){}
//	array(std::tuple<int>){assert(0);}
#endif
	template<class It> static auto distance(It a, It b){using std::distance; return distance(a, b);}
protected:
	void deallocate(){
		alloc_traits::deallocate(this->alloc(), this->base_, static_cast<typename alloc_traits::size_type>(this->num_elements()));
		this->base_ = nullptr;
	}
	void clear() noexcept{this->destroy(); deallocate(); layout_t<D>::operator=({});}
public:
	~static_array() noexcept{clear();}
	using element_const_ptr = typename std::pointer_traits<typename static_array::element_ptr>::template rebind<typename static_array::element const>;
	using reference = std::conditional_t<
		static_array::dimensionality != 1, 
		basic_array<typename static_array::element, static_array::dimensionality-1, typename static_array::element_ptr>, 
		typename pointer_traits<typename static_array::element_ptr>::element_type&
	>;
	using const_reference = std::conditional_t<
		static_array::dimensionality != 1, 
		basic_array<typename static_array::element, static_array::dimensionality-1, typename static_array::element_const_ptr>, // TODO should be const_reference, but doesn't work witn rangev3
		typename pointer_traits<typename static_array::element_ptr>::element_type const&
	>;
	using const_iterator = multi::array_iterator<T, static_array::dimensionality, typename static_array::element_const_ptr, const_reference>;
//	reference       
	HD decltype(auto) operator[](index i)      {return ref::operator[](i);}
	const_reference operator[](index i) const{return ref::operator[](i);}
	typename static_array::allocator_type get_allocator() const{return static_cast<typename static_array::allocator_type const&>(*this);}

	HD typename static_array::element_ptr       data()      {return ref::data();}
	HD typename static_array::element_const_ptr data() const{return ref::data();}
	friend typename static_array::element_ptr       data(static_array&       s){return s.data();}
	friend typename static_array::element_const_ptr data(static_array const& s){return s.data();}

	typename static_array::element_ptr       origin()      {return ref::origin();}
	typename static_array::element_const_ptr origin() const{return ref::origin();}
	friend typename static_array::element_ptr       origin(static_array&       s){return s.origin();}
	friend typename static_array::element_const_ptr origin(static_array const& s){return s.origin();}

//	using const_reverse_iterator = basic_reverse_iterator<const_iterator>;

	typename static_array::iterator begin(){return ref::begin();}
	typename static_array::iterator end()  {return ref::end();}
//	typename array::iterator begin() &&{return ref::begin();}
//	typename array::iterator end()   &&{return ref::end();}

	typename static_array::const_iterator begin() const{return ref::begin();}
	typename static_array::const_iterator end()   const{return ref::end();}
	const_iterator cbegin() const{return begin();}
	const_iterator cend() const{return end();}

	static_array& operator=(static_array const& other){
		assert( extensions(other) == static_array::extensions() );
		using std::copy_n;
		copy_n(other.data(), other.num_elements(), this->data());
		return *this;
	}
};

template<class T, dimensionality_type D, class Alloc>
struct array : static_array<T, D, Alloc>,
	boost::multi::random_iterable<array<T, D, Alloc> >
{
	using static_ = static_array<T, D, Alloc>;
	static_assert(std::is_same<typename array::alloc_traits::value_type, T>{} or std::is_same<typename array::alloc_traits::value_type, void>{}, "!");
public:
	using static_::static_;
	using typename array::ref::value_type;
//	using typename ref::reference;
//	using static_::operator==;
//	template<class Array> auto operator==(Array const& other) const{return static_::operator==(other);}
	using static_::ref::operator<;
	array() = default;
	array(array const&) = default;
	template<class O, typename = std::enable_if_t<std::is_base_of<array, O>{}> > 
	array(O const& o) : static_(o){}
//	array() noexcept(noexcept(static_::allocator_type())) : static_{}{} // 1a //allocator_type{}, ref{}{}      //1a
//	array(typename array::allocator_type a) : static_(a){}                  //1b
#if (not defined(__INTEL_COMPILER)) or (__GNUC >= 6)
	array(
		std::initializer_list<typename array::value_type> il, 
		typename array::allocator_type const& a={}
	) : static_(il, a){}
#endif
//	array(typename array::extensions_type const& x) //3
//	:	allocator_type{}, ref{allocate(typename array::layout_t{x}.num_elements()), x}{
//		uninitialized_value_construct();
//	}
//	template<class Extension, typename = decltype(array(std::array<Extension, D>{}, allocator_type{}, std::make_index_sequence<D>{}))>
//	array(std::array<Extension, D> const& x, allocator_type const& a = {}) : array(x, a, std::make_index_sequence<D>{}){}
//	array(multi::iextensions<D> const& ie) : array(typename array::extensions_type{ie}){}
private:
//	template<class Extension, size_t... Is>//, typename = decltype(typename array::extensions_type{std::array<Extension, D>{}})>
//	array(std::array<Extension, D> const& x, allocator_type const& a, std::index_sequence<Is...>) : array(typename array::extensions_type{std::get<Is>(x)...}, a){}
public:
	using static_::clear;
	friend void clear(array& self) noexcept{self.clear();}
//	explicit	
//	array(array const& other)                                              // 5a
//	:	allocator_type{other}, ref{allocate(other.num_elements()), extensions(other)}{
//		uninitialized_copy_(other.data());
//	}
//	explicit
//s	using static_::static_array;
#if 0
	template<
		class Array, 
		typename=std::enable_if_t<not std::is_constructible<typename array::extensions_type, Array>{}>,//, 
		typename=std::enable_if_t<not std::is_base_of<array, Array>{}>,
		typename=std::enable_if_t<multi::rank<std::remove_reference_t<Array>>{}()>=1>//,
	//	typename=decltype(typename array::element{std::declval<typename array_traits<Array>::element const&>()})
	>	array(Array const& o, typename array::allocator_type const& a = {})
	:	static_(o, a){
	//	using std::begin; using std::end;
	//	uninitialized_copy(alloc(), begin(o), end(o), ref::begin());
	}
#endif
#if 0
	template<
		class Array, 
		typename=std::enable_if_t<not std::is_constructible<typename array::extensions_type, std::decay_t<Array>>{}>,//, 
		typename=std::enable_if_t<not std::is_same<array, Array>{}>,
		typename=std::enable_if_t<not std::is_same<static_, Array>{}>,
		typename=std::enable_if_t<multi::rank<std::remove_reference_t<Array>>{}()>=1>,
		typename=decltype(typename array::element{std::declval<typename array_traits<Array>::element const&>()})
	//	typename = decltype(ref{typename alloc_traits::allocate(num_elements(std::declval<Array&&>())), extensions(std::declval<Array&&>())}) 
	>
	array(Array const& o, typename array::allocator_type const& a = {}) : static_{o, a}{}
#endif
//	template<class Array> array(Array const& arr) : static_{arr}{}
//	using static_::static_;
//	template<class... As>
//	array(typename array::extensions_type x, As&&... as) : static_{x, std::forward<As>(as)...}{} //2
//	array(array const& other) : static_{static_cast<static_ const&>(other)}{}
	array(array&& other) noexcept : static_{other.get_allocator()}{
		static_::layout_t::operator=(other);
		this->base_ = std::exchange(other.base_, nullptr);
		other.static_::layout_t::operator=({});
	}
//	array(array const& o)                                                    //5b
//	:	static_{o.get_allocator()}//, 
//		static_::ref{allocate(o.num_elements()), o.extensions()}
//	{
//		assert(0);
//		uninitialized_copy_(o.data());
//	}
//	array(array&& other) noexcept                                           //6a
//	://	array::allocator_type{other.get_allocator()},
//		static_{std::move(other)}//std::exchange(other.base_, nullptr), other.extensions()}
//	{
//		other.static_::layout_t::operator=({});
//	}
//	array(array&& other, typename array::allocator_type const& a)                           //6b
//	:	array::allocator_type{a},
//		static_{std::exchange(other.base_, nullptr), other.extensions()}
//	{
//		//TODO
//		other.static_::ref::layout_t::operator=({});
//	}
	template<class A, typename = std::enable_if_t<not std::is_base_of<array, std::decay_t<A>>{}> >
	array& operator=(A&& a){
		auto ext = extensions(a);
		if(ext==array::extensions()){
			const_cast<array const&>(*this).static_::ref::operator=(std::forward<A>(a));
		}else{
			this->clear(); //	this->ref::layout_t::operator=(layout_t<D>{extensions(a)}); //			this->base_ = allocate(this->num_elements());
			this->base_ = this->allocate(static_cast<typename array::alloc_traits::size_type>(this->static_::ref::layout_t::operator=(layout_t<D>{extensions(a)}).num_elements()));
			using std::begin; using std::end;
			uninitialized_copy(this->alloc(), begin(std::forward<A>(a)), end(std::forward<A>(a)), array::begin()); //	recursive_uninitialized_copy<D>(alloc(), begin(std::forward<A>(a)), end(std::forward<A>(a)), array::begin());
		}
		return *this;
	}
	array& operator=(array const& other){
		if(extensions(other)==array::extensions()){
			static_::operator=(other);
		//	using std::copy_n;
		//	copy_n(other.data(), other.num_elements(), this->data());
		}else{
			this->clear();
			this->static_::ref::layout_t::operator=(layout_t<D>{extensions(other)});
			this->base_ = this->allocate();//this->num_elements());
			uninitialized_copy_n(this->alloc(), other.data(), other.num_elements(), this->data());
		}
		return *this;
	}
	array& operator=(array&& other) noexcept{
	//	if(this!=std::addressof(other)) clear(); 
	//	swap(other); 
	//	return *this;
		using std::exchange;
		clear();
		this->base_ = exchange(other.base_, nullptr);
		this->alloc() = std::move(other.alloc());
		static_cast<typename array::layout_t&>(*this) = exchange(static_cast<typename array::layout_t&>(other), {});
	//	swap(
	//		static_cast<typename array::layout_t&>(*this), 
	//		static_cast<typename array::layout_t&>(other)
	//	);
		return *this;
	}
	void swap(array& other) noexcept{
		using std::swap;
		swap(this->alloc(), other.alloc());
		swap(this->base_, other.base_);
		swap(
			static_cast<typename array::layout_t&>(*this), 
			static_cast<typename array::layout_t&>(other)
		);
	}
	friend void swap(array& a, array& b){a.swap(b);}
	void assign(typename array::extensions_type x, typename array::element const& e){
		if(array::extensions()==x){
			fill_n(this->base_, this->num_elements(), e);
		}else{
			this->clear();
			this->layout_t<D>::operator=(layout_t<D>{x});
			this->base_ = this->allocate();
			uninitialized_fill_n(e);
		//	recursive_uninitialized_fill<dimensionality>(alloc(), begin(), end(), e);
		}
	}
	template<class It>
	array& assign(It first, It last){
		using std::next;
		using std::all_of;
		if(distance(first, last) == array::size() and multi::extensions(*first) == multi::extensions(*array::begin())){
			static_::ref::assign(first, last);
		}else{
			this->clear();
			this->layout_t<D>::operator=(layout_t<D>{std::tuple_cat(std::make_tuple(index_extension{array::extension().front(), array::extension().front() + distance(first, last)}), multi::extensions(*first))});
			using std::next;
			using std::all_of;
			if(first!=last) assert( all_of(next(first), last, [x=multi::extensions(*first)](auto& e){return extensions(e)==x;}) );
			this->base_ = this->allocate();
			multi::uninitialized_copy<D>(first, last, array::begin());
		}
		return *this;
	}
	array& operator=(std::initializer_list<value_type> il){return assign(begin(il), end(il));}
	void reextent(typename array::extensions_type const& e, typename array::element const& v = {}){
		array tmp(e, v, static_cast<Alloc const&>(*this));
		tmp.intersection_assign_(*this);
		swap(tmp);
	}



private:
#if 0
	allocator_type& alloc(){return static_cast<allocator_type&>(*this);}
	void destroy(){
		auto n = this->num_elements();
		while(n){
			alloc().destroy(to_address(this->data() + n - 1));
		//	multi::destroy(alloc(), this->data() + n - 1);
		//	destroy(alloc(), this->data() + n - 1);
			--n;
		}
	//	for(; first != last; ++first) a.destroy(to_address(first)); //	AT::destroy(a, to_address(first)); //	AT::destroy(a, addressof(*first)); //
	//	destroy_n(alloc(), this->data(), this->num_elements());
	//	destroy_n(alloc(), std::make_reverse_iterator(this->data() + this->num_elements()), this->num_elements());
	}
#endif
//	typename array::element_ptr allocate(typename array::index n){return alloc_traits::allocate(alloc(), n);}
//	auto allocate(){return allocate(this->num_elements());}
};

#if __cpp_deduction_guides
#define IL std::initializer_list
// clang cannot recognize templated-using, so don't replace IL<IL<T>> by IL2<T>, etc
template<class T, class A=std::allocator<T>> static_array(IL<T>                , A={})->static_array<T,1,A>; 
template<class T, class A=std::allocator<T>> static_array(IL<IL<T>>            , A={})->static_array<T,2,A>;
template<class T, class A=std::allocator<T>> static_array(IL<IL<IL<T>>>        , A={})->static_array<T,3,A>; 
template<class T, class A=std::allocator<T>> static_array(IL<IL<IL<IL<T>>>>    , A={})->static_array<T,4,A>; 
template<class T, class A=std::allocator<T>> static_array(IL<IL<IL<IL<IL<T>>>>>, A={})->static_array<T,5,A>;

template<class T, class A=std::allocator<T>> array(IL<T>                , A={})->array<T,1,A>; 
template<class T, class A=std::allocator<T>> array(IL<IL<T>>            , A={})->array<T,2,A>;
template<class T, class A=std::allocator<T>> array(IL<IL<IL<T>>>        , A={})->array<T,3,A>; 
template<class T, class A=std::allocator<T>> array(IL<IL<IL<IL<T>>>>    , A={})->array<T,4,A>; 
template<class T, class A=std::allocator<T>> array(IL<IL<IL<IL<IL<T>>>>>, A={})->array<T,5,A>;

template<class T, class A=std::allocator<T>> array(T[]                  , A={})->array<T,1,A>;
template<class Array, class A=std::allocator<typename multi::array_traits<Array>::element>> array(Array            , A={})->array<typename multi::array_traits<Array>::element, 1, A>;
#undef IL

template<class T, class A=std::allocator<T>> array(iextensions<1>, T)->array<T,1,A>;
template<class T, class A=std::allocator<T>> array(iextensions<2>, T)->array<T,2,A>;
template<class T, class A=std::allocator<T>> array(iextensions<3>, T)->array<T,3,A>;
template<class T, class A=std::allocator<T>> array(iextensions<4>, T)->array<T,4,A>;
template<class T, class A=std::allocator<T>> array(iextensions<5>, T)->array<T,5,A>;

template<class A> array(iextensions<1>, A)->array<typename std::allocator_traits<A>::value_type,1,A>;
template<class A> array(iextensions<2>, A)->array<typename std::allocator_traits<A>::value_type,2,A>;
template<class A> array(iextensions<3>, A)->array<typename std::allocator_traits<A>::value_type,3,A>;
template<class A> array(iextensions<4>, A)->array<typename std::allocator_traits<A>::value_type,4,A>;
template<class A> array(iextensions<5>, A)->array<typename std::allocator_traits<A>::value_type,5,A>;

template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<1>, T, MR*)->array<T,1,A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<2>, T, MR*)->array<T,2,A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<3>, T, MR*)->array<T,3,A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<4>, T, MR*)->array<T,4,A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<5>, T, MR*)->array<T,5,A>;
#endif

}}

#undef HD

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#if _TEST_BOOST_MULTI_ARRAY

#include<cassert>
#include<numeric> // iota
#include<iostream>
#include<algorithm>
#include<vector>

#include <random>
#include <boost/timer/timer.hpp>
#include<boost/multi_array.hpp>
using std::cout;
namespace multi = boost::multi;

#if 1
template<class Matrix, class Vector>
void solve(Matrix& m, Vector& y){
//	using std::size; // assert(size(m) == std::ptrdiff_t(size(y)));
	std::ptrdiff_t msize = size(m); 
	for(auto r = 0; r != msize; ++r){ //	auto mr = m[r]; //  auto const mrr = mr[r];// assert( mrr != 0 ); // m[r][r] = 1;
		auto mr = m[r];
		auto mrr = mr[r];
		for(auto c = r + 1; c != msize; ++c) mr[c] /= mrr;
		auto yr = (y[r] /= mrr);
		for(auto r2 = r + 1; r2 != msize; ++r2){ //	auto mr2 = m[r2]; //	auto const mr2r = mr2[r]; // m[r2][r] = 0;
			auto mr2 = m[r2];
			auto const& mr2r = mr2[r];
			auto const& mr = m[r];
			for(auto c = r + 1; c != msize; ++c) mr2[c] -= mr2r*mr[c];
			y[r2] -= mr2r*yr;
		}
	}
	for(auto r = msize - 1; r > 0; --r){ //	auto mtr = m.rotated(1)[r];
		auto const& yr = y[r];
		for(auto r2 = r-1; r2 >=0; --r2)
			y[r2] -= yr*m[r2][r];
	}
}
#endif

void f(boost::multi::array<double, 4> const& A){
	A[1][2];
	auto&& a = A[1][2]; (void)a; // careful, a is a reference here, don't use auto, 
	auto const& b = A[1][2]; (void)b; // use auto const& if possible
//	A[1][2][3][4] = 5; // fail, element is read-only
}

template<class C>
void set_99(C&& c){
	for(auto j : c.extension(0))
		for(auto k : c.extension(1))
				c[j][k] = 99.;
}

namespace multi = boost::multi;

template<class T> void fun(T const& t){
	std::cout << typeid(t).name() << std::endl;
}

template<class T> struct extension{};
template<class T> void gun(extension<T>){
	std::cout << typeid(T).name() << std::endl;
}

typedef double a1010[10][10];

struct A{
	double const* p;
	A(std::initializer_list<double> il){ p = &*(il.begin() + 1); };
};

double f(){return 5.;}
int main(){


#if __cpp_deduction_guides
{
	multi::array<double, 1> A1 = {1.,2.,3.}; 
	assert(A1.dimensionality==1 and A1.num_elements()==3);

	multi::array<double, 2> A2 {
		 {1.,2.,3.},
		 {4.,5.,6.}
	};
	*A2.begin()->begin() = 99;
	assert(A2[0][0] == 99 );
}
#endif

}
#endif
#endif

