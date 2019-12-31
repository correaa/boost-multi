#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -Wall -Wextra -D_TEST_BOOST_MULTI_ARRAY $0.cpp -o $0x &&$0x&&rm $0x $0.cpp;exit
#endif
//  © Alfredo A. Correa 2018-2019
#ifndef BOOST_MULTI_ARRAY_HPP 
#define BOOST_MULTI_ARRAY_HPP

#include "./array_ref.hpp"

#include "./memory/allocator.hpp"
#include "./detail/memory.hpp"

#include "utility.hpp"

//#include<boost/serialization/nvp.hpp>
//#include<boost/serialization/array_wrapper.hpp>
#include<iostream> // debug

#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD 
#endif

namespace boost{
namespace serialization{
	template<class> struct nvp;
	template<class T> const nvp<T> make_nvp(char const* name, T& t);
	template<class T> class array_wrapper;
	template<class T, class S> const array_wrapper< T > make_array(T* t, S s);
}
}

namespace boost{
namespace multi{

template<class Allocator> struct array_allocator{
	using allocator_type = Allocator;
protected:
#if __has_cpp_attribute(no_unique_address) >=201803
	[[no_unique_address]]
#endif
	allocator_type alloc_;
	allocator_type& alloc(){return alloc_;}
	array_allocator(allocator_type const& a = {}) : alloc_{a}{}
	typename std::allocator_traits<allocator_type>::pointer allocate(typename std::allocator_traits<allocator_type>::size_type n){
		return n?std::allocator_traits<allocator_type>::allocate(alloc_, n):nullptr;
	}
	auto uninitialized_fill_n(typename std::allocator_traits<allocator_type>::pointer base, typename std::allocator_traits<allocator_type>::size_type num_elements, typename std::allocator_traits<allocator_type>::value_type e){
		return uninitialized_fill_n(alloc_, base, num_elements, e);
	}
public:
	allocator_type get_allocator() const{return alloc_;}
};

template<class T, dimensionality_type D, class Alloc = std::allocator<T>>
struct static_array : 
	protected array_allocator<Alloc>,
	public array_ref<T, D, typename std::allocator_traits<typename array_allocator<Alloc>::allocator_type>::pointer>
{
private:
	using array_alloc = array_allocator<Alloc>;
public:	
	static_assert( std::is_same<typename std::allocator_traits<Alloc>::value_type, typename static_array::element>{}, 
		"allocator value type must match array value type");
	using array_alloc::get_allocator;
	using allocator_type = typename static_array::allocator_type;
protected:
	using alloc_traits = typename std::allocator_traits<typename static_array::allocator_type>;
	using ref = array_ref<T, D, typename std::allocator_traits<typename std::allocator_traits<Alloc>::template rebind_alloc<T>>::pointer>;
	auto uninitialized_value_construct(){return uninitialized_value_construct_n(static_array::alloc(), to_address(this->base_), this->num_elements());}
	template<typename It>
	auto uninitialized_copy_(It first){
		using boost::multi::uninitialized_copy_n;
		return uninitialized_copy_n(this->alloc(), first, this->num_elements(), this->data());
	}
	void destroy(){
		auto n = this->num_elements();
		while(n){
		//	std::allocator_traits<allocator_type>::destroy(this->alloc(), to_address(this->data() + n + (-1)));
			this->alloc().destroy(this->data() + n + (-1));//to_address(this->data() + n + (-1)));
			--n;
		}
	}
public:
	using typename ref::value_type;
	using typename ref::size_type;
	using typename ref::difference_type;
	static_array(typename static_array::allocator_type const& a) : array_alloc{a}{}
protected:
	static_array(static_array&& other, typename static_array::allocator_type const& a)                           //6b
	:	array_alloc{a},
		ref{other.base_, other.extensions()}
	{
		other.ref::layout_t::operator=({});
	}
public:
	using ref::operator==;
//	template<class Array> auto operator==(Array const& other) const{return ref::operator==(other);}
//	auto operator==(static_array const& other) const{return ref::operator==(other);}
	template<class It, typename = typename std::iterator_traits<It>::difference_type>//edecltype(std::distance(std::declval<It>(), std::declval<It>()), *std::declval<It>())>      
	static_array(It first, It last, typename static_array::allocator_type const& a = {}) :        //(4)
		array_alloc{a},
		ref{
			static_array::allocate(typename static_array::layout_t{index_extension(std::distance(first, last))*multi::extensions(*first)}.num_elements()), 
			index_extension(std::distance(first, last))*multi::extensions(*first)
		}
	{
	//	auto cat = std::tuple_cat(std::make_tuple(index_extension{std::distance(first, last)}), multi::extensions(*first).base() );
	//	std::cout << std::get<0>(cat) << std::endl;
	//	layout_t<D>::operator=(typename static_array::layout_t{std::tuple_cat(std::make_tuple(index_extension{std::distance(first, last)}), multi::extensions(*first))});
	//	this->base_ = this->allocate(typename static_array::layout_t{std::tuple_cat(std::make_tuple(index_extension{std::distance(first, last)}), multi::extensions(*first))}.num_elements());
	//	using std::next;
	//	using std::all_of;
	//	if(first!=last) assert( all_of(next(first), last, [x=multi::extensions(*first)](auto const& e){return extensions(e)==x;}) );
	//	recursive_uninitialized_copy<D>(alloc(), first, last, ref::begin());
		uninitialized_copy(static_array::alloc(), first, last, ref::begin());
	}
	static_array(typename static_array::extensions_type x, typename static_array::element const& e, typename static_array::allocator_type const& a) : //2
		array_alloc{a}, 
		ref(static_array::allocate(typename static_array::layout_t{x}.num_elements()), x)
	{
		uninitialized_fill(e);
	}
	template<class Element, typename = std::enable_if_t<std::is_convertible<Element, typename static_array::element>{} and D==0>>
	explicit static_array(Element const& e, typename static_array::allocator_type const& a)
	:	static_array(typename static_array::extensions_type{}, e, a){}
	auto uninitialized_fill(typename static_array::element const& e){
		return uninitialized_fill_n(this->alloc(), this->base_, this->num_elements(), e);
	}
	static_array(typename static_array::extensions_type const& x, typename static_array::element const& e)  //2
	:	array_alloc{}, ref(static_array::allocate(typename static_array::layout_t{x}.num_elements()), x){
		uninitialized_fill(e);
	}
	template<class Elem, typename = std::enable_if_t<std::is_convertible<Elem, typename static_array::element>{} and D==0>>
	static_array(Elem const& e)  //2
	:	static_array(multi::iextensions<D>{}, e){}

//	explicit static_array(typename static_array::index n, typename static_array::value_type const& v, typename static_array::allocator_type const& a = {})
//	: 	static_array(typename static_array::index_extension(n), v, a){}
	template<class ValueType, typename = std::enable_if_t<std::is_same<ValueType, typename static_array::value_type>{}>> 
	explicit static_array(typename static_array::index_extension const& e, ValueType const& v, typename static_array::allocator_type const& a = {}) //3
	: static_array(e*extensions(v), a){
	//	assert(0);
		using std::fill; fill(this->begin(), this->end(), v);
	}
//	template<class Allocator, typename = std::enable_if_t<std::is_same<Allocator, allocator_type>{}> >
//	explicit 
	static_array(typename static_array::extensions_type const& x, typename static_array::allocator_type const& a) //3
	: array_alloc{a}, ref{static_array::allocate(typename static_array::layout_t{x}.num_elements()), x}{
	//	assert(0);
	//	uninitialized_value_construct();
	}
	static_array(typename static_array::extensions_type const& x) //3
	:	array_alloc{}, ref{static_array::allocate(typename static_array::layout_t{x}.num_elements()), x}{
		if(not std::is_trivially_default_constructible<typename static_array::element>{})
			uninitialized_value_construct();
	}
	template<class TT, class... Args>
	static_array(multi::basic_array<TT, D, Args...> const& other, typename static_array::allocator_type const& a = {})
		: array_alloc{a}, ref(static_array::allocate(other.num_elements()), extensions(other))
	{
		using std::copy; copy(other.begin(), other.end(), this->begin());
	}
	template<class TT, class... Args>
	static_array(array_ref<TT, D, Args...> const& other)
	:	array_alloc{}, ref{static_array::allocate(other.num_elements()), extensions(other)}{
		uninitialized_copy_(other.data());
	}
	static_array(static_array const& other, typename static_array::allocator_type const& a)                      //5b
	:	array_alloc{a}, ref{static_array::allocate(other.num_elements()), extensions(other)}{
	//	assert(0);
		uninitialized_copy_(other.data());
	}
	static_array(static_array const& o) :                                  //5b
		array_alloc{o.get_allocator()}, 
		ref{static_array::allocate(o.num_elements()), o.extensions()}
	{
		uninitialized_copy_(o.data());
	}
	static_array(static_array&& o) :                                       //5b
		array_alloc{o.get_allocator()}, 
		ref{static_array::allocate(o.num_elements()), o.extensions()}
	{
		assert(0);
		uninitialized_copy_(o.data()); // TODO: uninitialized_move?
	}
	static_array(
		std::initializer_list<typename static_array::value_type> mil, 
		typename static_array::allocator_type const& a={}
	) 
	: static_array(mil.begin(), mil.end(), a)
	{}
	template<class It> static auto distance(It a, It b){using std::distance; return distance(a, b);}
protected:
	void deallocate(){
	//	if(this->base_){
	//		std::cout << "deallocated " << this->base_ <<" "<< this->num_elements() << std::endl;
//			if(this->num_elements()) 
		if(this->num_elements()) alloc_traits::deallocate(this->alloc(), this->base_, static_cast<typename alloc_traits::size_type>(this->num_elements()));
	//	}
	//	this->base_ = nullptr;
	}
	void clear() noexcept{
		this->destroy();
		deallocate();
		layout_t<D>::operator=({});
	}
public:
	static_array() = default;
	~static_array() noexcept{
		this->destroy();
		deallocate();
	//	alloc_traits::deallocate(this->alloc(), this->base_, static_cast<typename alloc_traits::size_type>(this->num_elements()));
	}
	using element_const_ptr = typename std::pointer_traits<typename static_array::element_ptr>::template rebind<typename static_array::element const>;
	using reference = typename std::conditional<
		(static_array::dimensionality > 1), 
		basic_array<typename static_array::element, static_array::dimensionality-1, typename static_array::element_ptr>, 
		typename std::conditional<
			static_array::dimensionality == 1,
			typename std::iterator_traits<typename static_array::element_ptr>::reference,
			void
		>::type
	//	typename pointer_traits<typename static_array::element_ptr>::element_type&
	>::type;
	using const_reference = typename std::conditional<
		(static_array::dimensionality > 1), 
		basic_array<typename static_array::element, static_array::dimensionality-1, typename static_array::element_const_ptr>, // TODO should be const_reference, but doesn't work witn rangev3
		typename std::conditional<
			static_array::dimensionality == 1,
			typename std::iterator_traits<typename static_array::element_const_ptr>::reference,
			void
		>::type
	//	typename pointer_traits<typename static_array::element_ptr>::element_type const&
	>::type;
	using iterator = multi::array_iterator<T, static_array::dimensionality, typename static_array::element_ptr, reference>;
	using const_iterator = multi::array_iterator<T, static_array::dimensionality, typename static_array::element_const_ptr, const_reference>;
//	reference
	HD decltype(auto) operator[](index i)      {return ref::operator[](i);}
	const_reference operator[](index i) const{return ref::operator[](i);}
//	typename static_array::allocator_type get_allocator() const{return static_cast<typename static_array::allocator_type const&>(*this);}
	friend typename static_array::allocator_type get_allocator(static_array const& self){return self.get_allocator();}
	HD typename static_array::element_ptr       data()      {return ref::data();}
	HD auto data() const{return typename static_array::element_const_ptr{ref::data()};}
	friend typename static_array::element_ptr       data(static_array&       s){return s.data();}
	friend typename static_array::element_const_ptr data(static_array const& s){return s.data();}

	HD typename static_array::element_ptr       base()      {return ref::base();}
	HD auto base() const{return typename static_array::element_const_ptr{ref::base()};}
	friend typename static_array::element_ptr       base(static_array&       s){return s.base();}
	friend typename static_array::element_const_ptr base(static_array const& s){return s.base();}

	typename static_array::element_ptr       origin()      {return ref::origin();}
	typename static_array::element_const_ptr origin() const{return ref::origin();}
	friend typename static_array::element_ptr       origin(static_array&       s){return s.origin();}
	friend typename static_array::element_const_ptr origin(static_array const& s){return s.origin();}

//	template<class... Args> decltype(auto) operator()(Args const&... args)&{return ref::operator()(args...);}
//	template<class... Args> decltype(auto) operator()(Args const&... args) const&{return ref::operator()(args...);}
	using ref::operator();

//	basic_array<T, D, typename static_array::element_ptr> 
	decltype(auto) operator()()&{
		return ref::operator()();
	//	return *this;
	}
	basic_array<T, D, typename static_array::element_const_ptr> operator()() const&{
		return basic_array<T, D, typename static_array::element_const_ptr>{this->layout(), this->base_};
	}
//	using const_reverse_iterator = basic_reverse_iterator<const_iterator>;
	auto rotated(dimensionality_type d = 1) const&{
		typename static_array::layout_t new_layout = *this;
		new_layout.rotate(d);
		return basic_array<T, D, typename static_array::element_const_ptr>{new_layout, this->base_};
	}
	auto rotated(dimensionality_type d = 1)&{
		typename static_array::layout_t new_layout = *this;
		new_layout.rotate(d);
		return basic_array<T, D, typename static_array::element_ptr>{new_layout, this->base_};
	}
	auto rotated(dimensionality_type d = 1)&&{
		typename static_array::layout_t new_layout = *this;
		new_layout.rotate(d);
		return basic_array<T, D, typename static_array::element_ptr>{new_layout, this->base_};
	}
//	friend decltype(auto) rotated(static_array const& self){return self.rotated();}
//	template<class Array, typename = std::enable_if_t<std::is_same<static_array, std::decay_t<Array>>{}> > 
	friend decltype(auto) rotated(static_array& self){return self.rotated();}
	friend decltype(auto) rotated(static_array const& self){return self.rotated();}

	auto unrotated(dimensionality_type d = 1) const&{
		typename static_array::layout_t new_layout = *this;
		new_layout.unrotate(d);
		return basic_array<T, D, typename static_array::element_const_ptr>{new_layout, this->base_};
	}
	auto unrotated(dimensionality_type d = 1)&{
		typename static_array::layout_t new_layout = *this;
		new_layout.unrotate(d);
		return basic_array<T, D, typename static_array::element_ptr>{new_layout, this->base_};
	}
	friend decltype(auto) unrotated(static_array& self){return self.unrotated();}
	friend decltype(auto) unrotated(static_array const& self){return self.unrotated();}

	decltype(auto) operator<<(dimensionality_type d){return rotated(d);}
	decltype(auto) operator>>(dimensionality_type d){return unrotated(d);}
	decltype(auto) operator<<(dimensionality_type d) const{return rotated(d);}
	decltype(auto) operator>>(dimensionality_type d) const{return unrotated(d);}

	typename static_array::iterator begin() HD {return ref::begin();}
	typename static_array::iterator end() HD {return ref::end();}

	typename static_array::const_iterator begin() const{return ref::begin();}
	typename static_array::const_iterator end()   const{return ref::end();}
	const_iterator cbegin() const{return begin();}
	const_iterator cend() const{return end();}
	friend const_iterator cbegin(static_array const& self){return self.cbegin();}
	friend const_iterator cend(static_array const& self){return self.cend();}

	static_array& operator=(static_array const& other){
		assert( extensions(other) == static_array::extensions() );
		using std::copy_n;
		copy_n(other.data(), other.num_elements(), this->data());
		return *this;
	}

	operator basic_array<typename static_array::value_type, static_array::dimensionality, typename static_array::element_const_ptr, typename static_array::layout_t>()&{
		return static_array_cast<typename static_array::value_type, typename static_array::element_const_ptr>(*this);
	}
};

template<class T, class Alloc>
struct static_array<T, dimensionality_type{0}, Alloc> : 
	protected array_allocator<Alloc>,
	public array_ref<T, 0, typename std::allocator_traits<typename array_allocator<Alloc>::allocator_type>::pointer>
{
private:
	using array_alloc = array_allocator<Alloc>;
public:	
	static_assert( std::is_same<typename std::allocator_traits<Alloc>::value_type, typename static_array::element>{}, 
		"allocator value type must match array value type");
	using array_alloc::get_allocator;
	using allocator_type = typename static_array::allocator_type;
protected:
	using alloc_traits = typename std::allocator_traits<typename static_array::allocator_type>;
	using ref = array_ref<T, 0, typename std::allocator_traits<typename std::allocator_traits<Alloc>::template rebind_alloc<T>>::pointer>;
	auto uninitialized_value_construct(){return uninitialized_value_construct_n(static_array::alloc(), to_address(this->base_), this->num_elements());}
	template<typename It>
	auto uninitialized_copy_(It first){
		using boost::multi::uninitialized_copy_n;
		return uninitialized_copy_n(this->alloc(), first, this->num_elements(), this->data());
	}
	void destroy(){
		auto n = this->num_elements();
		while(n){
		//	std::allocator_traits<allocator_type>::destroy(this->alloc(), to_address(this->data() + n + (-1)));
			this->alloc().destroy(this->data() + n + (-1));//to_address(this->data() + n + (-1)));
			--n;
		}
	}
public:
	using typename ref::value_type;
	using typename ref::size_type;
	using typename ref::difference_type;
	static_array(typename static_array::allocator_type const& a) : array_alloc{a}{}
protected:
	static_array(static_array&& other, typename static_array::allocator_type const& a)                           //6b
	:	array_alloc{a},
		ref{other.base_, other.extensions()}
	{
		other.ref::layout_t::operator=({});
	}
public:
	using ref::operator==;
	static_array(typename static_array::extensions_type x, typename static_array::element const& e, typename static_array::allocator_type const& a) : //2
		array_alloc{a}, 
		ref(static_array::allocate(typename static_array::layout_t{x}.num_elements()), x)
	{
		uninitialized_fill(e);
	}
	static_array(typename static_array::element_type const& e, typename static_array::allocator_type const& a)
	:	static_array(typename static_array::extensions_type{}, e, a){}
	auto uninitialized_fill(typename static_array::element const& e){
		return uninitialized_fill_n(this->alloc(), this->base_, this->num_elements(), e);
	}
	static_array(typename static_array::extensions_type const& x, typename static_array::element const& e)  //2
	:	array_alloc{}, ref(static_array::allocate(typename static_array::layout_t{x}.num_elements()), x){
		uninitialized_fill(e);
	}
	static_array(typename static_array::element const& e)  //2
	:	static_array(multi::iextensions<0>{}, e){}

//	explicit static_array(typename static_array::index n, typename static_array::value_type const& v, typename static_array::allocator_type const& a = {})
//	: 	static_array(typename static_array::index_extension(n), v, a){}
	template<class ValueType, typename = std::enable_if_t<std::is_same<ValueType, typename static_array::value_type>{}>> 
	explicit static_array(typename static_array::index_extension const& e, ValueType const& v, typename static_array::allocator_type const& a = {}) //3
	: static_array(e*extensions(v), a){
	//	assert(0);
		using std::fill; fill(this->begin(), this->end(), v);
	}
//	template<class Allocator, typename = std::enable_if_t<std::is_same<Allocator, allocator_type>{}> >
//	explicit 
	static_array(typename static_array::extensions_type const& x, typename static_array::allocator_type const& a) //3
	: array_alloc{a}, ref{static_array::allocate(typename static_array::layout_t{x}.num_elements()), x}{
	//	assert(0);
	//	uninitialized_value_construct();
	}
	static_array(typename static_array::extensions_type const& x) //3
	:	array_alloc{}, ref{static_array::allocate(typename static_array::layout_t{x}.num_elements()), x}{
		if(not std::is_trivially_default_constructible<typename static_array::element>{})
			uninitialized_value_construct();
	}
	template<class TT, class... Args>
	static_array(multi::basic_array<TT, 0, Args...> const& other, typename static_array::allocator_type const& a = {})
		: array_alloc{a}, ref(static_array::allocate(other.num_elements()), extensions(other))
	{
		using std::copy; copy(other.begin(), other.end(), this->begin());
	}
	template<class TT, class... Args>
	static_array(array_ref<TT, 0, Args...> const& other)
	:	array_alloc{}, ref{static_array::allocate(other.num_elements()), extensions(other)}{
		uninitialized_copy_(other.data());
	}
	static_array(static_array const& other, typename static_array::allocator_type const& a)                      //5b
	:	array_alloc{a}, ref{static_array::allocate(other.num_elements()), extensions(other)}{
	//	assert(0);
		uninitialized_copy_(other.data());
	}
	static_array(static_array const& o) :                                  //5b
		array_alloc{o.get_allocator()}, 
		ref{static_array::allocate(o.num_elements()), o.extensions()}
	{
		uninitialized_copy_(o.data());
	}
	static_array(static_array&& o) :                                       //5b
		array_alloc{o.get_allocator()}, 
		ref{static_array::allocate(o.num_elements()), o.extensions()}
	{
		assert(0);
		uninitialized_copy_(o.data()); // TODO: uninitialized_move?
	}
	template<class It> static auto distance(It a, It b){using std::distance; return distance(a, b);}
protected:
	void deallocate(){ // TODO move this to array_allocator
		if(this->num_elements()) alloc_traits::deallocate(this->alloc(), this->base_, static_cast<typename alloc_traits::size_type>(this->num_elements()));
	}
	void clear() noexcept{
		this->destroy();
		deallocate();
		layout_t<0>::operator=({});
	}
public:
	static_array() = default;
	~static_array() noexcept{
		this->destroy();
		deallocate();
	}
	using element_const_ptr = typename std::pointer_traits<typename static_array::element_ptr>::template rebind<typename static_array::element const>;
	friend typename static_array::allocator_type get_allocator(static_array const& self){return self.get_allocator();}
	HD typename static_array::element_ptr       data()      {return ref::data();}
	HD auto data() const{return typename static_array::element_const_ptr{ref::data()};}
	friend typename static_array::element_ptr       data(static_array&       s){return s.data();}
	friend typename static_array::element_const_ptr data(static_array const& s){return s.data();}

	HD typename static_array::element_ptr       base()      {return ref::base();}
	HD auto base() const{return typename static_array::element_const_ptr{ref::base()};}
	friend typename static_array::element_ptr       base(static_array&       s){return s.base();}
	friend typename static_array::element_const_ptr base(static_array const& s){return s.base();}

	typename static_array::element_ptr       origin()      {return ref::origin();}
	typename static_array::element_const_ptr origin() const{return ref::origin();}
	friend typename static_array::element_ptr       origin(static_array&       s){return s.origin();}
	friend typename static_array::element_const_ptr origin(static_array const& s){return s.origin();}

//	template<class... Args> decltype(auto) operator()(Args const&... args)&{return ref::operator()(args...);}
//	template<class... Args> decltype(auto) operator()(Args const&... args) const&{return ref::operator()(args...);}
	using ref::operator();

//	basic_array<T, D, typename static_array::element_ptr> 
	decltype(auto) operator()()&{
		return ref::operator()();
	//	return *this;
	}
	basic_array<T, 0, typename static_array::element_const_ptr> operator()() const&{
		return basic_array<T, 0, typename static_array::element_const_ptr>{this->layout(), this->base_};
	}
//	using const_reverse_iterator = basic_reverse_iterator<const_iterator>;
	auto rotated(dimensionality_type d = 1) const&{
		typename static_array::layout_t new_layout = *this;
		new_layout.rotate(d);
		return basic_array<T, 0, typename static_array::element_const_ptr>{new_layout, this->base_};
	}
	auto rotated(dimensionality_type d = 1)&{
		typename static_array::layout_t new_layout = *this;
		new_layout.rotate(d);
		return basic_array<T, 0, typename static_array::element_ptr>{new_layout, this->base_};
	}
	auto rotated(dimensionality_type d = 1)&&{
		typename static_array::layout_t new_layout = *this;
		new_layout.rotate(d);
		return basic_array<T, 0, typename static_array::element_ptr>{new_layout, this->base_};
	}
//	friend decltype(auto) rotated(static_array const& self){return self.rotated();}
//	template<class Array, typename = std::enable_if_t<std::is_same<static_array, std::decay_t<Array>>{}> > 
	friend decltype(auto) rotated(static_array& self){return self.rotated();}
	friend decltype(auto) rotated(static_array const& self){return self.rotated();}

	auto unrotated(dimensionality_type d = 1) const&{
		typename static_array::layout_t new_layout = *this;
		new_layout.unrotate(d);
		return basic_array<T, 0, typename static_array::element_const_ptr>{new_layout, this->base_};
	}
	auto unrotated(dimensionality_type d = 1)&{
		typename static_array::layout_t new_layout = *this;
		new_layout.unrotate(d);
		return basic_array<T, 0, typename static_array::element_ptr>{new_layout, this->base_};
	}
	friend decltype(auto) unrotated(static_array& self){return self.unrotated();}
	friend decltype(auto) unrotated(static_array const& self){return self.unrotated();}

	decltype(auto) operator<<(dimensionality_type d){return rotated(d);}
	decltype(auto) operator>>(dimensionality_type d){return unrotated(d);}
	decltype(auto) operator<<(dimensionality_type d) const{return rotated(d);}
	decltype(auto) operator>>(dimensionality_type d) const{return unrotated(d);}

//	typename static_array::iterator begin() HD {return ref::begin();}
//	typename static_array::iterator end() HD {return ref::end();}

//	typename static_array::const_iterator begin() const{return ref::begin();}
//	typename static_array::const_iterator end()   const{return ref::end();}
//	const_iterator cbegin() const{return begin();}
//	const_iterator cend() const{return end();}
//	friend const_iterator cbegin(static_array const& self){return self.cbegin();}
//	friend const_iterator cend(static_array const& self){return self.cend();}

	static_array& operator=(static_array const& other){
		assert( extensions(other) == static_array::extensions() );
		using std::copy_n;
		copy_n(other.data(), other.num_elements(), this->data());
		return *this;
	}

	operator basic_array<typename static_array::value_type, static_array::dimensionality, typename static_array::element_const_ptr, typename static_array::layout_t>()&{
		return static_array_cast<typename static_array::value_type, typename static_array::element_const_ptr>(*this);
	}
};

template<class Archive>
struct archive_traits{
	template<class T>
	static decltype(auto) make_nvp(char const* name, T&& t){
		return boost::serialization::make_nvp(name, std::forward<T>(t));
	}
	template<class P1, class P2>
	static decltype(auto) make_array(P1&& p1, P2&& p2){
		return boost::serialization::make_array(std::forward<P1>(p1), std::forward<P2>(p2));
	}
};

template<typename T, class Alloc>
struct array<T, dimensionality_type{0}, Alloc>
	: static_array<T, 0, Alloc>
{
	using static_ = static_array<T, 0, Alloc>;
	using static_::static_;
};

template<class T, dimensionality_type D, class Alloc>
struct array : static_array<T, D, Alloc>,
	boost::multi::random_iterable<array<T, D, Alloc> >
{
	using static_ = static_array<T, D, Alloc>;
	static_assert(std::is_same<typename array::alloc_traits::value_type, T>{} or std::is_same<typename array::alloc_traits::value_type, void>{}, "!");
public:
	template<class Archive>
	auto serialize(Archive& ar, const unsigned int)
	->decltype(ar & archive_traits<Archive>::make_nvp(nullptr, archive_traits<Archive>::make_array(this->data(), this->num_elements())), void())
	{
		auto extensions = this->extensions();
		ar & archive_traits<Archive>::make_nvp("extensions", extensions);
		if(extensions != this->extensions()){clear(); reextent(extensions);}
		ar & archive_traits<Archive>::make_nvp("data", archive_traits<Archive>::make_array(this->data(), this->num_elements()));
	}
	using static_::static_;
	using typename array::ref::value_type;
//	using static_::ref::operator<;
	array() = default;
	array(array const&) = default;
public:
	void reshape(typename array::extensions_type x) &{
		typename array::layout_t new_layout{x};
		assert( new_layout.num_elements() == this->num_elements() );
		static_cast<typename array::layout_t&>(*this)=new_layout;
	}
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
	array(array&& o) noexcept : static_{std::move(o), o.get_allocator()}{}
	friend typename array::allocator_type get_allocator(array const& self){return self.get_allocator();}
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
//	using ref_::operator=;
	template<class A, typename = std::enable_if_t<not std::is_base_of<array, std::decay_t<A>>{}>,
		typename = std::enable_if_t<not std::is_convertible<std::decay_t<A>, typename array::element_type>{}>
	>
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
		if(extensions(other)==array::extensions()) static_::operator=(other);
		else{
			this->clear(); // std::cerr << "here" << __LINE__ << std::endl;
			this->static_::ref::layout_t::operator=(layout_t<D>{extensions(other)}); // std::cerr << "here" << __LINE__ << std::endl;
			this->base_ = array::allocate(this->num_elements());
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
	void reextent(typename array::extensions_type const& e){
		array tmp(e, this->get_allocator());
		tmp.intersection_assign_(*this);
		swap(tmp);
	}
	void reextent(typename array::extensions_type const& e, typename array::element const& v){
		array tmp(e, v, this->get_allocator());//static_cast<Alloc const&>(*this));
		tmp.intersection_assign_(*this);
		swap(tmp);
	}
};


template<typename T, dimensionality_type D, class... Args>
auto size(basic_array<T, D, Args...> const& arr){return arr.size();}
//template<typename T, dimensionality_type D, class... Args>
//auto size(array<T, D, Args...> const& arr){return arr.size();}


#if __cpp_deduction_guides
// clang cannot recognize templated-using, so don't replace IL<IL<T>> by IL2<T>, etc
#ifndef __clang__
template<class T, dimensionality_type D, class A=std::allocator<T>> static_array(multi::initializer_list_t<T, D>, A={})->static_array<T, D, A>;
template<class T, dimensionality_type D, class A=std::allocator<T>> array(multi::initializer_list_t<T, D>, A={})->array<T, D, A>;
#else
#define IL std::initializer_list
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
#undef IL
#endif

//template<class T> 
//array(std::initializer_list<std::initializer_list<double>>                )->array<double, 2, std::allocator<double>>; 


template<class T, class A=std::allocator<T>> array(T[]                  , A={})->array<T,1,A>;
template<class Array, class A=std::allocator<typename multi::array_traits<Array>::element>> array(Array            , A={})->array<typename multi::array_traits<Array>::element, 1, A>;

template<dimensionality_type D, class T, typename = std::enable_if_t<not is_allocator<T>{}> > array(iextensions<D>, T)->array<T, D, std::allocator<T>>;
	template<class T, typename = std::enable_if_t<not is_allocator<T>{}> > array(iextensions<1>, T)->array<T,1, std::allocator<T>>;
	template<class T, typename = std::enable_if_t<not is_allocator<T>{}> > array(iextensions<2>, T)->array<T,2, std::allocator<T>>;
	template<class T, typename = std::enable_if_t<not is_allocator<T>{}> > array(iextensions<3>, T)->array<T,3, std::allocator<T>>;
	template<class T, typename = std::enable_if_t<not is_allocator<T>{}> > array(iextensions<4>, T)->array<T,4, std::allocator<T>>;
	template<class T, typename = std::enable_if_t<not is_allocator<T>{}> > array(iextensions<5>, T)->array<T,5, std::allocator<T>>;


template<dimensionality_type D, class A, typename T = typename std::allocator_traits<A>::value_type> array(iextensions<D>, A)->array<T, D, A>;
	template<class A, typename T = typename std::allocator_traits<A>::value_type> array(iextensions<1>, A)->array<T, 1, A>;
	template<class A, typename T = typename std::allocator_traits<A>::value_type> array(iextensions<2>, A)->array<T, 2, A>;
	template<class A, typename T = typename std::allocator_traits<A>::value_type> array(iextensions<3>, A)->array<T, 3, A>;
	template<class A, typename T = typename std::allocator_traits<A>::value_type> array(iextensions<4>, A)->array<T, 4, A>;
	template<class A, typename T = typename std::allocator_traits<A>::value_type> array(iextensions<5>, A)->array<T, 5, A>;

template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<1>, T, MR*)->array<T, 1, A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<2>, T, MR*)->array<T, 2, A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<3>, T, MR*)->array<T, 3, A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<4>, T, MR*)->array<T, 4, A>;
template<class T, class MR, class A=memory::allocator<T, MR>> array(iextensions<5>, T, MR*)->array<T, 5, A>;

template<typename T, dimensionality_type D, typename P> array(basic_array<T, D, P>)->array<T, D>;
#endif

template <class T, std::size_t N>
multi::array<typename std::remove_all_extents<T[N]>::type, std::rank<T[N]>{}> 
decay(const T(&t)[N]) noexcept{
	return multi::array_cref<typename std::remove_all_extents<T[N]>::type, std::rank<T[N]>{}>(data_elements(t), extensions(t));
}
#if 0
template<class Archive, class T, boost::multi::dimensionality_type D, class... Args>
auto serialize(Archive& ar, array_ref<T, D, Args...>& self, unsigned) 
->decltype(ar & boost::serialization::make_nvp(nullptr, boost::serialization::make_array(data_elements(self), num_elements(self))),void()){
	auto x = extensions(self);
	ar & boost::serialization::make_nvp("extensions", x);
	assert( x == extensions(self) );// {clear(self); self.reextent(x);}
	ar & boost::serialization::make_nvp("data", boost::serialization::make_array(data_elements(self), num_elements(self)));
}

template<class Archive, class T, dimensionality_type D, class... Args>
auto serialize_aux(Archive& ar, multi::array<T, D, Args...>& self, unsigned) 
->decltype(ar & boost::serialization::make_nvp(nullptr, boost::serialization::make_array(data_elements(self), num_elements(self))),void()){
	auto x = extensions(self);
	ar & boost::serialization::make_nvp("extensions", x);
	if(x != extensions(self)){clear(self); self.reextent(x);}
	ar & boost::serialization::make_nvp("data", boost::serialization::make_array(data_elements(self), num_elements(self)));
}

template<class Archive, class T, dimensionality_type D, class... Args>
auto serialize(Archive& ar, multi::array<T, D, Args...>& self, unsigned version)
->decltype(serialize_aux(ar, self, version)){
	return serialize_aux(ar, self, version);}

template<class Archive, class T, dimensionality_type D, class... Args>
auto serialize(Archive& ar, basic_array<T, D, Args...>& self, unsigned version)
->decltype(serialize_aux(ar, std::declval<array<T, D>&>(), version), void())
{
	if(Archive::is_saving()) serialize(ar, multi::array<T, D>{self}, version);
	else{
		boost::multi::array<T, D> tmp(extensions(self));
		serialize(ar, tmp, version); assert( extensions(self) == extensions(tmp) );
		self = tmp;
	}
}
#endif

}}


//namespace boost{
//namespace serialization{


//}}
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

{
	double A[2][3] = {{1.,2.,3.}, {4.,5.,6.}};
	using multi::decay;
	auto A_copy = decay(A);
}

}
#endif
#endif

