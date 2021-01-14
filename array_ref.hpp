// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo Correa 2018-2021

//#if (defined(__clang__) and defined(__CUDA__)) or defined(__NVCC__)
//#define BOOST_RESULT_OF_USE_TR1_WITH_DECLTYPE_FALLBACK //see comments https://www.boost.org/doc/libs/1_72_0/boost/utility/result_of.hpp
//#endif

#ifndef BOOST_MULTI_ARRAY_REF_HPP
#define BOOST_MULTI_ARRAY_REF_HPP

#if defined(__NVCC__)
#define HOSTDEVICE __host__ __device__
#else
#define HOSTDEVICE
#endif

#include "./memory/pointer_traits.hpp"
#include "utility.hpp" 

#include "./detail/layout.hpp"
#include "./detail/types.hpp"     // dimensionality_type
#include "./detail/operators.hpp" // random_iterable
#include "./detail/memory.hpp"    // pointer_traits

#include "./config/NODISCARD.hpp"
#include "./config/DELETE.hpp"
#include "./config/ASSERT.hpp"

//#include<iostream> // debug

#include<algorithm>  // copy_n
#include<cstring>    // for memset in reinterpret_cast
#include<functional> // invoke
#include<memory>     // pointer_traits

namespace std{
	template<class T>
	struct pointer_traits<std::move_iterator<T*>> : std::pointer_traits<T*>{
		template<class U> using rebind = 
			std::conditional_t<std::is_const<U>{}, 
				U*,
				std::pointer_traits<std::move_iterator<U*>>
			>;
	};
}

namespace boost{
namespace multi{

template<class T, class Ptr = T*>
struct move_ptr : std::move_iterator<Ptr>{
	using std::move_iterator<Ptr>::move_iterator;
	explicit operator Ptr() const{return std::move_iterator<Ptr>::base();}
};

template<class T> T& modify(T const& t){return const_cast<T&>(t);}

template<typename T, dimensionality_type D, typename ElementPtr = T*, class Layout = layout_t<D>>
class basic_array;

template<typename T, dimensionality_type D, class A = std::allocator<T>> struct array;

template<class To, class From, std::enable_if_t<    std::is_convertible<From, To>{} and std::is_constructible<To, From>{},int> =0> constexpr To implicit_cast(From&& f){return static_cast<To>(f);}
template<class To, class From, std::enable_if_t<not std::is_convertible<From, To>{} and std::is_constructible<To, From>{},int> =0> constexpr To explicit_cast(From&& f){return static_cast<To>(f);}

template<class T, dimensionality_type D, class ElementPtr = T*>
class basic_array_ptr :
	public equality_comparable2 <basic_array_ptr<T, D, ElementPtr>>
{
	mutable basic_array<T, D, ElementPtr> ref_;
	using layout_type = typename basic_array<T, D, ElementPtr>::layout_type;
	using element_ptr = typename basic_array<T, D, ElementPtr>::element_ptr;
public:
	constexpr basic_array_ptr(layout_type const& l, element_ptr const& p) : ref_{l, p}{}
	constexpr basic_array_ptr(basic_array_ptr const& o) : basic_array_ptr{o->layout(), o->base()}{}
	template<class BasicArrayPtr, decltype(implicit_cast<ElementPtr>(std::declval<BasicArrayPtr>()->base()))* =nullptr>
	constexpr          basic_array_ptr(BasicArrayPtr const& o) : basic_array_ptr{o->layout(), o->base()}{}
	template<class BasicArrayPtr, decltype(explicit_cast<ElementPtr>(std::declval<BasicArrayPtr>()->base()))* =nullptr>
	constexpr explicit basic_array_ptr(BasicArrayPtr const& o) : basic_array_ptr(o->layout(), o->base()){}

	constexpr basic_array_ptr(std::nullptr_t p) : basic_array_ptr{ {}, p}{}
	constexpr basic_array_ptr() : basic_array_ptr{ {}, {} }{}

	using reference = basic_array<T, D, ElementPtr>;
	using difference_type = typename basic_array<T, D, ElementPtr>::difference_type;

	constexpr basic_array_ptr(reference* p) : basic_array_ptr{p->layout(), p->base()}{}
	constexpr basic_array_ptr& operator=(basic_array_ptr const& other){
		ref_.layout_ = other.ref_.layout_;
		ref_.base_   = other.ref_.base_;
		return *this;
	}
	constexpr reference& operator* () const{return ref_;} // or reference
	constexpr reference* operator->() const{return std::addressof(ref_);}
	template<class BasicArrayPtr>
	bool operator==(BasicArrayPtr const& other) const{assert( (*this)->layout() == other->layout() );
		return (*this)->base() == other->base() and (*this)->layout() == other->layout();
	}
	template<class BasicArrayPtr>
	bool operator!=(BasicArrayPtr const& other) const{return not operator==(other);}
	
	explicit operator bool() const{return ref_.base();}
};

template<typename Element, dimensionality_type D, typename ElementPtr>
class array_iterator;

template<typename Element, dimensionality_type D, typename ElementPtr>
class array_iterator :
	public totally_ordered2<array_iterator<Element, D, ElementPtr>>
{
	basic_array_ptr<Element, D-1, ElementPtr> ptr_;
	using element = Element;
	using element_ptr = ElementPtr;
	using stride_type = index;//typename basic_array_ptr<Element, D-1, ElementPtr>::stride_type;
	stride_type stride_ = {1}; // nice non-zero default

public:
	using pointer   = basic_array_ptr<element, D-1, element_ptr>;
	using reference = typename pointer::reference;
	using value_type = array<element, D-1, typename multi::pointer_traits<element_ptr>::default_allocator_type>;// typename pointer::value_type;
	using difference_type = typename pointer::difference_type;
//	using element = typename Ref::value_type;
	using iterator_category = std::random_access_iterator_tag;

	using rank = std::integral_constant<dimensionality_type, D>;

	array_iterator() = default;
	constexpr array_iterator(std::nullptr_t p) : ptr_{p}, stride_{1}{}
	template<class, dimensionality_type, class> friend class array_iterator;
	constexpr array_iterator(array_iterator const& o) : ptr_{o.ptr_}, stride_{o.stride_}{}//= default;
	template<class ArrayIterator, decltype(implicit_cast<pointer>(std::declval<ArrayIterator>().ptr_))* =nullptr>
	constexpr          array_iterator(ArrayIterator const& o) : ptr_{o.ptr_}, stride_{o.stride_}{}
	template<class ArrayIterator, decltype(explicit_cast<pointer>(std::declval<ArrayIterator>().ptr_))* =nullptr>
	constexpr explicit array_iterator(ArrayIterator const& o) : ptr_(o.ptr_), stride_{o.stride_}{}

	constexpr array_iterator& operator=(array_iterator const& other){
		ptr_ = other.ptr_;
		stride_ = other.stride_;
		return *this;
	}
	explicit constexpr operator bool() const{return static_cast<bool>(ptr_);}
	constexpr reference operator*() const{/*assert(*this);*/ 
		return std::move(*ptr_);
	}
	constexpr decltype(auto) operator->() const{/*assert(*this);*/ return ptr_;}//return this;}
	constexpr basic_array<element, D-1, element_ptr> operator[](difference_type n) const{
		auto sum = (*this);
		sum += n;
		*sum;
		basic_array<element, D-1, element_ptr> ret{*sum};
		return ret;
	}
	template<class O> constexpr bool operator==(O const& o) const{return equal(o);}
	constexpr array_iterator(
		typename basic_array<element, D-1, element_ptr>::element_ptr const& p, 
		typename basic_array<element, D-1, element_ptr>::layout_type const& l, 
		index const& stride
	) : 
		ptr_{l, p}, 
		stride_{stride}
	{}
	template<class, dimensionality_type, class, class> friend class basic_array;
	template<class... As> constexpr decltype(auto) operator()(index i, As... as) const{return this->operator[](i)(as...);}
	                      constexpr decltype(auto) operator()(index i          ) const{return this->operator[](i)       ;}

private:
	template<typename Tuple, std::size_t ... I> 
	constexpr decltype(auto) apply_impl(Tuple const& t, std::index_sequence<I...>) const{return this->operator()(std::get<I>(t)...);}
public:
	template<typename Tuple> constexpr decltype(auto) apply(Tuple const& t) const{return apply_impl(t, std::make_index_sequence<std::tuple_size<Tuple>::value>());}
	constexpr bool operator==(array_iterator const& other) const{assert(stride_ == other.stride_);
		return ptr_ == other.ptr_;
	}
	constexpr bool operator!=(array_iterator const& other) const{return not operator==(other);}

private:
	constexpr bool equal(array_iterator const& o) const{return ptr_==o.ptr_ and stride_==o.stride_;}//base_==o.base_ && stride_==o.stride_ && ptr_.layout()==o.ptr_.layout();}
	constexpr difference_type distance_to(array_iterator const& other) const{
		assert( stride_ == other.stride_); assert( stride_ != 0 );
	//	assert( this->stride()==stride(other) and this->stride() );// and (base(other.ptr_) - base(this->ptr_))%stride_ == 0
	//	assert( stride_ == other.stride_ and stride_ != 0 and (other.ptr_.base_-ptr_.base_)%stride_ == 0 and ptr_.layout() == other.ptr_.layout() );
	//	assert( stride_ == other.stride_ and stride_ != 0 and (other.base_ - base_)%stride_ == 0 and layout() == other.layout() );
		return (other.ptr_->base_ - ptr_->base_)/stride_;
	}
public:
	constexpr array_iterator& operator+=(difference_type n){ptr_->base_ += stride_*n; return *this;}
	constexpr array_iterator& operator-=(difference_type n){return operator+=(-n);}

	constexpr array_iterator& operator++(){return operator+=(1);}
	constexpr array_iterator& operator--(){return operator-=(1);}

	constexpr array_iterator  operator+(difference_type n) const{return array_iterator{*this}+=n;}
	constexpr array_iterator  operator-(difference_type n) const{return array_iterator{*this}-=n;}

	constexpr bool            operator<(array_iterator const& s) const{
		assert(stride_ == s.stride_);
		assert(stride_ != 0);
		return (0 < stride_)?(ptr_->base_ < s.ptr_->base_):(s.ptr_->base_ < ptr_->base_);
	}
public:
	       constexpr element_ptr base()              const    {return ptr_->base();}
	friend constexpr element_ptr base(array_iterator const& s){return s.base();}
	       constexpr stride_type stride()              const    {return   stride_;}
	friend constexpr stride_type stride(array_iterator const& s){return s.stride_;}
	friend constexpr difference_type operator-(array_iterator const& self, array_iterator const& other){
		assert(self.stride_ == other.stride_); assert(self.stride_ != 0);
		return (self.ptr_->base() - other.ptr_->base())/self.stride_;
	}
};

template<class It>
struct biiterator : 
	boost::multi::iterator_facade<
		biiterator<It>,
		typename std::iterator_traits<It>::value_type, std::random_access_iterator_tag, 
		decltype(*(std::move((*std::declval<It>())).begin())), multi::difference_type
	>,
	multi::affine<biiterator<It>, multi::difference_type>,
	multi::decrementable<biiterator<It>>,
	multi::incrementable<biiterator<It>>,
	multi::totally_ordered2<biiterator<It>, void>
{
	It me_;
	std::ptrdiff_t pos_;
	std::ptrdiff_t stride_;
	biiterator() = default;
	biiterator(biiterator const& other) = default;// : me{other.me}, pos{other.pos}, stride{other.stride}{}
	constexpr biiterator(It me, std::ptrdiff_t pos, std::ptrdiff_t stride) : me_{me}, pos_{pos}, stride_{stride}{}
	constexpr decltype(auto) operator++(){
		++pos_;
		if(pos_==stride_){
			++me_;
			pos_ = 0;
		}
		return *this;
	}
	constexpr bool operator==(biiterator const& o) const{return me_==o.me_ and pos_==o.pos_;}
	constexpr biiterator& operator+=(multi::difference_type n){me_ += n/stride_; pos_ += n%stride_; return *this;}
	constexpr decltype(auto) operator*() const{
		auto meb = std::move(*me_).begin();
		return meb[pos_];
	}
	using difference_type = std::ptrdiff_t;
	using reference = decltype(*std::declval<biiterator>());
	using value_type = std::decay_t<reference>;
	using pointer = value_type*;
	using iterator_category = std::random_access_iterator_tag;
};

template<typename T, typename ElementPtr, class Layout>
class basic_array<T, dimensionality_type(0), ElementPtr, Layout>{
protected:
	Layout layout_;
	ElementPtr base_;
public:
	using layout_type = Layout;

	using element_ptr  = ElementPtr;
	using element      = typename std::pointer_traits<element_ptr>::element_type;
	using element_cptr = typename std::pointer_traits<element_ptr>::template rebind<element const>; //multi::const_iterator<ElementPtr>; 

	using  ptr = element_ptr;//BasicArrayPtr<T, dimensionality_type(0), ElementPtr>;
	using cptr = element_cptr;//BasicArrayPtr<T, dimensionality_type(0), ElementPtr>;

	using element_ref  = typename std::iterator_traits<typename basic_array::element_ptr >::reference;//decltype(*typename basic_array::element_ptr{});
	using element_cref = typename std::iterator_traits<typename basic_array::element_cptr>::reference;

	using value_type = void;
	using decay_type = multi::array<element, 0, typename multi::pointer_traits<typename basic_array::element_ptr>::default_allocator_type>;

	using extensions_type = typename layout_type::extensions_type;
	using index_extension = typename layout_type::index_extension;

	constexpr basic_array(layout_type const& l, element_ptr const& p) : layout_{l}, base_{p}{}
	
	constexpr auto extensions() const&{return layout_.extensions();}
	
	constexpr layout_type layout() const{return layout_;}
	constexpr auto num_elements() const{return layout_.num_elements();}
	friend constexpr auto num_elements(basic_array const& self){return self.num_elements();}

private:
	// TODO return array_ptr<T, 0>
	constexpr ptr addressof_() const&{return base_;}
public:
	constexpr cptr operator&() const&{return addressof_();}
	constexpr  ptr operator&()      &{return addressof_();}
	constexpr  ptr operator&()     &&{return addressof_();}

	constexpr decltype(auto) operator=(typename basic_array::element const& e) &{
		adl_copy_n(&e, 1, this->base_); return *this;
	}

private:
	constexpr element_ref element_ref_mutable() const&{return *(this->base_);}
public:
	constexpr operator element_cref() const&{return element_ref_mutable();}
	constexpr operator element_ref ()     &&{return element_ref_mutable();}
	constexpr operator element_ref ()      &{return element_ref_mutable();}

	template<class Value> constexpr bool operator==(Value const& o) const{
		return adl_equal(base_, base_ + 1, &o);
	}

	constexpr bool operator==(basic_array const& o) const&{
		return adl_equal(o.base_, o.base_ + 1, base_);
	}
	constexpr bool operator!=(basic_array const& o) const&{return not operator==(o);}

private:
	constexpr element_ptr base_mutable() const&{return base_;}
public:
	constexpr element_cptr base() const&{return base_mutable();}
	constexpr element_ptr  base()      &{return base_mutable();}
	constexpr element_ptr  base()     &&{return base_mutable();}

	template<class S, basic_array* =nullptr>
	friend constexpr auto base(S&& s){return std::forward<S>(s).base();}

private:
	constexpr element_cref paren() const&{return element_ref_mutable();}
	constexpr element_ref  paren()     &&{return element_ref_mutable();}
	constexpr element_ref  paren()      &{return element_ref_mutable();}
	template<typename, dimensionality_type, typename, class> friend class basic_array;
public:
	constexpr element_cref operator()() const&{return element_ref_mutable();}
	constexpr element_ref  operator()()     &&{return element_ref_mutable();}
	constexpr element_ref  operator()()      &{return element_ref_mutable();}
};

template<typename T, typename ElementPtr, class Layout>
class basic_array<T, dimensionality_type(1), ElementPtr, Layout> : 
	public totally_ordered2   <basic_array<T, dimensionality_type(1), ElementPtr, Layout>>,
	public swappable2         <basic_array<T, dimensionality_type(1), ElementPtr, Layout>>,
	public container_interface<basic_array<T, dimensionality_type(1), ElementPtr, Layout>, 
		typename multi::array_iterator<T, 1, ElementPtr                                                        >, 
		typename multi::array_iterator<T, 1, typename std::pointer_traits<ElementPtr>::template rebind<T const>>, 
		typename std::iterator_traits<ElementPtr>::reference, typename std::iterator_traits<typename std::pointer_traits<ElementPtr>::template rebind<T const>>::reference
	>
{
	using container_interface_ 
		 = container_interface<basic_array<T, dimensionality_type(1), ElementPtr, Layout>, 
		typename multi::array_iterator<T, 1, ElementPtr>, typename multi::array_iterator<T, 1, typename std::pointer_traits<ElementPtr>::template rebind<T const>>, 
		typename std::iterator_traits<ElementPtr>::reference, typename std::iterator_traits<typename std::pointer_traits<ElementPtr>::template rebind<T const>>::reference
	>;
	friend container_interface_;
protected:
	Layout layout_;
public:
	ElementPtr base_;
public:
	using layout_type = Layout;
	using element_ptr       = ElementPtr;
	using element           = typename std::pointer_traits<element_ptr>::element_type;
	using element_cptr = typename std::pointer_traits<element_ptr>::template rebind<element const>; //multi::const_iterator<ElementPtr>; 
	using element_ref       = typename std::iterator_traits<element_ptr>::reference;

	using typename container_interface_::      reference;
	using typename container_interface_::const_reference;
	using value_type      = T; // or std::decay_t<reference> or multi::decay_t<reference>

	using ptr             = basic_array_ptr<T, 1, element_ptr>;
	using cptr            = basic_array_ptr<T, 1, element_cptr>;
	using ref             = basic_array<T, 1, element_ptr , layout_type>;
	using cref            = basic_array<T, 1, element_cptr, layout_type>;
	using decay_type      = multi::array<T, 1, typename multi::pointer_traits<element_ptr>::default_allocator_type>;

//	using basic_array       = basic_array<T, 1, element_ptr , layout_type>;
	using basic_const_array = basic_array<T, 1, element_cptr, layout_type>;

	using typename container_interface_::      iterator;//typename multi::array_iterator<element, 1, element_ptr >;
	using typename container_interface_::const_iterator; //typename multi::array_iterator<element, 1, element_cptr>;

	using extensions_type = typename layout_type::extensions_type;
	using extension_type = typename layout_type::index_extension;

	using index_extension = typename layout_type::index_extension;
//	using difference_type = typename layout_type::difference_type;
//	using size_type       = typename layout_type::size_type;
//	using size_type       = typename container_interface_::size_type;
	using typename container_interface_::difference_type;
	using typename container_interface_::size_type;

	using rank = std::integral_constant<dimensionality_type, 1>;
	constexpr static dimensionality_type rank_v = rank{};
	
	constexpr dimensionality_type dimensionality() const{return layout_.dimensionality();}
protected:
	basic_array() = default;
	basic_array(basic_array const&) = default;
	constexpr basic_array(layout_type const& l, element_ptr const& p) : layout_{l}, base_{p}{}
	template<typename, dimensionality_type, typename, class> friend class basic_array;
	template<typename, dimensionality_type, typename> friend class basic_array_ptr;
//	friend class basic_array_ptr<element, 1, element_ptr>;
public:
	basic_array(basic_array&&) = default;
	template<class BasicArray> constexpr basic_array(BasicArray&& o) : basic_array(o.layout(), element_ptr{o.base()}){}
	template<class BasicArray> constexpr basic_array(BasicArray      &) = delete;
	template<class BasicArray> constexpr basic_array(BasicArray const&) = delete;

public:
	constexpr layout_type layout()     const{return layout_;}

	constexpr auto extension()  const{return layout_.extension() ;}
	constexpr auto extensions() const{return layout_.extensions();}
	constexpr auto sizes()      const{return layout_.sizes();}
	constexpr auto stride()     const{return layout_.stride();}
	friend constexpr auto stride(basic_array const& self){return self.stride();}

	constexpr auto strides()     const{return layout_.strides();}
	friend constexpr auto strides(basic_array const& self){return self.strides();}

	constexpr auto num_elements() const{return layout_.num_elements();}
	friend constexpr auto num_elements(basic_array const& self){return self.num_elements();}
	
	constexpr size_type size() const{return container_interface_::size();}

	using container_interface_::is_empty;
	using container_interface_::front;
	using container_interface_::back;

private:
	constexpr reference at_(index i) const{MULTI_ACCESS_ASSERT(this->extension().contains(i)&&"out of bounds");
		return *(base_ + layout_(i));} // in C++17 this is allowed even with syntethic references
public:
	using container_interface_::operator[];

private:
	constexpr element_ptr base_mutable() const{return base_;}
public:
	constexpr element_cptr base() const&{return base_mutable();}
	constexpr element_ptr  base()      &{return base_mutable();}
	constexpr element_ptr  base()     &&{return base_mutable();}

	template<class S, basic_array* =nullptr>
	constexpr auto base(S&& s){return std::forward<S>(s).base();}

	constexpr element_cptr cbase() const{return base();}
	template<class S, basic_array* =nullptr>
	constexpr auto cbase(S&& s){return std::forward<S>(s).cbase();}

private:
	constexpr auto begin_() const{return iterator(base_                   , layout_.stride());}
	constexpr auto end_  () const{return iterator(base_ + layout_.nelems(), layout_.stride());}
public:
	using container_interface_::begin;
	using container_interface_::end;

public:
	constexpr basic_const_array protect() const{return {layout(), base_};}
	friend constexpr basic_const_array protect(basic_array const& self){return self.protect();}

private:
	constexpr ref paren_() const&{return {layout_, base_};}
public:
	constexpr auto paren() const&{return paren_().protect();}
	constexpr auto paren()     &&{return paren_();}
	constexpr auto paren()      &{return paren_();}

	constexpr auto operator()() const&{return paren_().protect();}
	constexpr auto operator()()     &&{return paren_();}
	constexpr auto operator()()      &{return paren_();}

private:
	constexpr basic_array strided_(index s) const&{
		layout_type new_ = layout_;
		new_.stride_*=s;
		return {new_, base_};
	}
public:
	constexpr auto strided(index s) const&{return strided_(s).protect();}
	constexpr auto strided(index s)     &&{return strided_(s);}
	constexpr auto strided(index s)      &{return strided_(s);}

private:
	constexpr basic_array sliced_(index first, index last) const&{
		layout_type new_ = layout_; 
		(new_.nelems_ /= size())*=(last - first);
		return {new_, base_ + layout_(first)};
	}
public:
	constexpr auto sliced(index first, index last) const&{return sliced_(first, last).protect();}
	constexpr auto sliced(index first, index last)     &&{return sliced_(first, last);}
	constexpr auto sliced(index first, index last)      &{return sliced_(first, last);}

private:
	constexpr auto range(index_range ir) const&{return sliced(ir.front(), ir.last());}
	constexpr auto range(index_range ir)     &&{return sliced(ir.front(), ir.last());}
	constexpr auto range(index_range ir)      &{return sliced(ir.front(), ir.last());}

// TODO remove paren
private:
	constexpr auto paren_(index_range const& ir) const{return sliced_(ir.front(), ir.last());}
public:
	constexpr auto paren(index_range const& ir) const&{return paren_(ir).protect();}
	constexpr auto paren(index_range const& ir)     &&{return paren_(ir);}
	constexpr auto paren(index_range const& ir)      &{return paren_(ir);}

	constexpr decltype(auto) paren(index i) &     {return operator[](i);}
	constexpr decltype(auto) paren(index i) &&    {return operator[](i);}
	constexpr decltype(auto) paren(index i) const&{return operator[](i);}

	template<class S, basic_array* =nullptr>
	friend constexpr decltype(auto) get(S&& s, index i){return std::forward<S>(s).operator[](i);}

public:
	constexpr auto rotated(dimensionality_type = 1) const&{return operator()();}
	constexpr auto rotated(dimensionality_type = 1)     &&{return operator()();}
	constexpr auto rotated(dimensionality_type = 1)      &{return operator()();}

	template<class S, basic_array* =nullptr>
	friend constexpr auto rotated(S&& s, dimensionality_type d = 1){return std::forward<S>(s).rotated(d);}

public:
	constexpr auto unrotated(dimensionality_type = 1) const&{return operator()();}
	constexpr auto unrotated(dimensionality_type = 1)     &&{return operator()();}
	constexpr auto unrotated(dimensionality_type = 1)      &{return operator()();}

	template<class S, basic_array* =nullptr>
	friend constexpr auto unrotated(S&& s, dimensionality_type d = 1){return std::forward<S>(s).rotated(d);}

	template<class It> bool equal(It first) const{return adl_equal(this->begin(), end(), first);}

	template<class BasicArray>
	constexpr auto operator==(BasicArray const& other) const{
		return extension()==other.extension() and equal(other.begin());
	}

	template<class It> constexpr auto assign(It other) &{return adl_copy_n(other, size(), begin());}
	template<class It> constexpr auto assign(It other)&&{return assign(other);}

	constexpr void fill(value_type const& v) &{adl_fill_n(begin(), size(), v);}
	constexpr void fill(value_type const& v)&&{fill(v);}

	template<class BasicArray>
	constexpr basic_array&  operator=(BasicArray const& o)&{assert(this->extension() == o.extension());
		return this->assign(o.begin()), *this; // cannot be rotated in 1D
	}
	template<class BasicArray>
	constexpr basic_array&& operator=(BasicArray const& o)&&{return std::move(operator=(o));}

	constexpr basic_array&  operator=(basic_array const& o) &{return operator= <basic_array>(o);}
	constexpr basic_array&& operator=(basic_array const& o)&&{return std::move(operator=(o));}

private:
	constexpr basic_array<T, 2, ElementPtr> partitioned_(size_type s) const{
		assert( layout().nelems_%s==0 ); // TODO remove assert? truncate left over? (like mathematica)
		typename basic_array<T, 2, ElementPtr>::layout_type new_{layout_, layout_.nelems_/s, 0, layout_.nelems_};
		new_.sub_.nelems_/=s;
		return {new_, base_};
	}
public:
	constexpr auto partitioned(size_type s) const&{return partitioned_(s).protect();}
	constexpr auto partitioned(size_type s)     &&{return partitioned_(s);}
	constexpr auto partitioned(size_type s)      &{return partitioned_(s);}

	template<class It>
	constexpr bool lexicographical_compare(It first, It last) const{
		return adl_lexicographical_compare(begin(), end(), first, last);
	}
	template<class BasicArray>
	constexpr bool operator<(BasicArray const& other) const{
		if(extension().first() > other.extension().first()) return true ;
		if(extension().first() < other.extension().first()) return false;
		return lexicographical_compare(other.begin(), other.end());
	}

private:
	template<class It> constexpr void swap_ranges(It first) noexcept{
		adl_swap_ranges(this->begin(), this->end(), first);
	}
public:
	template<class Array> constexpr void swap(Array&& o)&& noexcept{return swap_ranges(adl_begin(std::forward<Array>(o)));}
	template<class Array> constexpr void swap(Array&& o)&  noexcept{return swap_ranges(adl_begin(std::forward<Array>(o)));}

protected:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>>
	using rebind = basic_array<std::decay_t<T2>, 1, P2>;

	template<class T2, class P2>
	constexpr rebind<T2> static_array_cast_() const{
		return {layout(), static_cast<typename std::pointer_traits<element_ptr>::template rebind<T2>>(base_)}; // pointer_cast
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2 const>> constexpr rebind<T2, P2> static_array_cast() const&{return static_array_cast_<T2, P2>();}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2      >> constexpr rebind<T2, P2> static_array_cast()     &&{return static_array_cast_<T2, P2>();}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2      >> constexpr rebind<T2, P2> static_array_cast()      &{return static_array_cast_<T2, P2>();}

public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>,
		class Element = typename basic_array::element,
		class PM = T2 std::decay_t<Element>::*
	>
	constexpr rebind<T2, P2> member_array_cast_(PM pm) const{
		static_assert(sizeof(T)%sizeof(T2) == 0, 
			"member_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements");
#if defined(__GNUC__) and (not defined(__INTEL_COMPILER))
		auto&& r1 = (*((typename basic_array::element*)(basic_array::base_))).*pm;//->*pm;
		auto p1 = &r1; auto p2 = (P2)p1;
		return {this->layout().scale(sizeof(T)/sizeof(T2)), p2};
#else
		return {this->layout().scale(sizeof(T)/sizeof(T2)), static_cast<P2>(&(this->base_->*pm))}; // this crashes the gcc compiler
#endif
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>, class Element = typename basic_array::element, class PM = T2 std::decay_t<Element>::*> constexpr auto member_array_cast(PM pm) const&{return member_array_cast_<T2, P2>(pm).protect();}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>, class Element = typename basic_array::element, class PM = T2 std::decay_t<Element>::*> constexpr auto member_array_cast(PM pm)     &&{return member_array_cast_<T2, P2>(pm);}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>, class Element = typename basic_array::element, class PM = T2 std::decay_t<Element>::*> constexpr auto member_array_cast(PM pm)      &{return member_array_cast_<T2, P2>(pm);}

private:
	constexpr ptr addressof_() const&{return {layout(), base_};}
public:
	constexpr cptr operator&() const&{return addressof_();}
	constexpr  ptr operator&()      &{return addressof_();}
	constexpr  ptr operator&()     &&{return addressof_();}
	
private:
	constexpr basic_array reindexed_(index first) const{
		layout_type new_ = layout_;
		new_.reindex(first);
		return {new_, base_};
	}
public:
	constexpr auto reindexed(index first) const&{return reindexed_(first).protect();}
	constexpr auto reindexed(index first)     &&{return reindexed_(first);}
	constexpr auto reindexed(index first)      &{return reindexed_(first);}

public:
	constexpr basic_const_array blocked(index first, index last) const&{return sliced(first, last).reindexed(first);}
	constexpr basic_array       blocked(index first, index last)     &&{return sliced(first, last).reindexed(first);}
	constexpr basic_array       blocked(index first, index last)      &{return sliced(first, last).reindexed(first);}

public:
	constexpr auto stenciled(index_extension x) const&{return blocked(x.start(), x.finish());}
	constexpr auto stenciled(index_extension x)     &&{return blocked(x.start(), x.finish());}
	constexpr auto stenciled(index_extension x)      &{return blocked(x.start(), x.finish());}

private:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>
	constexpr rebind<T2, P2> reinterpret_array_cast_() const{
		static_assert( sizeof(T)%sizeof(T2)== 0, "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );
//			this->layout().scale(sizeof(T)/sizeof(T2));
		static_assert( sizeof(P2) == sizeof(element_ptr), "reinterpret on equal size?");
		P2 new_base; std::memcpy(&new_base, &base_, sizeof(P2)); //reinterpret_cast<P2 const&>(thisbase) // TODO find a better way, fancy pointers wouldn't need reinterpret_cast
		return {this->layout().scale(sizeof(T)/sizeof(T2)), new_base};
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2 const>> constexpr auto reinterpret_array_cast() const&{return reinterpret_array_cast_<T2, P2>().protect();}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2      >> constexpr auto reinterpret_array_cast()     &&{return reinterpret_array_cast_<T2, P2>();}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2      >> constexpr auto reinterpret_array_cast()      &{return reinterpret_array_cast_<T2, P2>();}

private:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>
	constexpr auto reinterpret_array_cast_(size_type n) const{
		static_assert( sizeof(T)%sizeof(T2)== 0, 
			"error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );
	//	assert( sizeof(T )%(sizeof(T2)*n)== 0 );
		auto thisbase = base_;
		return basic_array<std::decay_t<T2>, 2, P2>{
			layout_t<2>{this->layout().scale(sizeof(T)/sizeof(T2)), 1, 0, n}, 
			static_cast<P2>(static_cast<void*>(thisbase))
		}.rotated();
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2 const>> constexpr auto reinterpret_array_cast(size_type n) const&{return reinterpret_array_cast_<T2, P2>(n).protect();}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2      >> constexpr auto reinterpret_array_cast(size_type n)     &&{return reinterpret_array_cast_<T2, P2>(n);}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2      >> constexpr auto reinterpret_array_cast(size_type n)      &{return reinterpret_array_cast_<T2, P2>(n);}

protected:
	template<class A>
	constexpr void intersection_assign_(A&& other)&{
		auto const is = intersection(extension(), other.extension());
		using std::for_each;
		for_each(
			is.begin(), is.end(), 
			[&](index idx){operator[](idx) = std::forward<A>(other)[idx];}
		);
	}
	template<class A> constexpr void intersection_assign_(A&& o)&&{intersection_assign_(std::forward<A>(o));}

public:
	constexpr auto operator()(index_range const& ir) const&{return range(ir).protect();}
	constexpr auto operator()(index_range const& ir)     &&{return range(ir);}
	constexpr auto operator()(index_range const& ir)      &{return range(ir);}

	constexpr decltype(auto) operator()(index i) const&{return operator[](i);}
	constexpr decltype(auto) operator()(index i)     &&{return operator[](i);}
	constexpr decltype(auto) operator()(index i)      &{return operator[](i);}

	constexpr decay_type decay() const{return *this;}
	constexpr decay_type operator+() const{return decay();}
};

struct general_order_layout_tag{};

struct c_order_layout_tag : general_order_layout_tag{};

template<typename T, dimensionality_type D, typename ElementPtr, class Layout>
class basic_array : 
	public totally_ordered2<basic_array<T, D, ElementPtr, Layout>>
{
protected:
	Layout layout_;
public:
	ElementPtr base_;
	using layout_ordering_category = general_order_layout_tag;
public:
	using layout_type = Layout;

	using element_ptr  = ElementPtr;
	using element      = typename std::pointer_traits<element_ptr>::element_type;
	using element_cptr = typename std::pointer_traits<element_ptr>::template rebind<element const>; //multi::const_iterator<ElementPtr>; 

	using element_ref  = typename std::iterator_traits<element_ptr >::reference;
	using element_cref = typename std::iterator_traits<element_cptr>::reference;


	using       reference = std::conditional_t<(D>1), basic_array<element, D-1, element_ptr >, element_ref >;
	using const_reference = std::conditional_t<(D>1), basic_array<element, D-1, element_cptr>, element_cref>;
	using value_type      = std::conditional_t<(D>1), array<element, D-1, typename multi::pointer_traits<typename basic_array::element_ptr>::default_allocator_type>, element>;

	using ptr             = basic_array_ptr<T, D, element_ptr >;
	using cptr            = basic_array_ptr<T, D, element_cptr>;

	using ref             = basic_array<T, D, element_ptr , layout_type>;
	using cref            = basic_array<T, D, element_cptr, layout_type>;

	using decay_type      = array<T, D, typename multi::pointer_traits<typename basic_array::element_ptr>::default_allocator_type>;

	using extensions_type = typename layout_type::extensions_type;
	using index_extension = typename layout_type::index_extension;
	using difference_type = typename layout_type::difference_type;
	using size_type = typename layout_type::size_type;

	using rank = std::integral_constant<dimensionality_type, D>;
	constexpr static dimensionality_type rank_v = rank{};

	constexpr layout_type layout() const{return layout_;}
	constexpr dimensionality_type dimensionality() const{return layout_.dimensionality();}

private:
	constexpr element_ptr base_mutable() const&{return base_;}
public:
	constexpr element_cptr base() const&{return base_mutable();}
	constexpr element_ptr  base()      &{return base_mutable();}
	constexpr element_ptr  base()     &&{return base_mutable();}

	template<class S, basic_array* =nullptr>
	friend constexpr auto base(S&& s){return std::forward<S>(s).base();}

	constexpr auto extension()  const{return layout_.extension();}
	constexpr auto extensions() const{return layout_.extensions();}
	constexpr auto size()       const{return layout_.size();}
	constexpr auto sizes()      const{return layout_.sizes();}
	constexpr auto is_empty()   const{return layout_.is_empty();}
	constexpr auto stride()     const{return layout_.stride();}
	constexpr auto strides()     const{return layout_.strides();}
	constexpr auto num_elements() const{return layout_.num_elements();}

	friend constexpr auto size  (basic_array const& self){return self.size()  ;}
	friend constexpr auto stride(basic_array const& self){return self.stride();}
	friend constexpr auto num_elements(basic_array const& self){return self.num_elements();}
	friend constexpr auto strides(basic_array const& self){return self.strides();}
	using basic_const_array = basic_array<
		T, 
		D, 
		typename std::pointer_traits<ElementPtr>::template rebind<
			typename std::pointer_traits<ElementPtr>::element_type const
		>, 
		Layout
	>;
protected:
	basic_array() = default;
	basic_array(basic_array const&) = default;
	constexpr basic_array(layout_type const& l, element_ptr const& p) : layout_{l}, base_{p}{}
	template<typename, dimensionality_type, class Alloc> friend struct static_array;
	template<class, dimensionality_type, class> friend class basic_array_ptr;
public:
	basic_array(basic_array&&) = default; // in C++ < 17 this is necessary to return references from functions
	template<class BasicArray> constexpr basic_array(BasicArray&& other) : basic_array(other.layout(), other.base()){}
	template<class BasicArray> basic_array(BasicArray&) = delete;
	template<class BasicArray> basic_array(BasicArray const&) = delete;

	template<class It> constexpr auto assign(It other) &{return adl_copy_n(other, size(), begin());}
	template<class It> constexpr auto assign(It other)&&{return assign(other);}

	constexpr void assign(value_type const& v) &{adl_fill_n(begin(), size(), v);}
	constexpr void assign(value_type const& v)&&{assign(v);}

	template<class BasicArray>
	constexpr basic_array& operator=(BasicArray const& o)&{assert(this->extension() == o.extension());
		return this->assign(o.begin()), *this; // cannot be rotated in 1D
	}
	template<class BasicArray>
	constexpr basic_array&& operator=(BasicArray const& o)&&{return std::move(operator=(o));}

	constexpr basic_array&  operator=(basic_array const& o) &{return operator= <basic_array>(o);}
	constexpr basic_array&& operator=(basic_array const& o)&&{return std::move(operator=(o));}

public:
	template<class T2> friend constexpr auto reinterpret_array_cast(basic_array&& a){
		return std::move(a).template reinterpret_array_cast<T2, typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>();
	}
	template<class T2> friend constexpr auto reinterpret_array_cast(basic_array const& a){
		return a.template reinterpret_array_cast<T2, typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>();
	}
	friend constexpr dimensionality_type dimensionality(basic_array const&){return D;}

	using default_allocator_type = typename multi::pointer_traits<typename basic_array::element_ptr>::default_allocator_type;

	constexpr default_allocator_type get_allocator() const{
		using multi::get_allocator;
		return get_allocator(this->base());
	}

	friend constexpr default_allocator_type get_allocator(basic_array const& s){return s.get_allocator();}
	template<class P>
	static constexpr default_allocator_type get_allocator_(P const& p){
		return multi::default_allocator_of(p);
	}

	template<class Archive>
	auto serialize(Archive& ar, unsigned int /*file version*/){
		std::for_each(this->begin(), this->end(), [&](auto&& e){ar & multi::archive_traits<Archive>::make_nvp("item", e);});
	}
	constexpr decay_type decay() const{
		decay_type ret = std::move(modify(*this));
		return ret;
	}
	constexpr decay_type operator+() const{return decay();}

	constexpr const_reference operator[](index i) const&{//MULTI_ACCESS_ASSERT(this->extension().contains(i)&&"out of bounds");
		return const_reference(layout_.sub_, base_ + layout_.operator()(i));
	}
	constexpr typename basic_array::      reference operator[](index i) &&{//MULTI_ACCESS_ASSERT(this->extension().contains(i)&&"out of bounds");
		return typename basic_array::      reference(layout_.sub_, base() + layout_.operator()(i));
	}
	constexpr reference operator[](index i) &{//MULTI_ACCESS_ASSERT(this->extension().contains(i)&&"out of bounds");
		return {layout_.sub_, base_ + layout_.operator()(i)};
	}
	template<class Tp = std::array<index, static_cast<std::size_t>(D)>, typename = std::enable_if_t<(std::tuple_size<std::decay_t<Tp>>{}>1)> >
	constexpr auto operator[](Tp&& t) const
	->decltype(operator[](std::get<0>(t))[detail::tuple_tail(t)]){
		return operator[](std::get<0>(t))[detail::tuple_tail(t)];}
	template<class Tp, typename = std::enable_if_t<std::tuple_size<std::decay_t<Tp>>::value==1> >
	constexpr auto operator[](Tp&& t) const
	->decltype(operator[](std::get<0>(t))){
		return operator[](std::get<0>(t));}
	template<class Tp = std::tuple<>, typename = std::enable_if_t<std::tuple_size<std::decay_t<Tp>>::value==0> >
	constexpr decltype(auto) operator[](Tp&&) const{return *this;}
	
private:
	constexpr basic_array reindexed_(index first) const{
		layout_type new_ = layout();
		new_.reindex(first);
		return {new_, base_};
	}
public:
	constexpr auto reindexed(index first) const&{return reindexed_(first).protect();}
	constexpr auto reindexed(index first)     &&{return reindexed_(first);}
	constexpr auto reindexed(index first)      &{return reindexed_(first);}

private:
	template<class... Indexes>
	constexpr basic_array reindexed_(index first, Indexes... idxs) const{return ((reindexed(first).rotated()).reindexed(idxs...)).unrotated();}
public:
	template<class... Indexes>
	constexpr basic_const_array reindexed(index first, Indexes... idxs) const&{return ((reindexed(first).rotated()).reindexed(idxs...)).unrotated();}
	// TODO reindexed

	constexpr basic_const_array sliced(index first, index last) const&{
		layout_type new_ = layout();
		(new_.nelems_ /= size())*=(last - first);
		return {new_, base_ + layout().operator()(first)};
	}
	constexpr basic_array sliced(index first, index last)&{
		layout_type new_ = layout();
		(new_.nelems_ /= size()) *= (last - first);
		return {new_, base_ + layout().operator()(first)};
	}
	constexpr basic_array sliced(index first, index last) &&{return sliced(first, last);}

	constexpr auto blocked(index first, index last) const&{return sliced(first, last).reindexed(first);}
	constexpr auto blocked(index first, index last)     &&{return sliced(first, last).reindexed(first);}
	constexpr auto blocked(index first, index last)      &{return sliced(first, last).reindexed(first);}
	
	using iextension = typename basic_array::layout_type::index_extension;

	NODISCARD("no side effects")
	constexpr basic_array stenciled(iextension x)                                             &{return blocked(x.start(), x.finish());}
	constexpr basic_array stenciled(iextension x, iextension x1)                              &{return ((stenciled(x).rotated()).stenciled(x1)).unrotated();}
	constexpr basic_array stenciled(iextension x, iextension x1, iextension x2)               &{return ((stenciled(x)<<1).stenciled(x1, x2))>>1;}
	constexpr basic_array stenciled(iextension x, iextension x1, iextension x2, iextension x3)&{return ((stenciled(x)<<1).stenciled(x1, x2, x3))>>1;}
	template<class... Xs>
	constexpr basic_array stenciled(iextension x, iextension x1, iextension x2, iextension x3, Xs... xs)&{return ((stenciled(x)<<1).stenciled(x1, x2, x3, xs...))>>1;}

	NODISCARD("no side effects")
	constexpr basic_array stenciled(iextension x)                                             &&{return blocked(x.start(), x.finish());}
	constexpr basic_array stenciled(iextension x, iextension x1)                              &&{return ((stenciled(x)<<1).stenciled(x1))>>1;}
	constexpr basic_array stenciled(iextension x, iextension x1, iextension x2)               &&{return ((stenciled(x)<<1).stenciled(x1, x2))>>1;}
	constexpr basic_array stenciled(iextension x, iextension x1, iextension x2, iextension x3)&&{return ((stenciled(x)<<1).stenciled(x1, x2, x3))>>1;}
	template<class... Xs>
	constexpr basic_array stenciled(iextension x, iextension x1, iextension x2, iextension x3, Xs... xs)&&{return ((stenciled(x)<<1).stenciled(x1, x2, x3, xs...))>>1;}

	NODISCARD("no side effects")
	constexpr basic_const_array stenciled(iextension x)                                             const&{return blocked(x.start(), x.finish());}
	constexpr basic_const_array stenciled(iextension x, iextension x1)                              const&{return ((stenciled(x)<<1).stenciled(x1))>>1;}
	constexpr basic_const_array stenciled(iextension x, iextension x1, iextension x2)               const&{return ((stenciled(x)<<1).stenciled(x1, x2))>>1;}
	constexpr basic_const_array stenciled(iextension x, iextension x1, iextension x2, iextension x3)const&{return ((stenciled(x)<<1).stenciled(x1, x2, x3))>>1;}
	template<class... Xs>
	constexpr basic_const_array stenciled(iextension x, iextension x1, iextension x2, iextension x3, Xs... xs)const&{return ((stenciled(x)<<1).stenciled(x1, x2, x3, xs...))>>1;}

	constexpr decltype(auto) elements_at(size_type n) const&{assert(n < this->num_elements()); 
		auto const sub_num_elements = this->begin()->num_elements();
		return operator[](n / sub_num_elements).elements_at(n % sub_num_elements);
	}
	constexpr decltype(auto) elements_at(size_type n) &&{assert(n < this->num_elements()); 
		auto const sub_num_elements = this->begin()->num_elements();
		return operator[](n / sub_num_elements).elements_at(n % sub_num_elements);
	}
	constexpr decltype(auto) elements_at(size_type n) &{assert(n < this->num_elements()); 
		auto const sub_num_elements = this->begin()->num_elements();
		return operator[](n / sub_num_elements).elements_at(n % sub_num_elements);
	}

	using index_range = typename basic_array::layout_type::index_range;
	constexpr decltype(auto) range(index_range ir) &     {return sliced(ir.front(), ir.front() + ir.size());}
	constexpr decltype(auto) range(index_range ir) &&    {return range(ir);}
	constexpr decltype(auto) range(index_range ir) const&{return sliced(ir.front(), ir.front() + ir.size());}

	constexpr auto range(typename basic_array::index_range const& ir, dimensionality_type n) const{
		return rotated(n).range(ir).rotated(-n);
	}
	constexpr decltype(auto) flattened()&&{
		multi::biiterator<std::decay_t<decltype(std::move(*this).begin())>> biit{std::move(*this).begin(), 0, size(*(std::move(*this).begin()))};
		return basic_array<typename std::iterator_traits<decltype(biit)>::value_type, 1, decltype(biit)>{
			multi::layout_t<1>(1, 0, this->size()*size(*(std::move(*this).begin()))),
			biit
		};
	}
	friend constexpr decltype(auto) flattened(basic_array&& self){return std::move(self).flattened();}
	constexpr bool is_flattable() const{return this->stride() == this->layout().sub_.nelems_;}
	constexpr auto flatted() const{
		assert(is_flattable() && "flatted doesn't work for all layouts!");//this->nelems());
		multi::layout_t<D-1> new_layout{this->layout().sub_};
		new_layout.nelems_*=this->size();
		return basic_array<T, D-1, ElementPtr>{new_layout, basic_array::base_};
	}
	friend constexpr auto flatted(basic_array const& self){return self.flatted();}

	NODISCARD("because it has no side-effect")
	constexpr auto diagonal()&&{return this->diagonal();}
	NODISCARD("because it has no side-effect")
	constexpr basic_array<T, D-1, typename basic_array::element_ptr> diagonal()&{
		auto L = std::min(std::get<0>(this->sizes()), std::get<1>(this->sizes()));
		multi::layout_t<D-1> new_layout{(*this)({0, L}, {0, L}).layout().sub_};
		new_layout.nelems_ += (*this)({0, L}, {0, L}).layout().nelems_;
		new_layout.stride_ += (*this)({0, L}, {0, L}).layout().stride_;
		return {new_layout, basic_array::base_};
	}
	NODISCARD("because it has no side-effect")
	constexpr basic_array<T, D-1, typename basic_array::element_cptr> diagonal() const&{
		auto L = std::min(std::get<0>(this->sizes()), std::get<1>(this->sizes()));
		multi::layout_t<D-1> new_layout{(*this)({0, L}, {0, L}).layout().sub_};
		new_layout.nelems_ += (*this)({0, L}, {0, L}).layout().nelems_;
		new_layout.stride_ += (*this)({0, L}, {0, L}).layout().stride_;
		return {new_layout, basic_array::base_};
	}
	friend constexpr auto diagonal(basic_array const& self){return           self .diagonal();}
	friend constexpr auto diagonal(basic_array&       self){return           self .diagonal();}
	friend constexpr auto diagonal(basic_array&&      self){return std::move(self).diagonal();}

	template<typename Size>
	constexpr auto partitioned(Size const& s) const{
		assert(s!=0);
		assert(this->layout().nelems_%s==0);
		multi::layout_t<D+1> new_layout{this->layout(), this->layout().nelems_/s, 0, this->layout().nelems_};
		new_layout.sub_.nelems_/=s;
		return basic_array<T, D+1, ElementPtr>{new_layout, basic_array::base_};
	}
	constexpr basic_array transposed() const&{//	typename types::layout_t new_layout = *this;
		auto new_layout = this->layout();//
		new_layout.transpose();
		return {new_layout, basic_array::base_};
	}
	friend constexpr basic_array transposed(basic_array const& s){return s.transposed();}
	friend constexpr basic_array operator~ (basic_array const& s){return s.transposed();}

private:
	constexpr basic_array rotated_() const&{
		layout_type new_ = layout(); new_.rotate();
		return basic_array{new_, base_};
	}
public:
	constexpr basic_array       rotated()      &{return rotated_();}
	constexpr basic_array       rotated()     &&{return rotated_();}
	constexpr basic_const_array rotated() const&{return rotated_();}

	template<class S, basic_array* =nullptr>
	friend constexpr auto rotated(S&& s){return std::forward<S>(s).rotated();}

private:
	constexpr basic_array unrotated_aux() const&{
		layout_type new_ = layout();
		new_.unrotate();
		return basic_array<T, D, ElementPtr>{new_, base_};
	}
public:
	constexpr basic_array       unrotated()      &{return unrotated_aux();}
	constexpr basic_array       unrotated()     &&{return unrotated_aux();}
	constexpr basic_const_array unrotated() const&{return unrotated_aux();}

	template<class S, basic_array* =nullptr>
	friend constexpr auto unrotated(S&& s){return std::forward<S>(s).unrotated();}

private:
	constexpr basic_array rotated_aux(dimensionality_type i) const&{
		layout_type new_ = layout(); new_.rotate(i);
		return {new_, base_};
	}
public:
	constexpr basic_array       rotated(dimensionality_type i)      &{return rotated_aux(i);}
	constexpr basic_array       rotated(dimensionality_type i)     &&{return rotated_aux(i);}
	constexpr basic_const_array rotated(dimensionality_type i) const&{return rotated_aux(i);}

private:
	constexpr basic_array unrotated_(dimensionality_type i) const&{
		layout_type new_ = layout(); new_.unrotate(i);
		return {new_, base_};
	}
public:
	constexpr basic_array       unrotated(dimensionality_type i)      &{return unrotated_(i);}
	constexpr basic_array       unrotated(dimensionality_type i)     &&{return unrotated_(i);}
	constexpr basic_const_array unrotated(dimensionality_type i) const&{return unrotated_(i);}

	template<class S, basic_array* =nullptr> friend constexpr auto operator<<(S&& s, dimensionality_type i){return std::forward<S>(s).  rotated(i);}
	template<class S, basic_array* =nullptr> friend constexpr auto operator>>(S&& s, dimensionality_type i){return std::forward<S>(s).unrotated(i);}

	template<class S, basic_array* =nullptr> friend constexpr auto operator|(S&& s, typename layout_type::size_type n){return std::forward<S>(s).paratitioned(n);}

private:
	constexpr basic_array       paren_() const&{return basic_array{layout(), base_};}
public:
	constexpr basic_array       operator()() &     {return paren_();}
	constexpr basic_array       operator()() &&    {return paren_();}
	constexpr basic_const_array operator()() const&{return paren_();}

public:
	constexpr basic_const_array protect() &{return std::move(*this);}
	constexpr basic_const_array protect()&&{return protect();}

public:
	template<typename, dimensionality_type, typename, class> friend class basic_array;
	constexpr basic_array       paren() &     {return *this;}
	constexpr basic_array       paren() &&    {return this->operator()();}
	constexpr basic_const_array paren() const&{return {this->layout(), this->base()};}

	template<class... As> constexpr auto paren(index_range a, As... as) &     {return                  range(a).rotated().paren(as...).unrotated();}
	template<class... As> constexpr auto paren(index_range a, As... as) &&    {return this->range(a).rotated().paren(as...).unrotated();}
	template<class... As> constexpr auto paren(index_range a, As... as) const&{return                  range(a).rotated().paren(as...).unrotated();}

	template<class... As> constexpr decltype(auto) paren(intersecting_range<index> inr, As... as) &     {return                  paren(intersection(this->extension(), inr), as...);}
	template<class... As> constexpr decltype(auto) paren(intersecting_range<index> inr, As... as) &&    {return 				paren(intersection(this->extension(), inr), as...);}
	template<class... As> constexpr decltype(auto) paren(intersecting_range<index> inr, As... as) const&{return                  paren(intersection(this->extension(), inr), as...);}

	template<class... As> constexpr decltype(auto) paren(index i, As... as) &     {return                  operator[](i).paren(as...);}
	template<class... As> constexpr decltype(auto) paren(index i, As... as) &&    {return                  operator[](i).paren(as...);}
	template<class... As> constexpr decltype(auto) paren(index i, As... as) const&{return                  operator[](i).paren(as...);}
public:

	// the default template parameters below help interpret for {first, last} simple syntax as iranges
	// do not remove default parameter = irange
	template<class B1 = irange>                                                                       constexpr decltype(auto) operator()(B1 b1)                                const&{return paren(b1);}
	template<class B1 = irange, class B2 = irange>                                                    constexpr decltype(auto) operator()(B1 b1, B2 b2)                         const&{return paren(b1, b2);}
	template<class B1 = irange, class B2 = irange, class B3 = irange>                                 constexpr decltype(auto) operator()(B1 b1, B2 b2, B3 b3)                  const&{return paren(b1, b2, b3);}
	template<class B1 = irange, class B2 = irange, class B3 = irange, class B4 = irange, class... As> constexpr decltype(auto) operator()(B1 b1, B2 b2, B3 b3, B4 b4, As... as) const&{return paren(b1, b2, b3, b4, as...);}

	template<class B1 = irange>                                                                       constexpr decltype(auto) operator()(B1 b1)                                     &{return paren(b1);}
	template<class B1 = irange, class B2 = irange>                                                    constexpr decltype(auto) operator()(B1 b1, B2 b2)                              &{return paren(b1, b2);}
	template<class B1 = irange, class B2 = irange, class B3 = irange>                                 constexpr decltype(auto) operator()(B1 b1, B2 b2, B3 b3)                       &{return paren(b1, b2, b3);}
	template<class B1 = irange, class B2 = irange, class B3 = irange, class B4 = irange, class... As> constexpr decltype(auto) operator()(B1 b1, B2 b2, B3 b3, B4 b4, As... as)      &{return paren(b1, b2, b3, b4, as...);}

	template<class B1 = irange>                                                                       constexpr decltype(auto) operator()(B1 b1)                                    &&{return paren(b1);}
	template<class B1 = irange, class B2 = irange>                                                    constexpr decltype(auto) operator()(B1 b1, B2 b2)                             &&{return this->paren(b1, b2);}
	template<class B1 = irange, class B2 = irange, class B3 = irange>                                 constexpr decltype(auto) operator()(B1 b1, B2 b2, B3 b3)                      &&{return paren(b1, b2, b3);}
	template<class B1 = irange, class B2 = irange, class B3 = irange, class B4 = irange, class... As> constexpr decltype(auto) operator()(B1 b1, B2 b2, B3 b3, B4 b4, As... as)     &&{return paren(b1, b2, b3, b4, as...);}

private:
	template<typename Tuple, std::size_t ... I> constexpr decltype(auto) apply_impl(Tuple const& t, std::index_sequence<I...>) const&{return this->operator()(std::get<I>(t)...);}
	template<typename Tuple, std::size_t ... I> constexpr decltype(auto) apply_impl(Tuple const& t, std::index_sequence<I...>)      &{return this->operator()(std::get<I>(t)...);}
	template<typename Tuple, std::size_t ... I> constexpr decltype(auto) apply_impl(Tuple const& t, std::index_sequence<I...>)     &&{return std::move(*this).operator()(std::get<I>(t)...);}

public:
	template<typename Tuple> constexpr decltype(auto) apply(Tuple const& t) const&{return apply_impl(t, std::make_index_sequence<std::tuple_size<Tuple>::value>());}
	template<typename Tuple> constexpr decltype(auto) apply(Tuple const& t)     &&{return apply_impl(t, std::make_index_sequence<std::tuple_size<Tuple>::value>());}
	template<typename Tuple> constexpr decltype(auto) apply(Tuple const& t)      &{return apply_impl(t, std::make_index_sequence<std::tuple_size<Tuple>::value>());}

	constexpr ptr addressof() &&{return {layout(), base_};}
	constexpr ptr operator&() &&{return {layout(), base_};}

	using       iterator = array_iterator<typename basic_array::element, D, typename basic_array::element_ptr >;//, typename types::reference      >;
	using const_iterator = array_iterator<typename basic_array::element, D, typename basic_array::element_cptr>;//, typename types::const_reference>;

private:
	constexpr iterator begin_() const&{return {base_                   , layout_.sub_, layout_.stride_};}
public:
	constexpr const_iterator begin() const&{return begin_();}
	constexpr       iterator begin()     &&{return begin_();}
	constexpr       iterator begin()      &{return begin_();}

	template<class S, basic_array* =nullptr> friend constexpr auto begin(S&& s){return std::forward<S>(s).begin();}

private:
	constexpr iterator end_() const&{return {base_ + layout_.nelems_, layout_.sub_, layout_.stride_};}
public:
	constexpr const_iterator end() const&{return end_();}
	constexpr       iterator end()     &&{return end_();}
	constexpr       iterator end()      &{return end_();}

	template<class S, basic_array* =nullptr> friend constexpr auto end  (S&& s){return std::forward<S>(s).end()  ;}

protected:
	template<class A> constexpr void intersection_assign_(A&& other)&{// using multi::extension
		auto const is = intersection(basic_array::extension(), other.extension());
		using std::for_each;
		for_each(
			is.begin(), 
			is.end(), 
			[&](index i){operator[](i).intersection_assign_(std::forward<A>(other)[i]);}
		);
	}
	template<class A> constexpr void intersection_assign_(A&& o)&&{intersection_assign_(std::forward<A>(o));}

public:
	template<class Array> constexpr void swap(Array&& o) &&{assert( std::move(*this).extension() == std::forward<Array>(o).extension() );
		adl_swap_ranges(this->begin(), this->end(), adl_begin(std::forward<Array>(o)));
	}
	template<class A> constexpr void swap(A&& o) &{return swap(std::forward<A>(o));}

	friend constexpr void swap(basic_array&& a, basic_array&& b){std::move(a).swap(std::move(b));}
	template<class Array> constexpr void swap(basic_array const& s, Array&& a){s.swap(a);}
	template<class Array> constexpr void swap(Array&& a, basic_array const& s){s.swap(a);}

	template<class It> constexpr auto equal(It other) const{return adl_equal(begin(), end(), other);}

	template<class BasicArray>
	constexpr auto operator==(BasicArray const& other) const{
		return extension()==other.extension() and equal(other.begin());
	}

	template<class It>
	constexpr bool lexicographical_compare(It first, It last) const{
		return adl_lexicographical_compare(begin(), end(), first, last);
	}
	template<class BasicArray>
	constexpr bool operator<(BasicArray const& other) const{
		if(extension().first() > other.extension().first()) return true ;
		if(extension().first() < other.extension().first()) return false;
		return lexicographical_compare(other.begin(), other.end());
	}

public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>
	constexpr basic_array<T2, D, P2> static_array_cast() const{
		P2 p2{this->base_};
		return basic_array<T2, D, P2>{this->layout(), p2};
	}

	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>>
	using rebind = basic_array<std::decay_t<T2>, D, P2>;

private:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2 const>,
		class Element = typename basic_array::element,
		class PM = T2 std::decay_t<Element>::*
	>
	constexpr rebind<T2, P2> member_array_cast_(PM pm) const{
		static_assert(sizeof(T)%sizeof(T2) == 0, 
			"member_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements");
	//	return {this->layout().scale(sizeof(T)/sizeof(T2)), &(this->base_->*pm)};
		return basic_array<T2, D, P2>{this->layout().scale(sizeof(T)/sizeof(T2)), static_cast<P2>(&(this->base_->*pm))};
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, class Element = typename basic_array::element, class PM = T2 std::decay_t<Element>::*> constexpr auto member_array_cast(PM pm) const&{return member_array_cast_<T2, P2>(pm).protect();}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, class Element = typename basic_array::element, class PM = T2 std::decay_t<Element>::*> constexpr auto member_array_cast(PM pm)     &&{return member_array_cast_<T2, P2>(pm);}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2>, class Element = typename basic_array::element, class PM = T2 std::decay_t<Element>::*> constexpr auto member_array_cast(PM pm)      &{return member_array_cast_<T2, P2>(pm);}

private:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2 const> >
	constexpr rebind<T2, P2> reinterpret_array_cast_() const{
		static_assert( sizeof(T)%sizeof(T2)== 0, 
			"error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );
		auto const thisbase = base_;
		return {
			this->layout().scale(sizeof(T)/sizeof(T2)), 
			static_cast<P2>(static_cast<void*>(thisbase))
		};
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2 const> > constexpr auto reinterpret_array_cast() const&{return reinterpret_array_cast_<T2, P2>().protect();}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2      > > constexpr auto reinterpret_array_cast()     &&{return reinterpret_array_cast_<T2, P2>();}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2      > > constexpr auto reinterpret_array_cast()      &{return reinterpret_array_cast_<T2, P2>();}
	
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2      > > constexpr rebind<std::decay_t<T2>, P2> const_array_cast()&&{return {this->layout(), const_cast<P2>(this->base())};}
	template<class T2, class P2 = typename std::pointer_traits<element_ptr>::template rebind<T2      > > constexpr rebind<std::decay_t<T2>, P2> const_array_cast() &{return {this->layout(), const_cast<P2>(this->base())};}

private:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2> >
	constexpr basic_array<T2, D + 1, P2> reinterpret_array_cast_(size_type n) const{
		static_assert( sizeof(T)%sizeof(T2) == 0,
			"error: reinterpret_array_cast is limited to integral stride values");
		assert( sizeof(T) == sizeof(T2)*n );
		return {
			layout_t<D+1>{this->layout().scale(sizeof(T)/sizeof(T2)), 1, 0, n}.rotate(), 
			static_cast<P2>(static_cast<void*>(base_))
		};
	}
public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2 const>> constexpr auto reinterpret_array_cast(size_type n) const&{return reinterpret_array_cast_<T2, P2>(n).protect();}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2      >> constexpr auto reinterpret_array_cast(size_type n)     &&{return reinterpret_array_cast_<T2, P2>(n);}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2      >> constexpr auto reinterpret_array_cast(size_type n)      &{return reinterpret_array_cast_<T2, P2>(n);}
};

template<class Element, typename Ptr>
class array_iterator<Element, 1, Ptr> :
	public totally_ordered2<array_iterator<Element, 1, Ptr>>
{
	using element = Element;
	using element_ptr = Ptr;
	using stride_type = typename std::pointer_traits<element_ptr>::difference_type;

	element_ptr base_;// = nullptr;
	stride_type stride_;
	friend class basic_array<Element, 1, Ptr>;

	using rank = std::integral_constant<dimensionality_type, 1>;

public:
	using difference_type = typename std::pointer_traits<element_ptr>::difference_type;
	using value_type = typename std::iterator_traits<element_ptr>::value_type;
	using reference = typename std::iterator_traits<element_ptr>::reference;
	using pointer = element_ptr;
	using iterator_category = std::random_access_iterator_tag;
	
	array_iterator() = default;
	array_iterator(array_iterator const&) = default;
	template<class Other, decltype(implicit_cast<element_ptr>(std::declval<typename Other::element_ptr const&>()))* =nullptr>
	         constexpr array_iterator(Other const& o) : base_{o.base_}, stride_{o.stride_}{}
	template<class Other, decltype(explicit_cast<element_ptr>(std::declval<typename Other::element_ptr const&>()))* =nullptr> 
	explicit constexpr array_iterator(Other const& o) : base_(o.base_), stride_{o.stride_}{}

	template<class, dimensionality_type, class> friend class array_iterator;

	constexpr array_iterator(std::nullptr_t nu)  : base_{nu}, stride_{1}{}
	explicit constexpr array_iterator(element_ptr const& p) : base_{p}, stride_{1}{}
//	template<class EElement, typename PPtr, 
//		typename = decltype(implicit_cast<Ptr>(std::declval<array_iterator<EElement, 1, PPtr>>().data_))
//	>
//	constexpr array_iterator(array_iterator<EElement, 1, PPtr> other) : data_{other.data_}, stride_{other.stride_}{} 
	explicit constexpr operator bool() const{return static_cast<bool>(this->base_);}
	constexpr reference operator[](difference_type n) const{return *((*this) + n);}
	constexpr pointer   operator->() const{return base_;}

	constexpr bool operator<(array_iterator const& o) const{
		assert( stride_ == o.stride_ );
		assert( stride_ != 0 );
		return (0 < stride_)?(base_ < o.base_):(o.base_ < base_);
	}
	explicit constexpr array_iterator(Ptr d,                     stride_type s) : base_{d}, stride_{s}{}
	explicit constexpr array_iterator(Ptr d, multi::layout_t<0>, stride_type s) : base_{d}, stride_{s}{}

	// constexpr here creates problems with intel 19
	       constexpr element_ptr base()              const    {return   base_;}
	friend constexpr element_ptr base(array_iterator const& s){return s.base_;}

	       constexpr stride_type stride()              const    {return   stride_;}
	friend constexpr stride_type stride(array_iterator const& s){return s.stride_;}

	constexpr array_iterator& operator+=(difference_type d){base_+=stride_*d; return *this;}
	constexpr array_iterator& operator-=(difference_type d){return operator+=(-d);}

	constexpr array_iterator& operator++(){return operator+=(1);}
	constexpr array_iterator& operator--(){return operator-=(1);}

	constexpr array_iterator  operator+(difference_type d) const{return array_iterator{*this}+=d;}
	constexpr array_iterator  operator-(difference_type d) const{return array_iterator{*this}-=d;}

	template<class ArrayIterator> 
	constexpr bool operator==(ArrayIterator const& o) const{return base_ == o.base_ and stride_ == o.stride_;}
	constexpr reference operator*() const{return *base_;}

	constexpr difference_type operator-(array_iterator const& other) const{
		assert(stride_==other.stride_ and stride_ !=0 and (base_ - other.base_)%stride_ == 0);
		return (base_ - other.base_)/stride_;
	}
};

template<class Element, dimensionality_type D, typename... Ts>
using iterator = array_iterator<Element, D, Ts...>;

template<typename T, dimensionality_type D, typename ElementPtr = T*>
struct array_ref : 
	basic_array<T, D, ElementPtr>,
	totally_ordered2<array_ref<T, D, ElementPtr>>
{
	using layout_ordering_category = c_order_layout_tag;
protected:
	using basic_ = basic_array<T, D, ElementPtr>;
	constexpr array_ref() noexcept
		: basic_array<T, D, ElementPtr>{typename array_ref::layout_type{}, nullptr}{}
protected:
	[[deprecated("references are not copyable, use auto&&")]]
	array_ref(array_ref const&) = default; // don't try to use `auto` for references, use `auto&&` or explicit value type
public:
	array_ref(array_ref&&) = default; // this needs to be public in c++14
	using basic_array<T, D, ElementPtr>::operator=;
public:
	auto  data_elements() const&{return this-> base();}
	auto  data_elements()     &&{return this-> base();}
	auto  data_elements()      &{return this-> base();}
	auto cdata_elements() const&{return this->cbase();}

	template<class S> auto data_elements(S&& s){return std::forward<S>(s).data_elements();}

	using layout_type = typename basic_array<T, D, ElementPtr>::layout_type;
	template<class OtherPtr, class=std::enable_if_t<not std::is_same<OtherPtr, ElementPtr>{}>>
	constexpr array_ref(array_ref<T, D, OtherPtr>&& other)
		: basic_array<T, D, ElementPtr>{other.layout(), ElementPtr{other.base()}}{}
	constexpr array_ref(typename array_ref::element_ptr p, typename array_ref::extensions_type e = {}) noexcept
		: basic_array<T, D, ElementPtr>{typename array_ref::layout_type{e}, p}{}

	constexpr array_ref(typename array_ref::extensions_type e, typename array_ref::element_ptr p) noexcept
		: basic_array<T, D, ElementPtr>{typename array_ref::layout_type{e}, p}{}

	template<class TT, std::size_t N> // doesn't work with gcc, (needs *array_ptr)
	constexpr array_ref(TT(&t)[N]) : array_ref((typename array_ref::element_ptr)&t, extensions(t)){}

	template<class ArrayRef>
	array_ref(ArrayRef&& o) : array_ref{std::forward<ArrayRef>(o).data_elements(), std::forward<ArrayRef>(o).extensions()}{}

private:
	template<class It> constexpr auto copy_elements(It first){
		return adl_copy_n(first, array_ref::num_elements(), array_ref::data_elements());
	}
	template<class It> constexpr auto equal_elements(It first) const{
		return adl_equal(first, first + this->num_elements(), this->data_elements());
	}
	template<class TT, std::size_t N> using const_carr = TT const[N];
	template<class TT, std::size_t N> using carr       = TT      [N];
public:
	template<class TT, std::size_t N, std::enable_if_t<std::is_same<typename array_ref::element, std::decay_t<std::remove_all_extents_t<const_carr<TT, N>>>>{}, int> =0>
	constexpr operator const_carr<TT, N>&() const&{assert(extensions(*(const_carr<TT, N>*)this)==this->extensions());
		return *reinterpret_cast<const_carr<TT, N>*>(this->base_);
	}
	template<class TT, std::size_t N, std::enable_if_t<std::is_same<typename array_ref::element, std::decay_t<std::remove_all_extents_t<carr<TT, N>>> >{}, int> =0>
	constexpr operator carr<TT, N>&()&{assert(extensions(*(carr<TT, N>*)this)==this->extensions());
		return *reinterpret_cast<carr<TT, N>*>(this->base_);
	}

	using  elements_type = array_ref<typename array_ref::element, 1, typename array_ref::element_ptr >;
	using celements_type = array_ref<typename array_ref::element, 1, typename array_ref::element_cptr>;

private:
	constexpr  elements_type elements_() const{return {this->base_, this->num_elements()};}
public:
	constexpr  elements_type elements() &     {return elements_();}
	constexpr  elements_type elements() &&    {return elements_();}
	constexpr celements_type elements() const&{return elements_();}

	friend constexpr  elements_type elements(array_ref &      self){return           self . elements();}	
	friend constexpr  elements_type elements(array_ref &&     self){return std::move(self). elements();}
	friend constexpr celements_type elements(array_ref const& self){return           self . elements();}

	       constexpr celements_type celements()         const&   {return {array_ref::data(), array_ref::num_elements()};}
	friend constexpr celements_type celements(array_ref const& s){return s.celements();}

private:
	template<class Array> constexpr auto equals(Array const& other, c_order_layout_tag) const{
		return this->extensions() == other.extensions() and equal_elements(other.data_elements());
	}
	template<class Array> constexpr auto equals(Array const& other, general_order_layout_tag) const{
		return basic_array<T, D, ElementPtr>::operator==(other);
	}
public:
	template<class Array>
	constexpr auto operator==(Array const& other) const
	->decltype(equals(other, typename Array::layout_ordering_category{})){
		return equals(other, typename Array::layout_ordering_category{});}

	constexpr typename array_ref::decay_type const& decay() const&{
		return static_cast<typename array_ref::decay_type const&>(*this);
	}
	friend constexpr typename array_ref::decay_type const& decay(array_ref const& s){return s.decay();}

	template<class Archive>
	auto serialize(Archive& ar, const unsigned int v){
	//	using boost::serialization::make_nvp;
//		if(this->num_elements() < (2<<8) ) 
			basic_array<T, D, ElementPtr>::serialize(ar, v);
//		else{
		//	using boost::serialization::make_binary_object;
		//	using boost::serialization::make_array;
//			if(std::is_trivially_copy_assignable<typename array_ref::element>{})
//				ar & multi::archive_traits<Archive>::make_nvp("binary_data", multi::archive_traits<Archive>::make_binary_object(this->data(), sizeof(typename array_ref::element)*this->num_elements())); //#include<boost/serialization/binary_object.hpp>
//			else ar & multi::archive_traits<Archive>::make_nvp("data", multi::archive_traits<Archive>::make_array(this->data(), this->num_elements()));
//		}
	}
};

template<class T, dimensionality_type D, class Ptr = T*> 
using array_cref = array_ref<
	std::decay_t<T>, D,
	typename std::pointer_traits<Ptr>::template rebind<T const>
>;

template<class T, dimensionality_type D, class Ptr = T*>
using array_mref = array_ref<
	std::decay_t<T>, D,
	std::move_iterator<Ptr>
>;


template<class T, dimensionality_type D, typename Ptr = T*>
struct array_ptr : basic_array_ptr<T, D, Ptr>{//basic_array_ptr<basic_array<T, D, Ptr>, typename array_ref<T, D, Ptr>::layout_t>{
	using basic_ptr = basic_array_ptr<T, D, Ptr>; //basic_array_ptr<basic_array<T, D, Ptr>, typename array_ref<T, D, Ptr>::layout_t>;
	constexpr array_ptr(multi::extensions_t<D> x, Ptr p) : basic_ptr{multi::layout_t<D>{x}, p}{}
public:
	array_ptr(array_ptr const&) = default;
	constexpr array_ptr(std::nullptr_t n = nullptr) : array_ptr({}, n){}
	template<class TT, std::size_t N> 
	constexpr array_ptr(TT(*t)[N]) : array_ptr(multi::extensions(*t), multi::data_elements(*t)){}
	constexpr array_ref<T, D, Ptr> operator*() const{
		return {
			static_cast<basic_array_ptr<T, D, Ptr> const&>(*this)->base(), 
			static_cast<basic_array_ptr<T, D, Ptr> const&>(*this)->layout().extensions()
		};
	}
};

template<class TT, std::size_t N>
// auto operator&(TT(&t)[N]){ // c++ cannot overload & for primitive types
constexpr auto addressof(TT(&t)[N]){
	return array_ptr<
		std::decay_t<std::remove_all_extents_t<TT[N]>>, std::rank<TT[N]>{}, std::remove_all_extents_t<TT[N]>*
	>(&t);
}

template<class T, dimensionality_type D, typename Ptr = T*>
using array_cptr = array_ptr<T, D, 	typename std::pointer_traits<Ptr>::template rebind<T const>>;

template<dimensionality_type D, class P>
constexpr
array_ref<typename std::iterator_traits<P>::value_type, D, P> 
make_array_ref(P p, index_extensions<D> x){return {p, x};}

template<class P> auto make_array_ref(P p, index_extensions<1> x){return make_array_ref<1>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<2> x){return make_array_ref<2>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<3> x){return make_array_ref<3>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<4> x){return make_array_ref<4>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<5> x){return make_array_ref<5>(p, x);}

template<class T, std::size_t N, typename V = typename std::remove_all_extents<T[N]>::type, std::size_t D = std::rank<T[N]>{}>
multi::array_ptr<V, D> ptr(T(*t)[N]){return multi::array_ptr<V, D>{t};}

template<class Vector>
multi::array_ptr<typename Vector::value_type, 1, decltype(std::declval<Vector>().data())>
ptr(Vector* v){
	return {v->size(), v->data()};
}

template<class Vector>
auto
ref(Vector&& v)
->decltype(*multi::ptr(&v)){
	return *multi::ptr(&v);}

#if defined(__cpp_deduction_guides)

template<class It, typename V = typename std::iterator_traits<It>::value_type> // pointer_traits doesn't have ::value_type
array_ptr(It, index_extensions<0> = {})->array_ptr<V, 0, It>;

template<class It, typename V = typename std::iterator_traits<It>::value_type>
array_ptr(It, index_extensions<1>)->array_ptr<V, 1, It>;
template<class It, typename V = typename std::iterator_traits<It>::value_type>
array_ptr(It, index_extensions<2>)->array_ptr<V, 2, It>;
template<class It, typename V = typename std::iterator_traits<It>::value_type>
array_ptr(It, index_extensions<3>)->array_ptr<V, 3, It>;

template<class T, std::size_t N, typename V = typename std::remove_all_extents<T[N]>::type, std::size_t D = std::rank<T[N]>{}>
array_ptr(T(*)[N])->array_ptr<V, D>;

//#if not defined(__clang__)
//template<class It, dimensionality_type D, typename V = typename std::iterator_traits<It>::value_type>
//array_ref(It, index_extensions<D>)->array_ref<V, D, It>;
//#else
template<class It> array_ref(It, index_extensions<1>)->array_ref<typename std::iterator_traits<It>::value_type, 1, It>;
template<class It> array_ref(It, index_extensions<2>)->array_ref<typename std::iterator_traits<It>::value_type, 2, It>;
template<class It> array_ref(It, index_extensions<3>)->array_ref<typename std::iterator_traits<It>::value_type, 3, It>;
template<class It> array_ref(It, index_extensions<4>)->array_ref<typename std::iterator_traits<It>::value_type, 4, It>;
template<class It> array_ref(It, index_extensions<5>)->array_ref<typename std::iterator_traits<It>::value_type, 5, It>;
//#endif

template<class It, class Tuple> array_ref(It, Tuple)->array_ref<typename std::iterator_traits<It>::value_type, std::tuple_size<Tuple>::value, It>;
#endif

#if 1
template<class T, std::size_t N>
constexpr auto rotated(const T(&t)[N]) noexcept{ // TODO move to utility
	return multi::array_ref<std::remove_all_extents<T[N]>, std::rank<T[N]>{}, decltype(base(t))>(
		base(t), extensions(t)
	).rotated();
}
template<class T, std::size_t N>
constexpr auto rotated(T(&t)[N]) noexcept{
	return multi::array_ref<std::remove_all_extents<T[N]>, std::rank<T[N]>{}, decltype(base(t))>(
		base(t), extensions(t)
	).rotated();
}
#endif

template<class TD, class Ptr> struct Array_aux;
template<class T, std::size_t D, class Ptr> struct Array_aux<T   [D], Ptr>{using type = array    <T, D, Ptr>  ;};
template<class T, std::size_t D, class Ptr> struct Array_aux<T(&)[D], Ptr>{using type = array_ref<T, D, Ptr>&&;};
template<class T, std::size_t D, class Ptr> struct Array_aux<T(*)[D], Ptr>{using type = array_ptr<T, D, Ptr>  ;};

template<class TD, class Second = 
	std::conditional_t<
		std::is_reference<TD>{} or std::is_pointer<TD>{}, 
		std::add_pointer_t<std::remove_all_extents_t<std::remove_reference_t<std::remove_pointer_t<TD>>>>,
		std::allocator<std::remove_all_extents_t<TD>>
	>
> using Array = typename Array_aux<TD, Second>::type;

template<class RandomAccessIterator, dimensionality_type D>
constexpr
multi::array_ptr<typename std::iterator_traits<RandomAccessIterator>::value_type, D, RandomAccessIterator>
operator/(RandomAccessIterator data, multi::iextensions<D> x){return {data, x};}

template<class T, dimensionality_type D, class... Ts>
constexpr std::true_type  is_basic_array_aux(basic_array<T, D, Ts...> const&);
constexpr std::false_type is_basic_array_aux(...);

template<class A> struct is_basic_array: decltype(is_basic_array_aux(std::declval<A>())){};

template<class In, class T, dimensionality_type N, class TP, class=std::enable_if_t<(N>1)>, class=decltype(adl_begin(*In{}), adl_end(*In{}))>
constexpr auto uninitialized_copy
// require N>1 (this is important because it forces calling placement new on the pointer
(In first, In last, multi::array_iterator<T, N, TP> dest){
	using std::begin; using std::end;
	while(first!=last){
		adl_uninitialized_copy(adl_begin(*first), adl_end(*first), adl_begin(*dest));
		++first;
		++dest;
	}
	return dest;
}

}}
#endif

