#ifdef COMPILATION_INSTRUCTIONS//-*-indent-tabs-mode: t; c-basic-offset: 4; tab-width: 4;-*-
for a in ./tests/*.cpp; do echo $a; sh $a || break; echo "\n"; done; exit;*/
$CXX -D_TEST_BOOST_MULTI_ARRAY_REF -xc++ $0 -o $0x&&$0x&&rm $0x;exit
#endif
// © Alfredo Correa 2018-2020

#ifndef BOOST_MULTI_ARRAY_REF_HPP
#define BOOST_MULTI_ARRAY_REF_HPP

#include "utility.hpp"

#include "./detail/layout.hpp"
#include "./detail/types.hpp"     // dimensionality_type
#include "./detail/operators.hpp" // random_iterable
#include "./detail/memory.hpp"    // pointer_traits

#include "./config/NODISCARD.hpp"

//#include<iostream> // debug
#include<boost/pointer_cast.hpp>

#include<algorithm> // copy_n

#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD
#endif

namespace boost{
namespace serialization{
//	template<class Archive> struct archive_traits;
//	template<class> struct nvp;
//	template<class T> const nvp<T> make_nvp(char const* name, T& t);
//	template<class T> class array_wrapper;
//	template<class T, class S> const array_wrapper<T> make_array(T* t, S s);
//	template<class T> 
	class binary_object;
	inline const binary_object make_binary_object(const void * t, std::size_t size);
}}

namespace boost{
namespace multi{

template<typename T, dimensionality_type D, typename ElementPtr = T*, class Layout = layout_t<D>> 
struct basic_array;

template<typename T, dimensionality_type D, class A = std::allocator<T>> struct array;

template<typename T, dimensionality_type D, typename ElementPtr = T*, class Layout = layout_t<D>>
struct array_types : Layout{
	using element = T;
	using element_type = element; // this follows more closely https://en.cppreference.com/w/cpp/memory/pointer_traits
	constexpr static dimensionality_type dimensionality = D;
	using element_ptr = ElementPtr;
	using layout_t = Layout;
	using value_type = typename std::conditional<
		(dimensionality>1),
		array<element, dimensionality-1, typename pointer_traits<element_ptr>::default_allocator_type>, 
		typename std::conditional<
			dimensionality == 1,
			element,
			element
		>::type
	>::type;
	using decay_type = array<element, dimensionality, typename pointer_traits<element_ptr>::default_allocator_type>;
	using reference = typename std::conditional<
		(dimensionality > 1), 
		basic_array<element, dimensionality-1, element_ptr>,
			typename std::conditional<
				dimensionality == 1, 
//#ifdef __CUDACC__
//		decltype(*std::declval<ElementPtr>())                  // this works with cuda fancy reference
//#else
			typename std::iterator_traits<element_ptr>::reference   // this seems more correct but it doesn't work with cuda fancy reference
//#endif
			, typename std::iterator_traits<element_ptr>::reference
		>::type
	// typename std::iterator_traits<element_ptr>::reference 	//	typename pointer_traits<element_ptr>::element_type&
	>::type;
	element_ptr base() const HD{return base_;} //	element_const_ptr cbase() const{return base();}
	friend element_ptr base(array_types const& s){return s.base();}
	layout_t const& layout() const HD{return *this;}
	friend layout_t const& layout(array_types const& s){return s.layout();}
	element_ptr            origin() const{return base_+Layout::origin();} //	element_const_ptr     corigin() const{return origin();}
	friend decltype(auto)  origin(array_types const& s){return s.origin();} //	friend decltype(auto) corigin(array_types const& s){return s.corigin();}
protected:
	using derived = basic_array<T, D, ElementPtr, Layout>;
	element_ptr base_;
	array_types() = delete;
	array_types(std::nullptr_t np) : Layout{}, base_{np}{}
	array_types(array_types const&) = default;
public:
	array_types(layout_t l, element_ptr data) HD : 
		Layout{l}, 
		base_{data}
	{}
//	template<class T2, class P2, class Array> friend decltype(auto) static_array_cast(Array&&);
public://TODO find why this needs to be public and not protected or friend
	template<class ArrayTypes, typename = std::enable_if_t<not std::is_base_of<array_types, std::decay_t<ArrayTypes>>{}>
		, typename = decltype(element_ptr{std::declval<ArrayTypes const&>().base_})
	>
	array_types(ArrayTypes const& a) : Layout{a}, base_{a.base_}{}
	template<typename ElementPtr2, 
		typename = decltype(Layout{std::declval<array_types<T, D, ElementPtr2, Layout> const&>().layout()}),
		typename = decltype(element_ptr{std::declval<array_types<T, D, ElementPtr2, Layout> const&>().base_})
	>
	array_types(array_types<T, D, ElementPtr2, Layout> const& other) : Layout{other.layout()}, base_{other.base_}{}
	template<class T2, dimensionality_type D2, class E2, class L2> friend struct array_types;
};

template<class Ref, class Layout>
struct basic_array_ptr : 
	private Ref,
	boost::multi::iterator_facade<
		basic_array_ptr<Ref, Layout>, void, std::random_access_iterator_tag, 
		Ref const&, typename Layout::difference_type
	>,
	boost::multi::totally_ordered2<basic_array_ptr<Ref, Layout>, void>
{
	using pointer = Ref const*;
	using element_type = typename Ref::decay_type;
	using difference_type = typename Layout::difference_type;

	using value_type = element_type;
	using reference = Ref const&;
	using iterator_category = std::random_access_iterator_tag;

	basic_array_ptr(std::nullptr_t p = nullptr) : Ref{p}{}
	template<class, class> friend struct basic_array_ptr;
	basic_array_ptr(typename Ref::element_ptr p, layout_t<Ref::dimensionality-1> l) HD : Ref{l, p}{}
	basic_array_ptr(typename Ref::element_ptr p, index_extensions<Ref::dimensionality> e) : Ref{p, e}{}

	template<class Array, typename = decltype(typename Ref::element_ptr{typename Array::element_ptr{}})> 
	basic_array_ptr(Array const& o) : Ref{o->layout(), o->base()}{}//, stride_{o.stride_}{}
	basic_array_ptr(basic_array_ptr const& o) HD : Ref{static_cast<Layout const&>(o), o.base_}{}//, stride_{o.stride_}{}
	basic_array_ptr& operator=(basic_array_ptr const& other){
		this->base_ = other.base_;
		static_cast<Layout&>(*this) = other;
		return *this;
	}
	explicit operator bool() const{return this->base_;}
	Ref operator*() const HD{return *this;}
	Ref const* operator->() const{return this;}
	Ref        operator[](difference_type n) const HD{return *(*this + n);}
	template<class O> bool operator==(O const& o) const{return equal(o);}
	bool operator<(basic_array_ptr const& o) const{return distance_to(o) > 0;}
	basic_array_ptr(typename Ref::element_ptr p, Layout l) HD : Ref{l, p}{}
	template<typename T, dimensionality_type D, typename ElementPtr, class LLayout>
	friend struct basic_array;
	auto base() const{return this->base_;}
	friend auto base(basic_array_ptr const& self){return self.base();}
	using Ref::base_;
	using Ref::layout;
protected:
	bool equal(basic_array_ptr const& o) const{return base_==o.base_ and layout()==o.layout();}
	void increment(){base_ += Ref::nelems();}
	void decrement(){base_ -= Ref::nelems();}
	void advance(difference_type n) HD{base_ += Ref::nelems()*n;}
	difference_type distance_to(basic_array_ptr const& other) const{
		assert( Ref::nelems() == other.Ref::nelems() and Ref::nelems() != 0 );
		assert( (other.base_ - base_)%Ref::nelems() == 0); 
		assert( layout() == other.layout() );
		return (other.base_ - base_)/Ref::nelems();
	}
public:
	basic_array_ptr& operator+=(difference_type n) HD{advance(n); return *this;}
};

template<class Element, dimensionality_type D, typename Ptr, class Ref 
#if 1
= 
	typename std::conditional<
			D != 1,
			basic_array<Element, D-1, 
				typename std::conditional<
					std::is_same<typename std::pointer_traits<Ptr>::element_type, void>{}, 
					typename std::pointer_traits<Ptr>::template rebind<Element>,
					Ptr
				>::type
			>,
			typename std::iterator_traits<Ptr>::reference
	//		typename std::iterator_traits<
	//				typename std::conditional<
	//					std::is_same<typename std::pointer_traits<Ptr>::element_type, void>{}, 
	//					typename std::pointer_traits<Ptr>::template rebind<Element>,
	//					Ptr
	//				>::type
	//		>::reference
		>::type
#endif
>
struct array_iterator;

template<class Element, dimensionality_type D, typename Ptr, class Ref>
struct array_iterator : 
	boost::multi::iterator_facade<
		array_iterator<Element, D, Ptr, Ref>, void, std::random_access_iterator_tag, 
		Ref const&, typename layout_t<D-1>::difference_type
	>,
	multi::decrementable<array_iterator<Element, D, Ptr, Ref>>,
	multi::incrementable<array_iterator<Element, D, Ptr, Ref>>,
	multi::affine<array_iterator<Element, D, Ptr, Ref>, multi::difference_type>,
	multi::totally_ordered2<array_iterator<Element, D, Ptr, Ref>, void>
{
	using difference_type = typename layout_t<D>::difference_type;
	using value_type = typename Ref::decay_type;
	using pointer = Ref*;
	using reference = Ref&&;//Ref const&;
//	using element_type = typename Ref::value_type;
	using iterator_category = std::random_access_iterator_tag;

	using rank = std::integral_constant<dimensionality_type, D>;

	using element = typename Ref::element;
	using element_ptr = typename Ref::element_ptr;
	array_iterator(std::nullptr_t p = nullptr) : ptr_{p}, stride_{1}{}//Ref{p}{}
	template<class, dimensionality_type, class, class> friend struct array_iterator;
	template<class Other, typename = decltype(typename Ref::types::element_ptr{typename Other::element_ptr{}})> 
	array_iterator(Other const& o) : /*Ref{o.layout(), o.base()},*/ ptr_{o.ptr_.base_, o.ptr_.layout()}, stride_{o.stride_}{}
	array_iterator(array_iterator const&) = default;
	array_iterator& operator=(array_iterator const& other){
		ptr_ = other.ptr_;
		stride_ = other.stride_;
		return *this;
	}
	explicit operator bool() const{return static_cast<bool>(ptr_.base_);}
	Ref operator*() const HD{/*assert(*this);*/ return *ptr_;}//return *this;}
	decltype(auto) operator->() const{/*assert(*this);*/ return ptr_;}//return this;}
	Ref operator[](difference_type n) const HD{return *(*this + n);}
	template<class O> bool operator==(O const& o) const{return equal(o);}
	bool operator<(array_iterator const& o) const{return distance_to(o) > 0;}
	array_iterator(typename Ref::element_ptr p, layout_t<D-1> l, index stride) HD : /*Ref{l, p},*/
		ptr_{p, l}, 
		stride_{stride}
	{}
	template<typename T, dimensionality_type DD, typename ElementPtr, class LLayout>
	friend struct basic_array;
	auto base() const{return ptr_.base_;}//this->base_;}
	friend auto base(array_iterator const& self){return self.base();}
	auto stride() const{return stride_;}
	friend index stride(array_iterator const& s){return s.stride();}
private:
	basic_array_ptr<Ref, layout_t<D-1>> ptr_;
	index stride_ = {1}; // nice non-zero default
	bool equal(array_iterator const& o) const{return ptr_==o.ptr_ and stride_==o.stride_;}//base_==o.base_ && stride_==o.stride_ && ptr_.layout()==o.ptr_.layout();}
	void increment(){ptr_.base_ += stride_;}
	void decrement(){ptr_.base_ -= stride_;}
	void advance(difference_type n) HD{ptr_.base_ += stride_*n;}
	difference_type distance_to(array_iterator const& other) const{
		assert( stride_ == other.stride_);
		assert( stride_ != 0 );
	//	assert( this->stride()==stride(other) and this->stride() );// and (base(other.ptr_) - base(this->ptr_))%stride_ == 0
	//	assert( stride_ == other.stride_ and stride_ != 0 and (other.ptr_.base_-ptr_.base_)%stride_ == 0 and ptr_.layout() == other.ptr_.layout() );
	//	assert( stride_ == other.stride_ and stride_ != 0 and (other.base_ - base_)%stride_ == 0 and layout() == other.layout() );
		return (other.ptr_.base_ - ptr_.base_)/stride_;
	}
//	friend class boost::iterator_core_access;
public:
	array_iterator& operator++(){increment(); return *this;}
	array_iterator& operator--(){decrement(); return *this;}
	bool operator==(array_iterator const& o) const{return equal(o);}
	friend difference_type operator-(array_iterator const& self, array_iterator const& other){
		assert(self.stride_ == other.stride_); assert(self.stride_ != 0);
		return (self.ptr_.base_ - other.ptr_.base_)/self.stride_;
	}
	array_iterator& operator+=(difference_type d) HD{advance( d); return *this;}
	array_iterator& operator-=(difference_type d) HD{advance(-d); return *this;}
};

template<class It>
struct biiterator : 
	boost::multi::iterator_facade<
		biiterator<It>,
		typename std::iterator_traits<It>::value_type, std::random_access_iterator_tag, 
		decltype(*(std::declval<It>()->begin())), multi::difference_type
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
	biiterator(It me, std::ptrdiff_t pos, std::ptrdiff_t stride) HD : me_{me}, pos_{pos}, stride_{stride}{}
	decltype(auto) operator++(){
		++pos_;
		if(pos_==stride_){
			++me_;
			pos_ = 0;
		}
		return *this;
	}
	bool operator==(biiterator const& o) const{return me_==o.me_ and pos_==o.pos_;}
	biiterator& operator+=(multi::difference_type n) HD{me_ += n/stride_; pos_ += n%stride_; return *this;}
	decltype(auto) operator*() const HD{
		auto meb = me_->begin();
		return meb[pos_];
	}
	using difference_type = std::ptrdiff_t;
	using reference = decltype(*std::declval<biiterator>());
	using value_type = std::decay_t<reference>;
	using pointer = value_type*;
	using iterator_category = std::random_access_iterator_tag;
};

template<typename T, dimensionality_type D, typename ElementPtr, class Layout /*= layout_t<D>*/ >
struct basic_array : 
	multi::partially_ordered2<basic_array<T, D, ElementPtr, Layout>, void>,
	multi::random_iterable<basic_array<T, D, ElementPtr, Layout>>,
	array_types<T, D, ElementPtr, Layout>
{
	using types = array_types<T, D, ElementPtr, Layout>;
	friend struct basic_array<typename types::element, typename Layout::rank{} + 1, typename types::element_ptr >;
	friend struct basic_array<typename types::element, typename Layout::rank{} + 1, typename types::element_ptr&>;
	using types::layout;
	HD decltype(auto) layout() const{return array_types<T, D, ElementPtr, Layout>::layout();}
protected:
	using types::types;
	template<typename, dimensionality_type, class Alloc> friend struct static_array;
	basic_array(basic_array const&) = default;
	template<class, class> friend struct basic_array_ptr;
public:
//	template<class Archive>
//	void serialize(Archive& ar, unsigned int file_version){
//		for(auto&& e : *this) ar & BOOST_SERIALIZATION_NVP(e);
//	}
	friend constexpr auto dimensionality(basic_array const& self){return self.dimensionality;}
	using typename types::reference;
	basic_array(basic_array&&) = default;
	auto get_allocator() const{
		using multi::get_allocator;
		return get_allocator(this->base());
	}
	friend auto get_allocator(basic_array const& self){return self.get_allocator();}
//	using decay_type = array<typename types::element, D, decltype(default_allocator_of(std::declval<ElementPtr>()))>;
	template<class P>
	static decltype(auto) get_allocator_(P const& p){
		using multi::default_allocator_of;
		return default_allocator_of(p);
	}
	using decay_type = array<typename types::element_type, D, decltype(get_allocator_(std::declval<ElementPtr>()))>;
//	decay_type 
//	auto
	decay_type decay() const{
		decay_type ret = *this;
		return ret;
	}
//	static decay_type remake(std::initializer_list<typename basic_array::value_type> il){return decay_type(il);}
//	template<class... As> static auto remake(As&&... as) -> decay_type{return decay_type(std::forward<As>(as)...);}
	template<class Archive>
	auto serialize(Archive& ar, const unsigned int){
		using boost::serialization::make_nvp;
		std::for_each(this->begin(), this->end(), [&](auto&& e){ar & make_nvp("item", e);});
	}
	friend /*NODISCARD("decayed type is ignored")*/ auto operator+(basic_array const& self){return self.decay();}
	friend /*NODISCARD("decayed type is ignored")*/ auto operator*(basic_array const& self){return self.decay();}
	friend auto decay(basic_array const& self){return self.decay();}
	typename types::reference operator[](index i) const HD{
		assert( this->extension().contains(i) );
		typename types::element_ptr new_base = typename types::element_ptr(this->base()) + std::ptrdiff_t{Layout::operator()(i)};
		return typename types::reference(this->layout().sub_, new_base);
	}
	template<class Tp = std::array<index, static_cast<std::size_t>(D)>, typename = std::enable_if_t<(std::tuple_size<std::decay_t<Tp>>{}>1)> >
	auto operator[](Tp&& t) const
	->decltype(operator[](std::get<0>(t))[detail::tuple_tail(t)]){
		return operator[](std::get<0>(t))[detail::tuple_tail(t)];}
	template<class Tp, typename = std::enable_if_t<std::tuple_size<std::decay_t<Tp>>::value==1> >
	auto operator[](Tp&& t) const
	->decltype(operator[](std::get<0>(t))){
		return operator[](std::get<0>(t));}
	template<class Tp = std::tuple<>, typename = std::enable_if_t<std::tuple_size<std::decay_t<Tp>>::value==0> >
	decltype(auto) operator[](Tp&&) const{return *this;}
	basic_array sliced(typename types::index first, typename types::index last) const{
		typename types::layout_t new_layout = *this;
		(new_layout.nelems_/=Layout::size())*=(last - first);
		return {new_layout, types::base_ + Layout::operator()(first)};
	}
	basic_array strided(typename types::index s) const{
		typename types::layout_t new_layout = *this; 
		new_layout.stride_*=s;
		return {new_layout, types::base_};
	}
	basic_array sliced(typename types::index first, typename types::index last, typename types::index stride) const{
		return sliced(first, last).strided(stride);
	}
	auto range(typename types::index_range const& ir) const{
		return sliced(ir.front(), ir.front() + ir.size());
	}
	auto range(typename types::index_range const& ir, dimensionality_type n) const{
		return rotated(n).range(ir).rotated(-n);
	}
	auto flattened() const{
		multi::biiterator<std::decay_t<decltype(this->begin())>> biit{this->begin(), 0, size(*(this->begin()))};
		return basic_array<typename std::iterator_traits<decltype(biit)>::value_type, 1, decltype(biit)>{
			multi::layout_t<1>(1, 0, this->size()*size(*(this->begin()))),
			biit
		};
	}
	friend auto flattened(basic_array const& self){return self.flattened();}
	bool is_flattable() const{return this->stride() == this->layout().sub_.nelems_;}
	auto flatted() const{
		assert(is_flattable() && "flatted doesn't work for all layouts!");//this->nelems());
		multi::layout_t<D-1> new_layout{this->layout().sub_};
		new_layout.nelems_*=this->size();
		return basic_array<T, D-1, ElementPtr>{new_layout, types::base_};
	}
	friend auto flatted(basic_array const& self){return self.flatted();}
	template<typename Size>
	auto partitioned(Size const& s) const{
		assert(s!=0);
		assert(this->layout().nelems_%s==0);
		multi::layout_t<D+1> new_layout{this->layout(), this->layout().nelems_/s, 0, this->layout().nelems_};
		new_layout.sub_.nelems_/=s;
		return basic_array<T, D+1, ElementPtr>{new_layout, types::base_};
	}
	decltype(auto) transposed() const&{
		typename types::layout_t new_layout = *this;
		new_layout.transpose();
		return basic_array<T, D, ElementPtr>{new_layout, types::base_};
	}
	friend decltype(auto) transposed(basic_array const& self){return self.transposed();}
	friend decltype(auto) operator~(basic_array const& self){return self.transposed();}

	decltype(auto) rotated()&&{
		typename types::layout_t new_layout = *this;
		new_layout.rotate();
		return basic_array<T, D, ElementPtr>{new_layout, types::base_};
	}
	decltype(auto) rotated() const&{
		typename types::layout_t new_layout = *this;
		new_layout.rotate();
		return basic_array<T, D, typename basic_array::element_ptr>{new_layout, types::base_};
	}
	friend decltype(auto) rotated(basic_array const&  self){return self.rotated();}
	friend decltype(auto) rotated(basic_array      && self){return std::move(self).rotated();}
	auto unrotated() const{
		typename types::layout_t new_layout = *this; 
		new_layout.unrotate();
		return basic_array<T, D, ElementPtr>{new_layout, types::base_};
	}
	friend auto unrotated(basic_array const& self){return self.unrotated();}
	auto rotated(dimensionality_type i) const{
		typename types::layout_t new_layout = *this; 
		new_layout.rotate(i);
		return basic_array<T, D, ElementPtr>{new_layout, types::base_};
	}
	decltype(auto) operator<<(dimensionality_type i) const{return rotated(i);}
	decltype(auto) operator>>(dimensionality_type i) const{return unrotated(i);}
	basic_array //const& 
		operator()() const&{return *this;}
	template<class... As>
	auto operator()(index_range a, As... as) const{return range(a).rotated()(as...).unrotated();}
	template<class... As>
	auto operator()(index i) const
	->decltype(operator[](i)){
		return operator[](i);}
	template<class... As>
	auto operator()(index i, As... as) const
	->decltype(operator[](i)(as...)){
		return operator[](i)(as...);}
//#define SARRAY1(A1) auto operator()(A1 a1) const{return operator()<>(a1);}
#define SARRAY2(A1, A2)	auto operator()(A1 a1, A2 a2) const{return operator()<A2>(a1, a2);}
	SARRAY2(index, index ); SARRAY2(irange, index );
	SARRAY2(index, irange); SARRAY2(irange, irange);
#undef SARRAY2
#define SARRAY3(A1, A2, A3) auto operator()(A1 a1, A2 a2, A3 a3) const{return operator()<A2, A3>(a1, a2, a3);}
	SARRAY3(index, index , index ); SARRAY3(irange, index , index );
	SARRAY3(index, index , irange); SARRAY3(irange, index , irange);
	SARRAY3(index, irange, index ); SARRAY3(irange, irange, index );
	SARRAY3(index, irange, irange); SARRAY3(irange, irange, irange);
#undef SARRAY3
#define SARRAY4(A1, A2, A3, A4) auto operator()(A1 a1, A2 a2, A3 a3, A4 a4) const{return operator()<A2, A3, A4>(a1, a2, a3, a4);}
	SARRAY4(index, index, index , index ); SARRAY4(index, irange, index , index );
	SARRAY4(index, index, index , irange); SARRAY4(index, irange, index , irange);
	SARRAY4(index, index, irange, index ); SARRAY4(index, irange, irange, index );
	SARRAY4(index, index, irange, irange); SARRAY4(index, irange, irange, irange);
	SARRAY4(irange, index, index , index ); SARRAY4(irange, irange, index , index );
	SARRAY4(irange, index, index , irange); SARRAY4(irange, irange, index , irange);
	SARRAY4(irange, index, irange, index ); SARRAY4(irange, irange, irange, index );
	SARRAY4(irange, index, irange, irange); SARRAY4(irange, irange, irange, irange);
#undef SARRAY4
private:
	using Layout::nelems_;
	using Layout::stride_;
	using Layout::sub_;
public:
	using iterator =
		array_iterator<typename types::element, D, typename types::element_ptr, typename types::reference>;
private:
	template<class Iterator>
	struct basic_reverse_iterator : 
		std::reverse_iterator<Iterator>,
		boost::multi::totally_ordered2<basic_reverse_iterator<Iterator>, void>
	{
		template<class O, typename = decltype(std::reverse_iterator<Iterator>{base(std::declval<O const&>())})>
		basic_reverse_iterator(O const& o) : std::reverse_iterator<Iterator>{base(o)}{}
		basic_reverse_iterator() : std::reverse_iterator<Iterator>{}{}
		explicit basic_reverse_iterator(Iterator it) : std::reverse_iterator<Iterator>(std::prev(it)){}
		explicit operator Iterator() const{auto ret = this->base(); if(ret!=Iterator{}) return ++ret; else return Iterator{};}
		explicit operator bool() const{return bool(this->base());}
		bool operator==(basic_reverse_iterator const& other) const{return (this->base() == other.base());}
		typename Iterator::reference operator*() const{return this->current;}
		typename Iterator::pointer operator->() const{return &this->current;}
		typename Iterator::reference operator[](typename Iterator::difference_type n) const{return *(this->current - n);}
		bool operator<(basic_reverse_iterator const& o) const{return o.base() < this->base();}
	};
public:
	using reverse_iterator = basic_reverse_iterator<iterator>;
	using ptr = basic_array_ptr<basic_array, Layout>;
	ptr operator&() const{return {this->base_, this->layout()};}
	iterator begin(index i) const{
		Layout l = static_cast<Layout const&>(*this); l.rotate(i);
		return {types::base_ + l(0       ), l.sub_, l.stride_};
	}
	iterator end(index i) const{
		Layout l = static_cast<Layout const&>(*this); l.rotate(i);
		return {types::base_ + l(l.size()), l.sub_, l.stride_};
	}
	iterator  begin() const HD{return {types::base_          , sub_, stride_};}
	iterator  end  () const HD{return {types::base_ + nelems_, sub_, stride_};}
protected:
	template<class A>
	void intersection_assign_(A&& other) const{
		// using multi::extension
		for(auto i : intersection(types::extension(), multi::extension(other)))
			operator[](i).intersection_assign_(std::forward<A>(other)[i]);
	}
public:
	template<class It> void assign(It first, It last) &&{assert( this->size() == std::distance(first, last) );
		adl::copy(first, last, this->begin());
	}
	template<class It> It assign(It first) &&{
		return adl::copy_n(first, this->size(), this->begin()), first + this->size();
	}
//	template<class It> void assign(It first) &&
//	->decltype(adl::copy_n(first, this->size(), this->begin())){
//		return adl::copy_n(first, this->size(), this->begin()), ;}
	template<class Range> auto assign(Range&& r) 
	->decltype(std::move(*this).assign(adl::begin(r), adl::end(r))) &&{
		return std::move(*this).assign(adl::begin(r), adl::end(r));}
	void assign(std::initializer_list<typename basic_array::value_type> il) const{assert( il.size() == this->size() );
		assign(il.begin(), il.end());
	}
	template<class A, typename = std::enable_if_t<not std::is_base_of<basic_array, std::decay_t<A>>{}>>
	basic_array const& operator=(A&& o) &&{
	//	assert(extension(*this) == extension(o));
		assert(this->extension() == o.extension());
		std::move(*this).assign(adl::begin(std::forward<A>(o)), adl::end(std::forward<A>(o)));
		return *this;
	}
	basic_array const& operator=(basic_array const& o) &&{
		assert( this->extension() == o.extension() );
		std::move(*this).assign( o.begin(), o.end() );
		return *this;
	}
	template<class Array> void swap(Array&& o) const{assert(this->extension() == extension(o));
		adl::swap_ranges(this->begin(), this->end(), adl::begin(std::forward<Array>(o)));
	}
	friend void swap(basic_array const& a, basic_array const& b){a.swap(b);}
	template<class Array> void swap(basic_array const& s, Array&& a){s.swap(a);}
	template<class Array> void swap(Array&& a, basic_array const& s){s.swap(a);}
	template<class Array> bool operator==(Array const& o) const{
		if(this->extension()!=o.extension()) return false;
		return adl::equal(this->begin(), this->end(), adl::begin(o));
	}
	bool operator==(basic_array const& o) const{return operator==<basic_array>(o);}
private:
	template<class A1, class A2>
	static auto lexicographical_compare(A1 const& a1, A2 const& a2){
		if(extension(a1).first() > extension(a2).first()) return true;
		if(extension(a1).first() < extension(a2).first()) return false;
		return adl::lexicographical_compare(adl::begin(a1), adl::end(a1), adl::begin(a2), adl::end(a2));
	}
public:
	template<class O>
	bool operator<(O const& o) const{return lexicographical_compare(*this, o);}
	template<class O>
	bool operator>(O const& o) const{return lexicographical_compare(o, *this);}
public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>
	basic_array<T2, D, P2> static_array_cast() const HD{
		P2 p2{this->base_};
		return {this->layout(), p2};
	}
//	template<class T2, class P2 = decltype(boost::static_pointer_cast<T2>(std::declval<typename basic_array::element_ptr>()))>
//	auto static_array_cast() const HD ->basic_array<T2, D, P2>{
//		return basic_array<T2, D, P2>{this->layout(), boost::static_pointer_cast<T2>(this->base_)};
//	}
//	template<class T2>
//	auto static_array_cast() const HD -> basic_array<T2, D, decltype(boost::static_pointer_cast<T2>(std::declval<typename basic_array::element_ptr>()))>{
//		return {this->layout(), boost::static_pointer_cast<T2>(this->base_)};
//	}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>,
		class Element = typename basic_array::element,
		class PM = T2 Element::*
	>
	basic_array<T2, D, P2> member_cast(PM pm) const{
		static_assert(sizeof(T)%sizeof(T2) == 0, 
			"array_member_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements");
	//	return {this->layout().scale(sizeof(T)/sizeof(T2)), &(this->base_->*pm)};
		return basic_array<T2, D, P2>{this->layout().scale(sizeof(T)/sizeof(T2)), static_cast<P2>(&(this->base_->*pm))};
	}
	template<class T2, class P2 = T2*>//typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2> >
	basic_array<std::decay_t<T2>, D, P2> reinterpret_array_cast() const{
		static_assert( sizeof(T)%sizeof(T2)== 0, "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );
		auto thisbase = this->base();
		return {
			this->layout().scale(sizeof(T)/sizeof(T2)), 
			reinterpret_cast<P2 const&>(thisbase)
		};
	}
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template<class Element, typename Ptr, typename Ref>
struct array_iterator<Element, 1, Ptr, Ref> : 
	boost::multi::iterator_facade<
		array_iterator<Element, 1, Ptr, Ref>, 
		Element, std::random_access_iterator_tag, 
		Ref, multi::difference_type
	>,
	multi::affine<array_iterator<Element, 1, Ptr, Ref>, multi::difference_type>,
	multi::decrementable<array_iterator<Element, 1, Ptr, Ref>>,
	multi::incrementable<array_iterator<Element, 1, Ptr, Ref>>,
	multi::totally_ordered2<array_iterator<Element, 1, Ptr, Ref>, void>
{
	using affine = multi::affine<array_iterator<Element, 1, Ptr, Ref>, multi::difference_type>;
	using difference_type = typename affine::difference_type;

	array_iterator() = default;
	array_iterator(array_iterator const& other) = default;
	template<class Other, typename = decltype(Ptr{typename Other::pointer{}})> 
	array_iterator(Other const& o) HD : data_{o.data_}, stride_{o.stride_}{}
	template<class EE, dimensionality_type, class PP, class RR> friend struct array_iterator;
	array_iterator(std::nullptr_t nu) HD : data_{nu}, stride_{1}{}
	array_iterator(Ptr const& p) HD : data_{p}, stride_{1}{}
	explicit operator bool() const{return static_cast<bool>(this->data_);}
	Ref operator[](typename array_iterator::difference_type n) const HD{/*assert(*this);*/ return *((*this) + n);}
	Ptr operator->() const{return data_;}
	using element = Element;
	using element_ptr = Ptr;
	using pointer = element_ptr;
	using rank = std::integral_constant<dimensionality_type, 1>;
	bool operator<(array_iterator const& o) const{return distance_to(o) > 0;}
private:
	array_iterator(Ptr d, typename basic_array<Element, 1, Ptr>::index s) HD : data_{d}, stride_{s}{}
	friend struct basic_array<Element, 1, Ptr>;
	Ptr data_ = nullptr;
	multi::index stride_;
	Ref dereference() const HD{return *data_;}
//	bool equal(array_iterator const& o) const{
//		assert(stride_ == o.stride_);
//		return data_==o.data_;// and stride_==o.stride_;
//	}
//	void increment(){data_ += stride_;}
//	void decrement(){data_ -= stride_;}
//	void advance(typename array_iterator::difference_type n) HD{data_ += stride_*n;}
	difference_type distance_to(array_iterator const& other) const{
		assert(stride_==other.stride_ and (other.data_-data_)%stride_ == 0);
		return (other.data_ - data_)/stride_;
	}
	auto base() const{return data_;}
	friend auto base(array_iterator const& self){return self.base();}
public:
	auto data() const HD{return data_;}
	auto stride() const{return stride_;}
	friend auto stride(array_iterator const& self){return self.stride();}
	array_iterator& operator++(){data_+=stride_; /*increment()*/; return *this;}
	array_iterator& operator--(){data_-=stride_; /*decrement()*/; return *this;}
	bool operator==(array_iterator const& o) const{return data_== o.data_;/*return equal(o);*/}
	bool operator!=(array_iterator const& o) const{return data_!= o.data_;/*return equal(o);*/}
	Ref operator*() const HD{return dereference();}
	difference_type operator-(array_iterator const& o) const{return -distance_to(o);}
	array_iterator& operator+=(difference_type d) HD{data_+=stride_*d; return *this;}
	array_iterator& operator-=(difference_type d) HD{data_-=stride_*d; return *this;}
};

template<class Element, dimensionality_type D, typename... Ts>
using iterator = array_iterator<Element, D, Ts...>;

template<typename T, typename ElementPtr, class Layout>
struct basic_array<T, dimensionality_type{0}, ElementPtr, Layout> :
	array_types<T, dimensionality_type(0), ElementPtr, Layout>
{
	using types = array_types<T, dimensionality_type{0}, ElementPtr, Layout>;
	using types::types;
	using element_ref = typename std::iterator_traits<typename basic_array::element_ptr>::reference;//decltype(*typename basic_array::element_ptr{});
	decltype(auto) operator=(typename basic_array::element_type const& e) const&{
		adl::copy_n(&e, 1, this->base_); return *this;
	}
	bool operator==(basic_array const& o) const&{
		return adl::equal(o.base_, o.base_ + 1, this->base_);
	}
	bool operator!=(basic_array const& o) const&{return not operator==(o);}
//	bool operator==(typename basic_array::element_type const& e) const&{
//		using std::equal; return equal(&e, &e + 1, this->base_);
//	}
//	bool operator!=(typename basic_array::element_type const& e) const&{return not operator==(e);}
//	operator element_ref() const{return *(this->base_);}
//	template<class TT> operator TT(){return static_cast<TT>(element_ref());}
	typename basic_array::element_ptr operator&() const{return this->base_;}
	using decay_type = typename types::element;
//	basic_array&
	element_ref operator()() const&{return *(this->base_);}
	operator element_ref() const&{return *(this->base_);}
//	decltype(auto) operator()() &&{return std::move(*(this->base_));}
	template<class Archive>
	auto serialize(Archive& ar, const unsigned int){
		using boost::serialization::make_nvp;
		ar & make_nvp("element",  *(this->base_));
	//	std::for_each(this->begin(), this->end(), [&](auto&& e){ar & make_nvp("item", e);});
	}
};

template<typename T, typename ElementPtr, class Layout>
struct basic_array<T, dimensionality_type{1}, ElementPtr, Layout> : 
	multi::partially_ordered2<basic_array<T, dimensionality_type(1), ElementPtr, Layout>, void>,
	multi::random_iterable<basic_array<T, dimensionality_type(1), ElementPtr, Layout> >,
	array_types<T, dimensionality_type(1), ElementPtr, Layout>
{
	using types = array_types<T, dimensionality_type{1}, ElementPtr, Layout>;
	using types::types;
	auto get_allocator() const{return default_allocator_of(basic_array::base());}
	friend auto get_allocator(basic_array const& self){return self.get_allocator();}
	using decay_type = array<typename types::element, dimensionality_type{1}, decltype(default_allocator_of(std::declval<ElementPtr>()))>;
	decay_type decay() const{return {*this};}
	friend decay_type decay(basic_array const& self){return self.decay();}
protected:
	template<class A>
	void intersection_assign_(A&& other) const{
		for(auto idx : intersection(types::extension(), extension(other)))
			operator[](idx) = std::forward<A>(other)[idx];
	}
protected:
	template<class TT, dimensionality_type DD, typename EP, class LLayout> friend struct basic_array;
	template<class TT, dimensionality_type DD, class Alloc> friend struct static_array;
	basic_array(basic_array const&) = default;
	template<class T2, class P2, class TT, dimensionality_type DD, class PP>
	HD friend decltype(auto) static_array_cast(basic_array<TT, DD, PP> const&);
public:
//	template<class Archive>
//	void serialize(Archive& ar, unsigned int){
//		for(auto&& e : *this) ar & BOOST_SERIALIZATION_NVP(e);
//	}
	basic_array(basic_array&&) = default; // ambiguos deep-copy a reference type, in C++14 use auto&& A_ref = Expression; or decay_t<decltype(Expression)> A = Expression
	// in c++17 things changed and non-moveable non-copyable types can be returned from functions and captured by auto
//	decay_type static remake(std::initializer_list<typename basic_array::value_type> il){return decay_type(il);}
//	template<class... As> static auto remake(As&&... as)->decltype(decay_type(std::forward<As>(as)...)){return decay_type(std::forward<As>(as)...);}
protected:
	template<class, class> friend struct basic_array_ptr;
	template<class, dimensionality_type D, class, class>
	friend struct array_iterator;
public:
	friend constexpr auto dimensionality(basic_array const& self){return self.dimensionality;}
	template<class BasicArray, typename = std::enable_if_t<not std::is_base_of<basic_array, std::decay_t<BasicArray>>{}>, typename = decltype(types(std::declval<BasicArray&&>()))> 
	basic_array(BasicArray&& other) : types{std::forward<BasicArray>(other)}{}
	basic_array_ptr<basic_array, Layout> operator&() const{
		return {this->base_, this->layout()};
	}
	void assign(std::initializer_list<typename basic_array::value_type> il) const{assert( il.size() == static_cast<std::size_t>(this->size()) );
		assign(il.begin(), il.end());
	}
	template<class It> void assign(It first, It last) const{assert( std::distance(first, last) == this->size() );
		adl::copy(first, last, this->begin());
	}

	template<class Archive>
	auto serialize(Archive& ar, const unsigned int){
		using boost::serialization::make_nvp;
		std::for_each(this->begin(), this->end(), [&](auto&& e){ar & make_nvp("item", e);});
	}
	template<class A, 
		typename = std::enable_if_t<not std::is_base_of<basic_array, std::decay_t<A>>{}>,
		typename = decltype(
			std::declval<typename basic_array::reference&&>() 
				= std::declval<typename multi::array_traits<typename std::remove_reference_t<A>>::reference&&>()
		)
	>
	basic_array const& operator=(A&& o) const{{using multi::extension; assert(this->extension() == extension(o));}
		this->assign(adl::begin(std::forward<A>(o)), adl::end(std::forward<A>(o)));
		return *this;
	}
	basic_array&& operator=(basic_array const& o)&&{ assert(this->extension() == o.extension());
		return std::move(*this).assign(o.begin()), std::move(*this);
	}
	template<class TT, dimensionality_type DD, class... As>
	basic_array&& operator=(basic_array<TT, DD, As...> const& o)&&{assert(this->extension() == o.extension());
		std::move(*this).assign(o.begin(), o.end()); return std::move(*this);
	}
	typename types::reference operator[](typename types::index i) const HD{
		assert( this->extension().contains(i) );
		return *(this->base() + Layout::operator()(i)); // in C++17 this is allowed even with syntethic references
	}
	template<class Tuple, typename = std::enable_if_t<(std::tuple_size<std::decay_t<Tuple>>{}>1) > >
	auto operator[](Tuple&& t) const
	->decltype(operator[](std::get<0>(t))[detail::tuple_tail(t)]){
		return operator[](std::get<0>(t))[detail::tuple_tail(t)];}
	template<class Tuple, typename = std::enable_if_t<std::tuple_size<std::decay_t<Tuple>>{}==1> >
	decltype(auto) operator[](Tuple&& t) const{return operator[](std::get<0>(t));}
	decltype(auto) operator[](std::tuple<>) const{return *this;}

	basic_array sliced(typename types::index first, typename types::index last) const{
		typename types::layout_t new_layout = *this; 
		(new_layout.nelems_/=Layout::size())*=(last - first);
		return {new_layout, types::base_ + Layout::operator()(first)};		
	}
	basic_array strided(typename types::index s) const{
		typename types::layout_t new_layout = this->layout();
		new_layout.stride_*=s;
		return {new_layout, types::base_};//+ Layout::operator()(this->extension().front())};
	}
	basic_array sliced(typename types::index first, typename types::index last, typename types::index stride) const{
		return sliced(first, last).strided(stride);
	}
	auto range(index_range const& ir) const{return sliced(ir.front(), ir.last());}
	decltype(auto) operator()()&&{return std::move(*this);}
	auto operator()(index_range const& ir) const{return range(ir);}
	decltype(auto) operator()(typename types::index i) const{return operator[](i);}
	template<typename Size>
	auto partitioned(Size const& s) const{
		assert( this->layout().nelems_%s==0 );
		multi::layout_t<2> new_layout{this->layout(), this->layout().nelems_/s, 0, this->layout().nelems_};
		new_layout.sub_.nelems_/=s;
		return basic_array<T, 2, ElementPtr>{new_layout, types::base_};
	}
	friend decltype(auto) rotated(basic_array const& self){return self.rotated();}
	friend decltype(auto) unrotated(basic_array const& self){return self.unrotated();}
//	friend decltype(auto) transposed(basic_array const& self){return self.transposed();}
//	friend decltype(auto) operator~(basic_array const& self){return transposed(self);}

	decltype(auto)   rotated(dimensionality_type = 1) const{return *this;}
	decltype(auto) unrotated(dimensionality_type = 1) const{return *this;}
	decltype(auto) operator<<(dimensionality_type i) const{return rotated(i);}
	decltype(auto) operator>>(dimensionality_type i) const{return unrotated(i);}

	using iterator = typename multi::array_iterator<typename types::element, 1, typename types::element_ptr, typename types::reference>;
	using reverse_iterator = std::reverse_iterator<iterator>;

	iterator begin() const HD{return {types::base_               , Layout::stride_};}
	iterator end  () const HD{return{types::base_+types::nelems_, Layout::stride_};}

	template<class It> auto assign(It f)&& //	->decltype(adl::copy_n(f, this->size(), begin(std::move(*this))), void()){
	->decltype(adl::copy_n(f, this->size(), std::declval<iterator>()), void()){
		return adl::copy_n(f, this->size(), std::move(*this).begin()), void();}

	template<typename Array, typename = std::enable_if_t<not std::is_base_of<basic_array, Array>{}> >
	bool operator==(Array const& o) const{ // TODO assert extensions are equal?
		return (basic_array::extension()==extension(o)) and adl::equal(this->begin(), this->end(), adl::begin(o));
	}
	bool operator==(basic_array const& other) const{
		return operator==<basic_array, void>(other);
	//	using multi::extension; using std::equal; using std::begin;
	//	return (this->extension()==extension(other)) and equal(this->begin(), this->end(), begin(other));
	}
// commented for nvcc
	bool operator<(basic_array const& o) const{return lexicographical_compare(*this, o);}//operator< <basic_array const&>(o);}
	template<class Array> void swap(Array&& o) const{{using multi::extension;assert(this->extension() == extension(o));}
		adl::swap_ranges(this->begin(), this->end(), adl::begin(std::forward<Array>(o)));
	}
	friend void swap(basic_array const& a, basic_array const& b){a.swap(b);}
	template<class A, typename = std::enable_if_t<not std::is_base_of<basic_array, std::decay_t<A>>{}> > friend void swap(basic_array&& s, A&& a){s.swap(a);}
	template<class A, typename = std::enable_if_t<not std::is_base_of<basic_array, std::decay_t<A>>{}> > friend void swap(A&& a, basic_array&& s){s.swap(a);}
private:
	template<class A1, class A2>
	static auto lexicographical_compare(A1 const& a1, A2 const& a2){
		using multi::extension;
		if(extension(a1).first() > extension(a2).first()) return true;
		if(extension(a1).first() < extension(a2).first()) return false;
		return adl::lexicographical_compare(adl::begin(a1), adl::end(a1), adl::begin(a2), adl::end(a2));
	}
public:
	template<class O>
	bool operator<(O const& o) const{return lexicographical_compare(*this, o);}
	template<class O>
	bool operator>(O const& o) const{return lexicographical_compare(o, *this);}
public:
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>
	HD basic_array<T2, 1, P2> static_array_cast() const{//(basic_array&& o){  // name taken from std::static_pointer_cast
		return {this->layout(), static_cast<P2>(this->base())};
	}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>, class... Args>
	HD basic_array<T2, 1, P2> static_array_cast(Args&&... args) const{//(basic_array&& o){  // name taken from std::static_pointer_cast
		return {this->layout(), P2{this->base(), std::forward<Args>(args)...}};
	}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>,
		class Element = typename basic_array::element,
		class PM = T2 Element::*
	>
	basic_array<T2, 1, P2> member_cast(PM pm) const{
		static_assert(sizeof(T)%sizeof(T2) == 0, 
			"array_member_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom alignas structures (to the interesting member(s) sizes) or custom pointers to allow reintrepreation of array elements");
		return {this->layout().scale(sizeof(T)/sizeof(T2)), static_cast<P2>(&(this->base_->*pm))};
	}
	template<class T2, class P2 = typename std::pointer_traits<typename basic_array::element_ptr>::template rebind<T2>>
	basic_array<T2, 1, P2> reinterpret_array_cast() const{
		static_assert( sizeof(T)%sizeof(T2)== 0, "error: reinterpret_array_cast is limited to integral stride values, therefore the element target size must be multiple of the source element size. Use custom pointers to allow reintrepreation of array elements in other cases" );
//			this->layout().scale(sizeof(T)/sizeof(T2));
		auto const thisbase = this->base();
		return {
			this->layout().scale(sizeof(T)/sizeof(T2)), 
			reinterpret_cast<P2 const&>(thisbase) // 
		};
	}
};

template<class T2, class P2, class Array, class... Args>
HD decltype(auto) static_array_cast(Array&& a, Args&&... args){return a.template static_array_cast<T2, P2>(std::forward<Args>(args)...);}

template<class T2, class Array, class P2 = typename std::pointer_traits<typename std::decay<Array>::type::element_ptr>::template rebind<T2> , class... Args>
HD decltype(auto) static_array_cast(Array&& a, Args&&... args){return a.template static_array_cast<T2, P2>(std::forward<Args>(args)...);}

template<
	class T2, class Array,
	class P2 = typename std::pointer_traits<typename std::decay<Array>::type::element_ptr>::template rebind<T2>
>
decltype(auto) reinterpret_array_cast(Array&& a){
	return a.template reinterpret_array_cast<T2, P2>();
}

template<class T2, class Array, class P2 = typename std::pointer_traits<typename std::decay_t<Array>::element_ptr>::template rebind<T2>,
	class PM = T2 std::decay_t<Array>::element::*
>
decltype(auto) member_array_cast(Array&& a, PM pm){return a.template member_cast<T2, P2>(pm);}

template<
	class T2, class Array, class P2 = typename std::pointer_traits<typename std::decay_t<Array>::element_ptr>::template rebind<T2>,
	class OtherT2, class OtherElement, std::enable_if_t<not std::is_same<T2, OtherElement>{}, int> =0
>
decltype(auto) member_array_cast(Array&& a, OtherT2 OtherElement::* pm){
	static_assert(sizeof(OtherElement)==sizeof(typename std::decay_t<Array>::element_type),
		"member cast does not implicitly reinterprets between types of different size");
	static_assert(sizeof(OtherT2)==sizeof(T2), 
		"member cast does not implicitly reinterprets between types of different size");
	return reinterpret_array_cast<OtherElement>(std::forward<Array>(a)).template member_cast<T2>(pm);
}

template<typename T, dimensionality_type D, typename ElementPtr = T*>
struct array_ref : 
//TODO	multi::partially_ordered2<array_ref<T, D, ElementPtr>, void>,
	basic_array<T, D, ElementPtr>
{
protected:
	constexpr array_ref() noexcept
		: basic_array<T, D, ElementPtr>{typename array_ref::types::layout_t{}, nullptr}{}
public:
	array_ref(array_ref const&) = default;
	constexpr array_ref(typename array_ref::element_ptr p, typename array_ref::extensions_type e = {}) noexcept
		: basic_array<T, D, ElementPtr>{typename array_ref::types::layout_t{e}, p}{}
//	constexpr array_ref(typename array_ref::element_ptr p, std::initializer_list<index_extension> il) noexcept
//		: array_ref(p, detail::to_tuple<D, index_extension>(il)){}
//	template<class Extension>//, typename = decltype(array_ref(std::array<Extension, D>{}, allocator_type{}, std::make_index_sequence<D>{}))>
//	constexpr array_ref(typename array_ref::element_ptr p, std::array<Extension, D> const& x) 
//		: array_ref(p, x, std::make_index_sequence<D>{}){}
//	using basic_array<T, D, ElementPtr>::operator[];
	using basic_array<T, D, ElementPtr>::operator=;
	using basic_array<T, D, ElementPtr>::operator==;
//	using basic_array<T, D, ElementPtr>::operator<;
//	using basic_array<T, D, ElementPtr>::operator>;
//	template<class ArrayRef> explicit array_ref(ArrayRef&& a) : array_ref(a.data(), extensions(a)){} 
	array_ref const& operator=(array_ref const& o) &&{assert(this->num_elements()==o.num_elements());
		auto e = adl::copy_n(o.data(), o.num_elements(), this->data()); (void)e; assert( e == this->data() + this->num_elements() );
		return *this;
	}
	template<typename TT, dimensionality_type DD = D, class... As>
	array_ref const& operator=(array_ref<TT, DD, As...> const& o)&&{//const{
		assert( this->extensions() == o.extensions() );
		adl::copy_n(o.data(), o.num_elements(), this->data());
		return *this;
	}
	template<typename TT, dimensionality_type DD = D, class... As>
	bool operator==(array_ref<TT, DD, As...> const& o) const{
		if( this->extensions() != o.extensions() ) return false; // TODO, or assert?
		return adl::equal(data_elements(), data_elements() + this->num_elements(), o.data());
	}
	typename array_ref::element_ptr data_elements() const{return array_ref::base_;}
	friend typename array_ref::element_ptr data_elements(array_ref const& s){return s.data();}
	typename array_ref::element_ptr data() const HD{return array_ref::base_;} 
	friend typename array_ref::element_ptr data(array_ref const& self){return self.data();}
	template<class Archive>
	auto serialize(Archive& ar, const unsigned int v){ (void)v;
		using boost::serialization::make_nvp;
		if(this->num_elements() < (2<<8) ) basic_array<T, D, ElementPtr>::serialize(ar, v);
		else{
			if(std::is_trivial<typename array_ref::element>{}){
				using boost::serialization::make_binary_object;
				ar & make_nvp("binary_data", make_binary_object(this->data(), sizeof(typename array_ref::element)*this->num_elements()));
			}else{
				using boost::serialization::make_array;
				ar & make_nvp("data", make_array(this->data(), this->num_elements()));
			}
		}
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

//template<class T, dimensionality_type D, typename Ptr = T*>
//using array_ptr = basic_array_ptr<array_ref<T, D, Ptr>, typename array_ref<T, D, Ptr>::layout_t>;

template<class T, dimensionality_type D, typename Ptr = T*>
struct array_ptr : basic_array_ptr<array_ref<T, D, Ptr>, typename array_ref<T, D, Ptr>::layout_t>{
	using basic_ptr = basic_array_ptr<array_ref<T, D, Ptr>, typename array_ref<T, D, Ptr>::layout_t>;
	using basic_array_ptr<array_ref<T, D, Ptr>, typename array_ref<T, D, Ptr>::layout_t>::basic_array_ptr;
	template<class TT, std::size_t N>
	array_ptr(TT(*t)[N]) : basic_ptr(data_elements(*t), extensions(*t)){}
};

template<dimensionality_type D, class P>
array_ref<typename std::iterator_traits<P>::value_type, D, P> 
make_array_ref(P p, index_extensions<D> x){return {p, x};}

template<class P> auto make_array_ref(P p, index_extensions<1> x){return make_array_ref<1>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<2> x){return make_array_ref<2>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<3> x){return make_array_ref<3>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<4> x){return make_array_ref<4>(p, x);}
template<class P> auto make_array_ref(P p, index_extensions<5> x){return make_array_ref<5>(p, x);}

//In ICC you need to specify the dimensionality in make_array_ref<D>
//#if defined(__INTEL_COMPILER)
//template<dimensionality_type D, class P> 
//auto make_array_ref(P p, std::initializer_list<index_extension> il){return make_array_ref(p, detail::to_tuple<D, index_extension>(il));}
//template<dimensionality_type D, class P> 
//auto make_array_ref(P p, std::initializer_list<index> il){return make_array_ref(p, detail::to_tuple<D, index_extension>(il));}
//#endif

#if __cpp_deduction_guides
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
constexpr auto rotated(const T(&t)[N]) noexcept{
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


#if _TEST_BOOST_MULTI_ARRAY_REF

#include<cassert>
#include<numeric> // iota
#include<iostream>
#include<vector>

using std::cout;
namespace multi = boost::multi;


int main(){

	{
		double a[4][5] = {{1.,2.},{2.,3.}};
		multi::array_ref<double, 2> A(&a[0][0], {4, 5});
		multi::array_ref<double, 2, double const*> B(&a[0][0], {4, 5});
		multi::array_ref<double const, 2> C(&a[0][0], {4, 5});
		multi::array_cref<double, 2> D(&a[0][0], {4, 5});
		A[1][1] = 2.;
//		A[1].cast<double const*>()[1] = 2.;
		double d[4][5] = {{1.,2.},{2.,3.}};
	//	typedef d45 = double const[4][5];
		auto dd = (double const(&)[4][5])(d);
		assert( &(dd[1][2]) == &(d[1][2]) );
		assert(( & A[1].static_array_cast<double, double const*>()[1] == &A[1][1] ));
		assert(( &multi::static_array_cast<double, double const*>(A[1])[1] == &A[1][1] ));
	}
	{
		double const d2D[4][5] = {{1.,2.},{2.,3.}};
		multi::array_ref<double, 2, const double*> d2Rce(&d2D[0][0], {4, 5});
		assert( &d2Rce[2][3] == &d2D[2][3] );
		assert( d2Rce.size() == 4 );
		assert( num_elements(d2Rce) == 20 );
	}
	{
		std::string const dc3D[4][2][3] = {
			{{"A0a", "A0b", "A0c"}, {"A1a", "A1b", "A1c"}},
			{{"B0a", "B0b", "B0c"}, {"B1a", "B1b", "B1c"}},
			{{"C0a", "C0b", "C0c"}, {"C1a", "C1b", "C1c"}}, 
			{{"D0a", "D0b", "D0c"}, {"D1a", "D1b", "D1c"}}, 
		};
		multi::array_cref<std::string, 3> A(&dc3D[0][0][0], {4, 2, 3});
		assert( num_elements(A) == 24 and A[2][1][1] == "C1b" );
		auto const& A2 = A.sliced(0, 3).rotated()[1].sliced(0, 2).unrotated();
		assert( multi::rank<std::decay_t<decltype(A2)>>{} == 2 and num_elements(A2) == 6 );
		assert( std::get<0>(sizes(A2)) == 3 and std::get<1>(sizes(A2)) == 2 );
		{auto x = extensions(A2);
		for(auto i : std::get<0>(x) ){
			for(auto j : std::get<1>(x) ) cout<< A2[i][j] <<' ';
			cout<<'\n';
		}}
		auto const& A3 = A({0, 3}, 1, {0, 2});
		assert( multi::rank<std::decay_t<decltype(A3)>>{} == 2 and num_elements(A3) == 6 );
		{
			auto x = extensions(A3);
			for(auto i : std::get<0>(x)){
				for(auto j : std::get<1>(x)) cout<< A3[i][j] <<' ';
				cout<<'\n';
			}
		}
	}
	return 0;
}
#undef HD

#endif
#endif

