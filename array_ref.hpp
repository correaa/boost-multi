#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ -O3 `#-fconcepts` -std=c++17 -Wall -Wextra `#-fmax-errors=2` `#-Wfatal-errors` -lboost_timer -I${HOME}/prj -D_TEST_BOOST_MULTI_ARRAY_REF -rdynamic -ldl $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_ARRAY_REF_HPP
#define BOOST_MULTI_ARRAY_REF_HPP


#include "../multi/ordering.hpp"
#include "../multi/index_range.hpp"

#include<cassert>
#include<iostream> // cerr
#include<memory>

namespace boost{
namespace multi{

using std::cerr;

using index = std::ptrdiff_t;
using size_type = std::ptrdiff_t;

struct strides_t{
	std::ptrdiff_t val = 1;
	strides_t const& next;
	strides_t(std::ptrdiff_t v, strides_t const& n) : val(v), next(n){}
};

/*
template<typename T, dimensionality_type D, typename ElementPtr = T const*>//, class Storage = c_ordering<D>>
class array_ref{
	using element_ptr = ElementPtr;
	element_ptr data_;
public:
	template<class I, class... Is>
	constexpr array_ref(element_ptr p, I, Is...)// extensions_type<D> const& e) 
	noexcept : data_{p}{}
};*/

template<typename T, std::size_t N>
auto tail(std::array<T, N> a){
	std::array<T, N-1> ret;
	std::copy(a.begin() + 1, a.end(), ret.begin());
	return ret;
}

template<dimensionality_type D, dimensionality_type DM = D>
struct layout_t{
	layout_t<D-1, DM-1> sub;
	index stride_;
	index offset_;
	index nelems_;
	auto operator()(index i) const{return i*stride_ - offset_;}
	layout_t() : sub{}{}
	template<typename ExtList, typename = std::enable_if_t<!std::is_same<std::decay_t<ExtList>, layout_t>{}>>
	constexpr layout_t(ExtList&& e) : 
		sub{tail(e)}, 
		stride_{sub.size()*sub.stride_}, offset_{0}, nelems_{std::get<0>(e).size()*sub.nelems_} 
	{}
	bool operator==(layout_t const& other) const{
		return sub==other.sub && stride_==other.stride_ && offset_==other.offset_ && nelems_==other.nelems_;
	}
	bool operator!=(layout_t const& other) const{return not(*this==other);}
	friend bool operator!=(layout_t const& self, layout_t const& other){return not(self==other);}
	constexpr size_type num_elements() const{return nelems_;}
	constexpr size_type size() const{
		assert(stride_ != 0);
		assert(nelems_%stride_ == 0);
		return nelems_/stride_;
	}
	layout_t& rotate(){
		if constexpr(D != 1){
			using std::swap;
			swap(stride_, sub.stride_);
			swap(offset_, sub.offset_);
			swap(nelems_, sub.nelems_);
			sub.rotate();
		}
		return *this;
	}
	layout_t& rotate(dimensionality_type r){
		while(r){rotate(); --r;}
		return *this;
	}
	template<dimensionality_type DD = 0> 
	constexpr size_type size() const{
		if constexpr(DD == 0) return size();
		else return sub.template size<DD - 1>();
	}
	constexpr size_type size(dimensionality_type d) const{
		return d?sub.size(d-1):size();
	};
	constexpr index stride(dimensionality_type d = 0) const{
		return d?sub.stride(d-1):stride_;
	}
	constexpr index offset() const{return offset_;}
	auto sizes() const{
		std::array<size_type, D> ret;
		sizes_aux(ret.begin());
		return ret;
	}
	void sizes_aux(size_type* it) const{
		*it = size(); 
		sub.sizes_aux(++it);
	}
	auto strides() const{
		std::array<index, D> ret;
		strides_aux(begin(ret));
		return ret;
	}
	void strides_aux(size_type* it) const{
		*it = stride();
		sub.strides_aux(++it);
	}
	constexpr index_extension extension_aux() const{
		assert(stride_ != 0 and nelems_%stride_ == 0);
		return {offset_/stride_, (offset_ + nelems_)/stride_};
	}
	template<dimensionality_type DD = 0>
	constexpr index_extension extension(dimensionality_type d = DD) const{
		return d?sub.extension(d-1):extension_aux();
	}
	auto extensions() const{
		std::array<index_extension, D> ret;
		extensions_aux(ret.begin());
		return ret;
	}
	void extensions_aux(index_extension* it) const{
		*it = extension();
		sub.extensions_aux(++it);
	}
};

template<dimensionality_type DM>
struct layout_t<0, DM>{
	index stride_ = 1;
	index nelems_ = 1;
	index size_ = 1;
	layout_t() = default;
	template<class T>
	layout_t(std::array<T, 0>){}
	void sizes_aux(size_type*) const{}
	void strides_aux(size_type*) const{}
	void extensions_aux(index_extension*) const{};
	constexpr size_type num_elements() const{return 1;}
	template<dimensionality_type DD = 0>
	constexpr size_type size() const{return size_;}
	constexpr index stride(dimensionality_type d = 0) const{
		assert(d == -1);
		return stride_;
	}
	layout_t& rotate(){return *this;}
	bool operator==(layout_t const&) const{return true;}
	bool operator!=(layout_t const&) const{return false;}
	size_type size(dimensionality_type d) const{
		assert(d == -1);
		return -1;
	}
	index_extension extension(dimensionality_type d) const{
		assert(d == -1);
		return {};
	}
};

template<typename T, dimensionality_type D, class Alloc = std::allocator<T>>
class array;

template<
	typename T, 
	dimensionality_type D, 
	typename ElementPtr = T const* const, 
	class Layout = layout_t<D> const
> 
struct basic_array{
	using value_type = T;
	static constexpr dimensionality_type dimensionality = D;
	using element_ptr = ElementPtr;
	using layout_t = Layout;
// for at least up to 3D, ...layout const> is faster than ...layout const&>
	using const_reference = basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>>; 
	using reference = basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>>;
protected:
	element_ptr data_;
	layout_t layout_;
	basic_array() = delete;
public:
	basic_array(element_ptr data, layout_t layout) : data_(data), layout_(layout){}
	const_reference operator[](index i) const{return const_reference{&data_[layout_(i)], layout_.sub};}
	reference operator[](index i){return reference{data_ + layout_(i), layout_.sub};}
	auto range(index_range const& ir) const{
		auto new_layout = layout_; (new_layout.nelems_/=size())*=ir.size();
		return basic_array<T, D, ElementPtr, multi::layout_t<D>>{
			data_ + ir.front()*layout_.stride() - layout_.offset(), new_layout
		};
	}
	auto rotated(dimensionality_type i) const{
		auto new_layout = layout_; new_layout.rotate(i);
		return basic_array<T, D, ElementPtr, multi::layout_t<D>>{
			data_, new_layout
		};
	}
	decltype(auto) front(){return *begin();}
	decltype(auto) back(){return *(begin() + size() - 1);}
	class const_iterator : private const_reference{
		index stride_;
		const_iterator(const_reference const& cr, index stride) : const_reference{cr}, stride_{stride}{}
		friend struct basic_array;
		explicit operator bool() const{return this->data_;}
	public:
		using difference_type = index;
		using value_type = void;//std::string; // this will be array
		using pointer = void*;
		using reference = const_reference;
		using iterator_category = std::random_access_iterator_tag;
		const_iterator(std::nullptr_t = nullptr) : const_reference{}{}
		const_iterator& operator=(const_iterator const& other) = default;
		const_reference const& operator*() const{assert(operator bool()); return *this;}
		const_reference const* operator->() const{return static_cast<const_reference const*>(this);}
		const_reference operator[](index i) const{return {&this->data_[i*stride_], layout_.sub};}
		const_iterator& operator++(){this->data_ += stride_; return *this;}
		const_iterator& operator--(){this->data_ -= stride_; return *this;}
		const_iterator& operator+=(index d){this->data_ += stride_*d; return *this;}
		const_iterator& operator-=(index d){this->data_ -= stride_*d; return *this;}
		const_iterator operator+(index d){const_iterator ret = *this; return ret += d;}
		const_iterator operator-(index d){const_iterator ret = *this; return ret -= d;}
		ptrdiff_t operator-(const_iterator const& other) const{
			assert(stride_ != 0 and (this->data_ - other.data_)%stride_ == 0);
			assert(this->layout_ == other.layout_);
			return (this->data_ - other.data_)/stride_;
		}
		bool operator==(const_iterator const& other) const{
			return this->data_ == other.data_ and this->stride_ == other.stride_;
		}
		bool operator!=(const_iterator const& other) const{return not((*this)==other);}
	};
	struct iterator : basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>>{
		index stride_;
		iterator(basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>> const& cr, index stride) : basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>>{cr}, stride_{stride}{}
		friend struct basic_array;
		explicit operator bool() const{return this->data_;}
	public:
		using difference_type = index;
		using value_type = void;
		using pointer = void*;
		using reference_ = basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>>;
		iterator(std::nullptr_t = nullptr) : basic_array<T, D-1, ElementPtr, multi::layout_t<D-1>>{}{}
		using iterator_category = std::random_access_iterator_tag;
		bool operator==(iterator const& other) const{
			return this->data_ == other.data_ and this->stride_ == other.stride_;
		}
		bool operator!=(iterator const& other) const{return not((*this)==other);}
		reference& operator*() const{return *const_cast<iterator*>(this);}
		reference* operator->(){return static_cast<reference*>(this);}
		iterator& operator++(){this->data_ += stride_; return *this;}
		iterator& operator--(){this->data_ -= stride_; return *this;}
		iterator operator+(index d){iterator ret = *this; return ret += d;}
		iterator operator-(index d){iterator ret = *this; return ret -= d;}
		iterator& operator+=(index d){this->data_ += stride_*d; return *this;}
		iterator& operator-=(index d){this->data_ -= stride_*d; return *this;}
		ptrdiff_t operator-(iterator const& other) const{
			assert(stride_ != 0 and (this->data_ - other.data_)%stride_ == 0);
			assert(this->layout_ == other.layout_);
			return (this->data_ - other.data_)/stride_;
		}
	};
	template<dimensionality_type DD = 0> 
	size_type size() const{return layout_.template size<DD>();}
	constexpr size_type size(dimensionality_type d) const{return layout_.size(d);}
	constexpr size_type stride(dimensionality_type d = 0) const{return layout_.stride(d);}
	friend size_type size(basic_array const& self){return self.size();}
	auto sizes() const{return layout_.sizes();}
	auto strides() const{return layout_.strides();}
	auto extension() const{return layout_.extension();}
	friend auto sizes(basic_array const& self){return self.sizes();}
	friend auto strides(basic_array const& self){return self.strides();}
	friend auto extension(basic_array const& self){return self.extension();}
	auto extensions() const{return layout_.extensions();}
	friend auto extensions(basic_array const& self){return self.extensions();}
	template<dimensionality_type DD = 0>
	constexpr index_range extension(dimensionality_type d = DD) const{return layout_.extension(d);}
	const_iterator begin(index i) const{
		auto new_layout = layout_;
		new_layout.rotate(i);
		return const_iterator{
			const_reference{data_ + new_layout(0), new_layout.sub},
			new_layout.stride_
		};
	}
	const_iterator end(index i) const{
		auto new_layout = layout_;
		new_layout.rotate(i);
		return const_iterator{
			const_reference{&data_[new_layout(new_layout.size())], new_layout.sub},
			new_layout.stride_
		};
	}
	iterator begin(index i){
		auto new_layout = layout_;
		new_layout.rotate(i);
		return iterator{
			reference{&data_[new_layout(0)], new_layout.sub},
			new_layout.stride_
		};
	}
	iterator end(index i){
		auto new_layout = layout_;
		new_layout.rotate(i);
		return iterator{
			reference{&data_[new_layout(new_layout.size())], new_layout.sub},
			new_layout.stride_
		};
	}
	const_iterator begin()  const{return {operator[](0), layout_.stride_};}
	const_iterator end()    const{return {operator[](layout_.size()), layout_.stride_};}
	const_iterator cbegin()  const{return begin();}
	const_iterator cend()    const{return end();}
	iterator begin(){return {operator[](0), layout_.stride_};}
	iterator end()  {return {operator[](layout_.size()), layout_.stride_};}

	friend const_iterator begin(basic_array const& self){return self.begin();}
	friend const_iterator end(basic_array const& self){return self.end();}
	friend iterator begin(basic_array& self){return self.begin();}
	friend iterator end(basic_array& self){return self.end();}
	friend const_iterator cbegin(basic_array const& self){return self.begin();}
	friend const_iterator cend(basic_array const& self){return self.end();}
	size_type num_elements() const{return layout_.num_elements();}
	friend size_type num_elements(basic_array const& self){return self.num_elements();}
	template<class Array>
	bool operator<(Array const& other) const{
		using std::lexicographical_compare;
		using std::begin;
		using std::end;
		return lexicographical_compare(this->begin(), this->end(), begin(other), end(other)); // needs assignable iterator
	}
	template<class Array>
	bool operator<=(Array const& other) const{return (*this)==other or (*this)<other;}
	template<class Array>
	bool operator>(Array const& other) const{return not ((*this)<=other);}
	template<class Array>
	bool operator>=(Array const& other) const{return not ((*this)<other);}
};

template<class Array1, class Array2, typename = std::enable_if_t<Array1::dimensionality == Array2::dimensionality> >
bool operator==(Array1 const& self, Array2 const& other){
	if(self.extension() != other.extension()) return false;
	using std::equal;
	using std::begin;
	using std::end;
	return equal(begin(self), end(self), begin(other));
}
template<class Array1, class Array2>
bool operator!=(Array1 const& self, Array2 const& other){return not(self == other);}

template<class T, std::size_t N>
index_range extension(T(&)[N]){
	return {0, N};
}

template<typename T, typename ElementPtr, class Layout>
struct basic_array<T, dimensionality_type{1}, ElementPtr, Layout>{
	using value_type = T;
	using element_ptr = ElementPtr;
	using layout_t = Layout;
	using const_reference = T const&;
	using reference = decltype(*ElementPtr{});
protected:
	element_ptr data_;
	layout_t layout_;
	basic_array() = default;
	basic_array(element_ptr data, layout_t layout) : data_(data), layout_(layout){}
	friend struct basic_array<T, dimensionality_type{2}, ElementPtr, multi::layout_t<dimensionality_type{2}> >;
	friend struct basic_array<T, dimensionality_type{2}, ElementPtr, multi::layout_t<dimensionality_type{2}> const&>;
	friend struct basic_array<T, dimensionality_type{2}, ElementPtr, multi::layout_t<dimensionality_type{2}> const>;
public:
	const_reference operator[](index i) const{return data_[layout_(i)];}
	reference operator[](index i){return data_[layout_(i)];}
	class const_iterator{
		friend struct basic_array;
		const_iterator(element_ptr data, index stride) : data_(data), stride_(stride){}
	protected:
		element_ptr data_;
		index stride_;
	public:
		using difference_type = index;
		using value_type = std::string;
		using pointer = void*;
		using reference = const_reference;
		using iterator_category = std::random_access_iterator_tag;
		const_reference operator*() const{return *data_;}
		element_ptr const* operator->() const{return data_;}
		const_reference operator[](index i) const{return data_[i*stride_];}
		const_iterator& operator++(){data_ += stride_; return *this;}
		const_iterator& operator--(){data_ += stride_; return *this;}
		const_iterator& operator+=(ptrdiff_t d){this->data_ += stride_*d; return *this;}
		const_iterator operator+(ptrdiff_t d) const{return const_iterator(*this)+=d;}
		std::ptrdiff_t operator-(const_iterator const& other) const{
			assert(stride_ != 0 and (data_ - other.data_)%stride_ == 0);
			return (data_ - other.data_)/stride_;
		}
		bool operator==(const_iterator const& other) const{
			return data_ == other.data_ and stride_ == other.stride_;
		}
		bool operator!=(const_iterator const& other) const{return not((*this)==other);}
	};
	struct iterator : const_iterator{
		friend struct basic_array;
		using const_iterator::const_iterator;
		reference operator*() const{return *this->data_;}
		element_ptr* operator->() const{return this->data_;}
	};
	size_type num_elements() const{return layout_.num_elements();}
	friend size_type num_elements(basic_array const& self){return self.num_elements();}
	size_type size(dimensionality_type d = 0) const{return layout_.size(d);}
	friend auto size(basic_array const& self){return self.size();}
	index_range extension() const{return layout_.extension();}
	friend index_range extension(basic_array const& self){return self.extension();}
	const_iterator begin()  const{return const_iterator{data_ + layout_(0     ), layout_.stride_};}
	const_iterator end()    const{return const_iterator{data_ + layout_(size()), layout_.stride_};}
	iterator begin(){return iterator{data_ + layout_(0)     , layout_.stride_};}
	iterator end()  {return iterator{data_ + layout_(size()), layout_.stride_};}
	friend const_iterator begin(basic_array const& self){return self.begin();}
	friend const_iterator end(basic_array const& self){return self.end();}
	template<class Array>
	friend bool operator==(basic_array const& self, Array const& other){
	//	using multi::extension;
	//	auto& e = (index_range(*)(basic_array const&))&boost::multi::extension;
		using boost::multi::extension;
		if(self.extension() != extension(other)) return false;
		using std::equal;
		using std::begin;
		return equal(self.begin(), self.end(), begin(other));
	}
	template<class Array>
	bool operator!=(Array const& other) const{return not((*this)==other);}
	template<class Array>
	bool operator<(Array const& other) const{
		using std::lexicographical_compare;
		using std::begin;
		using std::end;
		return lexicographical_compare(this->begin(), this->end(), begin(other), end(other));
	}
	template<class Array>
	bool operator<=(Array const& other) const{return (*this)==other or (*this)<other;}
	template<class Array>
	bool operator>(Array const& other) const{return not ((*this)<=other);}
	template<class Array>
	bool operator>=(Array const& other) const{return not ((*this)<other);}
};

template<class Array1, class Array2>
void swap(Array1&& self, Array2&& other){
	assert(self.extension() == other.extension());
	using std::swap;
	for(auto i : self.extension()) swap(self[i], other[i]);
}

template<typename T, dimensionality_type D, typename ElementPtr = T const*>//, class Storage>
struct const_array_ref : basic_array<T, D, ElementPtr, layout_t<D>>{
	const_array_ref() = delete;
	const_array_ref(const_array_ref const&) = default;
//	template<class ExtList>
//	constexpr const_array_ref(typename const_array_ref::element_ptr p, ExtList&& e) noexcept : 
//	basic_array<T, D, ElementPtr, layout_t<D>>{p, typename const_array_ref::layout_t(std::forward<ExtList>(e))}{}
	constexpr const_array_ref(typename const_array_ref::element_ptr p, std::array<index_extension, D> const& e) noexcept : 
	basic_array<T, D, ElementPtr, layout_t<D>>{p, typename const_array_ref::layout_t{e}}{}
	typename const_array_ref::element_ptr  data() const{return cdata();}
	typename const_array_ref::element_ptr cdata() const{return this->data_;}
	friend typename const_array_ref::element_ptr  data(const_array_ref const& self){return self. data();}
	friend typename const_array_ref::element_ptr cdata(const_array_ref const& self){return self.cdata();}
};

template<typename T, dimensionality_type D, typename ElementPtr = T const*>
using array_cref = const_array_ref<T, D, ElementPtr>;

#if 0
template<typename T, typename ElementPtr>//, class Storage>
class array_ref<T, 2, ElementPtr> : public basic_array<T, 2, ElementPtr, layout_t<2> const>{//, Storage>{
	using element_ptr = typename array_ref::element_ptr;
	using basic_array<T, 2, ElementPtr, layout_t<2> const>::layout_;
//	element_ptr data_;// = nullptr;
//	layout_t<2> const layout_;

public:
	array_ref() = delete;
	array_ref(array_ref const&) = default;
	array_ref(array_ref&&) = delete;
//	template<class I>
//	constexpr array_ref(element_ptr p, I e0, I e1) noexcept : basic_array<T, 2, ElementPtr>{p, layout_t<2>(e0, e1)}{}
	template<typename ExtList>
	constexpr array_ref(element_ptr p, ExtList&& e) noexcept : basic_array<T, 2, ElementPtr>{p, layout_t<2>(std::forward<ExtList>(e))}{}
	constexpr array_ref(element_ptr p, std::array<index_extension, 2> const& e) noexcept : basic_array<T, 2, ElementPtr>{p, layout_t<2>(e)}{}
	class subarray{
		public:
		element_ptr data_;
		layout_t<1> const& layout_;
		friend class array_ref;
		public:
		subarray(element_ptr data, layout_t<1> const& lo) : data_{data}, layout_{lo}{}
		decltype(auto) operator[](index idx) const&{return data_[layout_(idx)];}
		class const_iterator{
			element_ptr data_;
			index stride_;
			const_iterator(std::nullptr_t n = nullptr) : data_(n){}
			const_iterator(element_ptr data, index stride) : data_{data}, stride_{stride}{}//, offsetp_{offsetp}{}
			friend class subarray;
		public:
			using value_type = std::string;
			decltype(auto) operator*() const{return *data_;}
			decltype(auto) operator->() const{return data_;}
			const_iterator& operator++(){data_ += stride_; return *this;}
			const_iterator& operator--(){data_ += stride_; return *this;}
			friend bool operator==(const_iterator const& self, const_iterator const& other){
				return self.data_ == other.data_ and self.stride_ == other.stride_;
			}
			friend bool operator!=(const_iterator const& self, const_iterator const& other){return not(self==other);}
		};
	};
	auto operator[](index idx) const{return subarray{&this->data_[layout_(idx)], layout_.sub};}
	class const_iterator : subarray{
		index stride_;
	//	const_iterator(subarray&& other, index const* stridep) : subarray(std::move(other)), stridep_{stridep}{}
		friend class array_ref;
	public:
		using value_type = std::string;
		subarray const& operator*() const{return *this;}
		subarray const* operator->() const{return static_cast<subarray const*>(this);}
		const_iterator& operator++(){this->data_ += stride_; return *this;}
		const_iterator& operator--(){this->data_ -= stride_; return *this;}
		subarray operator[](index i) const{
			return {&this->data_[i*stride_], this->layout_.sub};
		//	return {operator->()->data_ + i*(stride_), operator->()->stride0p_, operator->()->offset0p_, operator->()->size0p_};
		}
		friend bool operator==(const_iterator const& self, const_iterator const& other){
			return self->data_ == other->data_ and self.stride_ == other.stride_;
		}
		friend bool operator!=(const_iterator const& self, const_iterator const& other){return not(self==other);}
	};
	size_t size(){return layout_.size();}
	index_range extension(){return {0, layout_.size()};}
	const_iterator begin()  const{return {operator[](0), layout_.stride_};}
	const_iterator end()    const{
		assert(0);
		return {operator[](layout_.size()), layout_.stride_};
	}
//	const_iterator begin1() const{return {subarray{data_                 - offset1_, &stride0_, &offset1_, &size0_}, &stride1_};}
//	const_iterator end1()   const{return {subarray{data_+size1_*stride1_ - offset1_, &stride0_, &offset1_, &size0_}, &stride1_};}
};
#endif

}}

namespace boost{
namespace multi{

template<typename T, dimensionality_type D, typename ElementPtr = T*>
struct array_ref : const_array_ref<T, D, ElementPtr>{
	using const_array_ref<T, D, ElementPtr>::const_array_ref;
//	typename array_ref::element_ptr data(){return this->data_;}
	typename std::pointer_traits<typename array_ref::element_ptr>::template rebind<T const> cdata() const{return this->data_;}
	friend auto cdata(array_ref const& self){return self.cdata();}
};

}}



#if _TEST_BOOST_MULTI_ARRAY_REF

#include<cassert>
#include<numeric> // iota
#include<iostream>

using std::cout;
namespace bm = boost::multi;
#include <string_view>

int main(){

/*	std::string s = "hola";
	std::string_view sv = s;
	std::string s1 = "chau";
	sv = s1;
	std::cout << s;
	std::cout << sv;
*/

#if 0
	{
		double const d2D[4][5] = {{1.,2.},{2.,3.}};
		bm::array_ref<double, 2> d2Rce{&d2D[0][0], {4, 5}};
		assert( &d2Rce[2][3] == &d2D[2][3] );
		assert( d2Rce.size() == 4 );
	//	assert( d2Rce.size<0>() == 4);
	//	assert( d2Rce.size<1>() == 5);
		cout << d2Rce.num_elements() << std::endl;
		assert( d2Rce.num_elements() == 20 );
	}
#endif
	{
		double const dc2D[4][5] = {{1.,2.},{2.,3.}};
		bm::array_cref<double, 2> acrd2D{&dc2D[0][0], {4, 5}};
		assert( &acrd2D[2][3] == &dc2D[2][3] );
		assert( acrd2D.size() == 4);
		assert( acrd2D.size<0>() == acrd2D.size() );
		assert( acrd2D.size<1>() == 5);
		assert( acrd2D.num_elements() == 20 );

		bm::array_cref<double, 2> acrd2Dprime{acrd2D};
		assert( &acrd2D[2][3] == &dc2D[2][3] );
	
		assert( acrd2D.begin() == begin(acrd2D) );
		assert( acrd2D.begin() != acrd2D.end() );
	}
	return 0;
#if 0
	{
		double* d2p = new double[4*5]; std::iota(d2p, d2p + 4*5, 0);

		bm::array_ref<double, 2> d2R{d2p, 4, 5};
		assert(d2R.size()==4);
	}
	cout << "ddd " << d2R[1][1] << '\n';
	for(int i = 0; i != 4 ||!(cout << '\n'); ++i)
		for(int j = 0; j != 5 ||!(cout << '\n'); ++j)
			cout << d2R[i][j] << ' ';

	for(auto it1 = d2R.begin(); it1 != d2R.end() ||!(cout << '\n'); ++it1)
		for(auto it2 = it1->begin(); it2 != it1->end() ||!(cout << '\n'); ++it2)
			cout << *it2 << ' ';

	for(auto&& row : d2R){
		for(auto&& e : row) cout << e << ' '; cout << '\n';
	} cout << '\n';

	for(auto i : d2R.extension()){
		for(auto j : d2R[i].extension())
			cout << d2R[i][j] << ' ';
		cout << '\n';
	}
	cout << '\n';

	for(auto it1 = d2R.begin1(); it1 != d2R.end1() ||!(cout << '\n'); ++it1)
		for(auto it2 = it1->begin(); it2 != it1->end() ||!(cout << '\n'); ++it2)
			cout << *it2 << ' ';

	assert( d2R.begin()[1][1] == 6 );

	assert(d2R.size() == 4);
	auto it = d2R.begin();
	assert((*it)[1] == 1);
	assert( it->operator[](0) == 0 );
	assert( it->operator[](1) == 1 );
	++it;
	assert( it->operator[](0) == 5 );
	assert( it->operator[](1) == 6 );

	assert( *(it->begin()) == 5 );


#if 0
	if(double* d3p = new double[3*4*5]){
		std::iota(d3p, d3p + 3*4*5, 0);
		bm::array_ref<double, 3> d3R{d3p, 3, 4, 5};
		assert(d3R.size() == 3);
		for(int i = 0; i != 3; ++i, cout << '\n')
			for(int j = 0; j != 4; ++j, cout << '\n')
				for(int k = 0; k != 5; ++k)
					cout << d3R[i][j][k] << ' ';
		auto b = d3R.begin();
		
	}
#endif
#endif
}
#endif
#endif

