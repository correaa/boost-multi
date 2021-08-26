#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
$CXXX $CXXFLAGS $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
#ifndef MULTI_LAYOUT_HPP
#define MULTI_LAYOUT_HPP
// © Alfredo A. Correa 2018-2020

#include "types.hpp"

#include "../detail/operators.hpp"
#include "../config/NODISCARD.hpp"
#include "../config/ASSERT.hpp"

#include<type_traits> // make_signed_t

#include<limits>

namespace boost{
namespace multi{

namespace detail{

template<typename T, typename... As>
inline constexpr void construct_from_initializer_list(T* p, As&&... as){
	::new(static_cast<void*>(p)) T(std::forward<As>(as)...);
}

template<class To, class From, size_t... I>
constexpr auto to_tuple_impl(std::initializer_list<From> il, std::index_sequence<I...>){
	(void)il;
	return std::make_tuple(To{il.begin()[I]}...);
}

//template<size_t N, class To, class From>
//constexpr auto to_tuple(std::initializer_list<From> il){
//	return il.size()==N?to_tuple_impl<To>(il, std::make_index_sequence<N>()):throw 0;
//}

template<class To, class From, size_t... I>
constexpr auto to_tuple_impl(std::array<From, sizeof...(I)> arr, std::index_sequence<I...>){
	return std::make_tuple(To{std::get<I>(arr)}...);
}

template<class To, size_t N, class From>
constexpr auto to_tuple(std::array<From, N> arr){
	return to_tuple_impl<To, From>(arr, std::make_index_sequence<N>());
}

template <class TT, class Tuple, std::size_t... I>
constexpr std::array<TT, std::tuple_size<std::decay_t<Tuple>>{}> to_array_impl(
	Tuple&& t, std::index_sequence<I...>
){
	return {static_cast<TT>(std::get<I>(std::forward<Tuple>(t)))...};
}
 
template<class T = void, class Tuple, class TT = std::conditional_t<std::is_same<T, void>{}, std::decay_t<decltype(std::get<0>(std::decay_t<Tuple>{}))>, T> >
constexpr std::array<TT, std::tuple_size<std::decay_t<Tuple>>{}> 
to_array(Tuple&& t){
	return 
		to_array_impl<TT>(
			std::forward<Tuple>(t),
			std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>{}>{}	
		)
	;
}

template <class Tuple, std::size_t... Ns>
constexpr auto tuple_tail_impl(Tuple&& t, std::index_sequence<Ns...>){
   return std::forward_as_tuple(std::forward<decltype(std::get<Ns + 1>(t))>(std::get<Ns + 1>(t))...);
}

template<class Tuple>
constexpr auto tuple_tail(Tuple&& t)
->decltype(tuple_tail_impl(t, std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>{} - 1>())){//std::tuple<Ts...> t){
	return tuple_tail_impl(t, std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>{} - 1>());}

}

}}

namespace boost{
namespace multi{

struct f_tag{};

template<dimensionality_type D, typename SSize=multi::size_type> struct layout_t;

template<dimensionality_type D> struct extensions_t;

template<> struct extensions_t<0> :
		std::tuple<>{
typedef std::tuple<> base_;
	static constexpr dimensionality_type dimensionality = 0;
	using nelems_type = index;
	using std::tuple<>::tuple;
	// cppcheck-suppress noExplicitConstructor ; why is it not taking the inheredited constructor above?
	extensions_t(std::tuple<> const& t) : std::tuple<>{t}{}
	extensions_t() = default;
	constexpr base_ const& base() const{return *this;}
	friend constexpr decltype(auto) base(extensions_t const& s){return s.base();}
	constexpr operator nelems_type() const{return 1;}
	template<class Archive> void serialize(Archive&, unsigned){}
	constexpr size_type num_elements() const{return 1;}
	constexpr std::tuple<> from_linear(nelems_type n) const{assert(n < num_elements()); (void)n;
		return {};
	}
	friend constexpr std::tuple<> operator%(nelems_type n, extensions_t const& s){return s.from_linear(n);}
	friend constexpr extensions_t intersection(extensions_t const&, extensions_t const&){return {};}
};

template<> struct extensions_t<1> : 
		std::tuple<multi::index_extension>{
typedef std::tuple<multi::index_extension> base_;
	static constexpr dimensionality_type dimensionality = 1;
	using nelems_type = index;
	using index_extension = multi::index_extension;
	using std::tuple<index_extension>::tuple;
	// cppcheck-suppress noExplicitConstructor ; I don't know why TODO
	constexpr extensions_t(index_extension const& ie) : base_{ie}{}
	extensions_t() = default;
	constexpr base_ const& base() const{return *this;}
	// cppcheck-suppress noExplicitConstructor ; I don't know why TODO
	constexpr extensions_t(base_ const& t) : std::tuple<index_extension>(t){}
	friend constexpr decltype(auto) base(extensions_t const& s){return s.base();}
	template<class Archive> void serialize(Archive& ar, unsigned){ar & multi::archive_traits<Archive>::make_nvp("extension", std::get<0>(*this));}
	constexpr size_type num_elements() const{return std::get<0>(*this).size();}
	constexpr std::tuple<multi::index> from_linear(nelems_type n) const{assert(n < num_elements());
		return std::tuple<multi::index>{n};
	}
	friend constexpr std::tuple<multi::index> operator%(nelems_type n, extensions_t const& s){return s.from_linear(n);}
	friend extensions_t intersection(extensions_t const& x1, extensions_t const& x2){
		return extensions_t(std::tuple<index_extension>(intersection(std::get<0>(x1), std::get<0>(x2))));
	}
};

template<dimensionality_type D>
struct extensions_t : 
		std::decay_t<decltype(std::tuple_cat(std::make_tuple(std::declval<index_extension>()), std::declval<typename extensions_t<D-1>::base_>()))>{
typedef std::decay_t<decltype(std::tuple_cat(std::make_tuple(std::declval<index_extension>()), std::declval<typename extensions_t<D-1>::base_>()))> base_;
	using base_::base_;
	static constexpr dimensionality_type dimensionality = D;
	extensions_t() = default;
	using nelems_type = multi::index;
	template<class Array, typename = decltype(std::get<D-1>(std::declval<Array>()))> 
	constexpr extensions_t(Array const& t) : extensions_t(t, std::make_index_sequence<static_cast<std::size_t>(D)>{}){}
	constexpr extensions_t(index_extension const& ie, typename layout_t<D-1>::extensions_type const& other) : extensions_t(std::tuple_cat(std::make_tuple(ie), other.base())){}
	constexpr base_ const& base() const{return *this;}
	friend constexpr decltype(auto) base(extensions_t const& s){return s.base();}
	friend constexpr typename layout_t<D + 1>::extensions_type operator*(index_extension const& ie, extensions_t const& self){
		return {std::tuple_cat(std::make_tuple(ie), self.base())};
	}
	constexpr explicit operator bool() const{return not layout_t<D>{*this}.empty();}
	template<class Archive, std::size_t... I>
	void serialize_impl(Archive& ar, std::index_sequence<I...>){
	//	using boost::serialization::make_nvp;
	//	(void)std::initializer_list<int>{(ar & make_nvp("extension", std::get<I>(*this)),0)...};
		(void)std::initializer_list<int>{(ar & multi::archive_traits<Archive>::make_nvp("extension", std::get<I>(*this)),0)...};
	//	(void)std::initializer_list<int>{(ar & boost::serialization::nvp<std::remove_reference_t<decltype(std::get<I>(*this))> >{"extension", std::get<I>(*this)},0)...};
	}
	constexpr auto from_linear(nelems_type n) const{
		auto const sub_extensions = extensions_t<D-1>(detail::tuple_tail(this->base()));
		auto const sub_num_elements = sub_extensions.num_elements();
		return std::tuple_cat(std::make_tuple(n/sub_num_elements), sub_extensions.from_linear(n%sub_num_elements));
	}
	friend constexpr auto operator%(nelems_type n, extensions_t const& s){return s.from_linear(n);}
	template<class Archive>
	void serialize(Archive& ar, unsigned){
		serialize_impl(ar, std::make_index_sequence<D>{});
	}
private:
	template<class Array, std::size_t... I, typename = decltype(base_{std::get<I>(std::declval<Array const&>())...})> 
	constexpr extensions_t(Array const& t, std::index_sequence<I...>) : base_{std::get<I>(t)...}{}
//	template<class T, std::size_t N, std::size_t... I> extensions_type_(std::array<T, N> const& t, std::index_sequence<I...>) : extensions_type_{std::get<I>(t)...}{}
	static constexpr size_type multiply_fold(){return 1;}
	static constexpr size_type multiply_fold(size_type const& a0){return a0;}
	template<class...As> static constexpr size_type multiply_fold(size_type const& a0, As const&...as){return a0*multiply_fold(as...);}
	template<std::size_t... I> constexpr size_type num_elements_impl(std::index_sequence<I...>) const{return multiply_fold(std::get<I>(*this).size()...);}
public:
	constexpr size_type num_elements() const{return num_elements_impl(std::make_index_sequence<D>{});}
	friend constexpr extensions_t intersection(extensions_t const& x1, extensions_t const& x2){
		return extensions_t(
			std::tuple_cat(
				std::tuple<index_extension>(intersection(std::get<0>(x1), std::get<0>(x2))),
				intersection( extensions_t<D-1>(detail::tuple_tail(x1.base())), extensions_t<D-1>(detail::tuple_tail(x2.base())) ).base()
			)
		);
	}
};

template<typename SSize>
struct layout_t<dimensionality_type{0}, SSize>{
	using size_type = SSize;
	using difference_type = std::make_signed_t<size_type>;
	using index_extension = multi::index_extension;
	using index = difference_type;
	using stride_type=index;
	using offset_type=index;
	using nelems_type=index;
	using index_range = multi::range<index>;
	using rank = std::integral_constant<dimensionality_type, 0>;
	static constexpr dimensionality_type dimensionality = 0;
	friend constexpr auto dimensionality(layout_t const& l){return l.dimensionality;}
	using strides_type  = std::tuple<>;
	using sizes_type    = std::tuple<>;
	nelems_type nelems_ = 1;//std::numeric_limits<nelems_type>::max(); // 1
	void* stride_ = nullptr;
	void* sub = nullptr;
	using extensions_type = extensions_t<0>;
	constexpr layout_t(extensions_type const& = {}){}// : nelems_{1}{}
	constexpr extensions_type extensions() const{return extensions_type{};}
	friend constexpr auto extensions(layout_t const& self){return self.extensions();}
	constexpr auto sizes() const{return std::tuple<>{};}
	constexpr bool is_empty() const{return false;}
	friend constexpr auto sizes(layout_t const& s){return s.sizes();}
	constexpr nelems_type num_elements() const{return 1;}//nelems_;}
	constexpr bool operator==(layout_t const&) const{return true ;}
	constexpr bool operator!=(layout_t const&) const{return false;}
};

template<typename SSize>
struct layout_t<dimensionality_type{1}, SSize>{
	using size_type=SSize; 
	using difference_type=std::make_signed_t<size_type>;
	using index_extension = multi::index_extension;
	using index = difference_type;
	using stride_type=index; 
	using offset_type=index; 
	using nelems_type=index;
	using index_range = multi::range<index>;
	using rank = std::integral_constant<dimensionality_type, 1>;
	static constexpr dimensionality_type dimensionality = rank{};
	friend constexpr auto dimensionality(layout_t const& l){return l.dimensionality;}
	using sub_t = layout_t<dimensionality_type{0}, SSize>;
protected:
public:
	stride_type stride_ = 1;//std::numeric_limits<stride_type>::max(); 
	offset_type offset_ = 0; 
	nelems_type nelems_ = 0;
	using extensions_type = extensions_t<1>;
	using strides_type = std::tuple<stride_type>;
	using sizes_type = std::tuple<size_type>;
	layout_t() = default;
	layout_t(layout_t const&) = default;
	constexpr layout_t(index_extension ie, layout_t<0> const&) :
		stride_{1},
		offset_{ie.first()},
		nelems_{
//			ie.size()<=1?ie.size()*std::numeric_limits<stride_type>::max():ie.size()
			ie.size()
		}{}
	constexpr explicit layout_t(extensions_type e) : layout_t(std::get<0>(e), {}){}
	constexpr layout_t(stride_type stride, offset_type offset, nelems_type nelems) : 
		stride_{stride}, offset_{offset}, nelems_{nelems}
	{}
	constexpr auto offset() const{return offset_;}
	friend constexpr index offset(layout_t const& self){return self.offset();}
	constexpr auto offset(dimensionality_type d) const{assert(d==0); (void)d; return offset_;}
	constexpr auto nelems() const{return nelems_;}
	constexpr auto nelems(dimensionality_type d) const{return d==0?nelems_:throw 0;}
	friend constexpr auto nelems(layout_t const& self){return self.nelems();}
	constexpr size_type size() const{//assert(stride_!=0 and nelems_%stride_==0);
		MULTI_ACCESS_ASSERT(stride_);
		return nelems_/stride_;
	}
	friend constexpr size_type size(layout_t const& self){return self.size();}
	constexpr size_type size(dimensionality_type d) const{
		return d==0?nelems_/stride_:throw 0; // assert(d == 0 and stride_ != 0 and nelems_%stride_ == 0);
	}
	constexpr layout_t& reindex(index i){offset_ = i*stride_; return *this;}
	constexpr auto base_size() const{return nelems_;}
	       constexpr auto is_compact() const{return base_size() == num_elements();}
	friend constexpr auto is_compact(layout_t const& self){return self.is_compact();}
public:
	constexpr auto stride(dimensionality_type d = 0) const{assert(!d); (void)d; return stride_;}
	friend constexpr index stride(layout_t const& self){return self.stride();}
public:
	constexpr auto strides() const{return std::make_tuple(stride());}
	friend constexpr auto strides(layout_t const& self){return self.strides();}
	constexpr auto sizes() const{return std::make_tuple(size());}
	template<class T=void> constexpr auto sizes_as() const{return detail::to_array<T>(sizes());}
	constexpr auto offsets() const{return std::make_tuple(offset());}
	constexpr auto nelemss() const{return std::make_tuple(nelems_);}
	constexpr size_type num_elements() const{return this->size();}
	friend constexpr size_type num_elements(layout_t const& s){return s.num_elements();}
//	constexpr bool empty() const{return nelems_ == 0;}
//	friend constexpr bool empty(layout_t const& s){return s.empty();}
	       constexpr bool is_empty()        const    {return not nelems_;}
	friend constexpr bool is_empty(layout_t const& s){return s.is_empty();}
	[[deprecated("use ::is_empty()")]]
	       constexpr bool    empty()        const    {return is_empty();}

	constexpr index_extension extension()        const&{
		assert(stride_);
		return {offset_/stride_, (offset_+nelems_)/stride_};
	} friend
	constexpr index_extension extension(layout_t const& s){return s.extension();}
	constexpr auto extension(dimensionality_type d) const{
		assert(stride_);
		return d==0?index_extension{offset_/stride_, (offset_ + nelems_)/stride_}:throw 0;
	}
	constexpr extensions_type extensions() const{return extensions_type{extension()};}//std::make_tuple(extension());}
	friend constexpr extensions_type extensions(layout_t const& self){return self.extensions();}
private:
	friend struct layout_t<2u>;
	void constexpr strides_aux(size_type* it) const{*it = stride();}
	void constexpr sizes_aux(size_type* it) const{*it = size();}
	void constexpr offsets_aux(index* it) const{*it = offset();}
	void constexpr extensions_aux(index_extension* it) const{*it = extension();}
public:
	constexpr auto operator()(index i) const{return i*stride_ - offset_;}
	constexpr std::ptrdiff_t at(index i) const{return offset_ + i*stride_;}
	constexpr std::ptrdiff_t operator[](index i) const{return at(i);}
	constexpr auto origin() const{return -offset_;}
	constexpr bool operator==(layout_t const& other) const{
		return stride_==other.stride_ and offset_==other.offset_ and nelems_==other.nelems_;
	}
	constexpr bool operator!=(layout_t const& other) const{return not(*this==other);}
	template<typename Size>
	constexpr layout_t& partition(Size const&){return *this;}
	constexpr layout_t&   rotate(dimensionality_type = 1){return *this;}
	constexpr layout_t& unrotate(dimensionality_type = 1){return *this;}
	constexpr layout_t scale(size_type s) const{return {stride_*s, offset_*s, nelems_*s};}
	constexpr layout_t& reverse(){return *this;}
};

inline constexpr typename layout_t<1>::extensions_type operator*(layout_t<0>::index_extension const& ie, layout_t<0>::extensions_type const&){
	return typename layout_t<1>::extensions_type{std::make_tuple(ie)};
}

template<dimensionality_type D, typename SSize>
struct layout_t : multi::equality_comparable2<layout_t<D>, void>{
	using dimensionality_type = multi::dimensionality_type;
	using rank = std::integral_constant<dimensionality_type, D>;
	static constexpr dimensionality_type dimensionality(){return rank{};}
	friend constexpr dimensionality_type dimensionality(layout_t const& l){
		return l.dimensionality();
	}
	using sub_type = layout_t<D-1>;
	using size_type = multi::size_type;
	using index = multi::index;
	using difference_type = multi::difference_type;
	using index_extension = multi::index_extension;
	using index_range = multi::range<index>;
	using stride_type = index;
	using offset_type = index;
	using nelems_type = index;
 	sub_type    sub_ = {};
	stride_type stride_ = 1;//std::numeric_limits<stride_type>::max();
	offset_type offset_ = 0;
	nelems_type nelems_ = 0;
	using extensions_type = extensions_t<D>;
	using strides_type    = decltype(tuple_cat(std::make_tuple(std::declval<index>()), std::declval<typename sub_type::strides_type>()));
	using sizes_type      = decltype(tuple_cat(std::make_tuple(std::declval<size_type>()), std::declval<typename sub_type::sizes_type>()));
//	using extensions_type = typename detail::repeat<index_extension, D>::type;
//	using extensions_io_type = std::array<index_extension, D>;
	constexpr auto operator()(index i) const{return i*stride_ - offset_;}
	constexpr auto origin() const{return sub_.origin() - offset_;}
	constexpr sub_type at(index i) const{//assert( this->extension().contains(i) ); see why it gives false positives
		auto ret = sub_;
		ret.offset_ += offset_ + i*stride_;
		return ret;
	}
	constexpr sub_type operator[](index i) const{return at(i);}
	constexpr layout_t(
		sub_type sub, stride_type stride, offset_type offset, nelems_type nelems
	) : sub_{sub}, stride_{stride}, offset_{offset}, nelems_{nelems}{}
	layout_t() = default;
	constexpr explicit layout_t(extensions_type const& e) :
		sub_{detail::tail(e)}, 
		stride_{sub_.size()*sub_.stride()},//std::get<0>(e).size()*sub_.num_elements()!=0?sub_.size()*sub_.stride():1}, 
		offset_{std::get<0>(e).first()*stride_}, //sub_.offset_ + std::get<0>(e).first()*sub_.stride()}, //sub_.stride()  offset_ = i*stride_}, 
		nelems_{std::get<0>(e).size()*sub_.num_elements()} 
	{}
	template<class StdArray, typename = std::enable_if_t<std::is_same<StdArray, std::array<index_extension, static_cast<std::size_t>(D)>>{}> >
	constexpr explicit layout_t(StdArray const& e) : 
		sub_{detail::tail(e)}, 
		stride_{1},//std::get<0>(e).size()*sub_.num_elements()!=0?sub_.size()*sub_.stride():1}, 
		offset_{sub_.offset_ + std::get<0>(e).first()*sub_.stride()}, 
		nelems_{std::get<0>(e).size()*sub_.num_elements()}
	{}
	       constexpr auto nelems()        const&    -> index{return   nelems_;}
	friend constexpr auto nelems(layout_t const& s) -> index{return s.nelems();}

	constexpr auto nelems(dimensionality_type d) const{return (d!=0)?sub_.nelems(d-1):nelems_;}

public:
	constexpr auto operator!=(layout_t const& o) const -> bool{return not((*this)==o);}
	constexpr auto operator==(layout_t const& o) const -> bool{
		return sub_==o.sub_ and stride_==o.stride_ and offset_==o.offset_ and nelems_==o.nelems_;
	}

	constexpr auto reindex(index i) -> layout_t&{offset_ = i*stride_; return *this;}
	template<class... Idx>
	constexpr auto reindex(index i, Idx... is) -> layout_t&{reindex(i).rotate().reindex(is...).unrotate(); return *this;}

	       constexpr auto num_elements()        const&    -> size_type{return size()*sub_.num_elements();}
	friend constexpr auto num_elements(layout_t const& s) -> size_type{return s.num_elements();}

	       constexpr auto is_empty()        const     -> bool{return nelems_ == 0;}
	friend constexpr auto is_empty(layout_t const& s) -> bool{return s.is_empty();}

	NODISCARD(".empty() means .is_empty()")
	       constexpr auto    empty()        const -> bool{return is_empty();}

	friend constexpr auto size(layout_t const& l) -> size_type{return l.size();}
	       constexpr auto size()        const&    -> size_type{
		if(nelems_ == 0){return 0;}
		MULTI_ACCESS_ASSERT(stride_); // NOLINT(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay) : normal in a constexpr function
		return nelems_/stride_;
	}

	       constexpr auto size(dimensionality_type d) const -> size_type{return (d!=0)?sub_.size(d-1):size();}

	constexpr auto stride() const -> index{return stride_;}

	       constexpr auto stride(dimensionality_type d) const& -> index{return (d!=0)?sub_.stride(d-1):stride();}
	friend constexpr auto stride(layout_t const& s) -> index{return s.stride();}

	       constexpr auto strides()        const&    -> strides_type{return tuple_cat(std::make_tuple(stride()), sub_.strides());}
	friend constexpr auto strides(layout_t const& s) -> strides_type{return s.strides();}

	constexpr auto offset(dimensionality_type d) const -> index{return (d!=0)?sub_.offset(d-1):offset_;}
	       constexpr auto offset() const -> index {return offset_;}
	friend constexpr auto offset(layout_t const& self) -> index{return self.offset();}
	constexpr auto offsets() const{return tuple_cat(std::make_tuple(offset()), sub_.offsets());}
	constexpr auto nelemss() const{return tuple_cat(std::make_tuple(nelems()), sub_.nelemss());}

	constexpr auto base_size() const{using std::max; return max(nelems_, sub_.base_size());}

	       constexpr auto is_compact()        const&   {return base_size() == num_elements();}
	friend constexpr auto is_compact(layout_t const& s){return s.is_compact();}

	       constexpr auto shape()        const&    -> decltype(auto){return   sizes();}
	friend constexpr auto shape(layout_t const& s) -> decltype(auto){return s.shape();}

	constexpr auto sizes() const{return tuple_cat(std::make_tuple(size()), sub_.sizes());}
	template<class T = void>
	constexpr auto sizes_as() const{return detail::to_array<T>(sizes());}

	friend constexpr auto extension(layout_t const& s) -> index_extension{return s.extension();}
	       constexpr auto extension()        const&    -> index_extension{
		if(nelems_ == 0){return {};}
		assert(stride_); 
		return {offset_/stride_, (offset_ + nelems_)/stride_};
	}

	constexpr auto extension_aux() const -> index_extension{
		assert(stride_ != 0 and nelems_%stride_ == 0);
		return {offset_/stride_, (offset_ + nelems_)/stride_};
	}
	template<dimensionality_type DD = 0>
	constexpr auto extension(dimensionality_type d) const -> index_extension{return d?sub_.extension(d-1):extension();}
	constexpr auto extensions() const -> extensions_type{return tuple_cat(std::make_tuple(extension()), sub_.extensions().base());}
	friend constexpr auto extensions(layout_t const& self) -> extensions_type{return self.extensions();}
//	constexpr void extensions_aux(index_extension* it) const{
//		*it = extension();
//		++it;
//		sub_.extensions_aux(it);
//	}
	template<typename Size>
	constexpr auto partition(Size const& s) -> layout_t&{
		using std::swap;
		stride_ *= s;
	//	offset
		nelems_ *= s;
		sub_.partition(s);
		return *this;
	}
	constexpr auto transpose() -> layout_t&{
		using std::swap;
		swap(stride_, sub_.stride_);
		swap(offset_, sub_.offset_);
		swap(nelems_, sub_.nelems_);
		return *this;
	}
	constexpr auto reverse() -> layout_t&{
		unrotate();
		sub_.reverse();
		return *this;
	}
	constexpr auto   rotate() -> layout_t&{transpose(); sub_.  rotate(); return *this;}
	constexpr auto unrotate() -> layout_t&{sub_.unrotate(); transpose(); return *this;}

	constexpr auto   rotate(dimensionality_type r) -> layout_t&{
		if(r >= 0){
			while(r != 0){rotate(); --r;}
		}else{
			return rotate(D - r);
		}
		return *this;
	}
	constexpr auto unrotate(dimensionality_type r) -> layout_t&{
		if(r >= 0){
			while(r != 0){unrotate(); --r;}
		}else{
			return unrotate(D-r);
		}
		return *this;
	}
	constexpr auto scale(size_type s) const{
		return layout_t{sub_.scale(s), stride_*s, offset_*s, nelems_*s};
	}
};

inline constexpr auto operator*(layout_t<1>::index_extension const& ie, layout_t<1>::extensions_type const& self){
	return layout_t<2>::extensions_type(ie, self);
}

//template<dimensionality_type D>
//using extensions_type_ = typename layout_t<D>::extensions_type;

template<class T, class Layout>
constexpr auto sizes_as(Layout const& self)
->decltype(self.template sizes_as<T>()){
	return self.template sizes_as<T>();}

} // end namespace multi
} // end namespace boost

namespace std{
	template<> struct tuple_size<boost::multi::extensions_t<0>> : std::integral_constant<boost::multi::dimensionality_type, 0>{};
	template<> struct tuple_size<boost::multi::extensions_t<1>> : std::integral_constant<boost::multi::dimensionality_type, 1>{};
	template<> struct tuple_size<boost::multi::extensions_t<2>> : std::integral_constant<boost::multi::dimensionality_type, 2>{};
	template<> struct tuple_size<boost::multi::extensions_t<3>> : std::integral_constant<boost::multi::dimensionality_type, 3>{};
	template<> struct tuple_size<boost::multi::extensions_t<4>> : std::integral_constant<boost::multi::dimensionality_type, 4>{};
} // end namespace std

#endif

