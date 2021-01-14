// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
#ifndef BOOST_MULTI_DETAIL_OPERATORS_HPP
#define BOOST_MULTI_DETAIL_OPERATORS_HPP

#include<type_traits> // enable_if
#include<utility> // forward

namespace boost{
namespace multi{

struct empty_base{};

template<class T, class V = void> struct equality_comparable2;

template<class T> using equality_comparable = equality_comparable2<T, T>;

template<class T>
struct equality_comparable2<T, T>{
	using self_type = T;
	friend bool operator!=(self_type const& y, self_type const& x){return not(y==x);}
};

template <class T>
struct equality_comparable2<T, void>{
	using self_type = T;
	self_type const& self() const&{return static_cast<self_type const&>(*this);}
	template<class U> constexpr bool operator!=(U const& other) const{return not(self()==other);}
//	template<class U, 
//		typename = std::enable_if_t<not std::is_base_of<self_type, U>{}>
//	>//, typename = std::enable_if_t<
//		(not std::is_base_of<self_type, U>{}) and (not std::is_convertible<U, self_type>{})
//	>>
//	friend constexpr bool operator==(U const& y, self_type const& x){
//		return x.operator==(y);
//	}
};

template<class U, class T, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
constexpr bool operator==(U const& other, equality_comparable2<T> const& self){
	return self.self() == other;
}
template<class U, class T, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
constexpr bool operator!=(U const& other, equality_comparable2<T> const& self){
	return self.self() != other;
}


template<class Self, 
	class      Iterator , class ConstIterator,
	class      Reference, class ConstReference
>
class container_interface{
	using self_type = Self;
	self_type const& self() const&{return static_cast<self_type const&>(*this);}
protected:
	using const_iterator = ConstIterator;
	using       iterator =      Iterator;
	
//		static_assert( std::is_convertible<Iterator, ConstIterator>::value, "!" );

	using const_reference = ConstReference;
	using       reference =      Reference;

public:
	constexpr const_iterator begin() const&{return self().begin_();}
	constexpr       iterator begin()      &{return self().begin_();}
	constexpr       iterator begin()     &&{return self().begin_();}

	template<class S, Self* =nullptr> friend constexpr auto begin(S&& s){return std::forward<S>(s).begin();}

	constexpr const_iterator end() const&{return self().end_();}
	constexpr       iterator end()      &{return self().end_();}
	constexpr       iterator end()     &&{return self().end_();}

	template<class S, Self* =nullptr> friend constexpr auto end(S&& s){return std::forward<S>(s).end();}

public:
	constexpr const_iterator cbegin() const{return begin();}
	constexpr const_iterator cend  () const{return end  ();}

	//                                                             vvvvv needs && to win over std::cbegin
	template<class S, Self* =nullptr> friend constexpr auto cbegin(S&& s){return s.cbegin();}
	template<class S, Self* =nullptr> friend constexpr auto cend  (S&& s){return s.cend();}

protected:
	using size_type = typename std::iterator_traits<iterator>::difference_type;
	constexpr size_type size() const{return std::distance(begin(), end());}
	friend constexpr size_type size(self_type const& s){return s.size();}
	
	using difference_type = typename std::iterator_traits<iterator>::difference_type;
	
	constexpr bool is_empty() const{return size();}
	friend constexpr size_type is_empty(self_type const& s){return s.is_empty();}

	decltype(auto) front() const&{return *self().begin();}
	decltype(auto) front()     &&{return *std::move(self()).begin();}
	decltype(auto) front()      &{return *self().begin();}
	
	template<class S, Self* =nullptr> friend constexpr decltype(auto) front(S&& s){return std::forward<S>(s).front();}

	decltype(auto) back() const&{return *std::prev(self().end());}
	decltype(auto) back()     &&{return *std::prev(std::move(self()).end());}
	decltype(auto) back()      &{return *std::prev(self().end());}
	
	template<class S, Self* = nullptr> friend constexpr decltype(auto) back(S&& s){return std::forward<S>(s).back();}

	constexpr const_reference operator[](index i) const&{return self().at_(i);}
	constexpr       reference operator[](index i)      &{return self().at_(i);}
	constexpr       reference operator[](index i)     &&{return self().at_(i);}
};

//template<class Self>
//class equality_comparable{
//	Self const& self() const{return static_cast<Self const&>(*this);}
//	static void check(Self const& s1, Self const& s2){assert( s1 == s2 or s1 != s2 );}
//public:
////	template<class SSelf, std::enable_if_t<std::is_base_of<Self, SSelf>{}>>
////	constexpr auto operator==(SSelf const& other) const
////	->decltype(not bool{self()!=other}){
////		return not bool{self()!=other};}
//	template<class SSelf, std::enable_if_t<std::is_base_of<Self, SSelf>{}>>
//	constexpr auto operator!=(SSelf const& other) const
//	->decltype(not bool{self()==other}){
//		return not bool{self()==other};}
//};

template<class T, class V> struct partially_ordered2;

template <class T>
struct partially_ordered2<T, void>{
	template<class U, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
	friend constexpr bool operator>(const U& x, const T& y){return y < x;}
	template<class U, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
	friend constexpr bool operator<(const U& x, const T& y){return y > x;}

	template<class U>
	friend constexpr bool operator<=(T&& x, U&& y){return (std::forward<T>(x) < std::forward<T>(y)) or (std::forward<T>(x) == std::forward<T>(y));}
	template<class U, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
	friend constexpr bool operator<=(const U& x, const T& y){return (y > x) or (y == x);}
	template<class U>
	friend constexpr bool operator>=(const T& x, const U& y){return (x > y) or (x == y);}
	template<class U, typename = std ::enable_if_t<not std::is_base_of<T, U>{}>>
	friend constexpr bool operator>=(const U& x, const T& y){return (y < x) or (y == x);}
};

template<class T, class V = void> struct totally_ordered2;

template<class T> using totally_ordered = totally_ordered2<T, T>;

template<class T>
struct totally_ordered2<T, void> : public equality_comparable2<T, void>{
	template<class U>
	friend constexpr auto operator<=(const T& x, const U& y){return (x < y) or (x == y);}
	template<class U>
	friend constexpr auto operator>=(const T& x, const U& y){return (y < x) or (x == y);}
	template<class U>
	friend constexpr auto operator>(const T& x, const U& y){return y < x;}
};

template<class T, class V = void> struct swappable2;

template<class T> using swappable = swappable2<T, T>;

template<class T>
struct swappable2<T, void>{
	template<class U> friend void swap(T&  t, U&& u){t.swap(std::forward<U>(u));}
	template<class U> friend void swap(T&& t, U&& u){t.swap(std::forward<U>(u));}

	template<class U, std::enable_if_t<not std::is_base_of<T, std::decay_t<U>>{}> > 
	friend void swap(U&& u, T&  t){t.swap(std::forward<U>(u));}

	template<class U, std::enable_if_t<not std::is_base_of<T, std::decay_t<U>>{}> > 
	friend void swap(U&& u, T&& t){t.swap(std::forward<U>(u));}
};

template<class T>
struct copy_constructible{};

template<class T>
struct weakly_incrementable{
//	using self_type = T;
//	self_type& self()&{return static_cast<self_type&>(*this);}
//	friend T& operator++(weakly_incrementable& t){return ++self();}
};

template<class T>
struct weakly_decrementable{
//	friend T& operator--(weakly_decrementable& t){return --static_cast<T&>(t);}
};

template<class T>
struct incrementable : weakly_incrementable<T>{
	template<class U, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
	friend constexpr T operator++(U& self, int){T tmp{self}; ++self; return tmp;}
};

template<class T>
struct decrementable : weakly_decrementable<T>{
	template<class U, typename = std::enable_if_t<not std::is_base_of<T, U>{}>>
	friend constexpr T operator--(U& self, int){T tmp{self}; --self; return tmp;}
};

template<class T>
struct steppable : incrementable<T>, decrementable<T>{};

template<class T, class Reference>
struct dereferenceable{
	using reference = Reference;
	friend constexpr reference operator*(dereferenceable const& t){return *static_cast<T const&>(t);}
};

template<class T, class D>
struct addable2{
	using difference_type = D;
//	template<class TT, typename = std::enable_if_t<std::is_base_of<T, TT>{}> >
//	friend constexpr T operator+(TT&& t, difference_type const& d){T tmp{std::forward<TT>(t)}; tmp+=d; return tmp;}
//	template<class TT, typename = std::enable_if_t<std::is_base_of<T, TT>{}> >
//	friend constexpr T operator+(difference_type const& d, TT&& t) {return std::forward<TT>(t) + d;}
};

template<class T, class D>
struct subtractable2{
	using difference_type = D;
//	template<class TT, class = T>
//	friend T operator-(TT&& t, difference_type const& d){T tmp{std::forward<TT>(t)}; tmp-=d; return tmp;}
};

template<class T, class Difference>
struct affine : addable2<T, Difference>, subtractable2<T, Difference>{
	using difference_type = Difference;
};

template<class T>
struct random_iterable{
	constexpr auto rbegin(){return typename T::reverse_iterator{static_cast<T&>(*this).end  ()};}
	constexpr auto rend  (){return typename T::reverse_iterator{static_cast<T&>(*this).begin()};}
	friend 
	auto rbegin(T& s){return static_cast<random_iterable&>(s).rbegin();}
	friend
	auto rend  (T& s){return static_cast<random_iterable&>(s).rend  ();}

	constexpr decltype(auto) cfront() const{return static_cast<T const&>(*this).front();}
	constexpr decltype(auto) cback()  const{return static_cast<T const&>(*this).back() ;}
	friend constexpr auto cfront(T const& s){return s.cfront();}
	friend constexpr auto cback (T const& s){return s.cback() ;}
};

#if 0
// TODO random_iterable_container ??
template<class T, class B = empty_base>
struct random_iterable : B{
//	using iterator = Iterator;
///	template<typename U = T>
//	typename U::const_iterator
//	 cbegin() const{return typename T::const_iterator{static_cast<T const&>(*this).begin()};}
//	template<typename U = T>
//	typename U::const_iterator
//	 cend() const{return typename T::const_iterator{static_cast<T const&>(*this).end()};}
//	template<typename U = T>
//	friend typename U::const_iterator cbegin(U const& s, T* = 0, ...){
//		return static_cast<random_iterable const&>(s).cbegin();}
//	template<typename U = T>
//	friend typename U::const_iterator cend  (U const& s, T* = 0, ...){
//		return static_cast<random_iterable const&>(s).cend  ();}
//	auto cend()   const{return typename T::const_iterator{static_cast<T const*>(this)->end()};}
//	template<class TT, typename = std::enable_if_t<std::is_base_of<T, TT>{}> >
//	friend auto cbegin(TT const& s)->typename TT::const_iterator
//	{return typename TT::const_iterator{static_cast<T const&>(s).begin()};}
//	template<class TT, typename = std::enable_if_t<std::is_base_of<T, TT>{}> >
//	friend auto cend  (TT const& s)->typename TT::const_iterator
//	{return typename TT::const_iterator{static_cast<T const&>(s)->end()};}

//	auto crbegin() const{return typename T::const_reverse_iterator{cend  ()};}
//	auto crend  () const{return typename T::const_reverse_iterator{cbegin()};}
//	friend auto crbegin(T const& s){return static_cast<random_iterable const&>(s).cbegin();}
//	friend auto crend  (T const& s){return static_cast<random_iterable const&>(s).cend  ();}

};
#endif

//template<class T, class B>
//typename T::const_iterator cbegin(random_iterable<T, B> const& c){return c.cbegin();}
//template<class T, class B>
//typename T::const_iterator cend(random_iterable<T, B> const& c){return c.cend();}

}}

#endif

