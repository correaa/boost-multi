#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4-*-
$CXX $0 -o $0x -lboost_unit_test_framework&&$0x&&rm $0x;exit
#endif
// © Alfredo A. Correa 2018-2020

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi minimalistic pointer"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../array.hpp"

namespace multi = boost::multi;

namespace minimalistic{

template<class T> class ptr : public std::iterator_traits<T*>{ // minimalistic pointer
	T* impl_;
public:
	constexpr explicit ptr(T* impl) : impl_{impl}{}
	constexpr typename ptr::reference operator*() const{return *impl_;}
	constexpr auto operator+(typename ptr::difference_type n) const {return ptr{impl_ + n};}
//	T& operator[](std::ptrdiff_t n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;
	template<class> friend class ptr2;
};

template<class T> class ptr2 : public std::iterator_traits<T*>{ // minimalistic pointer
	T* impl_;
	friend class ptr2<T const>;
public:
	ptr2() = default;
	ptr2(std::nullptr_t n) : impl_{n}{}
	explicit ptr2(T* impl) : impl_{impl}{}
	template<class Other, decltype(multi::implicit_cast<T*>(std::declval<Other const&>().impl_))* =nullptr>
	ptr2(Other const& other) : impl_{other.impl_}{}
	typename ptr2::reference operator*() const{return *impl_;}
	auto operator+(typename ptr2::difference_type n) const{return ptr2{impl_ + n};}
	template<class O> bool operator==(O const& o) const{return impl_ == o.impl_;}
	template<class O> bool operator!=(O const& o) const{return impl_ != o.impl_;}
//	T& operator[](std::ptrdiff_t n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;
};

static_assert(     std::is_convertible<ptr2<double>, ptr2<double const>>{}, "!");
static_assert( not std::is_convertible<ptr2<double const>, ptr2<double>>{}, "!");

}

struct X{
	int a1;
	double a2;
	double b;
};


template<class T, class Ptr = T*>
class span{ // https://en.cppreference.com/w/cpp/container/span
	Ptr ptr_;
	typename std::pointer_traits<Ptr>::difference_type length_;
public:
	using pointer = Ptr;
	using difference_type = typename std::pointer_traits<pointer>::difference_type;
	using size_type = difference_type;
	using value_type = std::remove_cv_t<T>;
	span(pointer first, size_type count) : ptr_{first}, length_{count}{}
	span(pointer first, pointer last) : ptr_{first}, length_{last - first}{}
	pointer data() const{return ptr_;}
	size_type size() const{return length_;}
};

BOOST_AUTO_TEST_CASE(test_minimalistic_ptr){

//	int X::* s = &X::a1; // gives warning
//	X x{1, 2.5, 1.2};
//	assert( x.*s == x.a1 );

//	X X::*ss = &X::X;
	double d = 5;
	minimalistic::ptr2<double> p{&d};
	minimalistic::ptr2<double const> pc = p;
	minimalistic::ptr2<double const> pc2{&d};
	pc = pc2;
	
	BOOST_REQUIRE( pc == p );

	double* buffer = new double[100];
	
	auto&& C = multi::ref(span<double>(buffer, 100)).partitioned(10);

	C[2]; // requires operator+ 

	buffer[10 + 1] = 99.;
	BOOST_REQUIRE( C[1][1] == 99. ); // C[1][1]; // requires operator*

	C[1][1] = 88.;
	BOOST_REQUIRE(C[1][1] == 88.);

	auto&& C2 = C.template static_array_cast<double, minimalistic::ptr2<double>>();
	BOOST_REQUIRE( &C2[1][1] == &C[1][1] );

	delete[] buffer;

}

