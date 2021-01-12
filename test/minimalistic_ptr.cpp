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
template<class T> struct ptr : std::iterator_traits<T*>{ // minimalistic pointer
	T* impl_;
	ptr(T* impl) : impl_{impl}{}
	typename ptr::reference operator*() const{return *impl_;}
	auto operator+(typename ptr::difference_type n) const{return ptr{impl_ + n};}
//	T& operator[](std::ptrdiff_t n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;
};

template<class T> struct ptr2 : std::iterator_traits<T*>{ // minimalistic pointer
	T* impl_;
	ptr2(T* impl) : impl_{impl}{}
	explicit ptr2(ptr<T> p) : impl_{p.impl_}{} 
	typename ptr2::reference operator*() const{return *impl_;}
	auto operator+(typename ptr2::difference_type n) const{return ptr2{impl_ + n};}
//	T& operator[](std::ptrdiff_t n) const{return impl_[n];} // optional
	using default_allocator_type = std::allocator<T>;
};

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

