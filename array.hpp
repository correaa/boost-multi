#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ -O3 `#-fconcepts` -std=c++17 -Wall -Wextra `#-fmax-errors=2` `#-Wfatal-errors` -lboost_timer -I${HOME}/prj -D_TEST_BOOST_MULTI_ARRAY -rdynamic -ldl $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_ARRAY_HPP
#define BOOST_MULTI_ARRAY_HPP

#include "../multi/array_ref.hpp"
#include<iostream> // cerr
#include<algorithm>

namespace boost{
namespace multi{

using std::cerr;

template<class T, dimensionality_type D, class Allocator>
class array : 
	public array_ref<T, D, typename std::allocator_traits<Allocator>::pointer>
{
	using extensions_type = std::array<index_extension, D>;
public:
	using allocator_type = Allocator;
	allocator_type allocator_;
private:
	using alloc_traits = std::allocator_traits<allocator_type>;
public:
	template<class Array, typename = decltype(std::declval<Array>().extensions())>
	array(Array&& other) : array(other.extensions()){
		array::operator=(std::forward<Array>(other));
	}
	array(extensions_type e = {}, Allocator alloc = Allocator{}) : 
		array_ref<T, D, typename array::element_ptr>(nullptr, e),
		allocator_(std::move(alloc))
	{
		this->data_ = alloc_traits::allocate(allocator_, array::num_elements());
		uninitialized_construct();
	}
	array& operator=(array const& other){
		array tmp(other.extensions());
		for(auto i : tmp.extension()) tmp[i] = other[i];
		swap(tmp);
		return *this;
	}
	template<class Array>
	array& operator=(Array const& a){
		array tmp(a.extensions());
		tmp.array_ref<T, D, typename std::allocator_traits<Allocator>::pointer>::operator=(a);
		swap(tmp);
		return *this;
	}
	void swap(array& other) noexcept{
		using std::swap;
		swap(this->data_, other.data_);
		using layout_t = std::decay_t<decltype(this->layout())>;
		swap(
			static_cast<layout_t&>(*this), 
			static_cast<layout_t&>(other)
		);
	}
	friend void swap(array& self, array& other){self.swap(other);}
	//! Destructs the array
	~array(){
		destroy();
		allocator_.deallocate(this->data(), this->num_elements());
	}
private:
	void destroy(){
	//	destroy(this->data() + this->num_elements());
	//	for(auto cur = this->data(); cur != last; ++cur)
	///		alloc_traits::destroy(allocator_, std::addressof(*cur));
		using std::for_each;
		for_each(
			this->data(), this->data() + this->num_elements(), 
			[&](auto&& e){alloc_traits::destroy(allocator_, std::addressof(e));}
		);
	}
	void destroy(typename array::element_ptr last){
		for(auto cur = this->data(); cur != last; ++cur)
			alloc_traits::destroy(allocator_, std::addressof(*cur));
	}
	template<class... Args>
	auto uninitialized_construct(Args&&... args){
		typename array::element_ptr cur = this->data();
		try{
			using std::for_each;
			for_each(
				this->data(), this->data() + this->num_elements(), 
				[&](auto&& e){
					alloc_traits::construct(
						allocator_, std::addressof(e), 
						std::forward<Args>(args)...
					);
				}
			);
		//	for(size_type n = this->num_elements(); n > 0; --n, ++cur)
		//		alloc_traits::construct(allocator_, std::addressof(*cur), std::forward<Args>(args)...);
		//	return cur;
		}catch(...){destroy(cur); throw;}
	}
};

//template<class Array1, class Array2>
//void swap(Array1& a1, Array2& a2){a1.swap(a2);}

}}


#if _TEST_BOOST_MULTI_ARRAY

#include<cassert>
#include<numeric> // iota
#include<iostream>

using std::cout;
namespace multi = boost::multi;

int main(){

	multi::array<double, 2> MA({2, 3});
	MA[1][1] = 11.;
	assert( MA[1][1] == 11.);
	multi::array<double, 2> MA2({4, 5});
	using std::swap;
	swap(MA, MA2);
	assert( MA.size() == 4 );
	assert( MA2.size() == 2 );
	cout << MA2[1][1] << std::endl;
	assert( MA2[1][1] == 11. );
	multi::array<double, 2> MA3 = MA2;//({2, 3});
//	MA3 = MA2;
	cout << MA3[1][1] << std::endl;
	assert(MA3[1][1] == 11.);
#if 0
	multi::array<double, 2> MAswap({4, 5});
	multi::array<double, 1> MA1({3});
	using std::swap;
	swap(MA, MAswap);
	assert(MA[2][2] == 0.);
	MA[1][3] = 7.1;
	assert(MA[1][3] == 7.1);
	cout << MA.stride() << '\n';	
	cout << MA.strides()[0] << '\n';
	cout << MA.strides()[1] << '\n';
	cout << "s = " << MA.size() << std::endl;
	assert( MA.size() == 4 );
	assert( MA.size(0) == 4 );
	assert( MA.size(1) == 5 );

	double d2D[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	multi::array_ref<double, 2> d2D_ref{&d2D[0][0], {4, 5}};
	d2D_ref[1][1] = 8.1;
	multi::array<double, 2> MA2;
	MA2 = d2D_ref;
	d2D_ref[1][1] = 8.2;
	assert(MA2.extensions() == d2D_ref.extensions());
	cout << MA2[1][1] << std::endl;
	assert(MA2[1][1] == 8.2);
#endif
}
#endif
#endif

