#ifdef COMPILATION_INSTRUCTIONS
(echo "#include\""$0"\"" > $0x.cpp) && c++ -O3 `#-fconcepts` -std=c++17 -Wall -Wextra `#-fmax-errors=2` `#-Wfatal-errors` -lboost_timer -I${HOME}/prj -D_TEST_BOOST_MULTI_ARRAY -rdynamic -ldl $0x.cpp -o $0x.x && time $0x.x $@ && rm -f $0x.x $0x.cpp; exit
#endif
#ifndef BOOST_MULTI_ARRAY_HPP
#define BOOST_MULTI_ARRAY_HPP

#include "../multi/array_ref.hpp"

namespace boost{
namespace multi{


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
	array(extensions_type e, Allocator alloc = Allocator{}) : 
		array_ref<T, D, typename array::element_ptr>(nullptr, e),
		allocator_(std::move(alloc))
	{
		this->data_ = alloc_traits::allocate(allocator_, array::num_elements());
		uninitialized_construct();
	}
	void swap(array& other){
		using std::swap;
		swap(this->data_, other.data_);
		swap(this->layout_, other.layout_);
	}
	//! Destructs the array
	~array(){
		destroy();
		allocator_.deallocate(this->data(), this->num_elements());
	}
private:
	void destroy(){destroy(this->data() + this->num_elements());}
	void destroy(typename array::element_ptr last){
		for(auto cur = this->data(); cur != last; ++cur)
			alloc_traits::destroy(allocator_, std::addressof(*cur));
	}
	template<class... Args>
	auto uninitialized_construct(Args&&... args){
		typename array::element_ptr cur = this->data();
		try{
			for(size_type n = this->num_elements(); n > 0; --n, ++cur)
				alloc_traits::construct(allocator_, std::addressof(*cur), std::forward<Args>(args)...);
			return cur;
		}catch(...){destroy(cur); throw;}
	}
};

template<class Array1, class Array2>
void swap(Array1& a1, Array2& a2){a1.swap(a2);}

}}


#if _TEST_BOOST_MULTI_ARRAY

#include<cassert>
#include<numeric> // iota
#include<iostream>

using std::cout;
namespace multi = boost::multi;

int main(){

	multi::array<double, 2> MAswap({4, 5});
	multi::array<double, 2> MA({3, 2});
	using std::swap;
	swap(MA, MAswap);
	assert(MA[2][2] == 0.);
	MA[1][3] = 7.1;
	assert(MA[1][3] == 7.1);
	cout << MA.stride() << '\n';	
	cout << MA.strides()[0] << '\n';
	cout << MA.strides()[1] << '\n';
	assert( MA.size() == 4 );
	assert( MA.size(0) == 4 );
	assert( MA.size(1) == 5 );

	double const d2D[4][5] {
		{ 0,  1,  2,  3,  4}, 
		{ 5,  6,  7,  8,  9}, 
		{10, 11, 12, 13, 14}, 
		{15, 16, 17, 18, 19}
	};
	multi::array_cref<double, 2> d2D_cref{&d2D[0][0], {4, 5}};
//	multi::array<double, 2> MA2 = d2D_cref;

}
#endif
#endif

