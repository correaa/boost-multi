#ifdef COMPILATION_INSTRUCTIONS
$CXX -O3 -std=c++14 -Wall -Wextra -Wpedantic `#-Wfatal-errors` $0 -o $0.x && time $0.x $@ && rm -f $0.x; exit
#endif

//#include<boost/operators.hpp>

#include<iostream>
#include<cassert>

namespace operators{

template<class Self, class D>
struct addable{
	Self& operator++(){return static_cast<Self&>(*this)+=D{1};} // ++t
	template<class Self2>
	friend Self next(Self2&& s){Self ret{std::forward<Self2>(s)}; return ++ret;} 
};

}

namespace fancy{

template<class T> struct ref;

template<class T = void> class ptr{
	static double value;
public:
	using difference_type = std::ptrdiff_t;
	using value_type = T;
//	using element_type = T;
	using pointer = T*;
	using reference = ref<T>;
	using iterator_category = std::random_access_iterator_tag;
	reference operator*() const{return reference{&value};}
	ptr operator+(difference_type) const{return *this;}
	ptr& operator+=(difference_type){return *this;}
	ptr& operator++(){return operator+=(1);}
	ptr& operator=(std::nullptr_t){return *this;}
	friend difference_type operator-(ptr const&, ptr const&){return 0;}
	bool operator==(ptr const&) const{return true;}
	bool operator!=(ptr const&) const{return false;}
	explicit operator T*() const{return &value;}
	friend ptr to_address(ptr const& p){return p;}
};
template<> double ptr<double>::value = 42.;

template<class T> struct ref{
private:
	T* p_;
	ref(T* p) : p_{p}{}
	friend class ptr<T>;
public:
	bool operator==(ref const&) const{std::cout << "compared" << std::endl; return true;}
	bool operator!=(ref const&) const{return false;}
};

#if 0
template<> struct ptr<void>{ // minimal fancy ptr
	static double value;
	using difference_type = std::ptrdiff_t;
	using value_type = void;
	using element_type = void;
	using pointer = void*;
	using reference = void;
	using iterator_category = std::random_access_iterator_tag;
	double& operator*() const{return value;}
	ptr operator+(difference_type) const{return *this;}
	ptr& operator+=(difference_type){return *this;}
	friend difference_type operator-(ptr const&, ptr const&){return 0;}
	bool operator==(ptr const&) const{return true;}
};
double ptr<void>::value = 42.;
#endif

template<class T> struct allocator{
	using pointer = ptr<T>;
	using value_type = T;
	auto allocate(std::size_t){return pointer{};}
	void deallocate(pointer, std::size_t){}
	std::true_type operator==(allocator const&){return {};}
	allocator(){}
	template<class T2> allocator(allocator<T2> const&){}
	template<class... Args>
	void construct(pointer, Args&&...){}
	void destroy(pointer){}
};

// all these are optional, depending on the level of specialization needed
template<class Ptr, class T, class Size>
ptr<T> copy_n(Ptr, Size, ptr<T> d){ // custom copy_n, Boost.Multi uses copy_n
	std::cerr << "called Pointer-based copy_n(Ptr, n, fancy::ptr)" << std::endl; 
	return d;
}
template<class Ptr, class T, class Size>
Ptr copy_n(ptr<T>, Size, Ptr d){ // custom copy_n, Boost.Multi uses copy_n
	std::cerr << "called Pointer-based copy_n(fancy::ptr, n, Ptr)" << std::endl; 
	return d;
}
template<class T1, class T2, class Size>
ptr<T2> copy_n(ptr<T1>, Size, ptr<T2> d){ // custom copy_n, Boost.Multi uses copy_n
	std::cerr << "called Pointer-based copy_n(fancy::ptr, n, fancy::ptr)" << std::endl; 
	return d;
}

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// multi-fancy glue, where should this be? In boost/multi/adaptors/MyFancyApaptor.hpp if anything, or in user code if it is very special
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../array.hpp"

namespace boost{
namespace multi{

template<class It, class T>  // custom copy 1D (aka strided copy)
void copy(It first, It last, multi::array_iterator<T, 1, fancy::ptr<T> > dest){
	assert( stride(first) == stride(last) );
	std::cerr << "1D copy(it1D, it1D, it1D) with strides " << stride(first) << " " << stride(dest) << std::endl;
}

template<class It, class T> // custom copy 2D (aka double strided copy)
void copy(It first, It last, multi::array_iterator<T, 2, fancy::ptr<T> > dest){
	assert( stride(first) == stride(last) );
	std::cerr << "2D copy(It, It, it2D) with strides " << stride(first) << " " << stride(dest) << std::endl;
}

}}

////////////////////////////////////////////////////////////////////////////////
// user code
////////////////////////////////////////////////////////////////////////////////

using std::cout; using std::cerr;
namespace multi = boost::multi;

int main(){
	multi::array<double, 2, fancy::allocator<double> > A({5, 5});

	assert( A[1][1] == A[2][2] );

	multi::array<double, 2, fancy::allocator<double> > B = A;
	assert( A[1][1] == B[1][1] );

	multi::array<double, 2, fancy::allocator<double> > C({5, 5}, 42.);

}


