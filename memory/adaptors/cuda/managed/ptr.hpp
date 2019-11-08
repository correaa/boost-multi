#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc -Xcompiler -Wfatal-errors -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_MANAGED_PTR $0.cpp -o $0x &&$0x&&rm $0x; exit
#endif

#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_MANAGED_PTR_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_MANAGED_PTR_HPP

//#include "../../../adaptors/cuda/clib.hpp"

#include<cassert>
#include<cstddef> // nullptr_t
#include<iterator> // random_access_iterator_tag

#include<type_traits> // is_const

#include "../../cuda/ptr.hpp"

#ifndef _DISABLE_CUDA_SLOW
#define SLOW deprecated("WARNING: implies a slow access to GPU memory") 
#else
#define SLOW
#endif

#ifndef HD
#ifdef __CUDA_ARCH__
#define HD __host__ __device__
#else
#define HD
#endif
#endif

namespace boost{namespace multi{
namespace memory{namespace cuda{

namespace managed{
//template<class T> struct ref;

template<typename T, typename Ptr = T*> struct ptr;

template<typename RawPtr>
struct ptr<void const, RawPtr>{
	using T = void const;
	using raw_pointer = RawPtr;
	raw_pointer rp_;
	template<typename, typename> friend struct ptr;
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
	ptr(raw_pointer rp) : rp_{rp}{}
public:
	ptr() = default;
	ptr(ptr const&) = default;
	ptr(std::nullptr_t n) : rp_{n}{}
	template<class Other, typename = decltype(raw_pointer{std::declval<Other const&>().rp_})>
	ptr(Other const& o) : rp_{o.rp_}{}
	ptr& operator=(ptr const&) = default;

	using pointer = ptr<T>;
	using element_type = typename std::pointer_traits<raw_pointer>::element_type;
	using difference_type = void;//typename std::pointer_traits<impl_t>::difference_type;
	explicit operator bool() const{return rp_;}
//	explicit operator raw_pointer&()&{return rp_;}
	bool operator==(ptr const& other) const{return rp_==other.rp_;}
	bool operator!=(ptr const& other) const{return rp_!=other.rp_;}
	friend ptr to_address(ptr const& p){return p;}
	void operator*() const = delete;
};

template<typename RawPtr>
struct ptr<void, RawPtr>{
protected:
	using T = void;
	using raw_pointer = RawPtr;
	raw_pointer rp_;
private:
	ptr(ptr<void const> const& p) : rp_{const_cast<void*>(p.rp_)}{}
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
	template<class, class> friend struct ptr;
	template<class> friend class allocator;
public:
	template<class Other> ptr(ptr<Other> const& p) : rp_{p.rp_}{}
	explicit ptr(raw_pointer rp) : rp_{rp}{}
	ptr() = default;
	ptr(ptr const& p) : rp_{p.rp_}{}
	ptr(std::nullptr_t n) : rp_{n}{}
	template<class Other, typename = decltype(raw_pointer{std::declval<Other const&>().impl_})>
	ptr(Other const& o) : rp_{o.rp_}{}
	ptr& operator=(ptr const&) = default;
	bool operator==(ptr const& other) const{return rp_==other.rp_;}
	bool operator!=(ptr const& other) const{return rp_!=other.rp_;}
	operator cuda::ptr<void>(){return {rp_};}
	using pointer = ptr<T>;
	using element_type    = typename std::pointer_traits<raw_pointer>::element_type;
	using difference_type = typename std::pointer_traits<raw_pointer>::difference_type;
	template<class U> using rebind = ptr<U, typename std::pointer_traits<raw_pointer>::template rebind<U>>;

	explicit operator bool() const{return rp_;}
	explicit operator raw_pointer&()&{return rp_;}
	friend ptr to_address(ptr const& p){return p;}
	void operator*() = delete;
};

template<typename T, typename RawPtr>
struct ptr{
	using raw_pointer = RawPtr;
protected:
	raw_pointer rp_;
	template<class TT> friend class allocator;
	template<typename, typename> friend struct ptr;
	template<class TT, typename = typename std::enable_if<not std::is_const<TT>{}>::type> 
	ptr(ptr<TT const> const& p) : rp_{const_cast<T*>(p.impl_)}{}
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
public:
	template<class Other> ptr(Other const& o) HD : rp_{static_cast<raw_pointer>(o.rp_)}{}
	explicit ptr(raw_pointer p) HD : rp_{p}{}//Cuda::pointer::is_device(p);}
	ptr() = default;
	ptr(ptr const&) = default;
	ptr(std::nullptr_t n) : rp_{n}{}
	ptr& operator=(ptr const&) = default;
	bool operator==(ptr const& other) const{return rp_==other.rp_;}
	bool operator!=(ptr const& other) const{return rp_!=other.rp_;}

	using element_type = typename std::pointer_traits<raw_pointer>::element_type;
	using difference_type = typename std::pointer_traits<raw_pointer>::difference_type;
	using value_type = T;
	using pointer = ptr<T>;
	using iterator_category = typename std::iterator_traits<raw_pointer>::iterator_category;
//	using iterator_concept  = typename std::iterator_traits<impl_t>::iterator_concept;
	explicit operator bool() const{return rp_;}
	operator raw_pointer&()&{return rp_;}
	operator raw_pointer const&() const&{return rp_;}
	operator ptr<void>() const{return ptr<void>{rp_};}
//	template<class PM>
//	decltype(auto) operator->*(PM pm) const{return *ptr<std::decay_t<decltype(rp_->*pm)>, decltype(&(rp_->*pm))>{&(rp_->*pm)};}
	explicit operator typename std::pointer_traits<raw_pointer>::template rebind<void>() const{return rp_;}
	ptr& operator++(){++rp_; return *this;}
	ptr& operator--(){--rp_; return *this;}
	ptr  operator++(int){auto tmp = *this; ++(*this); return tmp;}
	ptr  operator--(int){auto tmp = *this; --(*this); return tmp;}
	ptr& operator+=(typename ptr::difference_type n){rp_+=n; return *this;}
	ptr& operator-=(typename ptr::difference_type n){rp_-=n; return *this;}
//	friend bool operator==(ptr const& s, ptr const& t){return s.impl_==t.impl_;}
//	friend bool operator!=(ptr const& s, ptr const& t){return s.impl_!=t.impl_;}
	ptr operator+(typename ptr::difference_type n) const{return ptr{rp_ + n};}
	ptr operator-(typename ptr::difference_type n) const{return ptr{rp_ - n};}
	using reference = typename std::pointer_traits<raw_pointer>::element_type&;//ref<element_type>;
	reference operator*() const{return *rp_;}
	HD reference operator[](difference_type n){return *((*this)+n);}
	friend ptr to_address(ptr const& p){return p;}
	typename ptr::difference_type operator-(ptr const& other) const{return rp_-other.rp_;}
	friend raw_pointer raw_pointer_cast(ptr const& self){return self.rp_;}
};

#if 0
template<
	class Alloc, class InputIt, class Size, class... T, class ForwardIt = ptr<T...>,
	typename InputV = typename std::pointer_traits<InputIt>::element_type, 
	typename ForwardV = typename std::pointer_traits<ForwardIt>::element_type
	, typename = std::enable_if_t<std::is_constructible<ForwardV, InputV>{}>
>
ForwardIt uninitialized_copy_n(Alloc&, InputIt f, Size n, ptr<T...> d){
	if(std::is_trivially_constructible<ForwardV, InputV>{})
		return memcpy(d, f, n*sizeof(ForwardV)) + n;
	else assert(0);
	return d;
}

template<class T> 
ptr<T> const_pointer_cast(ptr<T const> const& p){return ptr<T>{p.impl_};}

template<class T>
struct ref : private ptr<T>{
	using value_type = T;
	using reference = value_type&;
	using pointer = ptr<T>;
private:
	__host__ __device__ ref(pointer p) : ptr<T>{std::move(p)}{}
//	friend class ptr<T>;
public:
	template<class TT, class PP> friend struct ptr;
	ptr<T> operator&(){return *this;}
	struct skeleton_t{
		char buff[sizeof(T)]; T* p_;
		[[SLOW]] 
		__host__ __device__ skeleton_t(T* p) : p_{p}{
			#if __CUDA_ARCH__
			#else
			[[maybe_unused]] cudaError_t s = cudaMemcpy(buff, p_, sizeof(T), cudaMemcpyDeviceToHost); assert(s == cudaSuccess);
			#endif	
		}
		__host__ __device__ [[SLOW]] 
		operator T&()&&{return reinterpret_cast<T&>(buff);}
		__host__ __device__
		void conditional_copyback_if_not(std::false_type) const{
			#if __CUDA_ARCH__
		//	*p_ = reinterpret_cast<T const&>(
			#else
			[[maybe_unused]] cudaError_t s = cudaMemcpy(p_, buff, sizeof(T), cudaMemcpyHostToDevice); (void)s; assert(s == cudaSuccess);
			#endif
		}
		void conditional_copyback_if_not(std::true_type) const{}
		__host__ __device__ ~skeleton_t(){conditional_copyback_if_not(std::is_const<T>{});}
	};
	__host__ __device__
	skeleton_t skeleton()&&{return {this->impl_};}
public:
	__host__ __device__ ref(ref&& r) : ptr<T>(r){}
	ref& operator=(ref const&)& = delete;
private:
	ref& move_assign(ref&& other, std::true_type)&{
		[[maybe_unused]] cudaError_t s = cudaMemcpy(this->impl_, other.impl_, sizeof(T), cudaMemcpyDeviceToDevice); (void)s; assert(s == cudaSuccess);
		return *this;
	}
	ref& move_assign(ref&& other, std::false_type)&{
		[[maybe_unused]] cudaError_t s = cudaMemcpy(this->impl_, other.impl_, sizeof(T), cudaMemcpyDeviceToDevice); (void)s; assert(s == cudaSuccess);
		return *this;
	}
public:
	__host__ __device__
	[[SLOW]] ref&& operator=(ref&& other)&&{
		#ifdef __CUDA_ARCH__
		*(this->impl_) = *(other.impl_);
		return std::move(*this);
		#else
		return std::move(move_assign(std::move(other), std::is_trivially_copy_assignable<T>{}));
		#endif
	}
private:
public:
	template<class Other>
	__host__ __device__ auto operator+(Other&& o)&&
	->decltype(std::move(*this).skeleton() + std::forward<Other>(o)){
		return std::move(*this).skeleton() + std::forward<Other>(o);}
//	template<class Self, class O, typename = std::enable_if_t<std::is_same<std::decay_t<Self>, ref>{}> > 
//	friend auto operator+(Self&& self, O&& o)
//	->decltype(std::forward<Self>(self).skeleton() + std::forward<O>(o)){
//		return std::forward<Self>(self).skeleton() + std::forward<O>(o);}
	__host__ __device__ [[SLOW]]
	#ifndef __CUDA_ARCH__
	#else
	#endif
	ref&& operator=(value_type const& t)&&{
		#ifdef __CUDA_ARCH__
			*(this->impl_) = t;
		//	assert(0);
		#else
		if(std::is_trivially_copy_assignable<T>{}){
			[[maybe_unused]] cudaError_t s= cudaMemcpy(this->impl_, std::addressof(t), sizeof(T), cudaMemcpyHostToDevice);
			assert(s == cudaSuccess);
		}else{
			char buff[sizeof(T)];
			[[maybe_unused]] cudaError_t s1 = cudaMemcpy(buff, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); 
			assert(s1 == cudaSuccess);
			reinterpret_cast<T&>(buff) = t;
			[[maybe_unused]] cudaError_t s2 = cudaMemcpy(this->impl_, buff, sizeof(T), cudaMemcpyHostToDevice); 
			assert(s2 == cudaSuccess);
		}
		#endif
		return std::move(*this);
	}
#ifndef _MULTI_MEMORY_CUDA_DISABLE_ELEMENT_ACCESS
	bool operator!=(ref const& other) const&{return not(*this == other);}
	template<class Other>
	bool operator!=(ref<Other>&& other)&&{
		char buff1[sizeof(T)];
		[[maybe_unused]] cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess);
		char buff2[sizeof(Other)];
		[[maybe_unused]] cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(Other), cudaMemcpyDeviceToHost); assert(s2 == cudaSuccess);
		return reinterpret_cast<T const&>(buff1)!=reinterpret_cast<Other const&>(buff2);
	}
#else
//	bool operator==(ref const& other) const = delete;
#endif
#if 1
	template<class Other> 
	[[SLOW]]
	bool operator==(ref<Other>&& other)&&{
//#pragma message ("Warning goes here")
		char buff1[sizeof(T)];
		[[maybe_unused]] cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost);
		assert(s1 == cudaSuccess);
		char buff2[sizeof(Other)];
		[[maybe_unused]] cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(Other), cudaMemcpyDeviceToHost); 
		assert(s2 == cudaSuccess);
		return reinterpret_cast<T const&>(buff1)==reinterpret_cast<Other const&>(buff2);
	}
#if 1
	[[SLOW]] 
	bool operator==(ref const& other) const&{
		char buff1[sizeof(T)];
		{[[maybe_unused]] cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess);}
		char buff2[sizeof(T)];
		{[[maybe_unused]] cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s2 == cudaSuccess);}
		return reinterpret_cast<T const&>(buff1)==reinterpret_cast<T const&>(buff2);
	}
//	[[SLOW]] 
//	bool operator==(T const& other) const{
//		char buff1[sizeof(T)];
//		{cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess);}
//		return reinterpret_cast<T const&>(buff1)==other;
//	}
#endif
#endif
//	[[SLOW]] 
	#if __CUDA_ARCH__
	operator T()&&{return *(this->impl_);}
	#else
	[[SLOW]] operator T()&&{
		char buff[sizeof(T)];
		{[[maybe_unused]] cudaError_t s = cudaMemcpy(buff, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s == cudaSuccess);}
		return std::move(reinterpret_cast<T&>(buff));
	}
	#endif
	template<class Other, typename = decltype(std::declval<T&>()+=std::declval<Other&&>())>
	__host__ __device__
	ref& operator+=(Other&& o)&&{std::move(*this).skeleton()+=o; return *this;}
	template<class Other, typename = decltype(std::declval<T&>()-=std::declval<Other&&>())>
	ref&& operator-=(Other&& o)&&{std::move(*this).skeleton()-=o; return std::move(*this);}
	friend void swap(ref&& a, ref&& b){T tmp = std::move(a); std::move(a) = std::move(b); std::move(b) = tmp;}
	ref<T>&& operator++()&&{++(std::move(*this).skeleton()); return std::move(*this);}
	ref<T>&& operator--()&&{--(std::move(*this).skeleton()); return std::move(*this);}
};
#endif
}

}}
}}
#undef SLOW

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_MANAGED_PTR

#include "../../cuda/managed/clib.hpp" // cuda::malloc
#include "../../cuda/managed/malloc.hpp"

#include<cstring>
#include<iostream>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

void add_one(double& d){d += 1.;}
template<class T>
void add_one(T&& t){std::forward<T>(t) += 1.;}

// * Functions with a __global__ qualifier, which run on the device but are called by the host, cannot use pass by reference. 
//__global__ void set_5(cuda::ptr<double> const& p){
//__global__ void set_5(cuda::ptr<double> p){*p = 5.;}
//__global__ void check_5(cuda::ptr<double> p){assert(*p == 5.);}

double const* g(){double* p{nullptr}; return p;}

cuda::managed::ptr<double const> f(){
	return cuda::managed::ptr<double>{nullptr};
}

int main(){
	f();
	using T = double; static_assert( sizeof(cuda::managed::ptr<T>) == sizeof(T*) );
	std::size_t const n = 100;
	{
		auto p = static_cast<cuda::managed::ptr<T>>(cuda::managed::malloc(n*sizeof(T)));
		cuda::managed::ptr<void> pp = p;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
		*p = 99.; 
		if(*p != 99.) assert(0);
		if(*p == 11.) assert(0);
#pragma GCC diagnostic pop
		cuda::managed::free(p);
	}
	{
		auto p = static_cast<cuda::managed::ptr<T>>(cuda::managed::malloc(n*sizeof(T)));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
		double* ppp = p; *ppp = 3.14;
		assert( *p == 3.14 );
#pragma GCC diagnostic pop
		cuda::managed::ptr<T> P = nullptr;
	}
	{
		cuda::managed::ptr<double> p = nullptr;
		cuda::managed::ptr<double const> pc = nullptr; 
		pc = static_cast<cuda::managed::ptr<double const>>(p);
		double* dp = cuda::managed::ptr<double>{nullptr};
		auto f = [](double const*){};
		f(p);
	}
	std::cout << "Finish" << std::endl;
}
#endif
#endif


