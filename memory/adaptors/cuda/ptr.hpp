#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& `#nvcc -ccbin=cuda-`c++ -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_PTR $0.cpp -o $0x -lcudart &&$0x&& rm $0x; exit
#endif

#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_PTR_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_PTR_HPP


#include "../cuda/clib.hpp"

#include<cassert>
#include<cstddef> // nullptr_t
#include<iterator> // random_access_iterator_tag

#include<type_traits> // is_const

#ifndef _DISABLE_CUDA_SLOW
#define SLOW deprecated("WARNING: slow function") 
#else
#define SLOW
#endif

namespace boost{namespace multi{
namespace memory{namespace cuda{

template<class T> struct ref;

template<typename T, typename Ptr = T*> struct ptr;

template<typename Ptr>
struct ptr<void const, Ptr>{
	using T = void const;
	using impl_t = Ptr;
	impl_t impl_;
	template<typename, typename> friend struct ptr;
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
	ptr(impl_t impl) : impl_{impl}{}
public:
	ptr() = default;
	ptr(ptr const&) = default;
	ptr(std::nullptr_t n) : impl_{n}{}
	template<class Other, typename = decltype(impl_t{std::declval<Other const&>().impl_})>
	ptr(Other const& o) : impl_{o.impl_}{}
	ptr& operator=(ptr const&) = default;

	using pointer = ptr<T>;
	using element_type = typename std::pointer_traits<impl_t>::element_type;
	using difference_type = void;//typename std::pointer_traits<impl_t>::difference_type;
	explicit operator bool() const{return impl_;}
	explicit operator impl_t&()&{return impl_;}
	bool operator==(ptr const& other) const{return impl_==other.impl_;}
	bool operator!=(ptr const& other) const{return impl_!=other.impl_;}
	friend ptr to_address(ptr const& p){return p;}
};

template<typename Ptr>
struct ptr<void, Ptr>{
protected:
	using T = void;
	using impl_t = Ptr;
	impl_t impl_;
private:
	ptr(ptr<void const> const& p) : impl_{const_cast<void*>(p.impl_)}{}
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
	template<class, class> friend struct ptr;
public:
	explicit ptr(impl_t impl) : impl_{impl}{}
	ptr() = default;
	__host__ __device__ ptr(ptr const& other) : impl_{other.impl_}{}//= default;
	ptr(std::nullptr_t n) : impl_{n}{}
	template<class Other, typename = decltype(impl_t{std::declval<Other const&>().impl_})>
	__host__ __device__
	ptr(Other const& o) : impl_{o.impl_}{}
	ptr& operator=(ptr const&) = default;
	bool operator==(ptr const& other) const{return impl_==other.impl_;}
	bool operator!=(ptr const& other) const{return impl_!=other.impl_;}

	using pointer = ptr<T>;
	using element_type    = typename std::pointer_traits<impl_t>::element_type;
	using difference_type = typename std::pointer_traits<impl_t>::difference_type;
	template<class U> using rebind = ptr<U, typename std::pointer_traits<Ptr>::template rebind<U>>;

	explicit operator bool() const{return impl_;}
	explicit operator impl_t&()&{return impl_;}
	friend ptr to_address(ptr const& p){return p;}
};

template<typename T, typename Ptr>
struct ptr{
protected:
	using impl_t = Ptr;
	impl_t impl_;
private:
	template<class TT> friend class allocator;
	template<typename, typename> friend struct ptr;
	template<class TT, typename = typename std::enable_if<not std::is_const<TT>{}>::type> 
	__host__ __device__ ptr(ptr<TT const> const& p) : impl_{const_cast<T*>(p.impl_)}{}
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
public:
	template<class Other> explicit ptr(Other const& o) : impl_{static_cast<impl_t>(o.impl_)}{}
//	explicit ptr(ptr<void, void*> other) : impl_{static_cast<impl_t>(other.impl_)}{}
	__host__ __device__ 
	explicit ptr(impl_t p) : impl_{p}{}//Cuda::pointer::is_device(p);}
	ptr() = default;
	__host__ __device__ ptr(ptr const& other) : impl_{other.impl_}{}
//	ptr(ptr const&) = default;
	__host__ __device__ ptr(std::nullptr_t n) : impl_{n}{}
	template<class Other, typename = decltype(impl_t{std::declval<Other const&>().impl_}), typename = typename std::enable_if<not std::is_base_of<ptr, Other>{}>::type /*c++14*/>
	__host__ __device__ ptr(Other const& o) : impl_{o.impl_}{}
	ptr& operator=(ptr const&) = default;
	bool operator==(ptr const& other) const{return impl_==other.impl_;}
	bool operator!=(ptr const& other) const{return impl_!=other.impl_;}

	using element_type = typename std::pointer_traits<impl_t>::element_type;
	using difference_type = typename std::pointer_traits<impl_t>::difference_type;
	using value_type = T;
	using pointer = ptr<T>;
	using iterator_category = typename std::iterator_traits<impl_t>::iterator_category;
//	using iterator_concept  = typename std::iterator_traits<impl_t>::iterator_concept;
	explicit operator bool() const{return impl_;}
	__host__ __device__ explicit operator impl_t&()&{return impl_;}
	__host__ __device__ explicit operator impl_t const&() const&{return impl_;}
	explicit operator typename std::pointer_traits<impl_t>::template rebind<void>()&{return impl_;}
	ptr& operator++(){++impl_; return *this;}
	ptr& operator--(){--impl_; return *this;}
	ptr  operator++(int){auto tmp = *this; ++(*this); return tmp;}
	ptr  operator--(int){auto tmp = *this; --(*this); return tmp;}
	ptr& operator+=(typename ptr::difference_type n){impl_+=n; return *this;}
	ptr& operator-=(typename ptr::difference_type n){impl_+=n; return *this;}
	__host__ __device__ 
	ptr operator+(typename ptr::difference_type n) const{return ptr{impl_ + n};}
	ptr operator-(typename ptr::difference_type n) const{return ptr{impl_ - n};}
	using reference = ref<element_type>;
#ifdef __CUDA_ARCH__
	__device__ T& operator*() const{return *impl_;}
#else
	__host__ [[SLOW]] ref<element_type> operator*() const{return {*this};}
#endif
//	__host__ __device__ 
	reference operator[](difference_type n){return *((*this)+n);}
	friend ptr to_address(ptr const& p){return p;}
	typename ptr::difference_type operator-(ptr const& other) const{return impl_-other.impl_;}
};

template<class Alloc, class InputIt, class Size, class... T, class ForwardIt = ptr<T...>>//, typename AT = std::allocator_traits<Alloc> >
ForwardIt uninitialized_copy_n(Alloc&, InputIt f, Size n, ptr<T...> d){
	if(std::is_trivially_constructible<typename std::pointer_traits<ForwardIt>::element_type, typename std::pointer_traits<InputIt>::element_type>{}){
		return memcpy(d, f, n*sizeof(typename std::pointer_traits<ForwardIt>::element_type)) + n;
	} else assert(0);
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
	template<class TT, class PP> friend struct ptr;
	ptr<T> operator&(){return *this;}
	struct skeleton_t{
		char buff[sizeof(T)]; T* p_;
		[[SLOW]] 
		__host__ __device__ skeleton_t(T* p) : p_{p}{
			#if __CUDA_ARCH__
			#else
			cudaError_t s = cudaMemcpy(buff, p_, sizeof(T), cudaMemcpyDeviceToHost); assert(s == cudaSuccess);
			#endif	
		}
		__host__ __device__ [[SLOW]] 
		operator T&()&&{return reinterpret_cast<T&>(buff);}
		__host__ __device__
		void conditional_copyback_if_not(std::false_type) const{
			#if __CUDA_ARCH__
		//	*p_ = reinterpret_cast<T const&>(
			#else
			cudaError_t s = cudaMemcpy(p_, buff, sizeof(T), cudaMemcpyHostToDevice); assert(s == cudaSuccess);
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
		cudaError_t s = cudaMemcpy(this->impl_, other.impl_, sizeof(T), cudaMemcpyDeviceToDevice); assert(s == cudaSuccess);
		return *this;
	}
	ref& move_assign(ref&& other, std::false_type)&{
		cudaError_t s = cudaMemcpy(this->impl_, other.impl_, sizeof(T), cudaMemcpyDeviceToDevice); assert(s == cudaSuccess);
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
	__host__ __device__
	#ifndef __CUDA_ARCH__
//	[[SLOW]]
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
		cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess);
		char buff2[sizeof(Other)];
		cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(Other), cudaMemcpyDeviceToHost); assert(s2 == cudaSuccess);
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
		cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost);
		assert(s1 == cudaSuccess);
		char buff2[sizeof(Other)];
		cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(Other), cudaMemcpyDeviceToHost); 
		assert(s2 == cudaSuccess);
		return reinterpret_cast<T const&>(buff1)==reinterpret_cast<Other const&>(buff2);
	}
#if 1
	[[SLOW]] 
	bool operator==(ref const& other) const&{
		char buff1[sizeof(T)];
		{cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess);}
		char buff2[sizeof(T)];
		{cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s2 == cudaSuccess);}
		return reinterpret_cast<T const&>(buff1)==reinterpret_cast<T const&>(buff2);
	}
#endif
#endif
//	[[SLOW]] 
	#if __CUDA_ARCH__
	operator T()&&{return *(this->impl_);}
	#else
	[[SLOW]] operator T()&&{
		char buff[sizeof(T)];
		{cudaError_t s = cudaMemcpy(buff, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s == cudaSuccess);}
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

}}}}
#undef SLOW

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_PTR

#include "../cuda/clib.hpp" // cuda::malloc
#include "../cuda/malloc.hpp"

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

void add_one(double& d){d += 1.;}
template<class T>
void add_one(T&& t){std::forward<T>(t) += 1.;}

// * Functions with a __global__ qualifier, which run on the device but are called by the host, cannot use pass by reference. 
//__global__ void set_5(cuda::ptr<double> const& p){
//__global__ void set_5(cuda::ptr<double> p){*p = 5.;}
//__global__ void check_5(cuda::ptr<double> p){assert(*p == 5.);}

int main(){
	using T = double; static_assert( sizeof(cuda::ptr<T>) == sizeof(T*) );
	std::size_t const n = 100;
	{
		auto p = static_cast<cuda::ptr<T>>(cuda::malloc(n*sizeof(T)));
		[[maybe_unused]] cuda::ptr<void> pp = p;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
		*p = 99.; if(*p != 99.) assert(0);
#pragma GCC diagnostic pop
		free(p);
	}
}
#endif
#endif


