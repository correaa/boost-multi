#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&nvcc -std=c++14 -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_PTR -Wno-deprecated-declarations $0.cpp -o $0x -lboost_unit_test_framework&&$0x&&rm $0x; exit
#endif

#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_PTR_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_PTR_HPP

#ifndef HD
#if defined(__CUDACC__)
#define HD __host__ __device__
#else
#define HD 
#endif
#endif

#include "../../adaptors/cuda/clib.hpp"

#include<cassert> // debug

#ifndef _DISABLE_CUDA_SLOW
#define SLOW deprecated("WARNING: slow function") 
#else
#define SLOW
#endif

namespace boost{namespace multi{
namespace memory{namespace cuda{

template<class T> struct ref;

template<typename T, typename Ptr = T*> struct ptr;

#if 1
template<typename RawPtr>
struct ptr<void const, RawPtr>{ // used by memcpy
	using raw_pointer = RawPtr;
private:
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

	using pointer = ptr;
	using element_type = typename std::pointer_traits<raw_pointer>::element_type;
	using difference_type = void;//typename std::pointer_traits<impl_t>::difference_type;
	explicit operator bool() const{return rp_;}
//	explicit operator impl_t&()&{return impl_;}
	bool operator==(ptr const& other) const{return rp_==other.rp_;}
	bool operator!=(ptr const& other) const{return rp_!=other.rp_;}
	friend ptr to_address(ptr const& p){return p;}
};
#endif

namespace managed{
	template<typename T, typename RawPtr> struct ptr;
}

template<typename RawPtr>
struct ptr<void, RawPtr>{
protected:
	using T = void;
	using raw_pointer = RawPtr;
	using raw_pointer_traits = std::pointer_traits<raw_pointer>;
	static_assert(std::is_same<void, typename raw_pointer_traits::element_type>{}, "!");
	raw_pointer rp_;
	friend ptr<void> malloc(size_t);
	friend void free();
	friend ptr<void> memset(ptr<void> dest, int ch, std::size_t byte_count);
	template<class TT, class RRawPtr> friend struct managed::ptr;
private:
//	ptr(ptr<void const> const& p) : rp_{const_cast<void*>(p.rp_)}{}
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
	template<class, class> friend struct ptr;
	ptr(raw_pointer rp) : rp_{rp}{}
	operator raw_pointer() const{return rp_;}
	friend ptr<void> malloc(std::size_t);
	friend void free(ptr<void>);
public:
	ptr() = default;
	ptr(ptr const& other) : rp_{other.rp_}{}//= default;
	ptr(std::nullptr_t n) : rp_{n}{}
	template<class Other, typename = decltype(raw_pointer{std::declval<Other const&>().rp_})>
	ptr(Other const& o) : rp_{o.rp_}{}
	ptr& operator=(ptr const&) = default;
	bool operator==(ptr const& other) const{return rp_==other.rp_;}
	bool operator!=(ptr const& other) const{return rp_!=other.rp_;}

	using pointer = ptr<T>;
	using element_type    = typename std::pointer_traits<raw_pointer>::element_type;
	using difference_type = typename std::pointer_traits<raw_pointer>::difference_type;
	template<class U> using rebind = ptr<U, typename std::pointer_traits<raw_pointer>::template rebind<U>>;

	explicit operator bool() const{return rp_;}
//	explicit operator raw_pointer&()&{return impl_;}
	friend ptr to_address(ptr const& p){return p;}
};

template<typename T, typename RawPtr>
struct ptr{
protected:
	using raw_pointer = RawPtr;
	raw_pointer rp_;
	using raw_pointer_traits = typename std::pointer_traits<raw_pointer>;
	template<class TT> friend class allocator;
	template<typename, typename> friend struct ptr;
//	template<class TT, typename = typename std::enable_if<not std::is_const<TT>{}>::type> 
//	ptr(ptr<TT const> const& p) : rp_{const_cast<T*>(p.rp_)}{}
	template<class TT> friend ptr<TT> const_pointer_cast(ptr<TT const> const&);
	friend struct managed::ptr<T, RawPtr>;
public:
	template<class U> using rebind = ptr<U, typename std::pointer_traits<raw_pointer>::template rebind<U>>;

	template<class Other, typename = std::enable_if_t<std::is_convertible<std::decay_t<decltype(std::declval<ptr<Other>>().rp_)>, raw_pointer>{} and not std::is_same<Other, T>{} >>
	/*explicit(false)*/ ptr(ptr<Other> const& o) HD : rp_{static_cast<raw_pointer>(o.rp_)}{}
	template<class Other, typename = std::enable_if_t<not std::is_convertible<std::decay_t<decltype(std::declval<ptr<Other>>().rp_)>, raw_pointer>{} and not std::is_same<Other, T>{}>>
	explicit/*(true)*/ ptr(ptr<Other> const& o, void** = 0) HD : rp_{static_cast<raw_pointer>(o.rp_)}{}
	explicit ptr(raw_pointer rp) HD : rp_{rp}{}
	template<class Other> explicit ptr(Other const& o) : rp_{static_cast<raw_pointer>(o.rp_)}{}
	ptr() = default;
	ptr(ptr const&) = default;
	ptr(std::nullptr_t nu) : rp_{nu}{}
	ptr& operator=(ptr const&) = default;
	bool operator==(ptr const& other) const{return rp_==other.rp_;}
	bool operator!=(ptr const& other) const{return rp_!=other.rp_;}

	using element_type = typename raw_pointer_traits::element_type;
	using difference_type = typename raw_pointer_traits::difference_type;
	using value_type = element_type;
	using pointer = ptr<T, RawPtr>;//<T>;
	using iterator_category = typename std::iterator_traits<raw_pointer>::iterator_category;
	explicit operator bool() const HD{return rp_;}
	explicit operator typename raw_pointer_traits::template rebind<void>() const{return {rp_};}
	ptr& operator++(){++rp_; return *this;}
	ptr& operator--(){--rp_; return *this;}
	ptr  operator++(int){auto tmp = *this; ++(*this); return tmp;}
	ptr  operator--(int){auto tmp = *this; --(*this); return tmp;}
	ptr& operator+=(typename ptr::difference_type n) HD{rp_+=n; return *this;}
	ptr& operator-=(typename ptr::difference_type n) HD{rp_-=n; return *this;}
//	friend bool operator==(ptr const& s, ptr const& t){return s.impl_==t.impl_;}
//	friend bool operator!=(ptr const& s, ptr const& t){return s.impl_!=t.impl_;}
	ptr operator+(typename ptr::difference_type n) const HD{return ptr{rp_ + n};}
	ptr operator-(typename ptr::difference_type n) const HD{return ptr{rp_ - n};}
#ifdef __CUDA_ARCH__
	using reference = T&;
#else
	using reference = ref<element_type>;
#endif
	#ifdef __CUDA_ARCH__
	__device__  reference operator*() const{return *rp_;}
	#else
	__host__ reference operator*() const{return {*this};}
	#endif
//	__host__ ref<element_type> operator*() const{return {*this};}
//#endif
	#ifdef __CUDA_ARCH__
	__device__ T& operator[](difference_type n) const HD{return *(rp_+n);}
	#else
	__host__ decltype(auto) operator[](difference_type n) const HD{return *((*this)+n);}
	#endif
	friend ptr to_address(ptr const& p){return p;}
	typename ptr::difference_type operator-(ptr const& o) const{return rp_-o.rp_;}
	operator ptr<void>(){return {rp_};}
	explicit operator raw_pointer() const{return rp_;}
	friend raw_pointer raw_pointer_cast(ptr self){return self.rp_;}
	template<class PM>
	auto operator->*(PM&& pm) const
	->decltype(ref<std::decay_t<decltype(rp_->*std::forward<PM>(pm))>>{ptr<std::decay_t<decltype(rp_->*std::forward<PM>(pm))>>{&(rp_->*std::forward<PM>(pm))}}){
		return ref<std::decay_t<decltype(rp_->*std::forward<PM>(pm))>>{ptr<std::decay_t<decltype(rp_->*std::forward<PM>(pm))>>{&(rp_->*std::forward<PM>(pm))}};}
};

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
	ref(pointer const& p) HD: ptr<T>{p}{}
//	friend class ptr<T>;
public:
	template<class Other, typename = std::enable_if_t<std::is_convertible<std::decay_t<decltype(std::declval<ptr<Other>>())>, pointer>{}>>
	/*explicit(false)*/ ref(ref<Other> const& o) HD : ptr<T>{static_cast<pointer>(o)}{}
	template<class Other, typename = std::enable_if_t<not std::is_convertible<std::decay_t<decltype(std::declval<ptr<Other>>())>, pointer>{}>>
	explicit/*(true)*/ ref(ref<Other> const& o, void** = 0) HD : ptr<T>{static_cast<pointer>(o)}{}
//	ref(T const& r) HD : ptr<T>(&r){}
	template<class TT, class PP> friend struct ptr;
	ptr<T> operator&(){return *this;}
	struct skeleton_t{
		char buff[sizeof(T)]; T* p_;
		[[SLOW]] 
		skeleton_t(T* p) HD : p_{p}{
			#if __CUDA_ARCH__
			#else
//			[[maybe_unused]] 
			cudaError_t s = cudaMemcpy(buff, p_, sizeof(T), cudaMemcpyDeviceToHost); 
			(void)s; assert(s == cudaSuccess);
			#endif	
		}
		operator T&()&& HD{return reinterpret_cast<T&>(buff);}
		void conditional_copyback_if_not(std::false_type) const HD{
			#if __CUDA_ARCH__
		//	*p_ = reinterpret_cast<T const&>(
			#else
		//	[[maybe_unused]] 
			cudaError_t s = cudaMemcpy(p_, buff, sizeof(T), cudaMemcpyHostToDevice); 
			(void)s; assert(s == cudaSuccess);
			#endif
		}
		void conditional_copyback_if_not(std::true_type) const HD{
			#if __CUDA_ARCH__
		//	*p_ = reinterpret_cast<T const&>(
			#else
		//	[[maybe_unused]] 
			cudaError_t s = cudaMemcpy(p_, buff, sizeof(T), cudaMemcpyHostToDevice); 
			(void)s; assert(s == cudaSuccess);
			#endif
		}
		~skeleton_t() HD{conditional_copyback_if_not(std::is_const<T>{});}
	};
	skeleton_t skeleton()&& HD{return {this->rp_};}
public:
	ref(ref&& r) HD : ptr<T>(r){}
	ref& operator=(ref const&)& = delete;
private:
	ref& move_assign(ref&& other, std::true_type)&{
	//	[[maybe_unused]] 
		cudaError_t s = cudaMemcpy(this->rp_, other.rp_, sizeof(T), cudaMemcpyDeviceToDevice); (void)s; assert(s == cudaSuccess);
		return *this;
	}
	ref& move_assign(ref&& other, std::false_type)&{
	//	[[maybe_unused]] 
		cudaError_t s = cudaMemcpy(this->rp_, other.rp_, sizeof(T), cudaMemcpyDeviceToDevice); (void)s; assert(s == cudaSuccess);
		return *this;
	}
public:
	__host__ __device__
//	[[SLOW]] 
	ref&& operator=(ref&& other)&&{
		#ifdef __CUDA_ARCH__
		assert(0);
		*(this->rp_) = *(other.rp_);
		return std::move(*this);
		#else
		return std::move(move_assign(std::move(other), std::is_trivially_copy_assignable<T>{}));
		#endif
	}
	template<class Other>
	decltype(auto) operator/=(Other&& o){(void)o;
		#ifdef __CUDA_ARCH__
		return std::move(*(this->rp_) /= std::forward<Other>(o));
		#else
		assert(0);
		#endif
	}
private:
public:
#if 1
//	template<class Other>
//	decltype(auto) operator+(Other&& other)&&{return static_cast<T>(*this) + std::forward<Other>(other);}
#endif
//	template<class Other>
//	auto operator+(Other&& o)&& HD
//	->decltype(std::move(*this).skeleton() + std::forward<Other>(o)){
//		return std::move(*this).skeleton() + std::forward<Other>(o);}
//	template<class Self, class O, typename = std::enable_if_t<std::is_same<std::decay_t<Self>, ref>{}> > 
//	friend auto operator+(Self&& self, O&& o)
//	->decltype(std::forward<Self>(self).skeleton() + std::forward<O>(o)){
//		return std::forward<Self>(self).skeleton() + std::forward<O>(o);}
	__host__ __device__ // [[SLOW]]
	#ifndef __CUDA_ARCH__
	#else
	#endif
	ref&& operator=(value_type const& t)&&{
		#ifdef __CUDA_ARCH__
			*(this->rp_) = t;
		//	assert(0);
		#else
		if(std::is_trivially_copy_assignable<T>{}){
			/*[[maybe_unused]]*/ cudaError_t s= cudaMemcpy(this->rp_, std::addressof(t), sizeof(T), cudaMemcpyHostToDevice);
			assert(s == cudaSuccess); (void)s;
		}else{
			char buff[sizeof(T)];
			/*[[maybe_unused]]*/ cudaError_t s1 = cudaMemcpy(buff, this->rp_, sizeof(T), cudaMemcpyDeviceToHost); 
			assert(s1 == cudaSuccess); (void)s1;
			reinterpret_cast<T&>(buff) = t;
			/*[[maybe_unused]]*/ cudaError_t s2 = cudaMemcpy(this->rp_, buff, sizeof(T), cudaMemcpyHostToDevice); 
			assert(s2 == cudaSuccess); (void)s2;
		}
		#endif
		return std::move(*this);
	}
#ifndef _MULTI_MEMORY_CUDA_DISABLE_ELEMENT_ACCESS
	bool operator!=(ref const& other) const&{return not(*this == other);}
	template<class Other>
	bool operator!=(ref<Other>&& other)&&{
		char buff1[sizeof(T)];
		/*[[maybe_unused]]*/ cudaError_t s1 = cudaMemcpy(buff1, this->impl_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess); (void)s1;
		char buff2[sizeof(Other)];
		/*[[maybe_unused]]*/ cudaError_t s2 = cudaMemcpy(buff2, other.impl_, sizeof(Other), cudaMemcpyDeviceToHost); assert(s2 == cudaSuccess); (void)s2;
		return reinterpret_cast<T const&>(buff1)!=reinterpret_cast<Other const&>(buff2);
	}
#else
//	bool operator==(ref const& other) const = delete;
#endif
#if 1
	template<class Other> 
	auto operator==(Other const& other)&&
//	->decltype(static_cast<T>(std::move(*this))==other)
	{
#if __CUDA_ARCH__
		return *(this->rp_)==other;
#else
		return static_cast<T>(std::move(*this))==other;
#endif
	}
	template<class Other> 
	friend auto operator==(Other&& other, ref&& self){
#if __CUDA_ARCH__
		return std::forward<Other>(other)==*(this->rp_);
#else
		return std::forward<Other>(other)==static_cast<T>(std::move(self));
#endif
	}
	template<class Other>
	[[SLOW]]
	bool operator==(ref<Other>&& other)&&{
//#pragma message ("Warning goes here")
		char buff1[sizeof(T)];
	//	cuda::memcpy(buff1, ref::rp_, sizeof(T));
		/*[[maybe_unused]]*/ cudaError_t s1 = cudaMemcpy(buff1, this->rp_, sizeof(T), cudaMemcpyDeviceToHost);
		assert(s1 == cudaSuccess); (void)s1;
		char buff2[sizeof(Other)];
		/*[[maybe_unused]]*/ cudaError_t s2 = cudaMemcpy(buff2, other.rp_, sizeof(Other), cudaMemcpyDeviceToHost); 
		assert(s2 == cudaSuccess); (void)s2;
		return reinterpret_cast<T const&>(buff1)==reinterpret_cast<Other const&>(buff2);
	}
#if 1
	[[SLOW]] 
	bool operator==(ref const& other) const&{
		char buff1[sizeof(T)];
		{/*[[maybe_unused]]*/ cudaError_t s1 = cudaMemcpy(buff1, this->rp_, sizeof(T), cudaMemcpyDeviceToHost); assert(s1 == cudaSuccess); (void)s1;}
		char buff2[sizeof(T)];
		{/*[[maybe_unused]]*/ cudaError_t s2 = cudaMemcpy(buff2, other.rp_, sizeof(T), cudaMemcpyDeviceToHost); assert(s2 == cudaSuccess); (void)s2;}
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
//	operator T const&()&& HD{return *(this->rp_);}
	operator T&&()&& HD{return std::move(*(this->rp_));}
	operator T const&() const& HD{return *(this->rp_);}
	operator T&()& HD{return *(this->rp_);}
	#else
	[[SLOW]] operator T()&&{
		char buff[sizeof(T)];
		{/*[[maybe_unused]]*/ cudaError_t s = cudaMemcpy(buff, this->rp_, sizeof(T), cudaMemcpyDeviceToHost); assert(s == cudaSuccess); (void)s;}
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

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA allocators"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../cuda/malloc.hpp"

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

template<class T> T what() = delete;
template<class T> T what(T&&) =  delete;

BOOST_AUTO_TEST_CASE(multi_memory_cuda_ptr){
	using T = double; 
	static_assert( sizeof(cuda::ptr<T>) == sizeof(T*) );
	std::size_t const n = 100;
	{
		using cuda::ptr;
		auto p = static_cast<ptr<T>>(cuda::malloc(n*sizeof(T)));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
		*p = 99.;
		{
			ptr<T const> pc = p;
			BOOST_REQUIRE(*p == *pc);
		}
		BOOST_REQUIRE(*p == 99.);
		BOOST_REQUIRE(*p != 11.);
#pragma GCC diagnostic pop
		cuda::free(p);
		cuda::ptr<T> P = nullptr;
		ptr<void> pv = p;

	}
//	what<typename cuda::ptr<T>::rebind<T const>>();
//	what<typename std::pointer_traits<cuda::ptr<T>>::rebind<T const>>();
	static_assert( std::is_same<typename std::pointer_traits<cuda::ptr<T>>::rebind<T const>, cuda::ptr<T const>>{} , "!");	
}

template<class T> struct Complex_{T real; T imag;};

BOOST_AUTO_TEST_CASE(multi_memory_cuda_ptr_member_pointer){
	
	Complex_<double> c{10.,20.};
//	double Complex_<double>::* 
	Complex_<double>* p = &c;
	auto pm = &Complex_<double>::imag;
	BOOST_REQUIRE( p->*pm == 20. );
	BOOST_REQUIRE( *p.*pm == 20. );

	cuda::ptr<Complex_<double>> pcu;
//	pcu->*pm;
}

#endif
#endif

