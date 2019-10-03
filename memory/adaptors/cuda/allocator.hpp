#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& c++ -std=c++14 -Wall -Wextra -D_DISABLE_CUDA_SLOW -D_TEST_MULTI_MEMORY_CUDA_ALLOCATOR -D_MULTI_MEMORY_CUDA_DISABLE_ELEMENT_ACCESS `pkg-config cudart --cflags --libs` $0.cpp -o$0x   && $0x && rm $0x $0.cpp; exit
#endif

#include<cuda_runtime.h> // cudaMalloc

#include "../../adaptors/cuda/ptr.hpp"

#include "../../adaptors/cuda/clib.hpp" // cuda::malloc
#include "../../adaptors/cuda/cstring.hpp" // cuda::memcpy
#include "../../adaptors/cuda/malloc.hpp"

#include<new> // bad_alloc
#include<cassert>
#include<iostream> // debug

namespace boost{namespace multi{
namespace memory{namespace cuda{

struct bad_alloc : std::bad_alloc{};

struct allocation_counter{
	static long n_allocations;
	static long n_deallocations;
	static long bytes_allocated;
	static long bytes_deallocated;
};

long allocation_counter::n_allocations = 0;
long allocation_counter::n_deallocations = 0;
long allocation_counter::bytes_allocated = 0;
long allocation_counter::bytes_deallocated = 0;

template<class T=void> 
class allocator : allocation_counter{
public:
	using value_type = T;
	using pointer = ptr<T>;
	using size_type = ::size_t; // as specified by CudaMalloc
	pointer allocate(size_type n, const void* = 0){
		if(n == 0) return pointer{nullptr};
		auto ret = static_cast<pointer>(cuda::malloc(n*sizeof(T)));
		if(not ret) throw bad_alloc{};
		++n_allocations; bytes_allocated+=sizeof(T)*n;
		return ret;
	}
	void deallocate(pointer p, size_type n){
		cuda::free(p);
		++n_deallocations; bytes_deallocated+=sizeof(T)*n;
	}
	std::true_type operator==(allocator const&) const{return {};}
	std::false_type operator!=(allocator const&) const{return {};}
	template<class P, class... Args>
	void construct(P p, Args&&... args){
		if(sizeof...(Args) == 0 and std::is_trivially_default_constructible<T>{})
			cuda::memset(p, 0, sizeof(T));
		else{
			char buff[sizeof(T)];
			::new(buff) T(std::forward<Args>(args)...);
			cuda::memcpy(p, buff, sizeof(T));
		}
	}
	template<class P> void destroy(P p){
		if(not std::is_trivially_destructible<T>{}){
			char buff[sizeof(T)];
			cuda::memcpy(buff, p, sizeof(T)); // cudaMemcpy(buff, p.impl_, sizeof(T), cudaMemcpyDeviceToHost);
			((T*)buff)->~T();
		}
	}
};

template<> 
class allocator<std::max_align_t> : allocation_counter{
public:
	using T = std::max_align_t;
	using value_type = T;
	using pointer = ptr<T>;
	using size_type = ::size_t; // as specified by CudaMalloc
	auto allocate(size_type n, const void* = 0){
		if(n == 0) return pointer{nullptr};
		auto ret = static_cast<pointer>(cuda::malloc(n*sizeof(T)));
		if(not ret) throw bad_alloc{};
		++n_allocations; bytes_allocated+=sizeof(T)*n;
		return ret;
	}
	void deallocate(pointer p, size_type n){
		cuda::free(p); ++n_deallocations; bytes_deallocated+=sizeof(T)*n;
	}
	std::true_type operator==(allocator const&) const{return {};}
	std::false_type operator!=(allocator const&) const{return {};}
	template<class P, class... Args>
	void construct([[maybe_unused]] P p, Args&&...){assert(0);} // TODO investigate who is calling this
	template<class P>
	void destroy(P){} // TODO investigate who is calling this
};

}}}}

#ifdef _TEST_MULTI_MEMORY_CUDA_ALLOCATOR

#include<memory>
#include<iostream>
#include "../../../array.hpp"
#include "../cuda/algorithm.hpp"

namespace boost{namespace multi{
namespace cuda{
	template<class T, dimensionality_type D>
	using array = multi::array<T, D, multi::memory::cuda::allocator<T>>;
}
}}

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

void add_one(double& d){d += 1.;}
template<class T> void add_one(T&& t){std::forward<T>(t) += 1.;}

using std::cout;

int main(){
	{
		multi::static_array<double, 1> A1(32, double{}); A1[17] = 3.;
		multi::static_array<double, 1, cuda::allocator<double>> A1_gpu = A1;
		assert( A1_gpu[17] == 3 );
	}
	{
		multi::array<double, 1> A1(32, double{}); A1[17] = 3.;
		multi::array<double, 1, cuda::allocator<double>> A1_gpu = A1;
		assert( A1_gpu[17] == 3 );
	}
	{
		multi::static_array<double, 2> A2({32, 64}, double{}); A2[2][4] = 8.;
		multi::static_array<double, 2, cuda::allocator<double>> A2_gpu = A2;
		assert( A2_gpu[2][4] == 8. );
	}
	{
		multi::array<double, 2> A2({32, 64}, double{}); A2[2][4] = 8.;
		multi::static_array<double, 2, cuda::allocator<double>> A2_gpu = A2;
		assert( A2_gpu[2][4] == 8. );
	}
	{
		multi::array<double, 2> A2({32, 64}, double{}); A2[2][4] = 8.;
		multi::array<double, 2, cuda::allocator<double>> A2_gpu = A2;
		assert( A2_gpu[2][4] == 8. );
	}
	{
		multi::array<double, 2> A2({32, 8000000}, double{}); A2[2][4] = 8.;
		multi::array<double, 2, cuda::allocator<double>> A2_gpu = A2;
		int s; std::cin >> s;
		assert( A2_gpu[2][4] == 8. );
	}
	{
		static_assert(std::is_same<std::allocator_traits<cuda::allocator<double>>::difference_type, std::ptrdiff_t>{}, "!");
		static_assert(std::is_same<std::allocator_traits<cuda::allocator<double>>::pointer, cuda::ptr<double>>{}, "!");
		static_assert(
			std::is_same<
				std::allocator_traits<cuda::allocator<int>>::rebind_alloc<double>,
				cuda::allocator<double>
			>{}, "!"
		);
		cuda::allocator<double> calloc;
		assert(calloc == calloc);
		cuda::ptr<double> p = calloc.allocate(100);
		p[33] = 123.;
		p[99] = 321.;
		p[33]+=1;
		double p33 = p[33];
		assert( p33 == 124. );
		assert( p[33] == 124. );
		assert( p[33] == p[33] );
		swap(p[33], p[99]);
		assert( p[99] == 124. );
		assert( p[33] == 321. );
		std::cout << p[33] << std::endl;
		calloc.deallocate(p, 100);
		p = nullptr;
		cout<<"n_alloc/dealloc "<< cuda::allocation_counter::n_allocations <<"/"<< cuda::allocation_counter::n_deallocations <<"\n"
			<<"bytes_alloc/dealloc "<< cuda::allocation_counter::bytes_allocated <<"/"<< cuda::allocation_counter::bytes_deallocated <<"\n";
	}
}
#endif

