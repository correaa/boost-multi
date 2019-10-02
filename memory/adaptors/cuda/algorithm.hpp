#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& clang++ -fopenmp -std=c++14 -Wall -Wextra -Weffc++ -D_DISABLE_CUDA_SLOW -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM $0.cpp -o$0x -lcudart -lboost_timer&&$0x&&rm $0x $0.cpp; exit
#endif
#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM_HPP

#include "../cuda/cstring.hpp"
#include "../../../../multi/array_ref.hpp"
#include<iostream>

namespace boost{namespace multi{
namespace memory{namespace cuda{

//template<class U, class T, typename Size, typename = std::enable_if_t<std::is_trivially_assignable<T&, U>{}>>
//ptr<T> copy_n(U const* first, Size count, ptr<T> result){
//	memcpy(result, first, count*sizeof(T)); return result + count;
//}

//template<class U, class T, typename Size, typename = std::enable_if_t<std::is_trivially_assignable<T&, U>{}>>
//ptr<T> copy_n(ptr<U> first, Size count, ptr<T> result){
//	memcpy(result, first, count*sizeof(T)); return result + count;
//}

//	template<class P> 
//	using element_t = typename std::pointer_traits<P>::element_type;
//	using std::enable_if_t;
//	using std::is_trivially_assignable;

using memory::cuda::ptr;

template<
	class PtrU, class PtrT, typename Size, 
	typename U = typename std::pointer_traits<PtrU>::element_type, typename T = typename std::pointer_traits<PtrT>::element_type,
	typename = std::enable_if_t<std::is_trivially_assignable<T&, U>{}>
>
//[[deprecated]]
ptr<T> copy_n(PtrU first, Size count, PtrT result){
	std::cerr<<"copying " << count*sizeof(T) << " bytes" << std::endl;
	memcpy(result, first, count*sizeof(T));  return result + count;
}

template<class U, class PtrT, typename Size>
auto copy_n(ptr<U> first, Size count, PtrT result)
->decltype(copy_n<ptr<U>, PtrT>(first, count, result)){
	return copy_n<ptr<U>, PtrT>(first, count, result);}

//template<class PtrU, class PtrT>
//auto copy(PtrU first, PtrU last, PtrT result)
//->decltype(copy_n(first, std::distance(first, last), result)){
//	return copy_n(first, std::distance(first, last), result);}

template<class PtrU, class T>
auto copy(PtrU first, PtrU last, ptr<T> result){
	return copy_n(first, std::distance(first, last), result);
}

template<class U, class T>
auto copy(ptr<U> first, ptr<U> last, ptr<T> result){
	return copy_n(first, std::distance(first, last), result);
}

//->decltype(copy_n(first, std::distance(first, last), result)){
//	return copy_n(first, std::distance(first, last), result);}


template<class T>
auto fill(memory::cuda::ptr<T> first, memory::cuda::ptr<T> last, T const& value)
->decltype(fill_n(first, std::distance(first, last), value)){
	assert(0);
	return fill_n(first, std::distance(first, last), value);}

template<class U, class T, typename Size, typename = std::enable_if_t<std::is_trivially_assignable<T&, U>{}>>
memory::cuda::ptr<T> fill_n(ptr<T> const first, Size count, U const& value){
	assert(0);
	if(value == 0) cuda::memset(first, 0, count*sizeof(T));
	else if(count--) for(ptr<T> new_first = copy_n(&value, 1, first); count;){
		auto n = std::min(Size(std::distance(first, new_first)), count);
		new_first = copy_n(first, n, new_first);
		count -= n;
	}
	return first + count;
}

}}}}

namespace boost{
namespace multi{

template<class T1, class Q1, typename Size, class T2, class Q2, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
array_iterator<T2, 1, memory::cuda::ptr<Q2>> copy_n(
	array_iterator<T1, 1, memory::cuda::ptr<Q1>> first, Size count, 
	array_iterator<T2, 1, memory::cuda::ptr<Q2>> result
){
	std::cerr<<"copying " << count << " elements " << std::endl;
	[[maybe_unused]] cudaError_t const s = cudaMemcpy2D(
		static_cast<void*>(base(result)), sizeof(T2)*stride(result),
		static_cast<void*>(base(first )), sizeof(T1)*stride(first ),
		sizeof(T1), count, cudaMemcpyDeviceToDevice
	); assert(cudaSuccess == s);
	return result + count;
}

template<class T1, class Q1, typename Size, class T2, class Q2, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
array_iterator<T2, 1, memory::cuda::ptr<Q2>> copy_n(
	iterator<T1, 1, Q1*> first, Size count, 
	iterator<T2, 1, memory::cuda::ptr<Q2>> result
){
	std::cerr<<"copying " << count << " elements " << std::endl;
	[[maybe_unused]] cudaError_t const s = cudaMemcpy2D(
		static_cast<Q2*>(base(result)), sizeof(T2)*stride(result),
		                 base(first ) , sizeof(T1)*stride(first ),
		sizeof(T1), count, cudaMemcpyHostToDevice
	); assert(cudaSuccess == s);
	return result + count;
}

template<class T1, class Q1, class T2, class Q2>
auto copy(
	array_iterator<T1, 1, memory::cuda::ptr<Q1>> f, array_iterator<T1, 1, memory::cuda::ptr<Q1>> l, 
	array_iterator<T2, 1, memory::cuda::ptr<Q2>> d
)
->decltype(copy_n(f, std::distance(f, l), d)){assert(stride(f)==stride(l));
	return copy_n(f, std::distance(f, l), d);}

template<class T1, class Q1, class T2, class Q2>
auto copy(
	array_iterator<T1, 1, Q1*> f, array_iterator<T1, 1, memory::cuda::ptr<Q1>> l, 
	array_iterator<T2, 1, memory::cuda::ptr<Q2>> d
)
->decltype(copy_n(f, std::distance(f, l), d)){assert(stride(f)==stride(l));
	return copy_n(f, std::distance(f, l), d);}


}}

#if 0
namespace boost{
namespace multi{

double* omp_copy_n(double const* first, std::size_t count, double* result){
	std::size_t i;
#pragma omp target parallel for shared(result,first) private(i)
	for(i = 0; i != count; ++i){
		result[i] = first[i];
	}
	return result + count;
}

}}
#endif

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM

#include "../../../array.hpp"
#include "../cuda/allocator.hpp"
#include<boost/timer/timer.hpp>
#include<numeric>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

template<class T> void what(T);

int main(){
#if 0
	{
		cuda::allocator<double> calloc;
		std::size_t n = 2e9/sizeof(double);
		[[maybe_unused]] cuda::ptr<double> p = calloc.allocate(n);
		{
			boost::timer::auto_cpu_timer t;
		//	using std::fill_n; fill_n(p, n, 99.);
		}
	//	assert( p[0] == 99. );
	//	assert( p[n/2] == 99. );
	//	assert( p[n-1] == 99. );
		[[maybe_unused]] cuda::ptr<double> q = calloc.allocate(n);
		{
			boost::timer::auto_cpu_timer t;
		//	multi::omp_copy_n(static_cast<double*>(p), n, static_cast<double*>(q));
		//	using std::copy_n; copy_n(p, n, q);
			using std::copy; copy(p, p + n, q);
		}
		{
			boost::timer::auto_cpu_timer t;
		//	using std::copy_n; copy_n(p, n, q);
		}
	//	assert( q[23] == 99. );
	//	assert( q[99] == 99. );
	//	assert( q[n-1] == 99. );
	}
	{
		multi::array<double, 1> const A(100);//, double{99.});
		multi::array<double, 1, cuda::allocator<double>> A_gpu = A;
				#pragma GCC diagnostic push // allow cuda element access
				#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	//	assert( A_gpu[1] == A_gpu[0] );
				#pragma GCC diagnostic pop
	}
#endif
	{
		multi::array<double, 2> A({32, 8}, double{99.});
	//	std::iota(A.data(), A.data()+A.num_elements(), 0.);
		multi::array<double, 2, cuda::allocator<double>> A_gpu = A;//({32, 8000});// = A;
		assert( A_gpu[2][2] == 99. );
	//	assert( A_gpu[10][10] == A[10][10] );
	//	std::cerr<< "initialized" << std::endl;
	//	assert( A_gpu[2][2] == 99. );
	//	assert( A_gpu[2][2] == A[2][2] );
	//	multi::array<double, 2> const A({32, 100000}, double{99.});
	//	multi::array<double, 2, cuda::allocator<double>> A_gpu = A;
	}
}
#endif
#endif


