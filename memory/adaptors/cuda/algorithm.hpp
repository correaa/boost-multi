#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -std=c++2a -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM -DNDEBUG $0.cpp -o$0x -lcudart -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
#endif
#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM_HPP

#include          "../cuda/cstring.hpp"
#include "../../../../multi/array_ref.hpp"

#include "../cuda/error.hpp"
//#include <boost/log/trivial.hpp>
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
	class PtrU, class PtrT, //typename Size, 
	typename U = typename std::iterator_traits<PtrU>::value_type, typename T = typename std::iterator_traits<PtrT>::value_type,
	typename = std::enable_if_t<std::is_trivially_assignable<T&, U>{}>
>
auto copy_n(PtrU first, std::ptrdiff_t count, PtrT result)
->decltype(memcpy(result, first, count*sizeof(T)), result + count){std::cerr<<"cudamemcpy " << count*sizeof(T) << " bytes" << std::endl;
	return memcpy(result, first, count*sizeof(T)), result + count;}

//template<class U, class PtrT, typename Size>
//auto copy_n(ptr<U> first, Size count, PtrT result)
//->decltype(copy_n<ptr<U>, PtrT>(first, count, result)){
//	return copy_n<ptr<U>, PtrT>(first, count, result);}

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

namespace memory{
namespace cuda{

#if 0
template<class T1, class Q1, typename Size, class T2, class Q2, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
auto copy_n(iterator<T1, 1, Q1*> first, Size count, iterator<T2, 1, ptr<Q2>> result)
->decltype(memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first ), sizeof(T1), count), result + count){
	return memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first ), sizeof(T1), count), result + count;}

template<class T1, class Q1, typename Size, class T2, class Q2, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
auto copy_n(iterator<T1, 1, ptr<Q1>> first, Size count, iterator<T2, 1, Q2*> result)
->decltype(memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first), sizeof(T1), count), result + count){
	return memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first), sizeof(T1), count), result + count;}

template<class T1, class Q1, typename Size, class T2, class Q2, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
auto copy_n(iterator<T1, 1, ptr<Q1>> first, Size count, iterator<T2, 1, ptr<Q2>> result)
->decltype(memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first), sizeof(T1), count), result + count){
	return memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first), sizeof(T1), count), result + count;}
#endif

template<class It1, class It2, class T1 = typename std::iterator_traits<It1>::value_type, class T2 = typename std::iterator_traits<It2>::value_type, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
auto copy_n(It1 first, typename std::iterator_traits<It1>::difference_type count, It2 result)
->decltype(memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first), sizeof(T1), count), result + count){
	return memcpy2D(base(result), sizeof(T2)*stride(result), base(first), sizeof(T1)*stride(first), sizeof(T1), count), result + count;}

template<class It1, class It2>
auto copy(It1 first, It1 last, It2 result)
->decltype(copy_n(first, last - first, result)){
	return copy_n(first, last - first, result);}

}}

/*
template<class T1, typename Size, class T2, class Q2, typename = std::enable_if_t<std::is_trivially_assignable<T2&, T1>{}>>
array_iterator<T2, 1, memory::cuda::ptr<Q2>> copy_n(
	T1* first, Size count, 
	array_iterator<T2, 1, memory::cuda::ptr<Q2>> result
){
	std::cout << "count " << std::endl;
	return copy_n<array_iterator<std::decay_t<T1>, 1, T1*>>(first, count, result);
}*/

#if 0
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
#endif

}}

#define FWD(x) std::forward<decltype(x)>(x)

namespace boost::multi{
	template<size_t I> struct priority_tag : priority_tag<I-1> {};
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM
#define BOOST_TEST_MODULE "C++ Unit Tests for Multi initializer_list"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../../array.hpp"
#include "../../../adaptors/cuda.hpp"

#include "../cuda/allocator.hpp"
#include<boost/timer/timer.hpp>
#include<numeric>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

BOOST_AUTO_TEST_CASE(multi_memory_adaptors_cuda_copy_2D){
	multi::array<double, 1> A(50, 99.);
	multi::cuda::array<double, 1> B(50);
	BOOST_REQUIRE( size(B) == 50 );

//	using std::copy_n;
//	using std::copy;
	using boost::multi::copy_n;
//	copy_n(&A[0], size(A), &B[0]);
	copy_n(begin(A), size(A), begin(B));

	multi::cuda::array<double, 1> D(50);
	copy_n(begin(B), size(B), begin(D));

	multi::array<double, 1> C(50, 88.);
	copy_n(begin(D), size(D), begin(C));
//	C = B;

//	BOOST_REQUIRE( C == A );
}

namespace utf = boost::unit_test;

#if 0
BOOST_AUTO_TEST_CASE(multi_memory_adaptors_cuda_algorithm, *utf::disabled()){
	BOOST_REQUIRE(false);
	multi::cuda::array<double, 1> const A(10, 99.);
	multi::cuda::array<double, 1> B(10, 88.);
	B = A;

	B() = A();

//	B.assign({1., 2., 3., 4.});
	B = {1., 2., 3., 4.};
	BOOST_REQUIRE( size(B) == 4 );

	B().assign({11., 22., 33., 44.});//.begin(), il.end());
//	BOOST_REQUIRE( B[2] == 33. );
///	B.assign


//	multi::cuda::array<double, 2> B({10, 10}, 88.);
//	B = A;
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
		multi::array<double, 2> A({32, 8}, 99.);
		multi::array<double, 2, cuda::allocator<double>> A_gpu({32, 8}, 0.);// = A;//({32, 8000});// = A;
	}
	
}
#endif

#endif
#endif


