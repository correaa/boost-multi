#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&& clang++ -std=c++14 -I/usr/include/cuda -Wfatal-errors -D_TEST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM $0.cpp -o$0x -lcudart -lboost_timer&& $0x && rm $0x $0.cpp; exit
#endif

#ifndef BOOST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM_HPP
#define BOOST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM_HPP

#include "../cuda/cstring.hpp"

namespace boost{
namespace multi{
namespace memory{
namespace cuda{

template<class U, class T, typename Size, typename = std::enable_if_t<std::is_trivially_assignable<T&, U>{}>>
ptr<T> copy_n(ptr<U> first, Size count, ptr<T> result){
	static_assert(sizeof(U) == sizeof(T), "!");
	memcpy(result, first, count*sizeof(T)); return result + count;
}

template<class T, typename Size, typename = std::enable_if_t<std::is_trivially_copy_assignable<T>{}>>
ptr<T> copy_n(T const* first, Size count, ptr<T> result){
	memcpy(result, first, count*sizeof(T)); return result + count;
}

template<class T, typename Size, typename = std::enable_if_t<std::is_trivially_copy_assignable<T>{}>>
ptr<T> fill_n(ptr<T> first, Size count, std::nullptr_t){
	memset(first, 0, count*sizeof(T)); return first + count;
}

template<class T>
auto fill(ptr<T> first, ptr<T> last, T const& value)
->decltype(fill_n(first, std::distance(first, last), value)){
	return fill_n(first, std::distance(first, last), value);}

template<class T, typename Size, typename = std::enable_if_t<std::is_trivially_copy_assignable<T>{}>>
ptr<T> fill_n(ptr<T> const first, Size count, T const& value){
	if(value == 0){
		memset(first, 0, count*sizeof(T));
	}else if(count--) for(ptr<T> new_first = copy_n(&value, 1, first); count;){
		auto n = std::min(Size(std::distance(first, new_first)), count);
		new_first = copy_n(first, n, new_first);
		count -= n;
	}
	return first + count;
}

}}}}

#ifdef _TEST_MULTI_MEMORY_ADAPTORS_CUDA_ALGORITHM

#include "../../../array.hpp"
#include "../cuda/allocator.hpp"
#include<boost/timer/timer.hpp>
#include<numeric>

namespace multi = boost::multi;
namespace cuda = multi::memory::cuda;

int main(){
	{
		cuda::allocator<double> calloc;
		std::size_t n = 2e9/sizeof(double);
		cuda::ptr<double> p = calloc.allocate(n);
		{
			boost::timer::auto_cpu_timer t;
			using std::fill_n; fill_n(p, n, 99.);
		}
		assert( p[0] == 99. );
		assert( p[n/2] == 99. );
		assert( p[n-1] == 99. );
		cuda::ptr<double> q = calloc.allocate(n);
		{
			boost::timer::auto_cpu_timer t;
			using std::copy_n; copy_n(p, n, q);
		}
		assert( q[23] == 99. );
		assert( q[99] == 99. );
		assert( q[n-1] == 99. );
	}
}
#endif
#endif


