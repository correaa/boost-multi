#ifndef MULTI_ADAPTORS_THRUST_HPP
#define MULTI_ADAPTORS_THRUST_HPP

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

namespace boost{
namespace multi{

#define CUDA_MAX_DIM1 2147483647ULL
#define CUDA_MAX_DIM23 65535

inline static void factorize(const std::size_t val, const std::size_t thres, std::size_t & fact1, std::size_t & fact2){
	fact1 = val;
	fact2 = 1;
	while (fact1 > thres){
		fact1 = (fact1 + 1)/2;
		fact2 *= 2;
	}

	assert(fact1*fact2 >= val);
}

template <class kernel_type>
__global__ void cuda_run_kernel_1(unsigned size, kernel_type kernel){
	auto ii = blockIdx.x*blockDim.x + threadIdx.x;
	if(ii < size) kernel(ii);
}

template <class kernel_type>
void run(size_t size, kernel_type kernel){
	if(size == 0) return;
		
	assert(size <= CUDA_MAX_DIM1);

	int mingridsize = 0;
	int blocksize = 0;
	auto s1 = cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,  cuda_run_kernel_1<kernel_type>);
	if(s1 != cudaSuccess) throw __LINE__;
	
	unsigned nblock = (size + blocksize - 1)/blocksize;

	cuda_run_kernel_1<<<nblock, blocksize>>>(size, kernel);
	auto s2 = cudaGetLastError();
	if(s2 != cudaSuccess) throw __LINE__;
	
	cudaDeviceSynchronize();
}

template <class kernel_type>
__global__ void cuda_run_kernel_2(unsigned sizex, unsigned sizey, unsigned dim2, kernel_type kernel){
	auto i1 = blockIdx.x*blockDim.x + threadIdx.x;
	auto i2 = blockIdx.y*blockDim.y + threadIdx.y;
	auto i3 = blockIdx.z*blockDim.z + threadIdx.z;
	
	auto ix = i1;
	auto iy = i2 + dim2*i3;
	if(ix < sizex && iy < sizey) kernel(ix, iy);
}

template <class kernel_type>
void run(size_t sizex, size_t sizey, kernel_type kernel){
	if(sizex == 0 or sizey == 0) return;

	int mingridsize = 0;
	int blocksize = 0;
	auto s1 = cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize, cuda_run_kernel_2<kernel_type>);
	if(s1 != cudaSuccess) throw __LINE__;
	
	std::cout<<"!!!!!!!!!!!!!!!!!blocksize mingridsize "<< blocksize <<", "<< mingridsize <<std::endl;

	//OPTIMIZATION, this is not ideal if sizex < blocksize
	unsigned nblock = (sizex + blocksize - 1)/blocksize;

	size_t dim2, dim3;
	factorize(sizey, CUDA_MAX_DIM23, dim2, dim3);
	
	struct dim3 dg{nblock, unsigned(dim2), unsigned(dim3)};
	struct dim3 db{unsigned(blocksize), 1, 1};
	cuda_run_kernel_2<<<dg, db>>>(sizex, sizey, dim2, kernel);
	auto s2 = cudaGetLastError();
	if(s1 != cudaSuccess) throw __LINE__;

	cudaDeviceSynchronize();
#if 0
	for(size_t iy = 0; iy < sizey; iy++){
		for(size_t ix = 0; ix < sizex; ix++){
			kernel(ix, iy);
		}
	}
#endif
}


template<class Difference>
struct stride_functor{
	Difference stride;
	constexpr std::ptrdiff_t operator()(Difference i) const{return stride*i;} // needs --expt-relaxed-constexpr
};

template<class It>
using strided_iterator = thrust::permutation_iterator<
	It, 
	thrust::transform_iterator<
		stride_functor<typename std::iterator_traits<It>::difference_type>, 
		thrust::counting_iterator<typename std::iterator_traits<It>::difference_type>
	>
>;

template<class It>
constexpr auto make_strided_iterator(It it, typename std::iterator_traits<It>::difference_type d){
	return strided_iterator<It>{it, thrust::make_transform_iterator(thrust::make_counting_iterator(0*d), stride_functor<decltype(d)>{d})};
}

template<class T1, class Q1, class Size, class T2, class Q2>
constexpr auto copy_n(
	array_iterator<T1, 1, thrust::device_ptr<Q1>> first, Size count, 
	array_iterator<T2, 1, thrust::device_ptr<Q2>> result
){
	throw std::runtime_error("96");
	thrust::copy_n(make_strided_iterator(first.base(), first.stride()), count, make_strided_iterator(result.base(), result.stride()));
	return result + count;
}

template<class T1, class Q1, class T2, class Q2>
constexpr auto copy(
	array_iterator<T1, 1, thrust::device_ptr<Q1>> first, array_iterator<T1, 1, thrust::device_ptr<Q1>> last, 
	array_iterator<T2, 1, thrust::device_ptr<Q2>> result
){
	throw std::runtime_error("106");
	return copy_n(first, last - first, result);
}

template<class T1, class Q1, class Size, class T2, class Q2>
constexpr 
array_iterator<T2, 2, thrust::device_ptr<Q2>> 
copy_n(
	array_iterator<T1, 2, thrust::device_ptr<Q1>> first, Size count, 
	array_iterator<T2, 2, thrust::device_ptr<Q2>> result
){
	assert(first->extensions() == result->extensions());
	thrust::for_each(
		thrust::make_counting_iterator(0l), 
		thrust::make_counting_iterator(count*first->num_elements()), 
		[first, count, result, x = first->extensions()] __device__ (auto n){
			tuple<index, index> ij = (count*x).from_linear(n);
			result[std::get<0>(ij)][std::get<1>(ij)] = first[std::get<0>(ij)][std::get<1>(ij)];
		}
	);
}


//	assert( first->size() == result->size() );
//	TODO: swap indeces
//	run(count, first->size(),  // first index should be faster
//		[=] __device__ (std::ptrdiff_t i, std::ptrdiff_t j){ // first index should be faster
//			result[i][j] = first[i][j];
//		}
//	);
	run(first->size(), count,  // first index should be faster
		[=] __device__ (std::ptrdiff_t j, std::ptrdiff_t i){ // first index should be faster
			result[i][j] = first[i][j];
		}
	);

//	thrust::copy_n(make_strided_iterator(first.base(), first.stride()), count, make_strided_iterator(result.base(), result.stride()));
	return result + count;
}

template<class T1, class Q1, class T2, class Q2>
constexpr auto copy(
	array_iterator<T1, 2, thrust::device_ptr<Q1>> first, array_iterator<T1, 2, thrust::device_ptr<Q1>> last, 
	array_iterator<T2, 2, thrust::device_ptr<Q2>> result
){
//	throw std::runtime_error("143");
	return copy_n(first, last - first, result);
}

}}

#endif

