// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2021

#pragma once

#include "../../array.hpp"

#include "./thrust/cuda/managed.hpp"

#include <thrust/device_allocator.h>
#include <thrust/system/cuda/memory.h> // ::thrust::cuda::allocator

#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include <thrust/host_vector.h>

namespace boost{
namespace multi{
namespace thrust{

template<class T, multi::dimensionality_type D> using device_array = multi::array<T, D, ::thrust::device_allocator<T>>;
template<class T, multi::dimensionality_type D> using host_array   = multi::array<T, D                               >;

namespace device{

template<class T, multi::dimensionality_type D> using array = device_array<T, D>;

}

namespace host{

template<class T, multi::dimensionality_type D> using array = host_array<T, D>;

}

namespace cuda{

template<class T, multi::dimensionality_type D> using array = multi::array<T, D, ::thrust::cuda::allocator<T>>;

namespace managed{

template<class T, multi::dimensionality_type D> using array = multi::array<T, D, boost::multi::thrust::cuda::managed::allocator<T>>;

}

}

}}}

namespace thrust{

template<>
struct iterator_system<boost::multi::basic_array<char, 2L, char *, boost::multi::layout_t<2L, boost::multi::size_type>>::elements_iterator_t<char *>>{
	using type = thrust::iterator_system<char *>::type;
};

template<>
struct iterator_system<boost::multi::basic_array<char, 2L, std::__detected_or_t<char *, std::__allocator_traits_base::__pointer, thrust::cuda_cub::allocator<char>>, boost::multi::layout_t<2L, boost::multi::size_type>>::elements_iterator_t<std::__detected_or_t<char *, std::__allocator_traits_base::__pointer, thrust::cuda_cub::allocator<char>>>>{
	using type = thrust::iterator_system<thrust::cuda::pointer<char>>::type;
};
//template<>
//template<class P>
//struct iterator_system<<boost::multi::basic_array<T, dimensionality_type D, typename ElementPtr, class Layout>
//struct basic_array

template<class T1, class PQ1, class Size, class T2, class Q2>
[[deprecated]]
auto copy(
	boost::multi::array_iterator<T1, 2, PQ1                   > first_ , boost::multi::array_iterator<T1, 2, PQ1> last_,
	boost::multi::array_iterator<T2, 2, thrust::device_ptr<Q2>> result_
)-> boost::multi::array_iterator<T2, 2, thrust::device_ptr<Q2>> {
	assert(0);
//	MULTI_MARK_SCOPE("cuda copy_n 2D");
//	array_iterator<T1, 2, ::thrust::device_ptr<Q1>> first ; std::memcpy((void*)&first , (void const*)&first_ , sizeof(first_));
//	array_iterator<T2, 2, ::thrust::device_ptr<Q2>> result; std::memcpy((void*)&result, (void const*)&result_, sizeof(first_));
//	static_assert( sizeof(first ) == sizeof(first_ ) );
//	static_assert( sizeof(result) == sizeof(result_) );
//	assert(first->extensions() == result->extensions());
//	::thrust::for_each(
//		::thrust::make_counting_iterator(0L),
//		::thrust::make_counting_iterator(count*first->num_elements()),
//		[first, count, result, x = first->extensions()] __device__ (auto n){
//			std::tuple<index, index> ij = (count*x).from_linear(n);
//			result[std::get<0>(ij)][std::get<1>(ij)] = T2(first[std::get<0>(ij)][std::get<1>(ij)]);
//		}
//	);
	return result_ + (last_ - first_);
}

template<class T1, class Q1, class Size, class T2, class Q2>
[[deprecated]]
auto copy(
	boost::multi::array_iterator<T1, 2, thrust::device_ptr<Q1>> first_ , boost::multi::array_iterator<T1, 2, thrust::device_ptr<Q1>> last_,
	boost::multi::array_iterator<T2, 2, thrust::device_ptr<Q2>> result_
)-> boost::multi::array_iterator<T2, 2, thrust::device_ptr<Q2>> {
	assert(0);
//	MULTI_MARK_SCOPE("cuda copy_n 2D");
//	array_iterator<T1, 2, ::thrust::device_ptr<Q1>> first ; std::memcpy((void*)&first , (void const*)&first_ , sizeof(first_));
//	array_iterator<T2, 2, ::thrust::device_ptr<Q2>> result; std::memcpy((void*)&result, (void const*)&result_, sizeof(first_));
//	static_assert( sizeof(first ) == sizeof(first_ ) );
//	static_assert( sizeof(result) == sizeof(result_) );
//	assert(first->extensions() == result->extensions());
//	::thrust::for_each(
//		::thrust::make_counting_iterator(0L),
//		::thrust::make_counting_iterator(count*first->num_elements()),
//		[first, count, result, x = first->extensions()] __device__ (auto n){
//			std::tuple<index, index> ij = (count*x).from_linear(n);
//			result[std::get<0>(ij)][std::get<1>(ij)] = T2(first[std::get<0>(ij)][std::get<1>(ij)]);
//		}
//	);
	return result_ + (last_ - first_);
}

template<class T1, class Q1, class Size, class T2, class Q2>
[[deprecated]]
auto copy(
	boost::multi::array_iterator<T1, 1, thrust::device_ptr<Q1>> first_ , boost::multi::array_iterator<T1, 1, thrust::device_ptr<Q1>> last_,
	boost::multi::array_iterator<T2, 1, thrust::device_ptr<Q2>> result_
)-> boost::multi::array_iterator<T2, 1, thrust::device_ptr<Q2>> {
	assert(0);
//	MULTI_MARK_SCOPE("cuda copy_n 2D");
//	array_iterator<T1, 2, ::thrust::device_ptr<Q1>> first ; std::memcpy((void*)&first , (void const*)&first_ , sizeof(first_));
//	array_iterator<T2, 2, ::thrust::device_ptr<Q2>> result; std::memcpy((void*)&result, (void const*)&result_, sizeof(first_));
//	static_assert( sizeof(first ) == sizeof(first_ ) );
//	static_assert( sizeof(result) == sizeof(result_) );
//	assert(first->extensions() == result->extensions());
//	::thrust::for_each(
//		::thrust::make_counting_iterator(0L),
//		::thrust::make_counting_iterator(count*first->num_elements()),
//		[first, count, result, x = first->extensions()] __device__ (auto n){
//			std::tuple<index, index> ij = (count*x).from_linear(n);
//			result[std::get<0>(ij)][std::get<1>(ij)] = T2(first[std::get<0>(ij)][std::get<1>(ij)]);
//		}
//	);
	return result_ + (last_ - first_);
}


template<class MultiIterator>
struct elements_range{
	using difference_type = std::ptrdiff_t;

 public:
	struct strides_functor {
		std::ptrdiff_t z_;
		typename MultiIterator::layout_type l_;

		constexpr difference_type operator()(const difference_type& n) const {
			auto const x = l_.extensions();
			return n/x.num_elements()*z_ + std::apply(l_, x.from_linear(n%x.num_elements()));
		}
	};

 protected:
	using CountingIterator = thrust::counting_iterator<difference_type>;
	using TransformIterator = thrust::transform_iterator<strides_functor, CountingIterator>;
	using PermutationIterator = thrust::permutation_iterator<typename MultiIterator::element_ptr, TransformIterator>;

 public:
	elements_range(MultiIterator it, std::ptrdiff_t count)
	: base_{it.base()}
	, sf_{it.stride(), it->layout()}
	, size_{count*(it->extensions().num_elements())} {}

	using iterator = typename thrust::permutation_iterator<typename MultiIterator::element_ptr, TransformIterator>;

	auto begin() const {return iterator{base_, TransformIterator{CountingIterator{    0}, sf_}};}
	auto end()   const {return iterator{base_, TransformIterator{CountingIterator{size_}, sf_}};}

	auto size() const {return size_;}

 protected:
	typename MultiIterator::element_ptr base_;
	strides_functor sf_;
	std::ptrdiff_t size_;
};

//template<class Base, class... As>
//struct iterator_system<elements_range<Base, As...>>{
//	using type = typename thrust::iterator_system<Base>::type;
//};

template<class T1, class Q1, class Size, class T2, class Q2>
[[deprecated]]
auto copy_n(
	boost::multi::array_iterator<T1, 2, Q1*                      > first_ , Size count,
	boost::multi::array_iterator<T2, 2, thrust::cuda::pointer<Q2>> result_
)-> boost::multi::array_iterator<T2, 2, thrust::cuda::pointer<Q2>> {
	assert(first_->extensions() == result_->extensions());

	auto const& source_range = boost::multi::ref(first_ , first_  + count).elements();

	cudaHostRegister((void*)&source_range.front(), (&source_range.back() - &source_range.front())*sizeof(T1), cudaHostRegisterPortable);  // cudaHostRegisterReadOnly not available in cuda < 11.1
	::thrust::copy_n(thrust::device,
		source_range.begin(), source_range.size(),
		boost::multi::ref(result_, result_ + count).template reinterpret_array_cast<T2, Q2*>().elements().begin()
	);
	cudaHostUnregister((void*)&source_range.front());

	return result_ + count;
}

template<class T1, class Q1, class Size, class T2, class Q2>
[[deprecated]]
auto copy_n(
	boost::multi::array_iterator<T1, 1, thrust::device_ptr<Q1>> first_ , Size count,
	boost::multi::array_iterator<T2, 1, thrust::device_ptr<Q2>> result_
)-> boost::multi::array_iterator<T2, 1, thrust::device_ptr<Q2>> {
	assert(0);
//	MULTI_MARK_SCOPE("cuda copy_n 2D");
//	array_iterator<T1, 2, ::thrust::device_ptr<Q1>> first ; std::memcpy((void*)&first , (void const*)&first_ , sizeof(first_));
//	array_iterator<T2, 2, ::thrust::device_ptr<Q2>> result; std::memcpy((void*)&result, (void const*)&result_, sizeof(first_));
//	static_assert( sizeof(first ) == sizeof(first_ ) );
//	static_assert( sizeof(result) == sizeof(result_) );
//	assert(first->extensions() == result->extensions());
//	::thrust::for_each(
//		::thrust::make_counting_iterator(0L),
//		::thrust::make_counting_iterator(count*first->num_elements()),
//		[first, count, result, x = first->extensions()] __device__ (auto n){
//			std::tuple<index, index> ij = (count*x).from_linear(n);
//			result[std::get<0>(ij)][std::get<1>(ij)] = T2(first[std::get<0>(ij)][std::get<1>(ij)]);
//		}
//	);
	return result_ + count;
}

}

