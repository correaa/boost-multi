// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;autowrap:nil;-*-
// © Alfredo A. Correa 2021

#pragma once

#include "../../array.hpp"

#include "./thrust/cuda/managed.hpp"

#include <thrust/device_allocator.h>
#include <thrust/system/cuda/memory.h> // ::thrust::cuda::allocator

#include <thrust/system/cuda/experimental/pinned_allocator.h>

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


template<class Base, class Extensions = boost::multi::extensions_t<1>,  class Layout = std::multi::layout_t<1>>//class Strides = std::tuple<std::ptrdiff_t> >
struct elements_range{
	using difference_type = std::ptrdiff_t;

	struct strides_functor {// : public thrust::unary_function<difference_type, difference_type>{
		std::ptrdiff_t z_;
		Layout    l_;

		__host__ __device__
		difference_type operator()(const difference_type& n) const {
			auto const x = l_.extensions();
			return
				  n / x.num_elements()*z_
				+ std::apply(l_, x.from_linear(n % x_.num_elements()));
		}
	};

	using CountingIterator = typename thrust::counting_iterator<difference_type>;
	using TransformIterator = typename thrust::transform_iterator<strides_functor, CountingIterator>;
	typedef typename thrust::permutation_iterator<Base, TransformIterator> PermutationIterator;

	using iterator = typename thrust::permutation_iterator<Base, TransformIterator>;

	elements_range(Base base, std::ptrdiff_t stride, Layout layout, std::ptrdiff_t count)
	: base_{base}, stride_{stride}, layout_{layout}, count_{count} {}

	iterator begin() const {
		return iterator{
			base_,
			TransformIterator{
				CountingIterator{0},
				strides_functor{stride_, layout_}
			}
		};
	}

	iterator end() const {
		return iterator{
			base_,
			TransformIterator{
				CountingIterator{count_*(layout_.extensions().num_elements())},
				strides_functor{stride_, layout_}
			}
		};
	}

	protected:
	Base base_;
	std::ptrdiff_t stride_;
	Layout layout_;

	std::ptrdiff_t count_;
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

	cudaHostRegister((void*)first_.base(), count*first_.stride()*sizeof(T1), cudaHostRegisterPortable);
	// cudaHostRegisterReadOnly not available in cuda < 11.1

	auto source_range = elements_range<Q1*>                      {first_ .base(), count, first_ ->extensions(), first_ .stride(), first_ ->strides()};
	auto destin_range = elements_range<thrust::cuda::pointer<Q2>>{result_.base(), count, result_->extensions(), result_.stride(), result_->strides()};

	::thrust::copy_n(
		thrust::device,
		source_range.begin(), count*first_->num_elements(),
		destin_range.begin()
	);

	cudaHostUnregister((void*)first_.base());

//	::thrust::for_each(
//		::thrust::make_counting_iterator(0L),
//		::thrust::make_counting_iterator(count*first_->num_elements()),
//		[tmpdata = tmp.data_elements(), first_, count, result_, x = first_->extensions()] __device__ (auto n){
//			auto const ij = (count*x).from_linear(n);
//			result_[std::get<0>(ij)][std::get<1>(ij)] = tmpdata[n];//first_[std::get<0>(ij)][std::get<1>(ij)];
//		}
//	);

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

