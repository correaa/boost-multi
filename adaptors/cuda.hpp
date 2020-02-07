#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&$CXX -D_TEST_MULTI_ADAPTORS_CUDA $0.cpp -o $0x -lcudart -lboost_unit_test_framework&&$0x&&rm $0x $0.cpp;exit
#endif
// © Alfredo A. Correa 2019-2020

#ifndef MULTI_ADAPTORS_CUDA_HPP
#define MULTI_ADAPTORS_CUDA_HPP

#include "../memory/adaptors/cuda/allocator.hpp"
#include "../memory/adaptors/cuda/managed/allocator.hpp"

#include "../array.hpp"

namespace boost{
namespace multi{
namespace cuda{

	template<class T>
	using allocator = multi::memory::cuda::allocator<T>;

	template<class T> using ptr = multi::memory::cuda::ptr<T>;

	template<class T, multi::dimensionality_type D>
	using array = multi::array<T, D, cuda::allocator<T>>;

	template<class T, multi::dimensionality_type D>
	using array_ref = multi::array_ref<T, D, cuda::ptr<T>>;

	template<class T, multi::dimensionality_type D>
	using static_array = multi::static_array<T, D, cuda::allocator<T>>;

	namespace managed{
		template<class T>
		using allocator = multi::memory::cuda::managed::allocator<T>;

		template<class T> using ptr = multi::memory::cuda::managed::ptr<T>;

		template<class T, multi::dimensionality_type D>
		using array = multi::array<T, D, cuda::managed::allocator<T>>;

		template<class T, multi::dimensionality_type D>
		using array_ref = multi::array<T, D, multi::memory::cuda::managed::ptr<T>>;

		template<class T, multi::dimensionality_type D>
		using static_array = multi::array<T, D, multi::memory::cuda::managed::ptr<T>>;
	}

}

/*
auto copy(const double* first, const double* last, boost::multi::array_iterator<double, 1, boost::multi::memory::cuda::managed::ptr<double, double*>, double&> d_first){
	return copy(
		boost::multi::array_iterator<double, 1, double const*, double const&>(first), 
		boost::multi::array_iterator<double, 1, double const*, double const&>(last), 
		d_first
	);
}*/

}}

#ifdef _TEST_MULTI_ADAPTORS_CUDA

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA adaptor"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

namespace multi = boost::multi;
namespace cuda = multi::cuda;

BOOST_AUTO_TEST_CASE(multi_adaptors_cuda_copy_1d){
	multi::array<double, 1> A(4, 99.);
	cuda::array<double, 1> Agpu(4);
	BOOST_REQUIRE( extensions(A) == extensions(Agpu) );
	Agpu = A;
//	BOOST_REQUIRE( Agpu[1] == 99. );
}

template<class... T> void what(T&&...) = delete;

BOOST_AUTO_TEST_CASE(multi_adaptors_cuda_copy_2d){
	multi::array<double, 2> A({4, 4}, 99.);
	cuda::array<double, 2> Agpu({4, 4}, 99.);
	BOOST_REQUIRE( extensions(A) == extensions(Agpu) );
	what( cuda::array<double, 2>::value_type() );
//	Agpu = A;
//	BOOST_REQUIRE( Agpu[1] == 99. );
}

#if 0
BOOST_AUTO_TEST_CASE(multi_adaptors_cuda_copy){

	multi::array<double, 2> A({4, 5}, 99.);
	cuda::array<double, 2> Agpu = A;

}

BOOST_AUTO_TEST_CASE(multi_adaptors_cuda){

	multi::array<double, 2> A({4, 5}, 99.);
	cuda::array<double, 2> Agpu = A;
	assert( Agpu == A );

	cuda::managed::array<double, 2> Amng = A;
	assert( Amng == Agpu );

	cuda::array_ref<double, 2> Rgpu(data_elements(Agpu), extensions(Agpu));

	{std::allocator<double> a = get_allocator(A);}

	{
		cuda::ptr<double> p;
		using multi::get_allocator;
		cuda::allocator<double> a = get_allocator(p); (void)a;
	}
	{
		cuda::managed::ptr<double> p;
		using multi::get_allocator;
		cuda::managed::allocator<double> a = get_allocator(p); (void)a;
	}
	{
		double* p = nullptr;
		using multi::get_allocator;
		std::allocator<double> a = get_allocator(p); (void)a;
	}
	{
		multi::array<double, 2> arr;
		std::allocator<double> a = get_allocator(arr);
	}
	{
		cuda::array<double, 2> arr;
		cuda::allocator<double> a = get_allocator(arr); (void)a;
	}
	{
//		cuda::array<double, 0> arr = 45.;
//		BOOST_REQUIRE( arr() == 45. );
	}
	{
//		cuda::managed::array<double, 0> arr = 45.;
//		BOOST_REQUIRE( arr() == 45. );
	}
	{
		cuda::managed::array<double, 1> arr = {1.2, 3.4, 4.5};
	}
	{
		using complex = std::complex<double>;
		cuda::managed::array<complex, 2> a({1000, 1000}, 99.);
		BOOST_REQUIRE( size(a) == 1000 );
		cuda::managed::array<complex, 2> b;
		b = std::move(a);
		BOOST_REQUIRE( size(b) == 1000 );
		BOOST_REQUIRE( size(a) == 0 );
	}
}
#endif
#endif
#endif

