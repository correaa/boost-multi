#ifdef COMPILATION// -*-indent-tabs-mode:t;c-basic-offset:4;tab-width:4;-*-
/usr/local/cuda-11.0/bin/nvcc -std=c++17 -ftemplate-backtrace-limit=0 $0 -o $0.$X `pkg-config --cflags --libs cudart-11.0 cuda-11.0` -lboost_timer&&$0.$X&&rm $0.$X;exit
#endif
// © Alfredo A. Correa 2020

#include<benchmark/benchmark.h>

#include<thrust/complex.h>
#include<thrust/device_allocator.h>
#include<thrust/device_vector.h>

#include "../../multi/array_ref.hpp"

#include "../../multi/adaptors/thrust.hpp"

#include<thrust/iterator/counting_iterator.h>

#include "../../multi/array.hpp"

#include<algorithm> // for_each
#include<execution>

namespace multi = boost::multi;

#if not defined(NDEBUG)
#warning "Benchmark in debug mode?"
#endif

using T = double;//thrust::complex<double>;

namespace std{
	template<class T>
	struct is_trivially_copy_assignable<thrust::complex<T>> : std::true_type{};
}

void BM_cpu_vector_assignment                     (benchmark::State& st){
	std::vector<T> const A(1<<27, 1.);
	std::vector<T>       B(A.size(), 2.);
	for(auto _ : st){
		thrust::copy(A.begin(), A.end(), B.begin());
		benchmark::DoNotOptimize(B.data());
	}
	std::cout << A.size()*sizeof(A.front())/1e6 << "MB" << std::endl;
	st.SetBytesProcessed(st.iterations()*A.size()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.size());
}

void BM_device_cudaMemcpy_assignment              (benchmark::State& st){
	thrust::device_vector<T> const A(1<<27, 1.);
	thrust::device_vector<T>       B(A.size(), 2.);
	for(auto _ : st){
		cudaMemcpy(raw_pointer_cast(B.data()), raw_pointer_cast(A.data()), A.size()*sizeof(T), cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		benchmark::DoNotOptimize(B.data());
	}
	st.SetBytesProcessed(st.iterations()*A.size()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.size());
}

void BM_device_vector_assignment                  (benchmark::State& st){
	thrust::device_vector<T> const A(1<<27, 1.);
	thrust::device_vector<T>       B(A.size(), 2.);
	for(auto _ : st){
		B = A;
		benchmark::DoNotOptimize(B.data());
	}
	st.SetBytesProcessed(st.iterations()*A.size()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.size());
}

void BM_device_array_assignment                   (benchmark::State& st){
	using alloc = thrust::device_allocator<T>; // std::allocator<T>;
	multi::array<T, 1, alloc> const A(1<<27, 1.);
	multi::array<T, 1, alloc>       B(extensions(A), 2.);
	for(auto _ : st){
		B() = A();
	}
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}

void BM_cpu_array_2D_assignment                   (benchmark::State& st){
	using alloc = std::allocator<T>; //thrust::device_allocator<T>; // std::allocator<T>;
	multi::array<T, 2, alloc> A({1<<14, 1<<13}, 1.); A[100][200] = 99.;
	multi::array<T, 2, alloc> B(extensions(A), 2.);
	for(auto _ : st){
	//	auto const ss = std::get<1>(A.sizes());
	//	for(auto n = 0l; n != A.num_elements(); ++n){
		//	B.data_elements()[n] = A.data_elements()[n];
		//	auto const i = n / ss; auto const j = n % ss; B[i][j] = A[i][j];
		//	B.elements_at(n) = A.elements_at(n);
	//	}
//		std::for_each(std::execution::par, 
//			thrust::make_counting_iterator(0l), 
//			thrust::make_counting_iterator(A.num_elements()), 
//			[&](auto n){B.data_elements()[n] = A.data_elements()[n];}
//		);
	//	std::copy(std::execution::par, A.elements().begin(), A.elements().end(), B.elements().begin());
		B() = A();
	}
	if( T{B[10][10]}   == T{2.})  throw std::runtime_error("120");;
	if( T{B[100][200]} != T{99.}) throw std::runtime_error("121");;

	std::cout << A.num_elements()*sizeof(T)/1e6 << "MB"<<std::endl;
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}

template<class T> void what(T&&) = delete;

template<class AP, class BP>
struct F{
	AP ap;
	BP bp;
	F(AP const& ap, BP const& bp) : ap{ap}, bp{bp}{}
	constexpr void operator()(std::ptrdiff_t n) const{
		auto p = n % ap->extensions();
		bp->apply(p) = ap->apply(p);
	}
};

void BM_device_array_2D_assignment                (benchmark::State& st){
	using alloc = thrust::device_allocator<T>; // std::allocator<T>;
	multi::array<T, 2, alloc>  A({1<<14, 1<<13}, 1.); A[100][200] = 99.;
	multi::array<T, 2, alloc>  B(extensions(A), 2.);
	if(B.extensions() != A.extensions()) throw std::runtime_error("mismatch");
//	throw std::runtime_error("hola");
	for(auto _ : st){
//		boost::multi::run( A.size(), (~A).size(), 
////			[
////				b1 = raw_pointer_cast(A.base()), s1 = A.stride(), ss1 = (~A).stride(), 
////				b2 = raw_pointer_cast(B.base()), s2 = B.stride(), ss2 = (~B).stride()  
////			] __device__ (std::ptrdiff_t ii, std::ptrdiff_t jj){
////				*(b2 + s2*ii + ss2*jj) = *(b1 + s1*ii + ss1*jj); // *(b1 + s1*ii + ss1*jj);
////			}
////			[
////				b1 = A.base(), s1 = A.stride(), ss1 = (~A).stride(), 
////				b2 = B.base(), s2 = B.stride(), ss2 = (~B).stride()  
////			] __device__ (std::ptrdiff_t ii, std::ptrdiff_t jj){
////				*(b2 + s2*ii + ss2*jj) =  *(b1 + s1*ii + ss1*jj);
////			}
//			[b = B.begin(), a = A.begin()] __device__ (std::ptrdiff_t i, std::ptrdiff_t j){
//				b[i][j] = a[i][j];
//			}
//		);
	//	auto b = &B();//std::decay_t<decltype(B())>::ptr{B.base(), B.layout()};
	//	auto a = &A();//std::decay_t<decltype(A())>::ptr{A.base(), A.layout()};
	//	F const f{&A(), &B()};
	//	what(f);
	//	boost::multi::run( A.num_elements(), 
	//		[bp = &B(), ap = &A(), x = A.extensions()] __device__ (std::ptrdiff_t n){bp->apply(n % x) = ap->apply(n % x);}
		//	[b = B.begin(), a = A.begin(), x = A.extensions()] __device__ (std::ptrdiff_t n){b.apply(n % x) = a.apply(n % x);}
		//	[b = B.begin(), a = A.begin(), N = A.size()] __device__ (std::ptrdiff_t n){b.apply(n % (N*b->extensions())) = a.apply(n % (N*a->extensions()));}
		//	f
	//	);
			//	(*bp)[std::get<0>(p)][std::get<1>(p)] = (*ap)[std::get<0>(p)][std::get<1>(p)];
			//	b(std::get<0>(p), std::get<1>(p)) = a(std::get<0>(p), std::get<1>(p));
			//	(*b)[std::get<0>(p)][std::get<1>(p)]
			//	*((*b).base() + b->stride()*std::get<0>(p) + std::get<1>(b->strides())*std::get<1>(p)) 
			//	(*b)[std::get<0>(p)][std::get<1>(p)]
			//		= *(a.base() + a->stride()*std::get<0>(p) + std::get<1>(a->strides())*std::get<1>(p)) ;
			//	(*b)[1][1];
			//	b( p ) = a( p );
			//	b[ std::get<0>(p) ][ std::get<1>(p) ] = a [ std::get<0>(p) ][ std::get<1>(p) ];
			//	std::apply(b , p) = Apply(a, p);
			//	b[std::get<0>(p)][std::get<1>(p)] = a[std::get<0>(p)][std::get<1>(p)];
			//	Apply(*b, p) = Apply(*a, p);
			//	b[ n/(b->size()) ][ n % (b->size()) ] = a[ n/(b->size()) ][ n % (b->size()) ];
			//}
		//);
		B() = A();
	}
	if( T{B[10][10]}   == T{2.})  throw std::runtime_error("120");;
	if( T{B[100][200]} != T{99.}) throw std::runtime_error("121");;
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}

void BM_cpu_array_5D_assignment                   (benchmark::State& st){
	using alloc = std::allocator<T>;
	multi::array<T, 4, alloc> const A({35, 69, 35*2, 384}, 1.);
	multi::array<T, 5, alloc>       B({2, 35, 69, 35, 384}, 2.);
	if(B.extensions() != A.unrotated(2).partitioned(2).transposed().rotated().transposed().rotated().extensions()) throw __LINE__;
	for(auto _ : st){
		B() = A.unrotated(2).partitioned(2).transposed().rotated().transposed().rotated();
	}
	std::cout << "sizees " << B.num_elements()*sizeof(T) <<std::endl;
	if( T{B[1][10][10][10][10]} == T{2.}) throw 0;
	st.SetBytesProcessed(st.iterations()*A.num_elements()*sizeof(T));
	st.SetItemsProcessed(st.iterations()*A.num_elements());
}

//BENCHMARK(BM_cpu_vector_assignment);
//BENCHMARK(BM_device_vector_assignment);
//BENCHMARK(BM_device_cudaMemcpy_assignment);
//BENCHMARK(BM_device_array_assignment);
BENCHMARK(BM_cpu_array_2D_assignment);
BENCHMARK(BM_device_array_2D_assignment);
//BENCHMARK(BM_cpu_array_5D_assignment);

BENCHMARK_MAIN();


