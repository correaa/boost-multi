// Copyright 2021-2025 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#include <boost/core/lightweight_test.hpp>

#include <boost/multi/adaptors/thrust.hpp>
#include <boost/multi/adaptors/thrust/managed_allocator.hpp>
#include <boost/multi/array.hpp>

#include <thrust/complex.h>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/universal_allocator.h>

#include <boost/timer/timer.hpp>

#include <numeric>

namespace multi = boost::multi;

#ifdef __NVCC__
template<>
inline constexpr bool ::boost::multi::force_element_trivial_default_construction<::std::complex<double>> = true;
template<>
inline constexpr bool ::boost::multi::force_element_trivial_default_construction<::std::complex<float>> = true;
template<>
inline constexpr bool ::boost::multi::force_element_trivial_default_construction<::thrust::complex<double>> = true;
template<>
inline constexpr bool ::boost::multi::force_element_trivial_default_construction<::thrust::complex<float>> = true;
#else  // vvv nvcc (12.1?) doesn't support this kind of customization: "error: expected initializer before ‘<’"
template<class T>
inline constexpr bool ::boost::multi::force_element_trivial_default_construction<::std::complex<T>> = std::is_trivially_default_constructible<T>::value;
template<class T>
inline constexpr bool ::boost::multi::force_element_trivial_default_construction<::thrust::complex<T>> = std::is_trivially_default_constructible<T>::value;
#endif

namespace {

template<class T> using test_allocator =
	//  multi::thrust::cuda::managed_allocator<T>
	thrust::cuda::allocator<T>;
}

auto universal_memory_supported() -> bool {
	int d;
	cudaGetDevice(&d);
	int is_cma = 0;
	cudaDeviceGetAttribute(&is_cma, cudaDevAttrConcurrentManagedAccess, d);
	return (is_cma == 1)?true:false;
}

auto main() -> int {  // NOLINT(readability-function-cognitive-complexity,bugprone-exception-escape)
	
	// BOOST_AUTO_TEST_CASE(cuda_universal_empty)
	if(universal_memory_supported())
	{
		using complex = thrust::complex<double>;
		multi::array<complex, 2, thrust::cuda::universal_allocator<complex>> A;
		multi::array<complex, 2, thrust::cuda::universal_allocator<complex>> B = A;
		BOOST_TEST( A.is_empty() );
		BOOST_TEST( B.is_empty() );
		BOOST_TEST( A == B );
	}

	// BOOST_AUTO_TEST_CASE(cuda_allocators)
	{

		multi::array<double, 1, thrust::cuda::allocator<double>> A1(200, 0.0);

		BOOST_TEST( size(A1) == 200 );
		A1[100] = 1.0;

		multi::array<double, 1, thrust::cuda::allocator<double>> const B1(200, 2.0);
		BOOST_TEST( B1[10] == 2.0 );

		A1[10] = B1[10];
		BOOST_TEST( A1[10] == 2.0 );
	}

	// BOOST_AUTO_TEST_CASE(cuda_1d_initlist)
	{
		multi::array<double, 1, thrust::device_allocator<double>> A1 = {1.0, 2.0, 3.0};
		BOOST_TEST( A1.size() == 3 );

		// BOOST_TEST( size(A1) == 200 );
		// A1[100] = 1.0;

		// multi::array<double, 1, thrust::cuda::allocator<double>> const B1(200, 2.0);
		// BOOST_TEST( B1[10] == 2.0 );

		// A1[10] = B1[10];
		// BOOST_TEST( A1[10] == 2.0 );

		{
			multi::array<int, 1, thrust::device_allocator<int>> A = {1, 2, 3};
			multi::array<int, 1, thrust::device_allocator<int>> B(3, 0);

			BOOST_TEST( A[1] == 2 );

			thrust::transform(
				A.begin(), A.end(),
				B.begin(),
				[] __host__ __device__(int elem) {
					return elem * 2;
				}  // *2.0;}
			);

			BOOST_TEST( B[1] == 4 );
		}

		{
			multi::array<double, 1, thrust::device_allocator<double>> A = {1.0, 2.0, 3.0};
			multi::array<double, 1, thrust::device_allocator<double>> B(3);

			// // for(int i = 0; i != A.size(); ++i) { B[i] = A[i]*2.0; }
			// // for(auto i : A.extension()) { B[i] = A[i]*2.0; }

			thrust::transform(
				A.begin(), A.end(),
				B.begin(),
				[] __host__ __device__(double const& elem) { return elem * 2.0; }
			);

			BOOST_TEST( B[1] == 4.0 );
		}
	}

	// BOOST_AUTO_TEST_CASE(test_univ_alloc)
	if(universal_memory_supported())
	{
		multi::array<double, 2, thrust::cuda::universal_allocator<double>> Dev({128, 128});
		*raw_pointer_cast(Dev.base()) = 99.0;
	}

	// BOOST_AUTO_TEST_CASE(mtc_universal_array)
	if(universal_memory_supported())
	{
		multi::thrust::cuda::universal_array<double, 2> Dev({128, 128});
		*raw_pointer_cast(Dev.base()) = 99.0;
	}

	// BOOST_AUTO_TEST_CASE(mtc_universal_coloncolon_array)
	if(universal_memory_supported())
	{
		multi::thrust::cuda::universal::array<double, 2> Dev({128, 128});
		*raw_pointer_cast(Dev.base()) = 99.0;
	}

	BOOST_AUTO_TEST_CASE(test_alloc) {
		multi::array<double, 2, thrust::cuda::allocator<double>> Dev({128, 128});
		// *raw_pointer_cast(Dev.base()) = 99.0;  // segmentation fault (correct behavior)
	}

#ifdef NDEBUG
	auto const n = 1024;

	// BOOST_AUTO_TEST_CASE(thrust_copy_1D_issue123_double)
	{  // BOOST_AUTO_TEST_CASE(fdfdfdsfds) { using T = char;
		using T = double;

		static_assert(multi::is_trivially_default_constructible<T>{});
		static_assert(std::is_trivially_copy_constructible<T>{});
		static_assert(std::is_trivially_assignable<T&, T>{});

		multi::array<T, 1, test_allocator<T>> Devc(multi::extensions_t<1>{n * n});
		multi::array<T, 1, test_allocator<T>> Dev2(multi::extensions_t<1>{n * n});
		multi::array<T, 1>                    Host(multi::extensions_t<1>{n * n});
		std::iota(Host.elements().begin(), Host.elements().end(), 12.0);
		multi::array<T, 1> Hos2(multi::extensions_t<1>{n * n});

		std::cout << "| 1D `" << typeid(T).name() << "` total data size: " << Host.num_elements() * sizeof(T) / 1073741824. << " GB | speed |\n|---|---|" << std::endl;
		{
			Devc = Host;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc = Host;
			cudaDeviceSynchronize();
			std::cout << "| contiguous host -> devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.sliced(0, n * n / 2) = Host.sliced(0, n * n / 2);
			cudaDeviceSynchronize();
			std::cout << "| sliced     host -> devc | " << Host.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.strided(2) = Host.strided(2);
			cudaDeviceSynchronize();
			std::cout << "| strided    host -> devc | " << Host.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc -> host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Hos2 == Host );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n * n / 2) = Devc.sliced(0, n * n / 2);
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc -> host | " << Host.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Hos2 == Host );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.strided(2) = Devc.strided(2);
			cudaDeviceSynchronize();
			std::cout << "| strided    devc -> host | " << Host.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Hos2 == Host );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc -> devc | " << Devc.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev3 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.sliced(0, n * n / 2) = Devc.sliced(0, n * n / 2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc -> devc | " << Dev2.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.strided(2) = Devc.strided(2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| strided    devc -> devc | " << Dev2.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Host;
			cudaDeviceSynchronize();
			std::cout << "| contiguous host -> host | " << Hos2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n * n / 2) = Host.sliced(0, n * n / 2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     host -> host | " << Hos2.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.strided(2) = Host.strided(2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| strided    host -> host | " << Hos2.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		std::cout << "   " << std::endl;
	}

	// BOOST_AUTO_TEST_CASE(thrust_copy_1D_issue123_complex)
	{
		using T = thrust::complex<double>;

		static_assert(multi::is_trivially_default_constructible<T>{});
		static_assert(std::is_trivially_copy_constructible_v<T>);
		static_assert(std::is_trivially_assignable_v<T&, T>);

		multi::array<T, 1, test_allocator<T>> Devc(multi::extensions_t<1>{n * n});
		multi::array<T, 1, test_allocator<T>> Dev2(multi::extensions_t<1>{n * n});
		multi::array<T, 1>                    Host(multi::extensions_t<1>{n * n});
		std::iota(Host.elements().begin(), Host.elements().end(), 12.);
		multi::array<T, 1> Hos2(multi::extensions_t<1>{n * n});

		std::cout << "| 1D `" << typeid(T).name() << "` total data size: " << Host.num_elements() * sizeof(T) / 1073741824.0 << " GB | speed |\n|---|---|" << std::endl;
		{
			Devc = Host;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc = Host;
			cudaDeviceSynchronize();
			std::cout << "| contiguous host -> devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.sliced(0, n * n / 2) = Host.sliced(0, n * n / 2);
			cudaDeviceSynchronize();
			std::cout << "| sliced     host -> devc | " << Host.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.strided(2) = Host.strided(2);
			cudaDeviceSynchronize();
			std::cout << "| strided    host -> devc | " << Host.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc -> host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Hos2 == Host );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n * n / 2) = Devc.sliced(0, n * n / 2);
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc -> host | " << Host.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Hos2 == Host );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.strided(2) = Devc.strided(2);
			cudaDeviceSynchronize();
			std::cout << "| strided    devc -> host | " << Host.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Hos2 == Host );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc -> devc | " << Devc.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev3 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.sliced(0, n * n / 2) = Devc.sliced(0, n * n / 2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc -> devc | " << Dev2.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.strided(2) = Devc.strided(2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| strided    devc -> devc | " << Dev2.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Host;
			cudaDeviceSynchronize();
			std::cout << "| contiguous host -> host | " << Hos2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n * n / 2) = Host.sliced(0, n * n / 2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     host -> host | " << Hos2.sliced(0, n * n / 2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.strided(2) = Host.strided(2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| strided    host -> host | " << Hos2.strided(2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		std::cout << "   " << std::endl;
	}

	// BOOST_AUTO_TEST_CASE(thrust_cpugpu_2D_issue123_double)
	{
		using T = double;

		auto const exts = multi::extensions_t<2>({n, n});

		std::cout << "| 2D `" << typeid(T).name() << "` max data size " << exts.num_elements() * sizeof(T) / 1073741824.0 << " GB | speed |\n|---|---|" << std::endl;

		multi::array<T, 2, test_allocator<T>> Devc(exts);
		multi::array<T, 2, test_allocator<T>> Dev2(exts);

		multi::array<T, 2> Host(exts);
		std::iota(Host.elements().begin(), Host.elements().end(), 12.);
		multi::array<T, 2> Hos2(exts);

		{
			Devc({0, n/2}, {0, n/2}) = Host({0, n/2}, {0, n/2});  // 0.002859s
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc = Host;
			std::cout << "| contiguous host to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.sliced(0, n/2) = Host.sliced(0, n/2);  //  0.005292s
			std::cout << "| sliced     host to devc | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc({0, n/2}, {0, n/2}) = Host({0, n/2}, {0, n/2});  // 0.002859s
			std::cout << "| strided    host to devc | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Devc;
			std::cout << "| contiguous devc to host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n/2) = Devc.sliced(0, n/2);  //  0.005292s
			std::cout << "| sliced     devc to host | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, n/2}, {0, n/2}) = Devc({0, n/2}, {0, n/2});  // 0.002859s
			std::cout << "| strided    devc to host | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc -> devc | " << Devc.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev3 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.sliced(0, n/2) = Devc.sliced(0, n/2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc to devc | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2({0, n/2}, {0, n/2}) = Devc({0, n/2}, {0, n/2});  // 0.002859s
			cudaDeviceSynchronize();
			std::cout << "| strided    devc to devc | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Host;
			std::cout << "| contiguous host to host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n/2) = Host.sliced(0, n/2);  //  0.005292s
			std::cout << "| sliced     host to host | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, n/2}, {0, n/2}) = Host({0, n/2}, {0, n/2});  // 0.002859s
			std::cout << "| strided    host to host | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		std::cout << "  " << std::endl;
	}

	// BOOST_AUTO_TEST_CASE(thrust_cpugpu_2D_issue123_complex)
	{
		using T = thrust::complex<double>;

		auto const exts = multi::extensions_t<2>({n, n});

		std::cout << "| 2D `" << typeid(T).name() << "` max data size " << exts.num_elements() * sizeof(T) / 1073741824.0 << " GB | speed |\n|---|---|" << std::endl;

		multi::array<T, 2, test_allocator<T>> Devc(exts);
		multi::array<T, 2, test_allocator<T>> Dev2(exts);

		multi::array<T, 2> Host(exts);
		std::iota(Host.elements().begin(), Host.elements().end(), 12.);
		multi::array<T, 2> Hos2(exts);

		{
			Devc({0, n/2}, {0, n/2}) = Host({0, n/2}, {0, n/2});  // 0.002859s
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc = Host;
			std::cout << "| contiguous host to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.sliced(0, n/2) = Host.sliced(0, n/2);  //  0.005292s
			std::cout << "| sliced     host to devc | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc({0, n/2}, {0, n/2}) = Host({0, n/2}, {0, n/2});  // 0.002859s
			std::cout << "| strided    host to devc | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824.0 << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Devc;
			std::cout << "| contiguous devc to host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n/2) = Devc.sliced(0, n/2);  //  0.005292s
			std::cout << "| sliced     devc to host | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, n/2}, {0, n/2}) = Devc({0, n/2}, {0, n/2});  // 0.002859s
			std::cout << "| strided    devc to host | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc -> devc | " << Devc.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev3 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.sliced(0, n/2) = Devc.sliced(0, n/2);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc to devc | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2({0, n/2}, {0, n/2}) = Devc({0, n/2}, {0, n/2});  // 0.002859s
			cudaDeviceSynchronize();
			std::cout << "| strided    devc to devc | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Host;
			std::cout << "| contiguous host to host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, n/2) = Host.sliced(0, n/2);  //  0.005292s
			std::cout << "| sliced     host to host | " << Host.sliced(0, n/2).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, n/2}, {0, n/2}) = Host({0, n/2}, {0, n/2});  // 0.002859s
			std::cout << "| strided    host to host | " << Host({0, n/2}, {0, n/2}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		std::cout << "  " << std::endl;
	}

	// BOOST_AUTO_TEST_CASE(thrust_cpugpu_issue123_3D_double)
	{
		using T         = double;
		auto const exts = multi::extensions_t<3>({1024, 1024, 100});

		std::cout << "| 3D `" << typeid(T).name() << "` max data size " << exts.num_elements() * sizeof(T) / 1073741824. << " GB | speed |\n|---|---|" << std::endl;

		multi::array<T, 3, test_allocator<T>> Devc(exts);
		multi::array<T, 3, test_allocator<T>> Dev2(exts);
		multi::array<T, 3>                    Host(exts);
		std::iota(Host.elements().begin(), Host.elements().end(), 12.);
		multi::array<T, 3> Hos2(exts);

		{
			Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc = Host;
			std::cout << "| contiguous host to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << " GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.sliced(0, 512) = Host.sliced(0, 512);  //  0.005292s
			std::cout << "| sliced     host to devc | " << Host.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << " GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			std::cout << "| strided    host to devc | " << Host({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Devc;
			std::cout << "| contiguous devc to host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, 512) = Devc.sliced(0, 512);  //  0.005292s
			std::cout << "| sliced     devc to host | " << Host.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, 512}, {0, 512}, {0, 512}) = Devc({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			std::cout << "| strided    devc to host | " << Host({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc to devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Dev2 == Devc);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc -> devc | " << Devc.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev3 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.sliced(0, 512) = Devc.sliced(0, 512);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc to devc | " << Dev2.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Dev2 == Devc);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2({0, 512}, {0, 512}, {0, 512}) = Devc({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			cudaDeviceSynchronize();
			std::cout << "| strided    devc to devc | " << Dev2({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Dev2 == Devc);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Host;
			std::cout << "| contiguous host to host | " << Hos2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, 512) = Host.sliced(0, 512);  //  0.005292s
			std::cout << "| sliced     host to host | " << Hos2.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			std::cout << "| strided    host to host | " << Hos2({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		std::cout << "   " << std::endl;
	}

	// BOOST_AUTO_TEST_CASE(thrust_cpugpu_issue123_3D_complex)
	{
		using T         = thrust::complex<double>;
		auto const exts = multi::extensions_t<3>({1024, 1024, 100});

		std::cout << "| 3D `" << typeid(T).name() << "` max data size " << exts.num_elements() * sizeof(T) / 1073741824. << " GB | speed |\n|---|---|" << std::endl;

		multi::array<T, 3, test_allocator<T>> Devc(exts);
		multi::array<T, 3, test_allocator<T>> Dev2(exts);
		multi::array<T, 3>                    Host(exts);
		std::iota(Host.elements().begin(), Host.elements().end(), 12.);
		multi::array<T, 3> Hos2(exts);

		{
			Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc = Host;
			std::cout << "| contiguous host to devc | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << " GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc.sliced(0, 512) = Host.sliced(0, 512);  //  0.005292s
			std::cout << "| sliced     host to devc | " << Host.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << " GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			std::cout << "| strided    host to devc | " << Host({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Devc;
			std::cout << "| contiguous devc to host | " << Host.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, 512) = Devc.sliced(0, 512);  //  0.005292s
			std::cout << "| sliced     devc to host | " << Host.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, 512}, {0, 512}, {0, 512}) = Devc({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			std::cout << "| strided    devc to host | " << Host({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Hos2 == Host);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| contiguous devc to devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Dev2 == Devc);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			auto                         Dev3 = Devc;
			cudaDeviceSynchronize();
			std::cout << "| copy_ctr   devc -> devc | " << Devc.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev3 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			cudaMemcpy(raw_pointer_cast(Dev2.data_elements()), raw_pointer_cast(Devc.data_elements()), Devc.num_elements() * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaDeviceSynchronize();
			std::cout << "| cudaMemcpy devc -> devc | " << Dev2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST( Dev2 == Devc );
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2.sliced(0, 512) = Devc.sliced(0, 512);  //  0.005292s
			cudaDeviceSynchronize();
			std::cout << "| sliced     devc to devc | " << Dev2.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Dev2 == Devc);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Dev2({0, 512}, {0, 512}, {0, 512}) = Devc({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			cudaDeviceSynchronize();
			std::cout << "| strided    devc to devc | " << Dev2({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
			// BOOST_TEST(Dev2 == Devc);
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2 = Host;
			std::cout << "| contiguous host to host | " << Hos2.num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2.sliced(0, 512) = Host.sliced(0, 512);  //  0.005292s
			std::cout << "| sliced     host to host | " << Hos2.sliced(0, 512).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		{
			boost::timer::auto_cpu_timer t{""};
			Hos2({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
			std::cout << "| strided    host to host | " << Hos2({0, 512}, {0, 512}, {0, 512}).num_elements() * sizeof(T) / (t.elapsed().wall / 1e9) / 1073741824. << "GB/sec |" << std::endl;
		}
		std::cout << "   " << std::endl;
	}
#endif

	return boost::report_errors();
}
