#define BOOST_TEST_MODULE "C++ Unit Tests for Multi CUDA thrust"
#include<boost/test/unit_test.hpp>

#include "../../../adaptors/thrust/fix_complex_traits.hpp"
#include          "../../../detail/fix_complex_traits.hpp"

#include <boost/timer/timer.hpp>

#include "../../../adaptors/thrust.hpp"

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/memory.h>
#include <thrust/uninitialized_copy.h>

#include <boost/mpl/list.hpp>

namespace multi = boost::multi;

BOOST_AUTO_TEST_CASE(thrust_array) {
	multi::thrust::cuda::array<double, 2> C({2, 3});

	C[0][0] = 0. ;
	C[1][1] = 11.;
	BOOST_REQUIRE( C[1][1] == 11. );
}

BOOST_AUTO_TEST_CASE(issue_118) {

	using Allocator = thrust::device_allocator<double>;
	multi::array<double, 2, Allocator> M({3, 3}, Allocator{});

	M[1][2] = 12.;

	BOOST_REQUIRE( M[1][2] == 12. );

	thrust::cuda::pointer<double> cpd = thrust::device_ptr<double>{};
}

using types_list = boost::mpl::list<char, double, std::complex<double>, thrust::complex<double> >;
BOOST_AUTO_TEST_CASE_TEMPLATE(thrust_cpugpu_1D_issue123, T, types_list) {

	static_assert( multi::is_trivially_default_constructible<T>{}, "!");
	static_assert( std::is_trivially_copy_constructible<T>{}   , "!");
	static_assert( std::is_trivially_assignable<T&, T>{}       , "!");

	std::cout<<"==="<<std::endl;

	multi::array<T, 1, thrust::cuda::allocator<T>> Devc(multi::extensions_t<1>{10240*10240});
	multi::array<T, 1>                             Host(multi::extensions_t<1>{10240*10240}, T{});

	std::cout<<"1D "<< typeid(T).name() <<" total data size: "<< Host.num_elements()*sizeof(T) / 1073741824. <<" GB\n---\n";
	{
		Devc = Host;
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc = Host;
		std::cout<<"contiguous host to devc: "<< Host.num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc.sliced(0, 10240*10240/2) = Host.sliced(0, 10240*10240/2);           //  0.005292s
		std::cout<<"sliced     host to devc: "<< Host.sliced(0, 10240*10240/2).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
//	{
//		boost::timer::auto_cpu_timer t{""};
//		Devc.sliced(0, 5120, 2) = Host.sliced(0, 5120, 2);  // 0.002859s
//		std::cout<<"strided host to device "<< Host.sliced(0, 5120, 2).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
//	}
//  BOOST_TEST_REQUIRE( T{Devc[123][456]} == T{} );
	{
		boost::timer::auto_cpu_timer t{""};
//		Host = Devc;
		std::cout<<"contiguous devc to host: "<< Host.num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Host.sliced(0, 10240*10240/2) = Devc.sliced(0, 10240*10240/2);           //  0.005292s
		std::cout<<"sliced     devc to host: "<< Host.sliced(0, 10240*10240/2).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	std::cout<<"==="<<std::endl;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(thrust_cpugpu_issue123, T, types_list) {

	multi::array<T, 2, thrust::cuda::allocator<T>> Devc({10240, 10240});
	multi::array<T, 2>                             Host({10240, 10240}, T{});

	std::cout<<"2D "<< typeid(T).name() <<" max data size "<< Host.num_elements()*sizeof(T) / 1073741824. <<" GB\n---\n";
	{
		Devc({0, 5120},{0, 5120}) = Host({0, 5120},{0, 5120});  // 0.002859s
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc = Host;
		std::cout<<"contiguous host to devc: "<< Host.num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc.sliced(0, 5120) = Host.sliced(0, 5120);           //  0.005292s
		std::cout<<"sliced     host to devc: "<< Host.sliced(0, 5120).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc({0, 5120},{0, 5120}) = Host({0, 5120},{0, 5120});  // 0.002859s
		std::cout<<"strided    host to devc: "<< Host({0, 5120},{0, 5120}).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Host = Devc;
		std::cout<<"contiguous devc to host: "<< Host.num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Host.sliced(0, 5120) = Devc.sliced(0, 5120);           //  0.005292s
		std::cout<<"sliced     devc to host: "<< Host.sliced(0, 5120).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Host({0, 5120},{0, 5120}) = Devc({0, 5120},{0, 5120});  // 0.002859s
		std::cout<<"strided    devc to host: "<< Host({0, 5120},{0, 5120}).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	std::cout<<"==="<<std::endl;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(thrust_cpugpu_issue123_3D, T, types_list) {

	multi::array<T, 3, thrust::cuda::allocator<T>> Devc({1024, 1024, 100});
	multi::array<T, 3>                             Host({1024, 1024, 100}, T{});

	std::cout<<"3D "<< typeid(T).name() <<" max data size "<< Host.num_elements()*sizeof(T) / 1073741824. <<" GB\n---\n";
	{
		Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc = Host;
		std::cout<<"contiguous host to devc: "<< Host.num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc.sliced(0, 512) = Host.sliced(0, 512);           //  0.005292s
		std::cout<<"sliced     host to devc: "<< Host.sliced(0, 512).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
		Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
		std::cout<<"strided    host to devc: "<< Host({0, 512},{0, 512}, {0, 512}).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Host = Devc;
		std::cout<<"contiguous devc to host: "<< Host.num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Devc.sliced(0, 512) = Host.sliced(0, 512);           //  0.005292s
		std::cout<<"sliced     devc to host: "<< Host.sliced(0, 512).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	{
		boost::timer::auto_cpu_timer t{""};
//		Devc({0, 512}, {0, 512}, {0, 512}) = Host({0, 512}, {0, 512}, {0, 512});  // 0.002859s
		std::cout<<"strided    devc to host: "<< Host({0, 512},{0, 512}, {0, 512}).num_elements()*sizeof(T) / (t.elapsed().wall/1e9) / 1073741824. << "GB/sec\n";
	}
	std::cout<<"==="<<std::endl;
}

#if 0
namespace inq {
	using complex = thrust::complex<double>;
}

BOOST_AUTO_TEST_CASE(thrust_complex_cached_1D) {
	using T = inq::complex;
	multi::array<T, 1, boost::multi::memory::cuda::cached::allocator<T> > aa(10, T{1., 1.});
	multi::array<T, 1, boost::multi::memory::cuda::cached::allocator<T> > bb(10, T{2., 2.});

	bb = aa;

	BOOST_REQUIRE(( bb[0] == T{1., 1.} ));
}

BOOST_AUTO_TEST_CASE(thrust_complex_cached_without_values_1D) {
	using T = inq::complex;
	multi::array<T, 1, boost::multi::memory::cuda::cached::allocator<T> > aa(10);
	multi::array<T, 1, boost::multi::memory::cuda::cached::allocator<T> > bb(10);
	BOOST_REQUIRE( aa.size() == 10 );
	BOOST_REQUIRE( bb.size() == 10 );

	bb = aa;

	BOOST_REQUIRE(( bb[0] == aa[0] ));
}

BOOST_AUTO_TEST_CASE(thrust_complex_cached_2D) {
	using T = inq::complex;
	multi::array<T, 2, boost::multi::memory::cuda::cached::allocator<T> > aa({10, 20}, T{1., 1.});
	multi::array<T, 2, boost::multi::memory::cuda::cached::allocator<T> > bb({10, 20}, T{2., 2.});

	bb = aa;

	BOOST_REQUIRE(( bb[0][0] == T{1., 1.} ));
}

BOOST_AUTO_TEST_CASE(thrust_complex_cached_without_values_2D) {
	using T = inq::complex;
	multi::array<T, 2, boost::multi::memory::cuda::cached::allocator<T> > aa({10, 20});
	multi::array<T, 2, boost::multi::memory::cuda::cached::allocator<T> > bb({10, 20});
	BOOST_REQUIRE( aa.size() == 10 );
	BOOST_REQUIRE( bb.size() == 10 );

	bb = aa;

	BOOST_REQUIRE(( bb[0][0] == aa[0][0] ));
}

BOOST_AUTO_TEST_CASE(array) {

//{
//	multi::thrust::cuda::array<double, 2> C({2, 3});

//	C[0][0] = 0. ;
//	C[1][1] = 11.;
//	BOOST_TEST_REQUIRE( C[1][1] == 11. );
//}

//{
//	multi::array<double, 2> const H = {
//		{00., 01., 02.},
//		{10., 11., 12.},
//	};

//	BOOST_TEST_REQUIRE( H[1][1] == 11. );

//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		BOOST_REQUIRE( C.num_elements() == H.num_elements() );

//		thrust::copy_n(H.data_elements(), H.num_elements(), C.data_elements());
//		BOOST_TEST_REQUIRE( C[1][1] == 11. );
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		BOOST_REQUIRE( C.num_elements() == H.num_elements() );

//		std::copy_n(H.data_elements(), H.num_elements(), C.data_elements());
//		BOOST_TEST_REQUIRE( C[1][1] == 11. );
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		BOOST_REQUIRE( C.num_elements() == H.num_elements() );

//		std::uninitialized_copy_n(H.data_elements(), H.num_elements(), C.data_elements());
//		BOOST_TEST_REQUIRE( C[1][1] == 11. );
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		BOOST_REQUIRE( C.num_elements() == H.num_elements() );

//		thrust::uninitialized_copy_n(H.data_elements(), H.num_elements(), C.data_elements());
//		BOOST_TEST_REQUIRE( C[1][1] == 11. );
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		BOOST_REQUIRE( C.extensions() == H.extensions() );
//		thrust::copy_n(H.begin(), H.size(), C.begin());
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		BOOST_REQUIRE( C.extensions() == H.extensions() );
//		std::copy_n(H.begin(), H.size(), C.begin());
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C(H.extensions());
//		C = H;
//		BOOST_REQUIRE( C == H );
//	}
//	{
//		multi::thrust::cuda::array<double, 2> C = H;
//		BOOST_REQUIRE( C == H );
//	}
//}

}
#endif
