#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&c++ -Wall -Wextra -Wpedantic -Wno-deprecated-declarations $0 -o $0x -lcudart -lcublas -lboost_unit_test_framework \
`pkg-config --cflags --libs blas` -lboost_timer &&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../../adaptors/blas/cuda.hpp" // must be included before blas/gemm.hpp
#include "../../../adaptors/blas/gemm.hpp"

#include "../../../adaptors/cuda.hpp"
#include "../../../array.hpp"

BOOST_AUTO_TEST_CASE(multi_blas_cuda_cpu){

	namespace multi = boost::multi;
	{
		multi::array<double, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<double, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<double, 2> C({size(rotated(B)), size(A)});

		multi::cuda::array<double, 2> const Agpu = A;
		multi::cuda::array<double, 2> const Bgpu = B;
		multi::cuda::array<double, 2> Cgpu({size(rotated(B)), size(A)});

		using multi::blas::gemm;

		gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		gemm('T', 'T', 1., Agpu, Bgpu, 0., Cgpu);

		BOOST_REQUIRE( C == Cgpu );

		multi::cuda::managed::array<double, 2> const Amgpu = A;
		multi::cuda::managed::array<double, 2> const Bmgpu = B;
		multi::cuda::managed::array<double, 2> Cmgpu({size(rotated(B)), size(A)});

		using multi::blas::gemm;
		gemm('T', 'T', 1., Amgpu, Bmgpu, 0., Cmgpu);

		BOOST_REQUIRE( C == Cmgpu );
	}
	{
		using complex = std::complex<double>;
		multi::array<complex, 2> const A = {
			{ 1., 3., 4.},
			{ 9., 7., 1.}
		};
		multi::array<complex, 2> const B = {	
			{ 11., 12., 4., 3.},
			{  7., 19., 1., 2.},
			{ 11., 12., 4., 1.}
		};
		multi::array<complex, 2> C({size(rotated(B)), size(A)});

		multi::cuda::array<complex, 2> const Agpu = A;
		multi::cuda::array<complex, 2> const Bgpu = B;
		multi::cuda::array<complex, 2> Cgpu({size(rotated(B)), size(A)});

		using multi::blas::gemm;

		gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
		gemm('T', 'T', 1., Agpu, Bgpu, 0., Cgpu);

		BOOST_REQUIRE( C == Cgpu );

		multi::cuda::managed::array<complex, 2> const Amgpu = A;
		multi::cuda::managed::array<complex, 2> const Bmgpu = B;
		multi::cuda::managed::array<complex, 2> Cmgpu({size(rotated(B)), size(A)});

		using multi::blas::gemm;
		gemm('T', 'T', 1., Amgpu, Bmgpu, 0., Cmgpu);

		BOOST_REQUIRE( C == Cmgpu );
	}

}

