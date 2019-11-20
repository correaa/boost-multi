#ifdef COMPILATION_INSTRUCTIONS
(echo '#include"'$0'"'>$0.cpp)&&clang++ `#-D_DISABLE_CUDA_SLOW` $0 -o $0x -lcudart -lcublas -lboost_unit_test_framework \
`pkg-config --cflags --libs blas`&&$0x&&rm $0x $0.cpp; exit
#endif
// © Alfredo A. Correa 2019

#define BOOST_TEST_MODULE "C++ Unit Tests for Multi cuBLAS gemm"
#define BOOST_TEST_DYN_LINK
#include<boost/test/unit_test.hpp>

#include "../../../memory/adaptors/cuda/managed/ptr.hpp"

#include "../../../adaptors/blas.hpp"
#include "../../../adaptors/blas/cuda.hpp"

#include "../../../adaptors/cuda.hpp"
#include "../../../array.hpp"

namespace multi = boost::multi;

template<class M> decltype(auto) print(M const& C){
	using std::cout;
	using multi::size;
	for(int i = 0; i != size(C); ++i){
		for(int j = 0; j != size(C[i]); ++j)
			cout<< C[i][j] <<' ';
		cout<<std::endl;
	}
	return cout<<std::endl;
}

BOOST_AUTO_TEST_CASE(multi_blas_cuda_cpu){
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
		multi::array<double, 2> C({size(A), size(rotated(B))});//, size(A)});
		using multi::blas::gemm;
		using multi::blas::operation;
		gemm(operation::identity, operation::identity, 1., A, B, 0., C);
		print(C);
		return;
	//	gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
#if 1
		{
			multi::cuda::array<double, 2> const Agpu = A;
			multi::cuda::array<double, 2> const Bgpu = B;
			multi::cuda::array<double, 2> Cgpu({size(rotated(B)), size(A)});

			gemm(operation::identity, operation::identity, 1., Agpu, Bgpu, 0., Cgpu);
		//	BOOST_REQUIRE( C == Cgpu );
		}
#endif
#if 0
		{
			multi::cuda::managed::array<double, 2> const Amgpu = A;
			multi::cuda::managed::array<double, 2> const Bmgpu = B;
			multi::cuda::managed::array<double, 2> Cmgpu({size(rotated(B)), size(A)});

			using multi::blas::gemm;
			using multi::blas::operation;
			gemm(operation::identity, operation::identity, 1., Amgpu, Bmgpu, 0., Cmgpu);
		//	BOOST_REQUIRE( C == Cmgpu );
		}
#endif
	}
#if 0
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
		using multi::blas::gemm;
	//	gemm('T', 'T', 1., A, B, 0., C); // C^T = A*B , C = (A*B)^T, C = B^T*A^T , if A, B, C are c-ordering (e.g. array or array_ref)
#if 0
		{
			multi::cuda::array<complex, 2> const Agpu = A;
			multi::cuda::array<complex, 2> const Bgpu = B;
			multi::cuda::array<complex, 2> Cgpu({size(rotated(B)), size(A)});

			using multi::blas::gemm;

			gemm('T', 'T', 1., Agpu, Bgpu, 0., Cgpu);
			BOOST_REQUIRE( C == Cgpu );
		}
		{
			multi::cuda::managed::array<complex, 2> const Amgpu = A;
			multi::cuda::managed::array<complex, 2> const Bmgpu = B;
			multi::cuda::managed::array<complex, 2> Cmgpu({size(rotated(B)), size(A)});

			using multi::blas::gemm;
			gemm('T', 'T', 1., Amgpu, Bmgpu, 0., Cmgpu);
			BOOST_REQUIRE( C == Cmgpu );
		}
#endif
	}
#endif
}

