#include <multi/array.hpp>
#include <multi/adaptors/thrust.hpp>

#include <thrust/system/cuda/memory.h>

namespace multi = boost::multi;

#define CUDA_CHECKED(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
        std::cerr<<"error: "<< cudaGetErrorString(code) <<" "<< file <<":"<< line <<std::endl;
        assert(0);
    }
}

#define REQUIRE(ans) { require((ans), __FILE__, __LINE__); }
inline void require(bool code, const char *file, int line, bool abort=true) {
   if (not code) {
        std::cerr<<"error: "<< file <<":"<< line <<std::endl;
        exit(666);
    }
}

// template<typename T>
// __global__ void kernel_setIdentity(int m, int n, T* A, int lda)
// {
//   int i = threadIdx.x + blockDim.x * blockIdx.x;
//   int j = threadIdx.y + blockDim.y * blockIdx.y;
//   if ((i < m) && (j < n))
//     if (i == j)
//     {
//       A[i * lda + i] = T(1.0);
//     }
//     else
//     {
//       A[j * lda + i] = T(0.0);
//     }
// }

// void set_identity(int m, int n, double* A, int lda) {
//     int xblock_dim = 16;
//     int xgrid_dim  = (m + xblock_dim - 1) / xblock_dim;
//     int ygrid_dim  = (n + xblock_dim - 1) / xblock_dim;
//     dim3 block_dim(xblock_dim, xblock_dim);
//     dim3 grid_dim(xgrid_dim, ygrid_dim);
//     kernel_setIdentity<<<grid_dim, block_dim>>>(m, n, A, lda);
//     CUDA_CHECKED(cudaGetLastError());
//     CUDA_CHECKED(cudaDeviceSynchronize());
// }

template<typename Array2DCursor>
__global__ void kernel_setIdentity(Array2DCursor home, int m, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if ((i < m) && (j < n)) {
        if (i == j) {
            home[i][j] = 1.0;
        } else {
            home[i][j] = 0.0;
        }
    }
}

template<class Array2D>
void set_identity(Array2D&& arr) {
    int xblock_dim = 16;
    auto [m, n] = arr.sizes();
    int xgrid_dim  = (m + xblock_dim - 1) / xblock_dim;
    int ygrid_dim  = (n + xblock_dim - 1) / xblock_dim;
    dim3 block_dim(xblock_dim, xblock_dim);
    dim3 grid_dim(xgrid_dim, ygrid_dim);
    kernel_setIdentity<<<grid_dim, block_dim>>>(arr.home(), m, n);
    CUDA_CHECKED(cudaGetLastError());
    CUDA_CHECKED(cudaDeviceSynchronize());
}

int main() {
    using T = double;

    {
        multi::array<double, 2, thrust::cuda::allocator<double>> A({20000, 20000});
        auto const size = A.num_elements()*sizeof(T)/1e9;
        std::cout<<"size is "<< size << "GB\n";

        auto start_time = std::chrono::high_resolution_clock::now();

        thrust::fill(A.elements().begin(), A.elements().end(), 0.0);
        thrust::fill(A.diagonal().begin(), A.diagonal().end(), 1.0);
        CUDA_CHECKED(cudaGetLastError());
        CUDA_CHECKED(cudaDeviceSynchronize());

        std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start_time;
        auto rate = size/time.count();
        std::cout<<"algorithm rate = "<< rate <<" GB/s (ratio = 1)\n";

        REQUIRE( A[0][0] == 1.0 );
        REQUIRE( A[1][1] == 1.0 );
        REQUIRE( A[2][1] == 0.0 );
    }

    {
    	multi::array<double, 2, thrust::cuda::allocator<double>> A({20000, 20000});
        auto const size = A.num_elements()*sizeof(T)/1e9;
        std::cout<<"size is "<< size << "GB\n";

        auto start_time = std::chrono::high_resolution_clock::now();
        set_identity(A);
        std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start_time;
        auto rate = size/time.count();
        std::cout<<"kernel rate = "<< rate <<" GB/s (ratio = 1)\n";

        REQUIRE( A[0][0] == 1.0 );
        REQUIRE( A[1][1] == 1.0 );
        REQUIRE( A[2][1] == 0.0 );
    }

}
