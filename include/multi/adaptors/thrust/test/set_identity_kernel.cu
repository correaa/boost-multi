#include <multi/array.hpp>

#include <thrust/device_allocator.h>

namespace multi = boost::multi;

#define CUDA_CHECKED(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
        std::cerr<<"error: "<< cudaGetErrorString(code) <<" "<< file <<":"<< line <<std::endl;
        assert(0);
    }
}

template<typename T>
__global__ void kernel_setIdentity(int m, int n, T* A, int lda)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int j = threadIdx.y + blockDim.y * blockIdx.y;
  if ((i < m) && (j < n))
    if (i == j)
    {
      A[i * lda + i] = T(1.0);
    }
    else
    {
      A[j * lda + i] = T(0.0);
    }
}

void set_identity(int m, int n, double* A, int lda) {
    int xblock_dim = 16;
    int xgrid_dim  = (m + xblock_dim - 1) / xblock_dim;
    int ygrid_dim  = (n + xblock_dim - 1) / xblock_dim;
    dim3 block_dim(xblock_dim, xblock_dim);
    dim3 grid_dim(xgrid_dim, ygrid_dim);
    kernel_setIdentity<<<grid_dim, block_dim>>>(m, n, A, lda);
    CUDA_CHECKED(cudaGetLastError());
    CUDA_CHECKED(cudaDeviceSynchronize());
}

template<typename Cursor>
__global__ void kernel_setIdentity(int m, int n, Cursor A) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if ((i < m) && (j < n)) {
        if (i == j) {
            A[i][j] = 1.0;
        } else {
            A[i][j] = 0.0;
        }
    }
}

template<class Array2D>
void set_identity(Array2D&& arr) {
    int xblock_dim = 16;
    int m = arr.size();
    int n = arr[0].size();
    int xgrid_dim  = (m + xblock_dim - 1) / xblock_dim;
    int ygrid_dim  = (n + xblock_dim - 1) / xblock_dim;
    dim3 block_dim(xblock_dim, xblock_dim);
    dim3 grid_dim(xgrid_dim, ygrid_dim);
    kernel_setIdentity<<<grid_dim, block_dim>>>(
        m, n, arr.home()
    );
    CUDA_CHECKED(cudaGetLastError());
    CUDA_CHECKED(cudaDeviceSynchronize());
}


#define REQUIRE(ans) { require((ans), __FILE__, __LINE__); }
inline void require(bool code, const char *file, int line, bool abort=true) {
   if (not code) {
        std::cerr<<"error: "<< file <<":"<< line <<std::endl;
        exit(666);
    }
}

int main() {
	multi::array<double, 2, thrust::device_allocator<double>> A({10,10});

    set_identity(A);

    REQUIRE( A[0][0] == 1.0 );
    REQUIRE( A[1][1] == 1.0 );
    REQUIRE( A[2][1] == 0.0 );
}
