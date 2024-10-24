
#define ENABLE_GPU 1

#include <boost/multi/array.hpp>

#include <thrust/universal_allocator.h>

#include <chrono>

template<class kernel_type, class array_type>
__global__ void reduce_kernel_vr(long sizex, long sizey, kernel_type kernel, array_type odata) {

	extern __shared__ char shared_mem[];
	auto                   reduction_buffer = (typename array_type::element*)shared_mem;  // {blockDim.x, blockDim.y}

	// each thread loads one element from global to shared mem
	unsigned int ix  = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.y;
	unsigned int iy  = blockIdx.y * blockDim.y + threadIdx.y;

	if(ix >= sizex)
		return;

	if(iy < sizey) {
		reduction_buffer[threadIdx.x + blockDim.x * tid] = kernel(ix, iy);
	} else {
		reduction_buffer[threadIdx.x + blockDim.x * tid] = (typename array_type::element)0.0;
	}

	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s = blockDim.y / 2; s > 0; s >>= 1) {
		if(tid < s) {
			reduction_buffer[threadIdx.x + blockDim.x * tid] += reduction_buffer[threadIdx.x + blockDim.x * (tid + s)];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if(tid == 0)
		odata[blockIdx.y][ix] = reduction_buffer[threadIdx.x];
}

namespace gpu {
template<class type, size_t dim, class allocator = thrust::universal_allocator<type>>
using array = boost::multi::array<type, dim, allocator>;

struct reduce {
	explicit reduce(long arg_size) : size(arg_size) {
	}
	long size;
};

template<typename array_type>
struct array_access {
	array_type array;

	__host__ __device__ auto operator()(long ii) const {
		return array[ii];
	}

	__host__ __device__ auto operator()(long ix, long iy) const {
		return array[ix][iy];
	}
};

template<class type, class kernel_type>
auto run(long sizex, reduce const& redy, kernel_type kernel) /*-> gpu::array<decltype(kernel(0, 0)), 1>*/ {

	auto const sizey = redy.size;

	// using type = decltype(kernel(0, 0));

#ifndef ENABLE_GPU

	gpu::array<type, 1> accumulator(sizex, 0.0);

	for(long iy = 0; iy < sizey; iy++) {
		for(long ix = 0; ix < sizex; ix++) {
			accumulator[ix] += kernel(ix, iy);
		}
	}

	return accumulator;

#else

	gpu::array<type, 2> result;

	auto blocksize = 256;  // max_blocksize(reduce_kernel_vr<kernel_type, decltype(begin(result))>);

	unsigned bsizex = 4;  // this seems to be the optimal value
	if(sizex <= 2)
		bsizex = sizex;
	unsigned bsizey = blocksize / bsizex;

	assert(bsizey > 1);

	unsigned nblockx = (sizex + bsizex - 1) / bsizex;
	unsigned nblocky = (sizey + bsizey - 1) / bsizey;

	result.reextent({nblocky, sizex});

	struct dim3 dg {
		nblockx, nblocky
	};
	struct dim3 db {
		bsizex, bsizey
	};

	auto shared_mem_size = blocksize * sizeof(type);

	assert(shared_mem_size <= 48 * 1024);

	reduce_kernel_vr<<<dg, db, shared_mem_size>>>(sizex, sizey, kernel, begin(result));
	// check_error(last_error());

	// thrust::transform_reduce(

	// );

	if(nblocky == 1) {
		cudaDeviceSynchronize();
		// gpu::sync();

		assert(result[0].size() == sizex);

		return gpu::array<type, 1>(result[0]);
	} else {
		return run<double>(sizex, reduce(nblocky), array_access<decltype(begin(result.transposed()))>{begin(result.transposed())});
	}

#endif
}

}  // namespace gpu

struct prod {
	/*__host__*/ __device__ auto operator()(long ix, long iy) const {
		return double(ix) * double(iy);
	}
};

int main() {

	long const maxsize = 390625;

	auto start = std::chrono::high_resolution_clock::now();

	int rank = 0;
	for(long nx = 1; nx <= 10000; nx *= 10) {
		for(long ny = 1; ny <= maxsize; ny *= 5) {

			auto pp = [] __device__(long ix, long iy) -> double { return double(ix) * double(iy); };

			auto res = gpu::run<double>(nx, gpu::reduce(ny), pp);

			assert(typeid(decltype(res)) == typeid(gpu::array<double, 1>));
			assert(res.size() == nx);

			for(long ix = 0; ix < nx; ix++)
				assert(res[ix] == double(ix) * ny * (ny - 1.0) / 2.0);
			rank++;
		}
	}
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() << "ms\n";
}
