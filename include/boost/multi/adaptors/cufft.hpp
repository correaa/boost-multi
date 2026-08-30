// Copyright 2020-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

#ifndef BOOST_MULTI_ADAPTORS_CUFFT_HPP
#define BOOST_MULTI_ADAPTORS_CUFFT_HPP

#include "boost/multi/array.hpp"
#include "boost/multi/detail/config/NODISCARD.hpp"
#include "boost/multi/utility.hpp"

#include <thrust/memory.h>  // for raw_pointer_cast

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#if !defined(__HIP_ROCclr__)
#include <cufft.h>
#include <cufftXt.h>
#endif

namespace boost::multi::cufft {

// cuFFT API errors
// static auto cuda_get_error_enum(cufftResult error) -> char const* {
// 	switch(error) {
// 	case CUFFT_SUCCESS: return "CUFFT_SUCCESS";

// 	case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
// 	case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";
// #ifdef CUFFT_INCOMPLETE_PARAMETER_LIST
// 	case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
// #endif
// 	case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
// 	case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";
// 	case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
// 	case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
// 	case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";
// 	case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";
// 	case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
// 	case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";
// 	case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
// 	// #if !defined(__HIP_PLATFORM_NVIDIA__)
// 	// case CUFFT_PARSE_ERROR:    return "CUFFT_PARSE_ERROR";
// 	// #endif
// 	case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";
// 	case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
// 	// #if !defined(__HIP_PLATFORM_NVIDIA__)
// 	// case CUFFT_LICENSE_ERROR:  return "CUFFT_LICENSE_ERROR";
// 	// #endif
// 	default: assert(0);
// 	}
// 	return "<unknown>";
// }

#define cufftSafeCall(err) implcufftSafeCall(err, __FILE__, __LINE__)
inline void implcufftSafeCall(cufftResult err, const char* /*file*/, const int /*line*/) {
	if(CUFFT_SUCCESS != err) {
		// std::cerr << "CUFFT error in file " << file << ", line " << line << "\nerror " << err << ": " << cuda_get_error_enum(err) << "\n";
		// fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n", __FILE__, __LINE__, err,
		//                         _cudaGetErrorEnum(err));
		cudaDeviceReset() == cudaSuccess ? void() : assert(0);
	}
}

class sign {
	int impl_ = 0;

 public:
	sign() = default;
	constexpr explicit sign(int impl) : impl_{impl} {}
	constexpr operator int() const { return impl_; }

	constexpr auto operator==(sign const& other) const { return impl_ == other.impl_; }
	constexpr auto operator!=(sign const& other) const { return impl_ != other.impl_; }
};

constexpr sign forward{CUFFT_FORWARD};
constexpr sign none{0};
constexpr sign backward{CUFFT_INVERSE};
// constexpr sign backward{CUFFT_BACKWARD};

static_assert(forward != none && none != backward && backward != forward);

struct cufft_iodim64 {
	std::ptrdiff_t n;
	std::ptrdiff_t is;
	std::ptrdiff_t os;
};  // based on fftw_iodim64

// geometry table for the pack/unpack (gather/scatter) kernels of the "packed" strategy:
// dims listed in packed (row-major) order: non-transformed (howmany) dims first, transformed dims last
struct pack_geom {
	static constexpr int max_dims = 8;

	int            nd = 0;             // number of dims
	std::ptrdiff_t n[max_dims]  = {};  // NOLINT(*-avoid-c-arrays) sizes; passed by value to kernels
	std::ptrdiff_t is[max_dims] = {};  // NOLINT(*-avoid-c-arrays) strides in the input  array (in elements)
	std::ptrdiff_t os[max_dims] = {};  // NOLINT(*-avoid-c-arrays) strides in the output array (in elements)
	std::ptrdiff_t ps[max_dims] = {};  // NOLINT(*-avoid-c-arrays) strides in the packed scratch
	int            v     = 0;          // dim with the smallest input stride: tile-transpose partner of dim nd-1
	std::ptrdiff_t total = 0;
};

#if defined(__CUDACC__)
// pack   (Pack=true):  dst[flat packed]     = src[strided offset]  (offsets from the input  strides is[])
// unpack (Pack=false): dst[strided offset]  = src[flat packed]     (offsets from the output strides os[])
template<bool Pack>
__global__ void pack_plain_kernel(cufftDoubleComplex const* src, cufftDoubleComplex* dst, pack_geom g) {
	for(std::ptrdiff_t flat = blockIdx.x * static_cast<std::ptrdiff_t>(blockDim.x) + threadIdx.x; flat < g.total; flat += static_cast<std::ptrdiff_t>(gridDim.x) * blockDim.x) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		std::ptrdiff_t rem = flat;
		std::ptrdiff_t off = 0;
		for(int d = g.nd - 1; d >= 0; --d) {  // NOLINT(altera-unroll-loops)
			off += (rem % g.n[d]) * (Pack ? g.is[d] : g.os[d]);
			rem /= g.n[d];
		}
		if constexpr(Pack) {
			dst[flat] = src[off];
		} else {
			dst[off] = src[flat];
		}
	}
}

// tiled variant: shared-memory transpose between the innermost packed dim (u = nd-1)
// and the smallest-stride dim v, so that both global reads and writes are coalesced
template<bool Pack>
__global__ void pack_tiled_kernel(cufftDoubleComplex const* src, cufftDoubleComplex* dst, pack_geom g) {
	__shared__ cufftDoubleComplex tile[32][33];  // NOLINT(*-avoid-c-arrays) +1 column avoids shared-memory bank conflicts

	int const            u  = g.nd - 1;
	std::ptrdiff_t const su = Pack ? g.is[u] : g.os[u];
	std::ptrdiff_t const sv = Pack ? g.is[g.v] : g.os[g.v];
	std::ptrdiff_t const pv = g.ps[g.v];  // note: g.ps[u] == 1

	std::ptrdiff_t const vtiles = (g.n[g.v] + 31) / 32;

	std::ptrdiff_t zn = 1;
	for(int d = 0; d != g.nd; ++d) {  // NOLINT(altera-unroll-loops)
		if(d != u && d != g.v) { zn *= g.n[d]; }
	}

	for(std::ptrdiff_t z = blockIdx.z; z < zn; z += gridDim.z) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		std::ptrdiff_t rem    = z;
		std::ptrdiff_t base_s = 0;
		std::ptrdiff_t base_p = 0;
		for(int d = g.nd - 1; d >= 0; --d) {  // NOLINT(altera-unroll-loops)
			if(d == u || d == g.v) { continue; }
			std::ptrdiff_t const c = rem % g.n[d];
			rem /= g.n[d];
			base_s += c * (Pack ? g.is[d] : g.os[d]);
			base_p += c * g.ps[d];
		}
		for(std::ptrdiff_t vt = blockIdx.y; vt < vtiles; vt += gridDim.y) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			std::ptrdiff_t const u0 = static_cast<std::ptrdiff_t>(blockIdx.x) * 32;
			std::ptrdiff_t const v0 = vt * 32;
			if constexpr(Pack) {
				{
					std::ptrdiff_t const uu = u0 + threadIdx.y;
					std::ptrdiff_t const vv = v0 + threadIdx.x;
					if(uu < g.n[u] && vv < g.n[g.v]) { tile[threadIdx.y][threadIdx.x] = src[base_s + uu * su + vv * sv]; }
				}
				__syncthreads();
				{
					std::ptrdiff_t const uu = u0 + threadIdx.x;
					std::ptrdiff_t const vv = v0 + threadIdx.y;
					if(uu < g.n[u] && vv < g.n[g.v]) { dst[base_p + uu + vv * pv] = tile[threadIdx.x][threadIdx.y]; }
				}
			} else {
				{
					std::ptrdiff_t const uu = u0 + threadIdx.x;
					std::ptrdiff_t const vv = v0 + threadIdx.y;
					if(uu < g.n[u] && vv < g.n[g.v]) { tile[threadIdx.x][threadIdx.y] = src[base_p + uu + vv * pv]; }
				}
				__syncthreads();
				{
					std::ptrdiff_t const uu = u0 + threadIdx.y;
					std::ptrdiff_t const vv = v0 + threadIdx.x;
					if(uu < g.n[u] && vv < g.n[g.v]) { dst[base_s + uu * su + vv * sv] = tile[threadIdx.y][threadIdx.x]; }
				}
			}
			__syncthreads();
		}
	}
}

// host-side launcher; a plan stores a pointer to (instantiations of) this function, so that
// translation units compiled without a CUDA compiler can still compile plan::execute
template<bool Pack>
void pack_launch(pack_geom const& geom, cufftDoubleComplex const* src, cufftDoubleComplex* dst) {
	if(geom.v == geom.nd - 1) {  // the innermost packed dim is already the smallest-stride dim: plain gather is coalesced
		auto const nblocks = static_cast<unsigned>(std::min<std::ptrdiff_t>((geom.total + 255) / 256, 1L << 20));
		pack_plain_kernel<Pack><<<nblocks, 256>>>(src, dst, geom);
	} else {
		std::ptrdiff_t zn = 1;
		for(int d = 0; d != geom.nd; ++d) {  // NOLINT(altera-unroll-loops)
			if(d != geom.nd - 1 && d != geom.v) { zn *= geom.n[d]; }
		}
		dim3 const grid(
			static_cast<unsigned>((geom.n[geom.nd - 1] + 31) / 32),
			static_cast<unsigned>(std::min<std::ptrdiff_t>((geom.n[geom.v] + 31) / 32, 65535)),
			static_cast<unsigned>(std::min<std::ptrdiff_t>(zn, 65535))
		);
		dim3 const block(32, 32);
		pack_tiled_kernel<Pack><<<grid, block>>>(src, dst, geom);
	}
}
#endif  // __CUDACC__

// thread-local grow-only device buffer shared by all packed-strategy plans (scratch + work area):
// growth is enqueued on the legacy stream, so it is ordered after any in-flight kernels using it
class scratch_pool {
	void*       ptr_   = nullptr;
	std::size_t bytes_ = 0;

	scratch_pool() = default;

 public:
	scratch_pool(scratch_pool const&)                    = delete;
	scratch_pool(scratch_pool&&)                         = delete;
	auto operator=(scratch_pool const&) -> scratch_pool& = delete;
	auto operator=(scratch_pool&&) -> scratch_pool&      = delete;
	~scratch_pool()                                      = delete;  // leaky singleton, avoids CUDA teardown-order problems (compare cached_plan)

	auto get(std::size_t bytes) -> void* {  // returns nullptr on failure; the returned pointer is only stable until the next get
		if(bytes > bytes_) {
			if(ptr_ != nullptr) {
				cudaFreeAsync(ptr_, cudaStream_t{}) == cudaSuccess ? void() : assert(0);
				ptr_   = nullptr;
				bytes_ = 0;
			}
			if(cudaMallocAsync(&ptr_, bytes, cudaStream_t{}) != cudaSuccess) {
				(void)cudaGetLastError();
				ptr_ = nullptr;
				return nullptr;
			}
			bytes_ = bytes;
		}
		return ptr_;
	}

	static auto instance() -> scratch_pool& {
		thread_local scratch_pool& inst = *new scratch_pool{};  // NOLINT(cppcoreguidelines-owning-memory) leaky singleton
		return inst;
	}
};

template<dimensionality_type DD, class Alloc = void*>
class plan {
	Alloc                                              alloc_;
	::size_t                                           workSize_ = 0;
	void*                                              workArea_{};
	cufftHandle                                        h_{};  // TODO(correaa) put this in a unique_ptr
	std::array<std::pair<bool, cufft_iodim64>, DD + 1> which_iodims_{};
	int                                                first_howmany_{};

	// mutable bool used_ = false;

	using complex_type = cufftDoubleComplex;

	// execution strategy for the dims that cuFFT cannot batch directly (see execute):
	//  - direct: a single cufftPlanMany covers everything (no leftover dims)
	//  - looped: loop over 1 or 2 leftover dims; for small transforms the iterations are
	//    round-robined over a pool of streams, each with its own subplan (a cuFFT handle's
	//    work area cannot be shared between concurrent executions)
	//  - packed: gather to a contiguous scratch, run one batched subplan, scatter back
	enum class strategy : char { direct, looped, packed };
	strategy strategy_ = strategy::direct;

	// looped strategy subplans
	std::vector<cufftHandle>  extra_handles_{};
	std::vector<void*>        extra_work_{};
	std::vector<cudaStream_t> streams_{};
	std::vector<cudaEvent_t>  events_{};  // [0, nstreams): join events, [nstreams]: fork event

	// packed strategy state
	using pack_fn = void (*)(pack_geom const&, complex_type const*, complex_type*);
	pack_geom   geom_{};
	void*       scratch_{};
	std::size_t scratch_bytes_  = 0;
	pack_fn     pack_launch_    = nullptr;  // set only by CUDA-compiled translation units
	pack_fn     unpack_launch_  = nullptr;

	// heuristic thresholds on the size of ONE transform (product of transformed sizes x sizeof(Z));
	// tuned on a Quadro RTX 5000, see adaptors/cufft/benchmark/many_loops.cu
	static constexpr std::ptrdiff_t pack_min_bytes_    = 512L * 1024L;        // above: pack+batch+unpack beats looping
	static constexpr std::ptrdiff_t stride1_max_bytes_ = 2L * 1024L * 1024L;  // below: batch over the smallest-stride dim
	static constexpr std::ptrdiff_t pool_max_bytes_    = 128L * 1024L;        // below: also overlap loop iterations on streams
	static constexpr int            pool_streams_      = 8;

 public:
	using allocator_type = Alloc;

	template<
		class ILayout, class OLayout, dimensionality_type D = std::decay_t<ILayout>::dimensionality,
		class = std::enable_if_t<D == std::decay_t<OLayout>::dimensionality>>
	plan(std::array<bool, +D> which, ILayout const& in, OLayout const& out) : plan(which, in, out, allocator_type{}) {}

	plan()            = delete;
	plan(plan const&) = delete;

	plan(plan&& other) noexcept
	: alloc_{std::move(other.alloc_)},
	  workSize_{std::exchange(other.workSize_, {})},
	  workArea_{std::exchange(other.workArea_, {})},
	  h_{std::exchange(other.h_, {})},
	  which_iodims_{std::exchange(other.which_iodims_, {})},
	  first_howmany_{std::exchange(other.first_howmany_, {})},
	  strategy_{std::exchange(other.strategy_, strategy::direct)},
	  extra_handles_{std::exchange(other.extra_handles_, {})},
	  extra_work_{std::exchange(other.extra_work_, {})},
	  streams_{std::exchange(other.streams_, {})},
	  events_{std::exchange(other.events_, {})},
	  geom_{std::exchange(other.geom_, {})},
	  scratch_{std::exchange(other.scratch_, {})},
	  scratch_bytes_{std::exchange(other.scratch_bytes_, {})},
	  pack_launch_{std::exchange(other.pack_launch_, {})},
	  unpack_launch_{std::exchange(other.unpack_launch_, {})} {
		// other.used_ = true;  // moved-from object cannot be used
		// used_       = false;
	}

	auto operator=(plan const&) = delete;
	auto operator=(plan&&)      = delete;

	template<
		class ILayout, class OLayout, dimensionality_type D = std::decay_t<ILayout>::dimensionality,
		class = std::enable_if_t<D == std::decay_t<OLayout>::dimensionality>>
	plan(std::array<bool, +D> which, ILayout const& in, OLayout const& out, allocator_type const& alloc) : alloc_{alloc} {
		// used_ = false;
		assert(in.sizes() == out.sizes());

		auto const sizes_tuple   = in.sizes();
		auto const istride_tuple = in.strides();
		auto const ostride_tuple = out.strides();

		using boost::multi::detail::get;
		auto which_iodims = std::apply([](auto... elems) {
			return std::array<std::pair<bool, cufft_iodim64>, sizeof...(elems) + 1>{
  // TODO(correaa) added one element to avoid problem with gcc 13 static analysis (out-of-bounds)
				std::pair<bool, cufft_iodim64>{
											   get<0>(elems),
											   cufft_iodim64{get<1>(elems), get<2>(elems), get<3>(elems)}
				}
				 ...,
				std::pair<bool, cufft_iodim64>{}
			};
		},
									   boost::multi::detail::tuple_zip(which, sizes_tuple, istride_tuple, ostride_tuple));

		std::stable_sort(which_iodims.begin(), which_iodims.end() - 1, [](auto const& alpha, auto const& omega) { return get<1>(alpha).is > get<1>(omega).is; });

		auto const part = std::stable_partition(which_iodims.begin(), which_iodims.end() - 1, [](auto elem) { return std::get<0>(elem); });

		std::array<cufft_iodim64, D> dims{};
		auto const                   dims_end = std::transform(which_iodims.begin(), part, dims.begin(), [](auto elem) { return elem.second; });

		// std::array<cufftw_iodim64, D> howmany_dims{};
		// auto const howmany_dims_end = std::transform(part, which_iodims.end() -1, howmany_dims.begin(), [](auto elem) {return elem.second;});

		which_iodims_  = which_iodims;
		first_howmany_ = part - which_iodims.begin();

		////////////////////////////////////////////////////////////////////////

		std::array<int, D> istrides{};
		std::array<int, D> ostrides{};
		std::array<int, D> ion{};

		auto const istrides_end = std::transform(dims.begin(), dims_end, istrides.begin(), [](auto elem) { return static_cast<int>(elem.is); });
		auto const ostrides_end = std::transform(dims.begin(), dims_end, ostrides.begin(), [](auto elem) { return static_cast<int>(elem.os); });
		auto const ion_end      = std::transform(dims.begin(), dims_end, ion.begin(), [](auto elem) { return static_cast<int>(elem.n); });

		int  istride = *(istrides_end - 1);
		auto inembed = istrides;
		inembed.fill(0);
		int  ostride = *(ostrides_end - 1);
		auto onembed = ostrides;
		onembed.fill(0);

		for(std::ptrdiff_t idx = 1; idx != ion_end - ion.begin(); ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch) TODO(correaa) replace with algorithm
			assert(ostrides[idx - 1] >= ostrides[idx]);
			assert(ostrides[idx - 1] % ostrides[idx] == 0);
			onembed[idx] = ostrides[idx - 1] / ostrides[idx];
			assert(istrides[idx - 1] % istrides[idx] == 0);
			inembed[idx] = istrides[idx - 1] / istrides[idx];
		}

		if(dims_end == dims.begin()) {
			throw std::runtime_error{"no ffts in any dimension is not supported"};
		}

		while(first_howmany_ < D - 1) {  // NOLINT(altera-id-dependent-backward-branch) TODO(correaa) replace with algorithm
			int nelems = 1;

			for(int idx = first_howmany_ + 1; idx != D; ++idx) {
				nelems *= which_iodims_[idx].second.n;
			}  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch) TODO(correaa) replace with algorithm
			if(
				which_iodims_[first_howmany_].second.is == nelems && which_iodims_[first_howmany_].second.os == nelems
			) {
				which_iodims_[first_howmany_ + 1].second.n *= which_iodims_[first_howmany_].second.n;
				++first_howmany_;
			} else {
				break;
			}
		}

		if(first_howmany_ == D) {
			if constexpr(std::is_same_v<Alloc, void*>) {
				assert(dims_end - dims.begin() < 4);  // cufft cannot do 4D FFT
				cufftSafeCall(::cufftPlanMany(
					/*cufftHandle *plan*/ &h_,
					/*int rank*/ dims_end - dims.begin(),
					/*int *n*/ ion.data(),
					/*int *inembed*/ inembed.data(),
					/*int istride*/ istride,
					/*int idist*/ 1,  // stride(first),
					/*int *onembed*/ onembed.data(),
					/*int ostride*/ ostride,
					/*int odist*/ 1,  // stride(d_first),
					/*cufftType type*/ CUFFT_Z2Z,
					/*int batch*/ 1  // BATCH
				));
			} else {
				cufftSafeCall(cufftCreate(&h_));
				cufftSafeCall(cufftSetAutoAllocation(h_, false));
				cufftSafeCall(cufftMakePlanMany(
					/*cufftHandle *plan*/ h_,
					/*int rank*/ dims_end - dims.begin(),
					/*int *n*/ ion.data(),
					/*int *inembed*/ inembed.data(),
					/*int istride*/ istride,
					/*int idist*/ 1,  // stride(first),
					/*int *onembed*/ onembed.data(),
					/*int ostride*/ ostride,
					/*int odist*/ 1,  // stride(d_first),
					/*cufftType type*/ CUFFT_Z2Z,
					/*int batch*/ 1,  // BATCH
					/*size_t **/ &workSize_
				));
				cufftSafeCall(cufftGetSize(h_, &workSize_));
				workArea_ = ::thrust::raw_pointer_cast(alloc_.allocate(workSize_));
				// auto s = cudaMalloc(&workArea_, workSize_);
				// if(s != cudaSuccess) {throw std::runtime_error{"L212"};}
				cufftSafeCall(cufftSetWorkArea(h_, workArea_));
			}
			if(!h_) {
				throw std::runtime_error{"cufftPlanMany null"};
			}
			return;
		}

		std::sort(which_iodims_.begin() + first_howmany_, which_iodims_.begin() + D, [](auto const& alpha, auto const& omega) { return get<1>(alpha).n > get<1>(omega).n; });

		std::ptrdiff_t transform_bytes = static_cast<std::ptrdiff_t>(sizeof(complex_type));
		std::for_each(dims.begin(), dims_end, [&transform_bytes](auto const& dim) { transform_bytes *= dim.n; });

#if defined(__CUDACC__) && defined(BOOST_MULTI_CUFFT_PACKED)
		// OPT-IN (moves data with kernels that are not part of cuFFT): for large transforms the
		// strided access pattern inside cuFFT costs ~3-4x; gather to a contiguous scratch, run
		// ONE batched subplan over all leftover dims, scatter back
		if(transform_bytes >= pack_min_bytes_ && try_pack_(dims.begin(), dims_end, ion.data())) {
			return;
		}
#endif
		// small transforms: batch over the smallest-stride leftover dim (the batch then walks
		// contiguous(-ish) blocks, measured ~4x faster) and loop over the large-stride dim(s)
		if(transform_bytes <= stride1_max_bytes_) {
			auto const min_it = std::min_element(
				which_iodims_.begin() + first_howmany_, which_iodims_.begin() + D,
				[](auto const& alpha, auto const& omega) {
					auto const abs_ = [](std::ptrdiff_t stride) { return stride < 0 ? -stride : stride; };
					return abs_(get<1>(alpha).is) < abs_(get<1>(omega).is);
				}
			);
			std::iter_swap(which_iodims_.begin() + first_howmany_, min_it);
		}
		strategy_ = strategy::looped;

		if(first_howmany_ <= D - 1) {
			if constexpr(std::is_same_v<Alloc, void*>) {  // NOLINT(bugprone-branch-clone) workaround bug in DeepSource
				cufftSafeCall(::cufftPlanMany(
					/*cufftHandle *plan*/ &h_,
					/*int rank*/ dims_end - dims.begin(),
					/*int *n*/ ion.data(),
					/*int *inembed*/ inembed.data(),
					/*int istride*/ istride,
					/*int idist*/ which_iodims_[first_howmany_].second.is,
					/*int *onembed*/ onembed.data(),
					/*int ostride*/ ostride,
					/*int odist*/ which_iodims_[first_howmany_].second.os,
					/*cufftType type*/ CUFFT_Z2Z,
					/*int batch*/ which_iodims_[first_howmany_].second.n
				));
			} else {
				cufftSafeCall(cufftCreate(&h_));
				cufftSafeCall(cufftSetAutoAllocation(h_, false));
				cufftSafeCall(cufftMakePlanMany(
					/*cufftHandle *plan*/ h_,
					/*int rank*/ dims_end - dims.begin(),
					/*int *n*/ ion.data(),
					/*int *inembed*/ inembed.data(),
					/*int istride*/ istride,
					/*int idist*/ which_iodims_[first_howmany_].second.is,
					/*int *onembed*/ onembed.data(),
					/*int ostride*/ ostride,
					/*int odist*/ which_iodims_[first_howmany_].second.os,
					/*cufftType type*/ CUFFT_Z2Z,
					/*int batch*/ which_iodims_[first_howmany_].second.n,
					/*size_t **/ &workSize_
				));
				cufftSafeCall(cufftGetSize(h_, &workSize_));
				workArea_ = ::thrust::raw_pointer_cast(alloc_.allocate(workSize_));
				cufftSafeCall(cufftSetWorkArea(h_, workArea_));
			}
			if(!h_) {
				throw std::runtime_error{"cufftPlanMany null"};
			}
			// tiny transforms don't fill the GPU: overlap the leftover loop iterations by
			// round-robining them over a pool of streams, one subplan (and work area) each
			if(transform_bytes <= pool_max_bytes_) {
				int nloop = 1;
				for(int idx = first_howmany_ + 1; idx != D; ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
					nloop *= static_cast<int>(which_iodims_[idx].second.n);
				}
				int const nstreams = std::min(pool_streams_, nloop);
				if(nstreams > 1) {
					make_pool_(
						nstreams,
						static_cast<int>(dims_end - dims.begin()), ion.data(),
						inembed.data(), istride, static_cast<int>(which_iodims_[first_howmany_].second.is),
						onembed.data(), ostride, static_cast<int>(which_iodims_[first_howmany_].second.os),
						static_cast<int>(which_iodims_[first_howmany_].second.n)
					);
				}
			}
			++first_howmany_;
			return;
		}
		// throw std::runtime_error{"cufft not implemented yet"};
	}

 private:
	// packed strategy: contiguous scratch + one batched subplan + gather/scatter kernels;
	// only CUDA-compiled translation units instantiate this (it sets the launcher pointers)
	template<class DimsIt>
	auto try_pack_(DimsIt dims_first, DimsIt dims_last, int* ion_data) -> bool {
#if !defined(__CUDACC__)
		(void)dims_first, (void)dims_last, (void)ion_data;  // NOLINT(clang-diagnostic-comma)
		return false;
#else
		pack_geom geom{};
		for(int idx = first_howmany_; idx != DD; ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			auto const& iodim = which_iodims_[idx].second;
			geom.n[geom.nd]  = iodim.n;
			geom.is[geom.nd] = iodim.is;
			geom.os[geom.nd] = iodim.os;
			++geom.nd;
		}
		int const nhowmany = geom.nd;
		std::for_each(dims_first, dims_last, [&geom](auto const& dim) {
			geom.n[geom.nd]  = dim.n;
			geom.is[geom.nd] = dim.is;
			geom.os[geom.nd] = dim.os;
			++geom.nd;
		});
		{
			std::ptrdiff_t stride = 1;
			for(int d = geom.nd - 1; d >= 0; --d) {  // NOLINT(altera-unroll-loops)
				geom.ps[d] = stride;
				stride *= geom.n[d];
			}
			geom.total = stride;
		}
		{
			auto const abs_ = [](std::ptrdiff_t stride) { return stride < 0 ? -stride : stride; };
			geom.v          = 0;
			for(int d = 1; d != geom.nd; ++d) {  // NOLINT(altera-unroll-loops)
				if(abs_(geom.is[d]) < abs_(geom.is[geom.v])) { geom.v = d; }
			}
		}

		int nelems = 1;
		for(int d = nhowmany; d != geom.nd; ++d) { nelems *= static_cast<int>(geom.n[d]); }  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
		int const batch = static_cast<int>(geom.total / nelems);
		int const rank  = geom.nd - nhowmany;

		std::size_t const bytes = static_cast<std::size_t>(geom.total) * sizeof(complex_type);

		// on ANY failure (out of memory, plan failure) return false: the caller falls back to the looped strategy
		if constexpr(std::is_same_v<Alloc, void*>) {
			// scratch AND work area live in the shared thread-local pool, fetched at each execute;
			// the cached plan itself then pins no array-sized memory
			cufftHandle handle{};
			if(cufftCreate(&handle) != CUFFT_SUCCESS) { return false; }
			if(cufftSetAutoAllocation(handle, false) != CUFFT_SUCCESS ||
			   cufftMakePlanMany(handle, rank, ion_data, nullptr, 1, nelems, nullptr, 1, nelems, CUFFT_Z2Z, batch, &workSize_) != CUFFT_SUCCESS ||
			   cufftGetSize(handle, &workSize_) != CUFFT_SUCCESS) {
				cufftDestroy(handle) == CUFFT_SUCCESS ? void() : assert(0);
				workSize_ = 0;
				return false;
			}
			if(scratch_pool::instance().get(((bytes + 255) & ~std::size_t{255}) + workSize_) == nullptr) {  // probe now so that we can still fall back
				cufftDestroy(handle) == CUFFT_SUCCESS ? void() : assert(0);
				workSize_ = 0;
				return false;
			}
			h_ = handle;
		} else {
			try {
				scratch_ = ::thrust::raw_pointer_cast(alloc_.allocate(bytes));
			} catch(...) { return false; }  // NOLINT(bugprone-empty-catch)
			scratch_bytes_ = bytes;

			auto const unpack_scratch = [&] {
				alloc_.deallocate(typename std::allocator_traits<Alloc>::pointer(reinterpret_cast<char*>(scratch_)), scratch_bytes_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
				scratch_       = nullptr;
				scratch_bytes_ = 0;
			};

			cufftHandle handle{};
			if(cufftCreate(&handle) != CUFFT_SUCCESS) {
				unpack_scratch();
				return false;
			}
			if(cufftSetAutoAllocation(handle, false) != CUFFT_SUCCESS ||
			   cufftMakePlanMany(handle, rank, ion_data, nullptr, 1, nelems, nullptr, 1, nelems, CUFFT_Z2Z, batch, &workSize_) != CUFFT_SUCCESS ||
			   cufftGetSize(handle, &workSize_) != CUFFT_SUCCESS) {
				cufftDestroy(handle) == CUFFT_SUCCESS ? void() : assert(0);
				unpack_scratch();
				workSize_ = 0;
				return false;
			}
			try {
				workArea_ = ::thrust::raw_pointer_cast(alloc_.allocate(workSize_));
			} catch(...) {  // NOLINT(bugprone-empty-catch)
				cufftDestroy(handle) == CUFFT_SUCCESS ? void() : assert(0);
				unpack_scratch();
				workSize_ = 0;
				return false;
			}
			cufftSafeCall(cufftSetWorkArea(handle, workArea_));
			h_ = handle;
		}

		scratch_bytes_ = bytes;
		geom_          = geom;
		pack_launch_   = &cufft::pack_launch<true>;
		unpack_launch_ = &cufft::pack_launch<false>;
		strategy_      = strategy::packed;
		return true;
#endif  // __CUDACC__
	}

	// looped strategy: create nstreams-1 extra subplans (same geometry as h_), one per extra stream
	void make_pool_(int nstreams, int rank, int const* ion, int const* inembed, int istride, int idist, int const* onembed, int ostride, int odist, int batch) {
		for(int idx = 0; idx != nstreams; ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			cudaStream_t stream{};
			if(cudaStreamCreate(&stream) != cudaSuccess) {
				abort_pool_();
				return;
			}
			streams_.push_back(stream);
		}
		for(int idx = 0; idx != nstreams + 1; ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			cudaEvent_t event{};
			if(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
				abort_pool_();
				return;
			}
			events_.push_back(event);
		}
		for(int idx = 1; idx != nstreams; ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			cufftHandle handle{};
			if constexpr(std::is_same_v<Alloc, void*>) {
				if(::cufftPlanMany(&handle, rank, const_cast<int*>(ion), const_cast<int*>(inembed), istride, idist, const_cast<int*>(onembed), ostride, odist, CUFFT_Z2Z, batch) != CUFFT_SUCCESS) {  // NOLINT(cppcoreguidelines-pro-type-const-cast) legacy interface
					abort_pool_();
					return;
				}
			} else {
				std::size_t worksize{};
				if(cufftCreate(&handle) != CUFFT_SUCCESS ||
				   cufftSetAutoAllocation(handle, false) != CUFFT_SUCCESS ||
				   cufftMakePlanMany(handle, rank, const_cast<int*>(ion), const_cast<int*>(inembed), istride, idist, const_cast<int*>(onembed), ostride, odist, CUFFT_Z2Z, batch, &worksize) != CUFFT_SUCCESS) {  // NOLINT(cppcoreguidelines-pro-type-const-cast) legacy interface
					abort_pool_();
					return;
				}
				try {
					extra_work_.push_back(::thrust::raw_pointer_cast(alloc_.allocate(workSize_)));
				} catch(...) {  // NOLINT(bugprone-empty-catch)
					cufftDestroy(handle) == CUFFT_SUCCESS ? void() : assert(0);
					abort_pool_();
					return;
				}
				cufftSafeCall(cufftSetWorkArea(handle, extra_work_.back()));
			}
			extra_handles_.push_back(handle);
			cufftSafeCall(cufftSetStream(handle, streams_[static_cast<std::size_t>(idx)]));
		}
		cufftSafeCall(cufftSetStream(h_, streams_[0]));
	}

	void abort_pool_() {  // undo a partial make_pool_: the plan falls back to the sequential loop
		for(auto& handle : extra_handles_) { cufftSafeCall(cufftDestroy(handle)); }
		extra_handles_.clear();
		if constexpr(!std::is_same_v<Alloc, void*>) {
			for(auto* work : extra_work_) {
				alloc_.deallocate(typename std::allocator_traits<Alloc>::pointer(reinterpret_cast<char*>(work)), workSize_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			}
		}
		extra_work_.clear();
		for(auto& event : events_) { cudaEventDestroy(event) == cudaSuccess ? void() : assert(0); }
		events_.clear();
		for(auto& stream : streams_) { cudaStreamDestroy(stream) == cudaSuccess ? void() : assert(0); }
		streams_.clear();
	}

	// fork/join the pool around the loop so that ordering against the (legacy) default
	// stream is preserved: execute stays asynchronous, exactly as without the pool
	void pool_fork_() const {
		if(streams_.empty()) { return; }
		cudaEventRecord(events_.back(), cudaStream_t{}) == cudaSuccess ? void() : assert(0);
		for(auto const& stream : streams_) { cudaStreamWaitEvent(stream, events_.back(), 0) == cudaSuccess ? void() : assert(0); }  // NOLINT(altera-unroll-loops)
	}
	void pool_join_() const {
		for(std::size_t idx = 0; idx != streams_.size(); ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
			cudaEventRecord(events_[idx], streams_[idx]) == cudaSuccess ? void() : assert(0);
			cudaStreamWaitEvent(cudaStream_t{}, events_[idx], 0) == cudaSuccess ? void() : assert(0);
		}
	}
	auto pooled_handle_(std::ptrdiff_t idx) const -> cufftHandle {
		if(streams_.empty()) { return h_; }
		auto const which = static_cast<std::size_t>(idx) % streams_.size();
		return which == 0 ? h_ : extra_handles_[which - 1];
	}

 public:

 private:
	template<typename = void>
	void ExecZ2Z_(complex_type const* idata, complex_type* odata, int direction) const {
		// used_ = true;
		cufftSafeCall(cufftExecZ2Z(h_, const_cast<complex_type*>(idata), odata, direction));  // NOLINT(cppcoreguidelines-pro-type-const-cast) wrap legacy interface
																							  // cudaDeviceSynchronize();
	}

 public:
	template<class IPtr, class OPtr>
	auto execute(IPtr idata, OPtr odata, int direction) const
		-> decltype((void)(reinterpret_cast<complex_type const*>(::thrust::raw_pointer_cast(idata)),
						   reinterpret_cast<complex_type*>(::thrust::raw_pointer_cast(odata)))) {  // TODO(correaa) make const
		// used_ = true;
		if(strategy_ == strategy::packed) {
			assert(pack_launch_ && unpack_launch_);  // a packed plan can only be built by a CUDA-compiled translation unit
			complex_type* tmp{};
			if constexpr(std::is_same_v<Alloc, void*>) {
				std::size_t const aligned = (scratch_bytes_ + 255) & ~std::size_t{255};
				auto* const       base    = static_cast<char*>(scratch_pool::instance().get(aligned + workSize_));
				if(base == nullptr) {
					throw std::runtime_error{"cufft packed plan: scratch allocation failed"};
				}
				tmp = reinterpret_cast<complex_type*>(base);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
				cufftSafeCall(cufftSetWorkArea(h_, base + aligned));
			} else {
				tmp = static_cast<complex_type*>(scratch_);
			}
			pack_launch_(geom_, reinterpret_cast<complex_type const*>(::thrust::raw_pointer_cast(idata)), tmp);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			cufftSafeCall(cufftExecZ2Z(h_, tmp, tmp, direction));
			unpack_launch_(geom_, tmp, reinterpret_cast<complex_type*>(::thrust::raw_pointer_cast(odata)));  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			return;
		}
		if(first_howmany_ == DD) {
			ExecZ2Z_(reinterpret_cast<complex_type const*>(::thrust::raw_pointer_cast(idata)), reinterpret_cast<complex_type*>(::thrust::raw_pointer_cast(odata)), direction);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) wrap a legacy interface
			return;
		}
		if(first_howmany_ == DD - 1) {
			if(which_iodims_[first_howmany_].first) {
				throw std::runtime_error{"logic error"};
			}

			pool_fork_();
			for(int idx = 0; idx != which_iodims_[first_howmany_].second.n; ++idx) {  // NOLINT(altera-unroll-loops,altera-id-dependent-backward-branch)
				cufftSafeCall(cufftExecZ2Z(
					pooled_handle_(idx),
					const_cast<complex_type*>(reinterpret_cast<complex_type const*>(::thrust::raw_pointer_cast(idata + idx * which_iodims_[first_howmany_].second.is))),  // NOLINT(cppcoreguidelines-pro-type-const-cast,cppcoreguidelines-pro-type-reinterpret-cast) legacy interface
					reinterpret_cast<complex_type*>(::thrust::raw_pointer_cast(odata + idx * which_iodims_[first_howmany_].second.os)),                                   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) legacy interface
					direction
				));
			}
			pool_join_();
			return;
		}
		if(first_howmany_ == DD - 2) {
			if(which_iodims_[first_howmany_ + 0].first) {
				throw std::runtime_error{"logic error0"};
			}
			if(which_iodims_[first_howmany_ + 1].first) {
				throw std::runtime_error{"logic error1"};
			}

			pool_fork_();
			for(int idx = 0; idx != which_iodims_[first_howmany_].second.n; ++idx) {          // NOLINT(altera-unroll-loops,altera-unroll-loops,altera-id-dependent-backward-branch) TODO(correaa) use an algorithm
				for(int jdx = 0; jdx != which_iodims_[first_howmany_ + 1].second.n; ++jdx) {  // NOLINT(altera-unroll-loops,altera-unroll-loops,altera-id-dependent-backward-branch) TODO(correaa) use an algorithm
					cufftSafeCall(cufftExecZ2Z(
						pooled_handle_(static_cast<std::ptrdiff_t>(idx) * which_iodims_[first_howmany_ + 1].second.n + jdx),
						const_cast<complex_type*>(reinterpret_cast<complex_type const*>(::thrust::raw_pointer_cast(idata + idx * which_iodims_[first_howmany_].second.is + jdx * which_iodims_[first_howmany_ + 1].second.is))),  // NOLINT(cppcoreguidelines-pro-type-const-cast,cppcoreguidelines-pro-type-reinterpret-cast) legacy interface
						reinterpret_cast<complex_type*>(::thrust::raw_pointer_cast(odata + idx * which_iodims_[first_howmany_].second.os + jdx * which_iodims_[first_howmany_ + 1].second.os)),                                   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) legacy interface
						direction
					));
				}
			}
			pool_join_();
			return;
		}
		throw std::runtime_error{"error2"};
	}

	template<class IPtr, class OPtr>
	void execute_forward(IPtr idata, OPtr odata) {  // TODO(correaa) make const
		execute(idata, odata, cufft::forward);
	}
	template<class IPtr, class OPtr>
	void execute_backward(IPtr idata, OPtr odata) {  // TODO(correaa) make const
		execute(idata, odata, cufft::backward);
	}

	template<class IPtr, class OPtr>
	void operator()(IPtr idata, OPtr odata, int direction) const {
		// used_ = true;
		ExecZ2Z_(reinterpret_cast<complex_type const*>(::thrust::raw_pointer_cast(idata)), reinterpret_cast<complex_type*>(::thrust::raw_pointer_cast(odata)), direction);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) legacy interface
	}
	template<class I, class O>
	auto execute_dft(I&& in, O&& out, int direction) const -> O&& {
		// used_ = true;
		ExecZ2Z_(
			const_cast<complex_type*>(reinterpret_cast<complex_type const*>(base(in))),   // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-type-const-cast) legay interface
			const_cast<complex_type*>(reinterpret_cast<complex_type const*>(base(out))),  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast,cppcoreguidelines-pro-type-const-cast) legay interface
			direction
		);
		return std::forward<O>(out);
	}

	~plan() {
		for(auto& handle : extra_handles_) { cufftSafeCall(cufftDestroy(handle)); }
		if constexpr(!std::is_same_v<Alloc, void*>) {
			if(workSize_ > 0) {
				alloc_.deallocate(typename std::allocator_traits<Alloc>::pointer(reinterpret_cast<char*>(workArea_)), workSize_);
			}  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast) legacy interface
			for(auto* work : extra_work_) {
				alloc_.deallocate(typename std::allocator_traits<Alloc>::pointer(reinterpret_cast<char*>(work)), workSize_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			}
			if(scratch_ != nullptr) {
				alloc_.deallocate(typename std::allocator_traits<Alloc>::pointer(reinterpret_cast<char*>(scratch_)), scratch_bytes_);  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
			}
		} else {
			if(scratch_ != nullptr) {
				cudaFree(scratch_) == cudaSuccess ? void() : assert(0);
			}
		}
		for(auto& event : events_) { cudaEventDestroy(event) == cudaSuccess ? void() : assert(0); }
		for(auto& stream : streams_) { cudaStreamDestroy(stream) == cudaSuccess ? void() : assert(0); }
		if(h_ != 0) {
			cufftSafeCall(cufftDestroy(h_));
		}
		// if(!used_) {
		//  std::cerr <<"Warning: cufft plan was never used\n";
		//  std::terminate();
		// }
	}

	using size_type  = int;
	using ssize_type = int;
};

template<dimensionality_type D, class Alloc = void*>
class cached_plan {
	typename std::map<std::tuple<std::array<bool, D>, multi::layout_t<D>, multi::layout_t<D>>, plan<D, Alloc>>::iterator it_;

 public:
	cached_plan(cached_plan const&) = delete;
	cached_plan(cached_plan&&)      = delete;

	auto operator=(cached_plan const&) -> cached_plan& = delete;
	auto operator=(cached_plan&&) -> cached_plan&      = delete;

	~cached_plan() = default;

	cached_plan(std::array<bool, D> which, boost::multi::layout_t<D, boost::multi::ssize_t> in, boost::multi::layout_t<D, boost::multi::ssize_t> out, Alloc const& alloc = {}) {  // NOLINT(fuchsia-default-arguments-declarations)
		thread_local std::map<std::tuple<std::array<bool, D>, multi::layout_t<D>, multi::layout_t<D>>, plan<D, Alloc>>& LEAKY_cache = *new std::map<std::tuple<std::array<bool, D>, multi::layout_t<D>, multi::layout_t<D>>, plan<D, Alloc>>;
		it_                                                                                                                         = LEAKY_cache.find(std::tuple<std::array<bool, D>, multi::layout_t<D>, multi::layout_t<D>>{which, in, out});
		if(it_ == LEAKY_cache.end()) {
			it_ = LEAKY_cache.insert(std::make_pair(std::make_tuple(which, in, out), plan<D, Alloc>(which, in, out, alloc))).first;
		}
	}
	template<class IPtr, class OPtr>
	auto execute(IPtr idata, OPtr odata, int direction)
		-> decltype((void)(std::declval<
							   typename std::map<std::tuple<std::array<bool, D>, multi::layout_t<D>, multi::layout_t<D>>, plan<D, Alloc>>::iterator&>()
							   ->second.execute(idata, odata, direction))) {
		// assert(it_ != LEAKY_cache.end());
		it_->second.execute(idata, odata, direction);
	}
};

// template<typename In, class Out, dimensionality_type D = In::rank::value, std::enable_if_t<!multi::has_get_allocator<In>::value, int> =0, typename = decltype(::thrust::raw_pointer_cast(std::declval<In const&>().base()))>
// auto dft(std::array<bool, +D> which, In const& in, Out&& out, int sgn)
// ->decltype(cufft::cached_plan<D>{which, in.layout(), out.layout()}.execute(in.base(), out.base(), sgn), std::forward<Out>(out)) {
// 	return cufft::cached_plan<D>{which, in.layout(), out.layout()}.execute(in.base(), out.base(), sgn), std::forward<Out>(out);
// }

template<typename In, class Out, dimensionality_type D = In::dimensionality>  // , std::enable_if_t<    multi::has_get_allocator<In>::value, int> =0, typename = decltype(raw_pointer_cast(std::declval<In const&>().base()))>
auto dft(std::array<bool, +D> which, In const& in, Out&& out, int sgn)
	-> decltype(cufft::cached_plan<D /*, typename std::allocator_traits<typename In::allocator_type>::rebind_alloc<char>*/>{which, in.layout(), out.layout() /*, i.get_allocator()*/}.execute(in.base(), out.base(), sgn), std::forward<Out>(out)) {
	if constexpr(D == 4) {
		if(which == std::array<bool, D>{true, true, true, true}) {
			auto const [is, js, ks, ls] = in.extents();
			for(auto i : is)
				for(auto j : js) {
					cufft::dft({true, true}, in[i][j], out[i][j], sgn);
				}
			for(auto k : ks)
				for(auto l : ls) {
					cufft::dft({true, true}, out.rotated().rotated()[k][l], out.rotated().rotated()[k][l], sgn);
				}
			return std::forward<Out>(out);
		}
	}
	return cufft::cached_plan<D /*, typename std::allocator_traits<typename In::allocator_type>::rebind_alloc<char>*/>{which, in.layout(), out.layout() /*, i.get_allocator()*/}.execute(in.base(), out.base(), sgn), std::forward<Out>(out);
}

template<typename In, class Out, dimensionality_type D = In::dimensionality>  //, std::enable_if_t<not multi::has_get_allocator<In>::value, int> =0>
auto dft_forward(std::array<bool, +D> which, In const& in, Out&& out) -> Out&& {
	//->decltype(cufft::plan<D>{which, i.layout(), o.layout()}.execute(i.base(), o.base(), cufft::forward), std::forward<Out>(o)) {
	return cufft::cached_plan<D>{which, in.layout(), out.layout()}.execute(in.base(), out.base(), cufft::forward), std::forward<Out>(out);
}

template<typename In, class Out, dimensionality_type D = In::dimensionality>  //, std::enable_if_t<not multi::has_get_allocator<In>::value, int> =0>
auto dft_backward(std::array<bool, +D> which, In const& in, Out&& out) -> Out&& {
	//->decltype(cufft::plan<D>{which, i.layout(), o.layout()}.execute(i.base(), o.base(), cufft::backward), std::forward<Out>(o)) {
	return cufft::cached_plan<D>{which, in.layout(), out.layout()}.execute(in.base(), out.base(), cufft::backward), std::forward<Out>(out);
}

// template<typename In, class Out, dimensionality_type D = In::rank::value, class = typename In::allocator_type, std::enable_if_t<    multi::has_get_allocator<In>::value, int> =0>
// auto dft_backward(std::array<bool, +D> which, In const& i, Out&& o) -> Out&& {
// //->decltype(cufft::plan<D, typename std::allocator_traits<typename In::allocator_type>::rebind_alloc<char> >{which, i.layout(), o.layout(), i.get_allocator()}.execute(i.base(), o.base(), cufft::backward), std::forward<Out>(o)) {
//  return cufft::cached_plan<D/*, typename std::allocator_traits<typename In::allocator_type>::rebind_alloc<char>*/>{which, i.layout(), o.layout()/*, i.get_allocator()*/}.execute(i.base(), o.base(), cufft::backward), std::forward<Out>(o); }

template<typename In, typename R = multi::array<typename In::element, In::dimensionality, decltype(get_allocator(std::declval<In>()))>>
BOOST_MULTI_NODISCARD("when first argument is const")
auto dft(In const& in, int sgn) -> R {
	static_assert(std::is_trivially_default_constructible<typename In::element>{});
	R ret(extents(in), get_allocator(in));
	cufft::dft(in, ret, sgn);
	// if(cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error{"Cuda error: Failed to synchronize"};
	return ret;
}

template<class Array, std::size_t... Ns>
constexpr auto array_tail_impl(Array const& arr, std::index_sequence<Ns...> /*unused*/) {
	return std::array<typename Array::value_type, std::tuple_size<Array>{} - 1>{std::get<Ns + 1>(arr)...};
}

template<class Array>
constexpr auto array_tail(Array const& arr)
	-> decltype(array_tail_impl(arr, std::make_index_sequence<std::tuple_size<Array>{} - 1>())) {
	return array_tail_impl(arr, std::make_index_sequence<std::tuple_size<Array>{} - 1>());
}

template<typename In, std::size_t D = In::dimensionality>
BOOST_MULTI_NODISCARD("when passing a const argument")
auto dft(std::array<bool, D> which, In const& in, int sign) -> std::decay_t<decltype(dft(which, in, typename In::decay_type(extents(in), get_allocator(in)), sign))> { return dft(which, in, typename In::decay_type(extents(in), get_allocator(in)), sign); }

template<typename In, std::size_t D = In::dimensionality>  // TODO(correaa) check that the type of In a decay_type (otherwise there is no ::dimensionality)
auto dft(std::array<bool, D> which, In&& in, int sign)
	-> decltype(dft(which, in, in, sign), std::forward<In>(in)) {
	return dft(which, in, in, sign), std::forward<In>(in);
}

template<typename Array, typename A> BOOST_MULTI_NODISCARD("when passing a const argument")
auto dft_forward(Array arr, A const& in)
	-> decltype(cufft::dft(arr, in, cufft::forward)) {
	return cufft::dft(arr, in, cufft::forward);
}

// template<typename Array, dimensionality_type D> NODISCARD("when passing a const argument")
// auto dft_forward(Array arr, multi::cuda::array<std::complex<double>, D>&& a)
// ->decltype(cufft::dft(arr, a, cufft::forward), multi::cuda::array<std::complex<double>, D>{}){//assert(0);
//  return cufft::dft(arr, a, cufft::forward), std::move(a);}

template<typename A> BOOST_MULTI_NODISCARD("when passing a const argument")
auto dft_forward(A const& arr)
	-> decltype(cufft::dft(arr, cufft::forward)) {
	return cufft::dft(arr, cufft::forward);
}

template<typename... As> auto dft_backward(As&&... as)
	-> decltype(cufft::dft(std::forward<As>(as)..., cufft::backward)) {
	return cufft::dft(std::forward<As>(as)..., cufft::backward);
}

template<typename Array, typename A> BOOST_MULTI_NODISCARD("when passing a const argument")
auto dft_backward(Array arr, A const& in)
	-> decltype(cufft::dft(arr, in, cufft::backward)) {
	return cufft::dft(arr, in, cufft::backward);
}

template<typename A> BOOST_MULTI_NODISCARD("when passing a const argument")
auto dft_backward(A const& arr)
	-> decltype(cufft::dft(arr, cufft::backward)) {
	return cufft::dft(arr, cufft::backward);
}

}  // end namespace boost::multi::cufft

#endif
