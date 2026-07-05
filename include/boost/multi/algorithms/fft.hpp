// Copyright 2024-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// This header contains a generic, header-only, in-place multidimensional FFT
// with a reusable "plan" (in the FFTW sense: auxiliary tables and scratch
// allocations are computed once and reused across repeated transforms).
//
// Public interface:
//   multi::fft_plan<T, D>            reusable transform state:
//     fft_plan<T, D>(extents, sign)  plan for a shape (any tuple-like extents)
//     fft_plan(array, sign)          plan deduced from an array (CTAD)
//     plan(A) / plan.execute(A)      transform A in place; A can be any array
//                                    or subarray with the planned sizes, of
//                                    any strided layout, as many times as
//                                    desired without re-allocating
//   multi::fft_inplace(A, sign)      one-shot convenience (plans, then runs)
//   multi::fft_forward/fft_backward  direction constants (FFTW convention)
//   multi::fft_real<T>               trait: real type underlying T
//
// It is a self-contained (no FFTW/cuFFT dependency) implementation that works
// for:
//   * arbitrary element types `T` that obey complex algebra
//     (closed under `+`, `-`, `*`, constructible as `T{re, im}`, and
//      value-initialized to zero, like `std::complex<Real>`),
//   * arbitrary number of dimensions `D`,
//   * arbitrary sizes: a self-sorting (Stockham autosort) mixed-radix engine
//     with specialized radix-4/2/3/5 kernels, table-driven generic kernels for
//     primes up to 64, and Bluestein's chirp-z algorithm (a power-of-two
//     convolution) for larger prime factors, so no size degenerates to O(n^2),
//   * arbitrary strided layouts supported by the Multi library.
//
// Performance notes (what makes it competitive with tuned libraries):
//   * All twiddle (root-of-unity) tables, factorizations, DFT matrices and
//     scratch buffers live in the plan and are reused across fibers, axes of
//     equal length, and repeated executions.
//   * The kernel is an iterative Stockham autosort FFT: no bit-reversal pass,
//     no per-stage allocation, no modulo in inner loops. Powers of two use
//     radix-4 stages (half the passes of radix-2).
//   * Every stage kernel is *batched*: it can transform `m` interleaved
//     fibers at once with the batch index innermost and contiguous, which
//     auto-vectorizes. Multi-fiber (multidimensional) transforms are tiled
//     through this batched path instead of transforming one fiber at a time.
//   * The N-D orchestration keeps the smallest-stride axis as the batch axis
//     (gathers then move contiguous runs), and prime factors > 64 are handled
//     by nested Bluestein sub-plans executed with the same batched kernels.
//
// Convention (matching FFTW): the transform is *unnormalized*; a forward
// followed by a backward transform multiplies each element by the number of
// transformed points.
//
// Thread safety: a plan owns mutable scratch, so concurrent `execute` calls on
// the *same* plan object require external synchronization (or one plan copy
// per thread); distinct plans are independent.

#pragma once
#ifndef BOOST_MULTI_ALGORITHM_FFT_HPP
#define BOOST_MULTI_ALGORITHM_FFT_HPP

#include <algorithm>    // for copy, fill, min, max, find_if
#include <array>        // for plan sizes
#include <cassert>      // for assert
#include <cmath>        // for cos, sin, acos
#include <complex>      // for the fft_ops<std::complex> fast product
#include <cstddef>      // for size_t, ptrdiff_t
#include <functional>   // for greater
#include <limits>       // for numeric_limits
#include <memory>       // for addressof
#include <type_traits>  // for decay_t, enable_if_t, void_t
#include <utility>      // for forward, index_sequence
#include <vector>       // for tables and scratch buffers

namespace boost {
namespace multi {

// Sign of the exponent in the discrete Fourier transform.
inline constexpr int fft_forward  = -1;  // exp(-2*pi*i*...), same as FFTW_FORWARD
inline constexpr int fft_backward = +1;  // exp(+2*pi*i*...), same as FFTW_BACKWARD

// Trait that maps a complex-algebra element type to its underlying real type.
// Specialize this for custom complex-like types that do not expose `value_type`.
template<class T> struct fft_real {
	using type = typename T::value_type;
};

// Multiplication customization point for the transform kernels. The default
// uses `operator*`. For `std::complex` the plain formula is used instead:
// the operator carries C-Annex-G infinity/NaN fixups (a branch and a libcall
// fallback) that prevent the batched inner loops from vectorizing. Users can
// specialize this for custom element types with a faster product.
template<class T> struct fft_ops {
	static constexpr auto mul(T const& a, T const& b) -> T { return a * b; }
};

template<class R> struct fft_ops<std::complex<R>> {
	static constexpr auto mul(std::complex<R> const& a, std::complex<R> const& b) -> std::complex<R> {
		return {(a.real() * b.real()) - (a.imag() * b.imag()), (a.real() * b.imag()) + (a.imag() * b.real())};
	}
};


// The stage kernels' source and destination never alias within one stage
// (even the fused in-place path reads user memory only in the first stage and
// writes it only in the last). Telling the compiler removes runtime overlap
// checks and loop versioning (measured: ~7-10% on 2-D/3-D).
#if defined(_MSC_VER)
#define BOOST_MULTI_FFT_RESTRICT __restrict
#else
#define BOOST_MULTI_FFT_RESTRICT __restrict__
#endif

namespace detail {

template<class T>
using fft_real_t = typename fft_real<T>::type;

template<class Real>
auto fft_pi() -> Real { return std::acos(Real{-1}); }

template<class T>
constexpr auto fft_mul(T const& a, T const& b) -> T { return fft_ops<T>::mul(a, b); }

// Largest prime handled by the direct (table-driven O(p^2)) kernel; larger
// prime factors use a Bluestein sub-plan, which is O(p log p).
inline constexpr std::size_t fft_max_direct_radix = 64;

// Single-fiber transforms at least this long use the six-step decomposition
// n = n1*n2 (column FFTs, twiddle-transpose, row FFTs): both FFT passes then
// run batched (vectorized) and cache-blocked instead of striding across the
// whole fiber. Threshold chosen by measurement (2^13 is neutral, 2^14..2^15
// gain 10-25%).
inline constexpr std::size_t fft_sixstep_min = std::size_t{1} << 13U;

// Detects Multi arrays/subarrays (as opposed to extents/shape tuples).
template<class A, class = void> struct fft_is_multi_like : std::false_type {};
template<class A>
struct fft_is_multi_like<A, std::void_t<typename A::element, decltype(A::dimensionality)>> : std::true_type {};

template<class Extent>
auto fft_extent_size(Extent const& ext) -> std::size_t {
	if constexpr(std::is_integral_v<Extent>) {
		return static_cast<std::size_t>(ext);
	} else {
		return static_cast<std::size_t>(ext.size());  // e.g. index_extension
	}
}

// Reusable single-length transform engine: twiddle tables, stage plan, and
// scratch for one fiber length `n` and one sign. All stage kernels are
// "batched": they transform `m` fibers at once, stored interleaved as
// buf[k*m + j] (element k of fiber j), with the batch index j contiguous so
// that inner loops vectorize.
template<class T>
struct fft_engine {
	// NOLINTBEGIN(misc-non-private-member-variables-in-classes) engine is an implementation-detail aggregate
	std::size_t n_    = 0;
	int         sign_ = fft_forward;
	std::size_t mb_   = 1;  // preferred batch width (scratch is sized so 2*n*mb stays cache-resident)

	struct stage_t {
		std::size_t radix;
		int         kind;  // 0: radix-2, 1: radix-3, 2: radix-4, 3: radix-5, 4: generic direct, 5: Bluestein sub-plan, 6: radix-8
		std::size_t aux;   // generic: offset into wmat_; sub-plan: index into sub_
	};

	std::vector<T>       tw_;      // tw_[k] = exp(sign*2*pi*i*k/n), k in [0, n)  (sign baked in)
	std::vector<stage_t> stages_;  // ordered stage factorization of n
	std::vector<T>       wmat_;    // concatenated p x p DFT matrices for direct generic radices
	std::size_t          max_gen_ = 0;

	// Six-step state (used for long single-fiber transforms):
	bool        sixstep_ = false;
	std::size_t six_n1_  = 0;
	std::size_t six_n2_  = 0;
	std::size_t six_i1_  = 0;  // index into sub_ of the length-n1 column plan
	std::size_t six_i2_  = 0;  // index into sub_ of the length-n2 row plan

	// Bluestein state (used when n_ is a prime > fft_max_direct_radix):
	// X_k = c_k * sum_n x_n c_n d_{k-n} with c_j = exp(sign*i*pi*j^2/n), d = 1/c,
	// evaluated as a circular convolution of power-of-two length conv_n_ >= 2n-1.
	bool           bluestein_ = false;
	std::size_t    conv_n_    = 0;
	std::vector<T> chirp_;      // c_j
	std::vector<T> postc_;      // c_k / conv_n_  (fused convolution normalization)
	std::vector<T> kernel_ft_;  // FFT of the wrapped d-kernel, precomputed

	std::vector<fft_engine> sub_;  // nested engines: Bluestein fwd/bwd pair, or large-prime stage sub-plans

	mutable std::vector<T> buf_;   // gathered input / ping-pong A, size >= n*m
	mutable std::vector<T> out_;   // ping-pong B, size >= n*m
	mutable std::vector<T> xbuf_;  // generic-stage gather scratch, size >= max_gen_*m
	// NOLINTEND(misc-non-private-member-variables-in-classes)

	fft_engine(std::size_t nn, int sign) : n_{nn}, sign_{sign} {  // NOLINT(readability-function-cognitive-complexity)
		if(nn < 2) {
			ensure(1);
			return;
		}

		// Factor n into stage radices. The power-of-two part uses radix-4
		// stages (the best-measured kernel) plus a single radix-8 stage when
		// the exponent is odd -- replacing a 4*2 tail by one 8 saves a whole
		// memory pass, while an all-8 plan loses to all-4 on register
		// pressure. A radix-2 stage only ever appears for n == 2. Odd primes
		// follow ascending, so large primes go last and their sub-plan runs
		// once with a wide batch. The Stockham engine is self-sorting for any
		// factorization order.
		std::vector<std::size_t> fac;
		std::size_t              rem = nn;
		std::size_t              k2  = 0;
		while(rem % 2 == 0) { ++k2; rem /= 2; }
		if(k2 == 1) {
			fac.push_back(2);
		} else {
			for(std::size_t k = (k2 % 2 == 1) ? k2 - 3 : k2; k >= 2; k -= 2) { fac.push_back(4); }
			if(k2 % 2 == 1 && k2 >= 3) { fac.push_back(8); }
		}
		for(std::size_t p = 3; p * p <= rem; p += 2) {
			while(rem % p == 0) { fac.push_back(p); rem /= p; }
		}
		if(rem > 1) { fac.push_back(rem); }

		if(fac.size() == 1 && fac.front() == nn && nn > fft_max_direct_radix) {
			init_bluestein_();
			mb_ = batch_width_(conv_n_);
			ensure(1);
			return;
		}

		using real      = fft_real_t<T>;
		real const step = static_cast<real>(sign) * real{2} * fft_pi<real>() / static_cast<real>(nn);
		tw_.resize(nn);
		for(std::size_t k = 0; k != nn; ++k) {
			real const theta = step * static_cast<real>(k);
			tw_[k]           = T{std::cos(theta), std::sin(theta)};
		}

		for(std::size_t const rr : fac) {
			stage_t st{rr, 4, 0};
			switch(rr) {
				case 2: st.kind = 0; break;
				case 3: st.kind = 1; break;
				case 4: st.kind = 2; break;
				case 5: st.kind = 3; break;
				case 8: st.kind = 6; break;
				default:
					if(rr <= fft_max_direct_radix) {  // direct kernel driven by a precomputed p x p DFT matrix
						st.kind  = 4;
						st.aux   = wmat_offset_(rr);
						max_gen_ = std::max(max_gen_, rr);
					} else {  // large prime: delegate the size-rr sub-DFTs to a nested (Bluestein) plan
						st.kind = 5;
						st.aux  = sub_index_(rr);
					}
					break;
			}
			stages_.push_back(st);
		}

		if(nn >= fft_sixstep_min) {
			// Balanced split n = n1*n2: distribute the factors, largest first,
			// onto the currently-smaller side.
			std::vector<std::size_t> desc = fac;
			std::sort(desc.begin(), desc.end(), std::greater<>{});
			std::size_t n1 = 1;
			std::size_t n2 = 1;
			for(std::size_t const f : desc) { (n1 <= n2 ? n1 : n2) *= f; }
			if(std::min(n1, n2) >= 16) {
				sixstep_ = true;
				six_n1_  = n1;
				six_n2_  = n2;
				six_i1_  = sub_index_(n1);
				if(n2 == n1) {  // distinct engine: the two passes' buffers must not alias
					sub_.emplace_back(n2, sign_);
					six_i2_ = sub_.size() - 1;
				} else {
					six_i2_ = sub_index_(n2);
				}
			}
		}

		mb_ = batch_width_(nn);
		ensure(1);
	}

	// Grow the scratch to hold batches of width m (never shrinks, so repeated
	// executions of a plan do not re-allocate).
	void ensure(std::size_t m) const {
		std::size_t const need = std::max<std::size_t>(n_, 1) * m;
		if(buf_.size() < need) {
			buf_.resize(need);
			if(!bluestein_) { out_.resize(need); }
		}
		if(max_gen_ != 0 && xbuf_.size() < max_gen_ * m) { xbuf_.resize(max_gen_ * m); }
	}

	// Transform the gathered data `in` (layout [n][m], batch contiguous) and
	// return a pointer to the result in the same layout. `in` defaults to the
	// plan's own gather buffer `buf_`; call ensure(m) before filling it.
	auto run(std::size_t m) const -> T const* { return run(m, buf_.data()); }

	auto run(std::size_t m, T const* in) const -> T const* {
		if(sixstep_ && m == 1 && n_ >= 2) { return run_sixstep_(in); }  // uses only the sub-plans' buffers
		ensure(m);
		if(n_ < 2) {
			if(in != buf_.data()) { std::copy(in, in + (n_ * m), buf_.data()); }
			return buf_.data();
		}
		if(bluestein_) { return run_bluestein_(m, in); }
		return (m == 1) ? run_stages_<false>(1, in) : run_stages_<true>(m, in);
	}

	// True when the stage pipeline can read/write user memory directly: the
	// first stage fully consumes the input (so it may alias the output) and a
	// distinct last stage exists to produce the final values.
	auto can_fuse() const -> bool { return !bluestein_ && stages_.size() >= 2; }

	// Transform directly between user tiles: the first stage reads
	// in[k*si + j] and the last stage writes out[k*so + j] (batch index j
	// contiguous), skipping the separate gather and scatter passes. `in` may
	// alias `out`. Only valid when can_fuse().
	void run_fused(T const* in, std::size_t si, T* out, std::size_t so, std::size_t m) const {
		assert(can_fuse());
		assert(m > 1 || (si == 1 && so == 1));  // the unbatched kernels fold strides to 1
		if(m == 1) {
			run_fused_impl_<false>(in, 1, out, 1, 1);
		} else {
			run_fused_impl_<true>(in, si, out, so, m);
		}
	}

	// In-place transform of one contiguous (stride-1) fiber in user memory;
	// the final pass writes straight back (no scatter copy).
	void run_contig_inplace(T* io) const {
		if(n_ < 2) { return; }
		if(sixstep_) { run_sixstep_(io, io); return; }
		if(bluestein_) { run_bluestein_(1, io, io); return; }
		if(stages_.size() >= 2) { run_fused_impl_<false>(io, 1, io, 1, 1); return; }
		T const* const res = run(1, io);
		std::copy(res, res + n_, io);
	}

 private:
	static auto batch_width_(std::size_t nn) -> std::size_t {
		// Two ping-pong buffers of n*mb elements should stay ~cache-resident.
		std::size_t const budget = (std::size_t{1} << 22) / (2 * sizeof(T) * std::max<std::size_t>(nn, 1));
		return std::clamp<std::size_t>(budget, 1, 64);
	}

	auto wmat_offset_(std::size_t rr) -> std::size_t {
		// The p x p matrix W[u*p + t] = exp(sign*2*pi*i*t*u/p) tabulates the
		// size-p sub-DFT, removing the inner-loop modulo of a naive kernel.
		std::size_t const off = wmat_.size();
		std::size_t const wr  = n_ / rr;  // step of the p-th roots of unity in tw_
		wmat_.resize(off + (rr * rr));
		for(std::size_t u = 0; u != rr; ++u) {
			for(std::size_t t = 0; t != rr; ++t) { wmat_[off + (u * rr) + t] = tw_[(t * u % rr) * wr]; }
		}
		return off;
	}

	auto sub_index_(std::size_t rr) -> std::size_t {
		for(std::size_t i = 0; i != sub_.size(); ++i) {
			if(sub_[i].n_ == rr) { return i; }
		}
		sub_.emplace_back(rr, sign_);
		return sub_.size() - 1;
	}

	// Cheapest 5-smooth (2^a * 3^b * 5^c) convolution length >= target, scored
	// with a simple per-stage cost model (measured relative kernel costs per
	// point per pass). All candidates up to the next power of two are
	// considered: a slightly larger, radix-4-heavy length often beats the
	// smallest smooth one (e.g. 2048 beats 2025 = 3^4*5^2 for n = 1009).
	static auto next_smooth_(std::size_t target) -> std::size_t {
		auto const cost = [](std::size_t c) -> double {
			double      w = 0.0;
			std::size_t r = c;
			std::size_t k = 0;
			while(r % 2 == 0) { ++k; r /= 2; }
			if(k == 1) { w += 0.7; }  // lone radix-2 stage
			else if(k != 0) { w += (k % 2 == 1 ? 1.2 + (static_cast<double>(k - 3) / 2) : static_cast<double>(k) / 2); }  // radix-4s + one 8 if odd
			while(r % 3 == 0) { w += 0.9; r /= 3; }
			while(r % 5 == 0) { w += 1.45; r /= 5; }
			if(r != 1) { return -1.0; }  // not smooth
			return static_cast<double>(c) * w;
		};
		std::size_t pow2 = 1;
		while(pow2 < target) { pow2 *= 2; }
		std::size_t best      = pow2;
		double      best_cost = cost(pow2);
		for(std::size_t c = target; c != pow2; ++c) {
			double const cc = cost(c);
			if(cc >= 0.0 && cc < best_cost) { best = c; best_cost = cc; }
		}
		return best;
	}

	void init_bluestein_() {
		using real = fft_real_t<T>;
		bluestein_ = true;

		conv_n_ = next_smooth_((2 * n_) - 1);

		chirp_.resize(n_);
		postc_.resize(n_);

		sub_.emplace_back(conv_n_, sign_);   // convolution "forward" transform
		sub_.emplace_back(conv_n_, -sign_);  // convolution "inverse" transform (unnormalized)

		// Wrapped convolution kernel b: b[j] = b[conv_n_ - j] = d_j = conj-chirp.
		auto const& fwd = sub_[0];
		fwd.ensure(1);
		std::fill(fwd.buf_.begin(), fwd.buf_.end(), T{});

		real const  pi_n = fft_pi<real>() / static_cast<real>(n_);
		std::size_t jsq  = 0;  // j^2 mod 2n, updated incrementally to avoid overflow
		for(std::size_t j = 0; j != n_; ++j) {
			real const theta = static_cast<real>(sign_) * pi_n * static_cast<real>(jsq);
			chirp_[j]        = T{std::cos(theta), std::sin(theta)};
			T const dj       = T{std::cos(theta), -std::sin(theta)};
			fwd.buf_[j]      = dj;
			if(j != 0) { fwd.buf_[conv_n_ - j] = dj; }
			jsq += (2 * j) + 1;
			while(jsq >= 2 * n_) { jsq -= 2 * n_; }
		}

		T const inv_m = T{real{1} / static_cast<real>(conv_n_), real{0}};
		for(std::size_t k = 0; k != n_; ++k) { postc_[k] = chirp_[k] * inv_m; }

		T const* kft = fwd.run(1);
		kernel_ft_.assign(kft, kft + conv_n_);
	}

	// --- batched Stockham stage kernels -----------------------------------
	// Data layout: element k of batch-fiber j lives at [k*m + j]. `Batched`
	// selects at compile time between the vector inner loop and the m == 1
	// fast path (no inner loop overhead for single fibers).

	template<bool Batched>
	void stage_radix2_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;  // folds all offset arithmetic when unbatched
		std::size_t const sa    = Batched ? sa_ : 1;  // input element stride (user tile when fused)
		std::size_t const sb    = Batched ? sb_ : 1;  // output element stride
		std::size_t const half  = n_ / 2;
		std::size_t const tstep = n_ / (2 * ns);
		for(std::size_t block = 0; block != half; block += ns) {
			std::size_t const base = block * 2;
			for(std::size_t r = 0; r != ns; ++r) {
				T const        w  = tw_[r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T const* const a1 = a0 + (half * sa);
				T* const       b0 = b + ((base + r) * sb);
				T* const       b1 = b0 + (ns * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const v0 = a0[j];
					T const v1 = fft_mul(w, a1[j]);
					b0[j]      = v0 + v1;
					b1[j]      = v0 - v1;
				}
			}
		}
	}

	// The multiply-by-(-/+ i) is expressed as a multiply by tw_[n/4] so it
	// stays generic over the element type and carries the correct sign.
	template<bool Batched>
	void stage_radix4_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const q     = n_ / 4;
		std::size_t const tstep = n_ / (4 * ns);
		T const           imu   = tw_[q];  // -i for forward, +i for backward
		for(std::size_t block = 0; block != q; block += ns) {
			std::size_t const base = block * 4;
			for(std::size_t r = 0; r != ns; ++r) {
				T const        w1 = tw_[r * tstep];
				T const        w2 = tw_[2 * r * tstep];
				T const        w3 = tw_[3 * r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T const* const a1 = a0 + (q * sa);
				T const* const a2 = a0 + (2 * q * sa);
				T const* const a3 = a0 + (3 * q * sa);
				T* const       b0 = b + ((base + r) * sb);
				T* const       b1 = b0 + (ns * sb);
				T* const       b2 = b0 + (2 * ns * sb);
				T* const       b3 = b0 + (3 * ns * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const x0 = a0[j];
					T const x1 = fft_mul(w1, a1[j]);
					T const x2 = fft_mul(w2, a2[j]);
					T const x3 = fft_mul(w3, a3[j]);
					T const t0 = x0 + x2;
					T const t1 = x0 - x2;
					T const t2 = x1 + x3;
					T const t3 = fft_mul(imu, x1 - x3);
					b0[j]      = t0 + t2;
					b1[j]      = t1 + t3;
					b2[j]      = t0 - t2;
					b3[j]      = t1 - t3;
				}
			}
		}
	}

	// One radix-8 stage, decomposed into two radix-4 sub-butterflies plus a
	// combining layer; all constants (W8, W8^2 = -/+i, W8^3) come from the
	// twiddle table so the kernel stays sign- and type-generic.
	template<bool Batched>
	void stage_radix8_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const q     = n_ / 8;
		std::size_t const tstep = n_ / (8 * ns);
		T const           imu   = tw_[2 * q];  // W8^2: -i for forward, +i for backward
		T const           w81   = tw_[q];
		T const           w83   = tw_[3 * q];
		for(std::size_t block = 0; block != q; block += ns) {
			std::size_t const base = block * 8;
			for(std::size_t r = 0; r != ns; ++r) {
				T const        w1 = tw_[r * tstep];
				T const        w2 = tw_[2 * r * tstep];
				T const        w3 = tw_[3 * r * tstep];
				T const        w4 = tw_[4 * r * tstep];
				T const        w5 = tw_[5 * r * tstep];
				T const        w6 = tw_[6 * r * tstep];
				T const        w7 = tw_[7 * r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T* const       b0 = b + ((base + r) * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const x0 = a0[j];
					T const x1 = fft_mul(w1, a0[(1 * q * sa) + j]);
					T const x2 = fft_mul(w2, a0[(2 * q * sa) + j]);
					T const x3 = fft_mul(w3, a0[(3 * q * sa) + j]);
					T const x4 = fft_mul(w4, a0[(4 * q * sa) + j]);
					T const x5 = fft_mul(w5, a0[(5 * q * sa) + j]);
					T const x6 = fft_mul(w6, a0[(6 * q * sa) + j]);
					T const x7 = fft_mul(w7, a0[(7 * q * sa) + j]);
					// radix-4 over the even legs (x0, x2, x4, x6)
					T const s0 = x0 + x4;
					T const s1 = x0 - x4;
					T const s2 = x2 + x6;
					T const s3 = fft_mul(imu, x2 - x6);
					T const e0 = s0 + s2;
					T const e1 = s1 + s3;
					T const e2 = s0 - s2;
					T const e3 = s1 - s3;
					// radix-4 over the odd legs (x1, x3, x5, x7), then W8^u twiddles
					T const u0 = x1 + x5;
					T const u1 = x1 - x5;
					T const u2 = x3 + x7;
					T const u3 = fft_mul(imu, x3 - x7);
					T const o0 = u0 + u2;
					T const o1 = fft_mul(w81, u1 + u3);
					T const o2 = fft_mul(imu, u0 - u2);
					T const o3 = fft_mul(w83, u1 - u3);
					b0[j]                = e0 + o0;
					b0[(1 * ns * sb) + j] = e1 + o1;
					b0[(2 * ns * sb) + j] = e2 + o2;
					b0[(3 * ns * sb) + j] = e3 + o3;
					b0[(4 * ns * sb) + j] = e0 - o0;
					b0[(5 * ns * sb) + j] = e1 - o1;
					b0[(6 * ns * sb) + j] = e2 - o2;
					b0[(7 * ns * sb) + j] = e3 - o3;
				}
			}
		}
	}

	template<bool Batched>
	void stage_radix3_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const n3    = n_ / 3;
		std::size_t const tstep = n_ / (3 * ns);
		T const           w1c   = tw_[n3];      // W_3
		T const           w2c   = tw_[2 * n3];  // W_3^2
		for(std::size_t block = 0; block != n3; block += ns) {
			std::size_t const base = block * 3;
			for(std::size_t r = 0; r != ns; ++r) {
				T const        w1 = tw_[r * tstep];
				T const        w2 = tw_[2 * r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T const* const a1 = a0 + (n3 * sa);
				T const* const a2 = a0 + (2 * n3 * sa);
				T* const       b0 = b + ((base + r) * sb);
				T* const       b1 = b0 + (ns * sb);
				T* const       b2 = b0 + (2 * ns * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const x0 = a0[j];
					T const x1 = fft_mul(w1, a1[j]);
					T const x2 = fft_mul(w2, a2[j]);
					b0[j]      = x0 + x1 + x2;
					b1[j]      = x0 + fft_mul(w1c, x1) + fft_mul(w2c, x2);
					b2[j]      = x0 + fft_mul(w2c, x1) + fft_mul(w1c, x2);
				}
			}
		}
	}

	template<bool Batched>
	void stage_radix5_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const n5    = n_ / 5;
		std::size_t const tstep = n_ / (5 * ns);
		T const           w1c   = tw_[n5];
		T const           w2c   = tw_[2 * n5];
		T const           w3c   = tw_[3 * n5];
		T const           w4c   = tw_[4 * n5];
		for(std::size_t block = 0; block != n5; block += ns) {
			std::size_t const base = block * 5;
			for(std::size_t r = 0; r != ns; ++r) {
				T const        w1 = tw_[r * tstep];
				T const        w2 = tw_[2 * r * tstep];
				T const        w3 = tw_[3 * r * tstep];
				T const        w4 = tw_[4 * r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T const* const a1 = a0 + (n5 * sa);
				T const* const a2 = a0 + (2 * n5 * sa);
				T const* const a3 = a0 + (3 * n5 * sa);
				T const* const a4 = a0 + (4 * n5 * sa);
				T* const       b0 = b + ((base + r) * sb);
				T* const       b1 = b0 + (ns * sb);
				T* const       b2 = b0 + (2 * ns * sb);
				T* const       b3 = b0 + (3 * ns * sb);
				T* const       b4 = b0 + (4 * ns * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const x0 = a0[j];
					T const x1 = fft_mul(w1, a1[j]);
					T const x2 = fft_mul(w2, a2[j]);
					T const x3 = fft_mul(w3, a3[j]);
					T const x4 = fft_mul(w4, a4[j]);
					b0[j]      = x0 + x1 + x2 + x3 + x4;
					b1[j]      = x0 + fft_mul(w1c, x1) + fft_mul(w2c, x2) + fft_mul(w3c, x3) + fft_mul(w4c, x4);
					b2[j]      = x0 + fft_mul(w2c, x1) + fft_mul(w4c, x2) + fft_mul(w1c, x3) + fft_mul(w3c, x4);
					b3[j]      = x0 + fft_mul(w3c, x1) + fft_mul(w1c, x2) + fft_mul(w4c, x3) + fft_mul(w2c, x4);
					b4[j]      = x0 + fft_mul(w4c, x1) + fft_mul(w3c, x2) + fft_mul(w2c, x3) + fft_mul(w1c, x4);
				}
			}
		}
	}

	// Direct radix-p stage for odd primes p <= fft_max_direct_radix, driven by
	// the precomputed p x p DFT matrix (no modulo in the inner loops).
	template<bool Batched>
	void stage_generic_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t rr, T const* wmat, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const nr    = n_ / rr;
		std::size_t const tstep = n_ / (rr * ns);
		T* const          x     = xbuf_.data();
		for(std::size_t block = 0; block != nr; block += ns) {
			std::size_t const base = block * rr;
			for(std::size_t r = 0; r != ns; ++r) {
				T const* const asrc = a + ((block + r) * sa);
				for(std::size_t j = 0; j != m; ++j) { x[j] = asrc[j]; }  // t == 0, twiddle == 1
				for(std::size_t t = 1; t != rr; ++t) {
					T const        w  = tw_[t * r * tstep];
					T const* const at = asrc + (t * nr * sa);
					T* const       xt = x + (t * m);
					for(std::size_t j = 0; j != m; ++j) { xt[j] = fft_mul(w, at[j]); }
				}
				for(std::size_t u = 0; u != rr; ++u) {
					T const* const wrow = wmat + (u * rr);
					T* const       dst  = b + ((base + r + (u * ns)) * sb);
					for(std::size_t j = 0; j != m; ++j) { dst[j] = x[j]; }  // wrow[0] == 1
					for(std::size_t t = 1; t != rr; ++t) {
						T const        wc = wrow[t];
						T const* const xt = x + (t * m);
						for(std::size_t j = 0; j != m; ++j) { dst[j] = dst[j] + fft_mul(wc, xt[j]); }
					}
				}
			}
		}
	}

	// Stage for a prime factor p > fft_max_direct_radix: after the input
	// twiddles, the p output legs form plain size-p DFTs, which are delegated
	// to a nested (Bluestein) sub-plan, batched over all ns*m interleaved
	// fibers at once. The sub-plan's [u][r*m+j] output layout coincides with
	// this stage's required b[(base + r + u*ns)*m + j] layout, so the result
	// is copied back in one contiguous block.
	template<bool Batched>
	void stage_subplan_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t rr, fft_engine const& sub, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const nr    = n_ / rr;
		std::size_t const tstep = n_ / (rr * ns);
		std::size_t const m2    = ns * m;
		sub.ensure(m2);
		for(std::size_t block = 0; block != nr; block += ns) {
			T* const y = sub.buf_.data();
			for(std::size_t r = 0; r != ns; ++r) {
				T const* const asrc = a + ((block + r) * sa);
				T* const       y0   = y + (r * m);
				for(std::size_t j = 0; j != m; ++j) { y0[j] = asrc[j]; }
				for(std::size_t t = 1; t != rr; ++t) {
					T const        w  = tw_[t * r * tstep];
					T const* const at = asrc + (t * nr * sa);
					T* const       yt = y + (((t * ns) + r) * m);
					for(std::size_t j = 0; j != m; ++j) { yt[j] = fft_mul(w, at[j]); }
				}
			}
			T const* const z = sub.run(m2);
			if(sb == m) {
				std::copy(z, z + (rr * ns * m), b + (block * rr * sb));
			} else {
				for(std::size_t idx = 0; idx != rr * ns; ++idx) {
					T const* const zr = z + (idx * m);
					T* const       br = b + (((block * rr) + idx) * sb);
					for(std::size_t j = 0; j != m; ++j) { br[j] = zr[j]; }
				}
			}
		}
	}

	template<bool Batched>
	void run_fused_impl_(T const* in, std::size_t si, T* out, std::size_t so, std::size_t m) const {
		ensure(m);
		T const*          src  = in;
		T*                dst  = out_.data();
		T*                alt  = buf_.data();
		std::size_t       ns   = 1;
		std::size_t const last = stages_.size() - 1;
		for(std::size_t i = 0; i != stages_.size(); ++i) {
			stage_t const&    st = stages_[i];
			std::size_t const sa = (i == 0) ? si : m;
			T* const          d  = (i == last) ? out : dst;
			std::size_t const sb = (i == last) ? so : m;
			switch(st.kind) {
				case 0: stage_radix2_<Batched>(src, d, ns, m, sa, sb); break;
				case 1: stage_radix3_<Batched>(src, d, ns, m, sa, sb); break;
				case 2: stage_radix4_<Batched>(src, d, ns, m, sa, sb); break;
				case 3: stage_radix5_<Batched>(src, d, ns, m, sa, sb); break;
				case 4: stage_generic_<Batched>(src, d, ns, st.radix, wmat_.data() + st.aux, m, sa, sb); break;
				case 6: stage_radix8_<Batched>(src, d, ns, m, sa, sb); break;
				default: stage_subplan_<Batched>(src, d, ns, st.radix, sub_[st.aux], m, sa, sb); break;
			}
			src = d;
			std::swap(dst, alt);
			ns *= st.radix;
		}
	}

	template<bool Batched>
	auto run_stages_(std::size_t m, T const* in) const -> T const* {
		T const*    src = in;
		T*          dst = out_.data();
		T*          alt = buf_.data();
		std::size_t ns  = 1;
		for(stage_t const& st : stages_) {
			switch(st.kind) {
				case 0: stage_radix2_<Batched>(src, dst, ns, m, m, m); break;
				case 1: stage_radix3_<Batched>(src, dst, ns, m, m, m); break;
				case 2: stage_radix4_<Batched>(src, dst, ns, m, m, m); break;
				case 3: stage_radix5_<Batched>(src, dst, ns, m, m, m); break;
				case 4: stage_generic_<Batched>(src, dst, ns, st.radix, wmat_.data() + st.aux, m, m, m); break;
				case 6: stage_radix8_<Batched>(src, dst, ns, m, m, m); break;
				default: stage_subplan_<Batched>(src, dst, ns, st.radix, sub_[st.aux], m, m, m); break;
			}
			src = dst;
			std::swap(dst, alt);
			ns *= st.radix;
		}
		return src;
	}

	// Six-step transform of one long fiber: with n = n1*n2 and the fiber seen
	// as a row-major [n1][n2] grid,
	//   X[k1 + n1*k2] = sum_j2 W_n2^{j2 k2} * W_n^{j2 k1} * sum_j1 W_n1^{j1 k1} x[j1][j2].
	// Step 1 is a length-n1 FFT batched over the contiguous j2 index (no
	// gather at all); the twiddle multiply is fused into a tiled transpose to
	// [n2][n1]; the final length-n2 FFT batched over k1 lands directly in
	// natural (flat) output order.
	auto run_sixstep_(T const* in, T* uout = nullptr) const -> T const* {
		auto const&       e1 = sub_[six_i1_];
		auto const&       e2 = sub_[six_i2_];
		std::size_t const n1 = six_n1_;
		std::size_t const n2 = six_n2_;

		T const* const z = e1.run(n2, in);  // column FFTs: layout [n1][n2] is already batched

		e2.ensure(n1);
		T* const              yt = e2.buf_.data();
		constexpr std::size_t tb = 32;  // 32 x 32 tiles staged through an L1 buffer, so both
		std::array<T, tb * tb> tile;    // the read and the write side stream contiguously
		for(std::size_t k10 = 0; k10 < n1; k10 += tb) {
			std::size_t const k1e = std::min(n1, k10 + tb);
			for(std::size_t j20 = 0; j20 < n2; j20 += tb) {
				std::size_t const j2e = std::min(n2, j20 + tb);
				for(std::size_t k1 = k10; k1 != k1e; ++k1) {
					std::size_t    idx = (k1 * j20) % n_;  // k1*j2 mod n, updated incrementally
					T const* const zr  = z + (k1 * n2);
					T* const       tr  = tile.data() + ((k1 - k10) * tb);
					for(std::size_t j2 = j20; j2 != j2e; ++j2) {
						tr[j2 - j20] = fft_mul(zr[j2], tw_[idx]);
						idx += k1;
						if(idx >= n_) { idx -= n_; }
					}
				}
				for(std::size_t j2 = j20; j2 != j2e; ++j2) {
					T* const       yr = yt + (j2 * n1) + k10;
					T const* const tc = tile.data() + (j2 - j20);
					for(std::size_t k1 = 0; k1 != k1e - k10; ++k1) { yr[k1] = tc[k1 * tb]; }
				}
			}
		}

		if(uout != nullptr && e2.can_fuse()) {  // final stage writes user memory directly
			e2.template run_fused_impl_<true>(e2.buf_.data(), n1, uout, n1, n1);
			return uout;
		}
		T const* const res = e2.run(n1);  // row FFTs, batched over k1
		if(uout != nullptr) {
			std::copy(res, res + (n2 * n1), uout);
			return uout;
		}
		return res;
	}

	auto run_bluestein_(std::size_t m, T const* in, T* out = nullptr) const -> T const* {
		auto const& fwd = sub_[0];
		auto const& bwd = sub_[1];
		fwd.ensure(m);
		bwd.ensure(m);

		T* const y = fwd.buf_.data();  // chirp-premultiplied input, zero-padded to conv_n_
		for(std::size_t k = 0; k != n_; ++k) {
			T const c = chirp_[k];
			for(std::size_t j = 0; j != m; ++j) { y[(k * m) + j] = fft_mul(c, in[(k * m) + j]); }
		}
		std::fill(y + (n_ * m), y + (conv_n_ * m), T{});

		T const* const yf = fwd.run(m);

		T* const z = bwd.buf_.data();  // pointwise product with the precomputed kernel spectrum
		for(std::size_t q = 0; q != conv_n_; ++q) {
			T const kq = kernel_ft_[q];
			for(std::size_t j = 0; j != m; ++j) { z[(q * m) + j] = fft_mul(kq, yf[(q * m) + j]); }
		}

		T const* const zc = bwd.run(m);

		T* const res = (out != nullptr) ? out : buf_.data();
		for(std::size_t k = 0; k != n_; ++k) {  // chirp-postmultiply (normalization fused into postc_)
			T const pc = postc_[k];
			for(std::size_t j = 0; j != m; ++j) { res[(k * m) + j] = fft_mul(pc, zc[(k * m) + j]); }
		}
		return res;
	}
};

// --- N-D orchestration ------------------------------------------------------

// Transform one (possibly strided) 1-D fiber through the engine.
template<class View1D, class T>
void fft_exec_fiber(View1D&& fib, fft_engine<T> const& eng) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	if constexpr(std::is_pointer_v<std::decay_t<decltype(fib.base())>>) {
		if(fib.stride() == 1) {  // contiguous fiber: no gather, and the final pass writes back directly
			eng.run_contig_inplace(fib.base());
			return;
		}
	}
	eng.ensure(1);
	std::copy(fib.begin(), fib.end(), eng.buf_.begin());  // gather strided fiber
	T const* const res = eng.run(1);
	std::copy(res, res + eng.n_, fib.begin());  // scatter result back
}

// Transform every row-fiber of a rank-2 slab [batch][n] in vector batches:
// tiles of up to eng.mb_ fibers are gathered interleaved (batch index
// contiguous) and pushed through the batched stage kernels together.
template<class View2D, class T>
void fft_exec_slab(View2D&& slab, fft_engine<T> const& eng) {  // NOLINT(cppcoreguidelines-missing-std-forward,readability-function-cognitive-complexity)
	using std::get;
	auto const yy = static_cast<std::size_t>(slab.size());
	auto const nn = eng.n_;
	if(yy == 0 || nn == 0) { return; }

	std::size_t const mb = std::min<std::size_t>(std::max<std::size_t>(eng.mb_, 1), yy);
	eng.ensure(mb);

	// Contiguous fibers transform faster one at a time straight from user
	// memory (no transpose gather) than through batched tiles.
	if constexpr(std::is_pointer_v<std::decay_t<decltype(slab.base())>>) {
		if(get<1>(slab.strides()) == 1) {
			for(std::ptrdiff_t y = 0; y != static_cast<std::ptrdiff_t>(yy); ++y) { fft_exec_fiber(slab[y], eng); }
			return;
		}
		// Batch axis contiguous in user memory: the batched stages read each
		// tile in place (first stage) and write it back (last stage) -- no
		// gather or scatter passes at all. Layout validity guarantees the
		// batch extent <= fiber stride, so tiles never self-overlap.
		if(get<0>(slab.strides()) == 1 && get<1>(slab.strides()) > 1 && eng.can_fuse()) {
			auto const sf = static_cast<std::size_t>(get<1>(slab.strides()));
			for(std::size_t y0 = 0; y0 < yy; y0 += mb) {
				std::size_t const mt = std::min(mb, yy - y0);
				if(mt == 1) {
					fft_exec_fiber(slab[static_cast<std::ptrdiff_t>(y0)], eng);
					continue;
				}
				T* const tile0 = std::addressof(slab[static_cast<std::ptrdiff_t>(y0)][0]);
				eng.run_fused(tile0, sf, tile0, sf, mt);
			}
			return;
		}
	}
	auto const abs_ = [](auto s) { return s < 0 ? -s : s; };
	// Pick the gather loop order from the layout: move along whichever axis is
	// closer in memory in the inner copy loop.
	bool const fiber_near = abs_(get<1>(slab.strides())) <= abs_(get<0>(slab.strides()));

	auto&& cols = slab.rotated();  // cols[k][y] == slab[y][k]

	for(std::size_t y0 = 0; y0 < yy; y0 += mb) {
		std::size_t const mt = std::min(mb, yy - y0);
		if(mt == 1) {
			fft_exec_fiber(slab[static_cast<std::ptrdiff_t>(y0)], eng);
			continue;
		}
		T* const bp = eng.buf_.data();
		if(fiber_near) {  // fibers contiguous-ish: blocked-transpose gather, reads stream along k
			constexpr std::size_t kb = 64;
			for(std::size_t k0 = 0; k0 < nn; k0 += kb) {
				std::size_t const ke = std::min(nn, k0 + kb);
				for(std::size_t j = 0; j != mt; ++j) {
					auto it = slab[static_cast<std::ptrdiff_t>(y0 + j)].begin();
					for(std::size_t k = k0; k != ke; ++k) { bp[(k * mt) + j] = it[static_cast<std::ptrdiff_t>(k)]; }
				}
			}
		} else {  // batch axis contiguous-ish: both reads and writes stream along j
			for(std::size_t k = 0; k != nn; ++k) {
				auto     it  = cols[static_cast<std::ptrdiff_t>(k)].begin() + static_cast<std::ptrdiff_t>(y0);
				T* const row = bp + (k * mt);
				for(std::size_t j = 0; j != mt; ++j) { row[j] = it[static_cast<std::ptrdiff_t>(j)]; }
			}
		}

		T const* const res = eng.run(mt);

		if(fiber_near) {
			constexpr std::size_t kb = 64;
			for(std::size_t k0 = 0; k0 < nn; k0 += kb) {
				std::size_t const ke = std::min(nn, k0 + kb);
				for(std::size_t j = 0; j != mt; ++j) {
					auto it = slab[static_cast<std::ptrdiff_t>(y0 + j)].begin();
					for(std::size_t k = k0; k != ke; ++k) { it[static_cast<std::ptrdiff_t>(k)] = res[(k * mt) + j]; }
				}
			}
		} else {
			for(std::size_t k = 0; k != nn; ++k) {
				auto           it  = cols[static_cast<std::ptrdiff_t>(k)].begin() + static_cast<std::ptrdiff_t>(y0);
				T const* const row = res + (k * mt);
				for(std::size_t j = 0; j != mt; ++j) { it[static_cast<std::ptrdiff_t>(j)] = row[j]; }
			}
		}
	}
}

template<class Strides, std::size_t... Is>
auto fft_min_abs_mid_stride(Strides const& strs, std::index_sequence<Is...> /*unused*/) -> std::ptrdiff_t {
	using std::get;
	std::ptrdiff_t ret  = std::numeric_limits<std::ptrdiff_t>::max();
	auto const     acc_ = [&ret](std::ptrdiff_t s) {
        s = (s < 0) ? -s : s;
        ret = std::min(ret, s);
	};
	(acc_(static_cast<std::ptrdiff_t>(get<Is + 1>(strs))), ...);
	return ret;
}

// Transform the last *two* axes of `view` together, slab by slab: both
// passes run while the rank-2 slab is still cache-resident. For D >= 3 a
// slab is a small fraction of the whole array, so this replaces two
// full-array memory sweeps by one (the second axis' fibers also become
// slab-local strides instead of array-wide ones).
template<class ViewND, class T>
void fft_apply_last_pair(ViewND&& view, fft_engine<T> const& last_eng, fft_engine<T> const& prev_eng) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	constexpr auto rank = std::decay_t<ViewND>::dimensionality;
	if constexpr(rank == 2) {
		fft_apply_last(view, last_eng);            // fibers along axis 1
		fft_apply_last(view.rotated(), prev_eng);  // fibers along axis 0, slab still hot
	} else {
		for(auto&& sub : view) { fft_apply_last_pair(sub, last_eng, prev_eng); }
	}
}

// Transform every fiber along the *last* axis of `view` through the engine.
// The rank-descent drops leading axes one at a time but keeps the leading axis
// of smallest stride alive (via transposed(), which swaps the first two axes),
// so that at rank 2 the batch axis is the one closest in memory.
template<class ViewND, class T>
void fft_apply_last(ViewND&& view, fft_engine<T> const& eng) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	constexpr auto rank = std::decay_t<ViewND>::dimensionality;
	if constexpr(rank == 1) {
		fft_exec_fiber(view, eng);
	} else if constexpr(rank == 2) {
		fft_exec_slab(view, eng);
	} else {
		using std::get;
		auto const strs = view.strides();
		auto const s0   = static_cast<std::ptrdiff_t>(get<0>(strs));
		auto const s0a  = (s0 < 0) ? -s0 : s0;
		if(s0a <= fft_min_abs_mid_stride(strs, std::make_index_sequence<static_cast<std::size_t>(rank) - 2>{})) {
			for(auto&& sub : view.transposed()) { fft_apply_last(sub, eng); }
		} else {
			for(auto&& sub : view) { fft_apply_last(sub, eng); }
		}
	}
}

}  // end namespace detail

// Reusable multidimensional FFT plan: precomputes twiddle tables, stage
// factorizations, DFT matrices and scratch buffers for a given shape and
// direction, and applies them to any array/subarray of that shape (any
// strided layout) with `plan(A)` or `plan.execute(A)`, repeatedly, without
// re-allocation.
template<class T, std::ptrdiff_t D>
class fft_plan {
	static_assert(D >= 1, "fft_plan requires at least one dimension");

	std::array<std::size_t, static_cast<std::size_t>(D)> sizes_{};
	int                                                  sign_;
	std::vector<detail::fft_engine<T>>                   engines_;  // one per distinct axis length
	std::array<std::size_t, static_cast<std::size_t>(D)> which_{};  // axis -> index into engines_

	void init_() {
		for(std::size_t a = 0; a != static_cast<std::size_t>(D); ++a) {
			auto const len = sizes_[a];
			auto       it  = std::find_if(engines_.begin(), engines_.end(), [len](auto const& e) { return e.n_ == len; });
			if(it == engines_.end()) {
				engines_.emplace_back(len, sign_);
				it = std::prev(engines_.end());
			}
			which_[a] = static_cast<std::size_t>(it - engines_.begin());
		}
	}

	// Engine serving axis `A` (compile-time axis index, resolved at plan build).
	template<std::ptrdiff_t A>
	auto engine_() const -> detail::fft_engine<T> const& {
		static_assert(A >= 0 && A < D, "axis out of range");
		return engines_[which_[static_cast<std::size_t>(A)]];
	}

	// Static recursion over the remaining axes D-3 .. 0: `view` is `arr`
	// rotated K times (rotated() sends axis 0 to the back), so its last axis
	// is original axis K-1. Each axis is a distinct instantiation, bound to
	// its engine at compile time; rotated() preserves rank and type, so the
	// recursion depth is exactly D-2 with a single View type.
	template<std::ptrdiff_t K, class View>
	void transform_middle_(View&& view) const {  // NOLINT(cppcoreguidelines-missing-std-forward)
		detail::fft_apply_last(view, engine_<K - 1>());
		if constexpr(K < D - 2) { transform_middle_<K + 1>(view.rotated()); }
	}

	template<class Extents, std::size_t... Is>
	static auto to_sizes_(Extents const& ext, std::index_sequence<Is...> /*unused*/) -> std::array<std::size_t, static_cast<std::size_t>(D)> {
		using std::get;
		return {detail::fft_extent_size(get<Is>(ext))...};
	}

	template<class Sizes, std::size_t... Is>
	auto matches_(Sizes const& szs, std::index_sequence<Is...> /*unused*/) const -> bool {
		using std::get;
		return ((static_cast<std::size_t>(get<Is>(szs)) == sizes_[Is]) && ...);
	}

 public:
	// Plan for a shape given as any tuple-like of extents or integral sizes
	// (e.g. multi::extents_t<D>, A.sizes(), std::array<int, D>).
	template<class Extents, std::enable_if_t<!detail::fft_is_multi_like<Extents>::value, int> = 0>  // NOLINT(modernize-use-constraints) for C++17 compatibility
	explicit fft_plan(Extents const& extents, int sign = fft_forward)
	: sizes_{to_sizes_(extents, std::make_index_sequence<static_cast<std::size_t>(D)>{})}, sign_{sign} { init_(); }

	// Plan deduced from a prototype array/subarray (only its shape is used).
	template<class MultiSubArray, std::enable_if_t<detail::fft_is_multi_like<MultiSubArray>::value && (MultiSubArray::dimensionality == D), int> = 0>  // NOLINT(modernize-use-constraints)
	explicit fft_plan(MultiSubArray const& arr, int sign = fft_forward)
	: sizes_{to_sizes_(arr.sizes(), std::make_index_sequence<static_cast<std::size_t>(D)>{})}, sign_{sign} { init_(); }

	auto sign() const -> int { return sign_; }

	// Execute the plan on `arr` in place. `arr` must have the planned sizes;
	// its layout (strides, subarray-ness) is free to differ between calls.
	template<class MultiSubArray>
	auto operator()(MultiSubArray&& arr) const -> MultiSubArray&& {  // NOLINT(cppcoreguidelines-missing-std-forward)
		static_assert(std::decay_t<MultiSubArray>::dimensionality == D, "array rank must match the plan");
		assert(matches_(arr.sizes(), std::make_index_sequence<static_cast<std::size_t>(D)>{}));
		if constexpr(D == 1) {
			detail::fft_apply_last(arr(), engine_<0>());
		} else {
			// Transform the last two axes together, slab by slab (cache
			// locality), then the remaining axes 0 .. D-3 by static recursion
			// over rotated views (each axis bound to its engine at compile
			// time; no runtime axis parameter anywhere).
			detail::fft_apply_last_pair(arr(), engine_<D - 1>(), engine_<D - 2>());
			if constexpr(D >= 3) { transform_middle_<1>(arr().rotated()); }
		}
		return std::forward<MultiSubArray>(arr);
	}

	template<class MultiSubArray>
	auto execute(MultiSubArray&& arr) const -> MultiSubArray&& {
		return operator()(std::forward<MultiSubArray>(arr));
	}
};

template<class MultiSubArray>
fft_plan(MultiSubArray const&, int) -> fft_plan<typename MultiSubArray::element, MultiSubArray::dimensionality>;

template<class MultiSubArray>
fft_plan(MultiSubArray const&) -> fft_plan<typename MultiSubArray::element, MultiSubArray::dimensionality>;

// Generic in-place multidimensional FFT (one-shot convenience: builds a
// throw-away plan; for repeated transforms of the same shape, build an
// fft_plan once and execute it).
template<class MultiSubArray>
auto fft_inplace(MultiSubArray&& A, int sign = fft_forward) -> MultiSubArray&& {
	using array_type = std::decay_t<MultiSubArray>;
	fft_plan<typename array_type::element, array_type::dimensionality> const plan{A, sign};
	return plan(std::forward<MultiSubArray>(A));
}

template<class MultiSubArray>
auto fft_inplace_forward(MultiSubArray&& A) -> MultiSubArray&& {
	return fft_inplace(std::forward<MultiSubArray>(A), fft_forward);
}

template<class MultiSubArray>
auto fft_inplace_backward(MultiSubArray&& A) -> MultiSubArray&& {
	return fft_inplace(std::forward<MultiSubArray>(A), fft_backward);
}

}  // end namespace multi
}  // end namespace boost

#endif  // BOOST_MULTI_ALGORITHM_FFT_HPP
