// Copyright 2024-2026 Alfredo A. Correa
// Distributed under the Boost Software License, Version 1.0.
// https://www.boost.org/LICENSE_1_0.txt

// This header contains a generic, header-only, in-place multidimensional FFT
// with a reusable "plan" (in the FFTW sense: auxiliary tables and scratch
// allocations are computed once and reused across repeated transforms).
//
// Public interface:
//   multi::fft_plan<D, TW = complex<double>>  reusable transform state, where
//                                    TW is the twiddle-table type, fixed at
//                                    construction (default complex<double>,
//                                    for ergonomics only -- not a claim that
//                                    double is the "right" twiddle precision)
//     fft_plan<D, TW>(extents, sign) plan for a shape (any tuple-like extents),
//                                    one direction broadcast to every axis
//     fft_plan<D, TW>(extents, dirs) per-axis direction plan: dirs is a
//                                    std::array<fft_direction, D>; an axis
//                                    set to fft_direction::none is left
//                                    completely untouched (also gives
//                                    batched lower-rank FFTs for free, e.g.
//                                    {none, forward} on a 2-D array is
//                                    "FFT each row")
//     plan.execute(A)                transform A in place; A can be any array
//                                    or subarray with the planned sizes, of
//                                    any strided layout, with element type T
//                                    deduced fresh per call -- independent of
//                                    TW, so one plan (built once for a shape)
//                                    can execute a complex<float> array today
//                                    and a complex<double> array tomorrow
//                                    without rebuilding any tables -- as many
//                                    times as desired without re-allocating
//                                    the tables (scratch is allocated fresh,
//                                    locally, on every execute() call)
//   multi::fft_inplace(A, sign)      one-shot convenience (plans, then runs;
//                                    TW = A's own element type)
//   multi::fft_inplace(dirs, A)      one-shot convenience, per-axis
//                                    directions (dirs first: see fft.hpp's
//                                    fft_inplace overload comment for the
//                                    deduction trick this argument order
//                                    enables, and its one documented gap)
//   multi::fft_forward/fft_backward  direction constants (FFTW convention)
//   multi::fft_direction             per-axis direction enum: forward/
//                                    none/backward (values match
//                                    fft_forward/fft_backward exactly)
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
// Thread safety: a plan owns no scratch (only precomputed sizes/offsets); all
// scratch is allocated locally inside `execute()`. Concurrent `execute` calls
// on the *same* plan object from multiple threads are therefore safe with no
// external synchronization needed.

#ifndef BOOST_MULTI_ALGORITHMS_FFT_HPP
#define BOOST_MULTI_ALGORITHMS_FFT_HPP

#include <boost/multi/array_ref.hpp>  // for layout_t and subarray (cursor -> strided view reconstruction)

#include <algorithm>    // for copy, copy_n, fill, min, max, find_if, transform
#include <array>        // for plan sizes
#include <cassert>      // for assert
#include <cmath>        // for cos, sin, acos
#include <complex>      // for the fft_ops<std::complex> fast product
#include <cstddef>      // for size_t, ptrdiff_t
#include <functional>   // for greater
#include <iterator>     // for prev, next, reverse iterators
#include <limits>       // for numeric_limits
#include <memory>       // for addressof
#include <type_traits>  // for decay_t, enable_if_t, void_t
#include <utility>      // for forward, index_sequence
#include <vector>       // for tables and scratch buffers

// NOLINTBEGIN(altera-id-dependent-backward-branch,altera-unroll-loops,bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic,misc-no-recursion,readability-function-cognitive-complexity,readability-identifier-length)
// The numeric kernels operate on raw contiguous scratch by design (see
// fft.NOTES.md, "Design boundary"): stage loop bounds are data-dependent (FFT
// stage structure), batched kernels take (pointer, stride, width) tuples,
// plans nest recursively (Bluestein/six-step sub-plans, rank descent), and
// single-letter names (a/b buffers, w twiddles, x values) follow the FFT
// literature.

namespace boost::multi {

// Sign of the exponent in the discrete Fourier transform.
inline constexpr int fft_forward  = -1;  // exp(-2*pi*i*...), same as FFTW_FORWARD
inline constexpr int fft_backward = +1;  // exp(+2*pi*i*...), same as FFTW_BACKWARD

// Per-axis transform direction for partial/mixed-direction FFTs (see
// fft.NOTES.md §10): `none` means "leave this axis completely untouched" --
// a plan with `none` on some axes is a batched lower-rank FFT for free
// (`{none, forward}` on a 2-D array = "FFT each row"). Values match
// fft_forward/fft_backward exactly (checked below) so `to_sign()` is a
// plain cast, not a branch.
enum class fft_direction : int { forward = fft_forward, none = 0, backward = fft_backward };

namespace detail {
constexpr auto fft_to_sign(fft_direction d) -> int { return static_cast<int>(d); }
}  // namespace detail
static_assert(detail::fft_to_sign(fft_direction::forward) == fft_forward);
static_assert(detail::fft_to_sign(fft_direction::backward) == fft_backward);

// Trait that maps a complex-algebra element type to its underlying real type.
// Specialize this for custom complex-like types that do not expose `value_type`.
template<class T> struct fft_real {
	using type = typename T::value_type;
};

// Multiplication customization point for the transform kernels: `w` is a
// table value (twiddle/root-of-unity/DFT-matrix entry, type `TW`, the plan's
// own fixed precision); `x` is a data value (type `T`, the array being
// transformed, deduced fresh per execute() call -- see fft.NOTES.md §9.2).
// The generic default widens `x` up to `TW`'s precision, multiplies at that
// precision, and narrows the *result* back to `T` once -- deliberately not
// narrowing `w` to `T` first, which would round the table value on every
// multiply instead of just the final result (see fft.NOTES.md §9.2 on the
// accuracy/throughput trade this exposes when TW is wider than T). For every
// std::complex pairing -- same-type AND mixed (e.g. complex<float> data
// through a complex<double>-twiddle plan) -- the specialization below uses
// the plain widen-multiply-narrow formula instead of `operator*`: the
// operator carries C-Annex-G infinity/NaN fixups (a branch and a __muldc3
// libcall fallback) that prevent the batched inner loops from vectorizing;
// routing the mixed case through it (as an earlier version of this file did,
// via the generic default) put a libcall in every twiddle multiply of the
// T != TW path. Users can specialize this for custom element types with a
// faster product.
template<class T, class TW = T> struct fft_ops {
	static constexpr auto mul(TW const& w, T const& x) -> T { return static_cast<T>(w * static_cast<TW>(x)); }
	// == mul(conj(w), x); ADL so a custom TW's own conj() (if any) is found.
	static constexpr auto conj_mul(TW const& w, T const& x) -> T {
		using std::conj;
		return static_cast<T>(conj(w) * static_cast<TW>(x));
	}
};

template<class R1, class R2> struct fft_ops<std::complex<R1>, std::complex<R2>> {
	static constexpr auto mul(std::complex<R2> const& w, std::complex<R1> const& x) -> std::complex<R1> {
		// Products form in the wider of the two precisions (arithmetic on
		// mixed operands promotes); each result component narrows once. For
		// R1 == R2 every conversion is an identity and this is byte-for-byte
		// the previous same-type-only specialization.
		using promoted   = std::common_type_t<R1, R2>;
		promoted const wr = w.real();
		promoted const wi = w.imag();
		promoted const xr = x.real();
		promoted const xi = x.imag();
		return {static_cast<R1>((wr * xr) - (wi * xi)), static_cast<R1>((wr * xi) + (wi * xr))};
	}
	// == mul(conj(w), x): same 4-mul/2-add shape as mul, two signs flipped
	// (conjugating w negates wi's contribution). Used for backward-direction
	// kernel dispatch (fft.NOTES.md §10, Phase B): engines store forward-sign
	// tables only, and a backward pass conjugates every table load instead of
	// keeping a second, sign-baked table.
	static constexpr auto conj_mul(std::complex<R2> const& w, std::complex<R1> const& x) -> std::complex<R1> {
		using promoted   = std::common_type_t<R1, R2>;
		promoted const wr = w.real();
		promoted const wi = w.imag();
		promoted const xr = x.real();
		promoted const xi = x.imag();
		return {static_cast<R1>((wr * xr) + (wi * xi)), static_cast<R1>((wr * xi) - (wi * xr))};
	}
};

// The stage kernels' source and destination never alias within one stage
// (even the fused in-place path reads user memory only in the first stage and
// writes it only in the last). Telling the compiler removes runtime overlap
// checks and loop versioning (measured: ~7-10% on 2-D/3-D).
#ifdef _MSC_VER
#define BOOST_MULTI_FFT_RESTRICT __restrict
#else
#define BOOST_MULTI_FFT_RESTRICT __restrict__
#endif

namespace detail {

template<class T>
using fft_real_t = typename fft_real<T>::type;

template<class Real>
auto fft_pi() -> Real { return std::acos(Real{-1}); }

template<class T, class TW>
constexpr auto fft_mul(TW const& w, T const& x) -> T { return fft_ops<T, TW>::mul(w, x); }

// Direction-dispatching multiply (fft.NOTES.md §10, Phase B): `Backward`
// picks conj_mul (conjugate the table operand `w` on load) or plain mul, so
// every kernel needs exactly one instantiation-time choice instead of a
// second, sign-baked table. Uniform-conjugation invariant (verified,
// NOTES §10.5): every direction-dependent value in every smooth-path kernel
// is a table load (`tw_`/`wmat_`, including the +-i constant and the fixed
// radix-3/5 roots -- they are table entries too, never hardcoded literals),
// so conjugating every `fft_mul_dir` call uniformly makes the whole smooth
// path direction-correct. Do not special-case any one call site "out" of
// the conjugation.
template<bool Backward, class T, class TW>
constexpr auto fft_mul_dir(TW const& w, T const& x) -> T {
	if constexpr(Backward) {
		return fft_ops<T, TW>::conj_mul(w, x);
	} else {
		return fft_ops<T, TW>::mul(w, x);
	}
}

// Largest prime handled by the direct (table-driven O(p^2)) kernel; larger
// prime factors use a Bluestein sub-plan, which is O(p log p).
inline constexpr std::size_t fft_max_direct_radix = 64;

// Single-fiber transforms at least this long use the six-step decomposition
// n = n1*n2 (column FFTs, twiddle-transpose, row FFTs): both FFT passes then
// run batched (vectorized) and cache-blocked instead of striding across the
// whole fiber. Threshold chosen by measurement (2^13 is neutral, 2^14..2^15
// gain 10-25%).
inline constexpr std::size_t fft_sixstep_min = std::size_t{1} << 13U;

// Whether skipping element default-construction (and destruction) of a
// scratch buffer is allowed for `T`. Three ways in:
//   1. the language already says construction/destruction are trivial;
//   2. the type is opted in through Multi's own customization point
//      (`multi::force_element_trivial_default_construction`, array_ref.hpp)
//      -- the same idiom multi::array itself uses to skip
//      zero-initialization for its elements;
//   3. `T` is a std::complex over a trivial real type: complex is trivially
//      copyable but NOT trivially default-constructible (its default
//      constructor zero-initializes through defaulted arguments), so
//      without this every "uninitialized" scratch buffer of complex
//      compiles to a full memset on each use. array_ref.hpp enables the
//      same thing through its opt-in only under the global
//      `_BOOST_MULTI_FORCE_TRIVIAL_STD_COMPLEX` macro; this header applies
//      it to its OWN scratch unconditionally (maintainer decision: safe --
//      scratch elements are always fully written before being read) without
//      defining that macro, which would change multi::array behavior for
//      every translation unit that includes this header.
template<class T> struct fft_is_trivial_complex : std::false_type {};
template<class R>
struct fft_is_trivial_complex<std::complex<R>> : std::bool_constant<std::is_trivially_copyable_v<R> && std::is_trivially_default_constructible_v<R>> {};

template<class T>
inline constexpr bool fft_skip_element_init =
	((std::is_trivially_default_constructible_v<T> || multi::force_element_trivial_default_construction<T>) &&
	 (std::is_trivially_destructible_v<T> || multi::force_element_trivial_destruction<T>)) ||
	fft_is_trivial_complex<T>::value;

// Small fixed-size local buffer that is deliberately NOT initialized when
// `fft_skip_element_init` allows it. A plain `std::array<T, N>` local
// default-initializes its elements, so for std::complex it would memset the
// whole buffer on every entry to the enclosing function; the six-step
// transpose tile below hit this once per fiber. Elements are always fully
// written before being read, so the zero-fill is pure waste. Cache-line
// aligned as a bonus. Types without the opt-in fall back to a properly
// constructed std::array.
template<class T, std::size_t N, bool = fft_skip_element_init<T>>
struct fft_tile_buffer {
	alignas(64) std::byte storage_[sizeof(T) * N];  // NOLINT(cppcoreguidelines-avoid-c-arrays,misc-non-private-member-variables-in-classes)
	// Cast via void* (not std::byte* -> T* directly): storage_ is already
	// alignas(64), so this is safe, but -Wcast-align=strict only reasons
	// about the STATIC pointer types (alignof(std::byte) == 1) and flags
	// the direct cast regardless of the runtime alignment guarantee; the
	// void* intermediate is the standard idiom to route around that.
	auto data() -> T* { return reinterpret_cast<T*>(static_cast<void*>(storage_)); }  // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
};

template<class T, std::size_t N>
struct fft_tile_buffer<T, N, false> {
	std::array<T, N> arr_{};  // NOLINT(misc-non-private-member-variables-in-classes)
	auto data() -> T* { return arr_.data(); }
};

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
template<class TW>
struct fft_engine {
	// NOLINTBEGIN(misc-non-private-member-variables-in-classes) engine is an implementation-detail aggregate
	std::size_t n_  = 0;
	std::size_t mb_ = 1;  // preferred batch width (scratch is sized so 2*n*mb stays cache-resident)

	struct stage_t {
		std::size_t radix;
		int         kind;  // 0: radix-2, 1: radix-3, 2: radix-4, 3: radix-5, 4: generic direct, 5: Bluestein sub-plan, 6: radix-8
		std::size_t aux;   // generic: offset into wmat_; sub-plan: index into sub_
	};

	std::vector<TW>      tw_;      // tw_[k] = exp(sign*2*pi*i*k/n), k in [0, n)  (sign baked in)
	std::vector<stage_t> stages_;  // ordered stage factorization of n
	std::vector<TW>      wmat_;    // concatenated p x p DFT matrices for direct generic radices
	std::size_t          max_gen_ = 0;

	// Six-step state (used for long single-fiber transforms):
	bool        sixstep_ = false;
	std::size_t six_n1_  = 0;
	std::size_t six_n2_  = 0;
	std::size_t six_i1_  = 0;  // index into sub_ of the length-n1 column plan
	std::size_t six_i2_  = 0;  // index into sub_ of the length-n2 row plan

	// Bluestein state (used when n_ is a prime > fft_max_direct_radix):
	// X_k = c_k * sum_n x_n c_n d_{k-n} with c_j = exp(-i*pi*j^2/n) (canonical
	// forward sign, Phase B), d = 1/c, evaluated as a circular convolution of
	// power-of-two length conv_n_ >= 2n-1. `chirp_`/`postc_` are plain
	// elementwise conjugates across direction, so conj-on-load
	// (`fft_mul_dir<Backward>`) handles them; `kernel_ft_` does not conjugate
	// cleanly on load (FFT of a conjugated sequence is a conjugated,
	// INDEX-REVERSED spectrum), so the backward-direction spectrum is a
	// second, separately precomputed table (fft.NOTES.md §10.5).
	bool            bluestein_     = false;
	std::size_t     conv_n_        = 0;
	std::vector<TW> chirp_;          // c_j (forward-canonical)
	std::vector<TW> postc_;          // c_k / conv_n_  (fused convolution normalization, forward-canonical)
	std::vector<TW> kernel_ft_;      // FFT of the wrapped d-kernel, forward direction
	std::vector<TW> kernel_ft_bwd_;  // == conj(kernel_ft_[(conv_n_ - k) % conv_n_]), backward direction

	std::vector<fft_engine> sub_;  // nested engines: single neutral Bluestein conv sub-engine (run forward then backward), or large-prime stage sub-plans

	// Scratch is not owned here: only the *sizes* (peak element counts,
	// `note_reach_`) and disjoint *offsets* (`assign_offsets_`) into an
	// external, execute()-time-local arena that covers this engine and its
	// whole `sub_` tree. Both are computed once, post-construction, and are
	// read-only from then on -- see fft.NOTES.md §9.2/§10.4(a).
	std::size_t buf_cap_  = 0;  // >= n*m, gathered input / ping-pong A
	std::size_t out_cap_  = 0;  // >= n*m, ping-pong B (0 for bluestein_ engines)
	std::size_t xbuf_cap_ = 0;  // >= max_gen_*m, generic-stage gather scratch

	std::size_t buf_off_  = 0;
	std::size_t out_off_  = 0;
	std::size_t xbuf_off_ = 0;
	// NOLINTEND(misc-non-private-member-variables-in-classes)

	// Non-owning accessors: locate this engine's scratch inside the caller's
	// arena. Never stored on the engine (that would make `execute()` racy
	// across threads); recomputed locally on every call. `T` is the array's
	// element type, deduced per execute() call -- independent of `TW`, the
	// engine's own (fixed, construction-time) twiddle-table type.
	template<class T>
	auto buf_ptr(T* arena) const -> T* { return arena + buf_off_; }
	template<class T>
	auto out_ptr(T* arena) const -> T* { return arena + out_off_; }
	template<class T>
	auto xbuf_ptr(T* arena) const -> T* { return arena + xbuf_off_; }

	// Record that this engine's own buf_/out_/xbuf_ must hold at least a
	// batch of width `m` (mirrors the old `ensure(m)` growth, but as a
	// static, monotonic-max size computation instead of a runtime resize).
	void note_own_(std::size_t m) {
		std::size_t const need = std::max<std::size_t>(n_, 1) * m;
		buf_cap_ = std::max(buf_cap_, need);
		if(!bluestein_) {
			out_cap_ = std::max(out_cap_, need);
		}
		if(max_gen_ != 0) {
			xbuf_cap_ = std::max(xbuf_cap_, max_gen_ * m);
		}
	}

	// Walk every code path `run(m, ...)` could take for this engine at batch
	// width `m`, recording scratch requirements for this engine and (with
	// the same `m` propagation `run`/`run_fused_impl_`/`run_sixstep_` use)
	// every reachable descendant in `sub_`. Deliberately conservative: both
	// the six-step and per-stage-subplan children are always visited when
	// they exist, even though a real call only ever takes one path for a
	// given `m` -- see fft.NOTES.md §10.5 (this trades a little unused
	// capacity for a much simpler, harder-to-get-wrong reachability rule).
	void note_reach_(std::size_t m) {
		note_own_(m);
		if(n_ < 2) {
			return;
		}
		if(bluestein_) {
			sub_[0].note_reach_(m);  // single neutral conv sub-engine (Phase B collapse), run forward then backward
			return;
		}
		if(sixstep_) {
			sub_[six_i1_].note_reach_(six_n2_);
			sub_[six_i2_].note_reach_(six_n1_);
		}
		std::size_t ns = 1;
		for(stage_t const& st : stages_) {
			if(st.kind == 5) {
				sub_[st.aux].note_reach_(ns * m);
			}
			ns *= st.radix;
		}
	}

	// Assign disjoint offsets (this engine, then recursively every `sub_`
	// child) within one flat arena, given the capacities `note_reach_` has
	// already computed. `cursor` is the running arena size; the final value
	// after the whole top-level walk is the plan's total scratch_elements().
	void assign_offsets_(std::size_t& cursor) {
		buf_off_ = cursor;
		cursor += buf_cap_;
		out_off_ = cursor;
		cursor += out_cap_;
		xbuf_off_ = cursor;
		cursor += xbuf_cap_;
		for(fft_engine& s : sub_) {
			s.assign_offsets_(cursor);
		}
	}

	// Default state == fft_engine(0): every member already has an in-class
	// default initializer matching what the nn<2 early-return below leaves
	// (n_=0, mb_=1, every table/vector empty) -- not a new/distinct
	// "invalid" state, just making the ALREADY-existing trivial, no-op
	// engine reachable without an explicit length. Used to default-fill
	// `fft_plan::engines_`'s unused (padding) slots -- NOTES §10.1 item 9.
	fft_engine() = default;

	// Direction-neutral (Phase B, NOTES §10.5): builds forward-canonical
	// tables only. A backward pass conjugates every table load at run time
	// (`fft_mul_dir<Backward>`) instead of building a second, sign-baked set
	// of tables -- see the kernel comments below for the invariant this
	// relies on.
	explicit fft_engine(std::size_t nn) : n_{nn} {  // NOLINT(readability-function-cognitive-complexity)
		if(nn < 2) {
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
		while(rem % 2 == 0) {
			++k2;
			rem /= 2;
		}
		if(k2 == 1) {
			fac.push_back(2);
		} else {
			for(std::size_t k = (k2 % 2 == 1) ? k2 - 3 : k2; k >= 2; k -= 2) {
				fac.push_back(4);
			}
			if(k2 % 2 == 1 && k2 >= 3) {
				fac.push_back(8);
			}
		}
		for(std::size_t p = 3; p * p <= rem; p += 2) {
			while(rem % p == 0) {
				fac.push_back(p);
				rem /= p;
			}
		}
		if(rem > 1) {
			fac.push_back(rem);
		}

		if(fac.size() == 1 && fac.front() == nn && nn > fft_max_direct_radix) {
			init_bluestein_();
			mb_ = batch_width_(conv_n_);
			return;
		}

		using real      = fft_real_t<TW>;
		real const step = static_cast<real>(fft_forward) * real{2} * fft_pi<real>() / static_cast<real>(nn);
		tw_.resize(nn);
		for(std::size_t k = 0; k != nn; ++k) {
			real const theta = step * static_cast<real>(k);
			tw_[k]           = TW{std::cos(theta), std::sin(theta)};
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
			std::sort(desc.begin(), desc.end(), std::greater<>{});  // NOLINT(modernize-use-ranges) C++17 compatibility
			std::size_t n1 = 1;
			std::size_t n2 = 1;
			for(std::size_t const f : desc) {
				(n1 <= n2 ? n1 : n2) *= f;
			}
			if(std::min(n1, n2) >= 16) {
				sixstep_ = true;
				six_n1_  = n1;
				six_n2_  = n2;
				six_i1_  = sub_index_(n1);
				if(n2 == n1) {  // distinct engine: the two passes' buffers must not alias
					sub_.emplace_back(n2);
					six_i2_ = sub_.size() - 1;
				} else {
					six_i2_ = sub_index_(n2);
				}
			}
		}

		mb_ = batch_width_(nn);
	}

	// Transform the gathered data `in` (layout [n][m], batch contiguous) and
	// return a pointer to the result in the same layout. `in` defaults to the
	// plan's own gather region `buf_ptr(arena)`; the caller fills it before
	// calling. `arena` is the caller-supplied scratch for the whole plan (see
	// `note_reach_`/`assign_offsets_`); sizes are precomputed, so no runtime
	// growth check is needed here. `T` (the array's element type) is deduced
	// from `arena`/`in`, independent of `TW` (this engine's table type).
	// `backward` is a runtime direction argument (Phase B, NOTES §10.5):
	// engines store forward-canonical tables only, and this dispatches ONCE
	// per invocation to the `<..., Backward>` instantiation -- one branch per
	// pass, nothing per element.
	template<class T>
	auto run(std::size_t m, bool backward, T* arena) const -> T const* { return run(m, backward, buf_ptr(arena), arena); }

	template<class T>
	auto run(std::size_t m, bool backward, T const* in, T* arena) const -> T const* {
		if(sixstep_ && m == 1 && n_ >= 2) {
			return backward ? run_sixstep_<true>(in, arena) : run_sixstep_<false>(in, arena);
		}  // uses only the sub-plans' buffers
		if(n_ < 2) {
			T* const b = buf_ptr(arena);
			if(in != b) {
				std::copy(in, in + (n_ * m), b);
			}
			return b;
		}
		if(bluestein_) {
			return backward ? run_bluestein_<true>(m, in, arena) : run_bluestein_<false>(m, in, arena);
		}
		if(m == 1) {
			return backward ? run_stages_<false, true>(1, in, arena) : run_stages_<false, false>(1, in, arena);
		}
		return backward ? run_stages_<true, true>(m, in, arena) : run_stages_<true, false>(m, in, arena);
	}

	// True when the stage pipeline can read/write user memory directly: the
	// first stage fully consumes the input (so it may alias the output) and a
	// distinct last stage exists to produce the final values.
	auto can_fuse() const -> bool { return !bluestein_ && stages_.size() >= 2; }

	// Transform directly between user tiles: the first stage reads
	// in[k*si + j] and the last stage writes out[k*so + j] (batch index j
	// contiguous), skipping the separate gather and scatter passes. `in` may
	// alias `out`. Only valid when can_fuse().
	template<class T>
	void run_fused(T const* in, std::size_t si, T* out, std::size_t so, std::size_t m, bool backward, T* arena) const {
		assert(can_fuse());
		assert(m > 1 || (si == 1 && so == 1));  // the unbatched kernels fold strides to 1
		if(m == 1) {
			if(backward) {
				run_fused_impl_<false, true>(in, 1, out, 1, 1, arena);
			} else {
				run_fused_impl_<false, false>(in, 1, out, 1, 1, arena);
			}
		} else {
			if(backward) {
				run_fused_impl_<true, true>(in, si, out, so, m, arena);
			} else {
				run_fused_impl_<true, false>(in, si, out, so, m, arena);
			}
		}
	}

	// In-place transform of one contiguous (stride-1) fiber in user memory;
	// the final pass writes straight back (no scatter copy).
	template<class T>
	void run_contig_inplace(T* io, T* arena, bool backward) const {
		if(n_ < 2) {
			return;
		}
		if(sixstep_) {
			if(backward) {
				run_sixstep_<true>(io, arena, io);
			} else {
				run_sixstep_<false>(io, arena, io);
			}
			return;
		}
		if(bluestein_) {
			if(backward) {
				run_bluestein_<true>(1, io, arena, io);
			} else {
				run_bluestein_<false>(1, io, arena, io);
			}
			return;
		}
		if(stages_.size() >= 2) {
			if(backward) {
				run_fused_impl_<false, true>(io, 1, io, 1, 1, arena);
			} else {
				run_fused_impl_<false, false>(io, 1, io, 1, 1, arena);
			}
			return;
		}
		T const* const res = run(1, backward, io, arena);
		std::copy(res, res + n_, io);
	}

 private:
	static auto batch_width_(std::size_t nn) -> std::size_t {
		// Two ping-pong buffers of n*mb elements should stay ~cache-resident.
		// Sized from `sizeof(TW)` since `mb_` is fixed at construction, before
		// any array type `T` is known -- a reasonable stand-in whenever T's
		// size is comparable to TW's (same-type, or float-vs-double).
		std::size_t const budget = (std::size_t{1} << 22U) / (2 * sizeof(TW) * std::max<std::size_t>(nn, 1));
		return std::clamp<std::size_t>(budget, 1, 64);
	}

	auto wmat_offset_(std::size_t rr) -> std::size_t {
		// The p x p matrix W[u*p + t] = exp(sign*2*pi*i*t*u/p) tabulates the
		// size-p sub-DFT, removing the inner-loop modulo of a naive kernel.
		std::size_t const off = wmat_.size();
		std::size_t const wr  = n_ / rr;  // step of the p-th roots of unity in tw_
		wmat_.resize(off + (rr * rr));
		for(std::size_t u = 0; u != rr; ++u) {
			for(std::size_t t = 0; t != rr; ++t) {
				wmat_[off + (u * rr) + t] = tw_[(t * u % rr) * wr];
			}
		}
		return off;
	}

	auto sub_index_(std::size_t rr) -> std::size_t {
		auto const it = std::find_if(sub_.begin(), sub_.end(), [rr](fft_engine const& e) { return e.n_ == rr; });
		if(it != sub_.end()) {
			return static_cast<std::size_t>(it - sub_.begin());
		}
		sub_.emplace_back(rr);
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
			while(r % 2 == 0) {
				++k;
				r /= 2;
			}
			if(k == 1) {
				w += 0.7;
			}  // lone radix-2 stage
			else if(k != 0) {
				w += (k % 2 == 1 ? 1.2 + (static_cast<double>(k - 3) / 2) : static_cast<double>(k) / 2);
			}  // radix-4s + one 8 if odd
			while(r % 3 == 0) {
				w += 0.9;
				r /= 3;
			}
			while(r % 5 == 0) {
				w += 1.45;
				r /= 5;
			}
			if(r != 1) {
				return -1.0;
			}  // not smooth
			return static_cast<double>(c) * w;
		};
		std::size_t pow2 = 1;
		while(pow2 < target) {
			pow2 *= 2;
		}
		std::size_t best      = pow2;
		double      best_cost = cost(pow2);
		for(std::size_t c = target; c != pow2; ++c) {
			double const cc = cost(c);
			if(cc >= 0.0 && cc < best_cost) {
				best      = c;
				best_cost = cc;
			}
		}
		return best;
	}

	void init_bluestein_() {
		using real = fft_real_t<TW>;
		bluestein_ = true;

		conv_n_ = next_smooth_((2 * n_) - 1);

		chirp_.resize(n_);
		postc_.resize(n_);

		// Single neutral conv sub-engine (Phase B, NOTES §10.5): the two
		// Phase-A engines ("forward conv" and "inverse conv") were only ever
		// distinguished by sign, and the convolution mechanism itself is fixed
		// regardless of the outer transform's direction (canonical fwd conv,
		// then inverse conv) -- so one engine, run twice with opposite
		// `Backward` values, replaces the pair. See run_bluestein_ for the
		// resulting buf_ptr_ aliasing between the two runs (safe, documented
		// there).
		sub_.emplace_back(conv_n_);

		// Wrapped convolution kernel b: b[j] = b[conv_n_ - j] = d_j = conj-chirp.
		// This is a one-time, construction-only bootstrap to precompute the
		// immutable kernel_ft_ table: `conv` needs *some* scratch to run once
		// at m=1, so it gets a private, throwaway local arena sized from its
		// own (self-contained) subtree requirement. These offsets are
		// provisional; fft_plan's constructor lays out the real, whole-plan
		// arena afterward (note_reach_ is monotonic-max, so re-running it
		// there is safe), and only the immutable table data computed here
		// (chirp_/postc_/kernel_ft_/kernel_ft_bwd_) survives past this
		// function. Entirely TW-typed: no array type T exists yet at
		// construction time.
		fft_engine& conv = sub_[0];
		conv.note_reach_(1);
		std::size_t boot_cursor = 0;
		conv.assign_offsets_(boot_cursor);
		std::vector<TW> boot_arena(boot_cursor);

		TW* const y = conv.buf_ptr(boot_arena.data());
		std::fill(y, y + conv_n_, TW{});

		real const  pi_n = fft_pi<real>() / static_cast<real>(n_);
		std::size_t jsq  = 0;  // j^2 mod 2n, updated incrementally to avoid overflow
		for(std::size_t j = 0; j != n_; ++j) {
			real const theta = static_cast<real>(fft_forward) * pi_n * static_cast<real>(jsq);  // forward-canonical (Phase B); backward via conj-on-load
			chirp_[j]        = TW{std::cos(theta), std::sin(theta)};
			TW const dj      = TW{std::cos(theta), -std::sin(theta)};
			y[j]             = dj;
			if(j != 0) {
				y[conv_n_ - j] = dj;
			}
			jsq += (2 * j) + 1;
			while(jsq >= 2 * n_) {
				jsq -= 2 * n_;
			}
		}

		TW const inv_m = TW{real{1} / static_cast<real>(conv_n_), real{0}};
		std::transform(chirp_.begin(), chirp_.end(), postc_.begin(), [inv_m](TW const& c) { return fft_mul(inv_m, c); });  // branch-free product, same as the kernels (construction-time, but no reason to take the operator* libcall path)

		TW const* kft = conv.run(1, /*backward=*/false, boot_arena.data());  // canonical forward, always -- see run_bluestein_
		kernel_ft_.assign(kft, kft + conv_n_);

		// kernel_ft_bwd_[k] == conj(kernel_ft_[(conv_n_ - k) % conv_n_]): FFT of
		// a conjugated sequence is a conjugated, INDEX-REVERSED spectrum, so
		// plain conj-on-load (fft_mul_dir) does not work for this table --
		// precompute the reversed/conjugated table once instead (fft.NOTES.md
		// §10.5). k == 0 is its own mirror; the rest is a reversed conjugate
		// copy: kernel_ft_.rbegin() is kernel_ft_[conv_n_-1] -> kernel_ft_bwd_[1],
		// down to kernel_ft_[1] -> kernel_ft_bwd_[conv_n_-1].
		kernel_ft_bwd_.resize(conv_n_);
		using std::conj;
		kernel_ft_bwd_[0] = conj(kernel_ft_[0]);
		std::transform(kernel_ft_.rbegin(), std::prev(kernel_ft_.rend()), std::next(kernel_ft_bwd_.begin()), [](TW const& v) { return conj(v); });
	}

	// --- batched Stockham stage kernels -----------------------------------
	// Data layout: element k of batch-fiber j lives at [k*m + j]. `Batched`
	// selects at compile time between the vector inner loop and the m == 1
	// fast path (no inner loop overhead for single fibers).

	template<bool Batched, bool Backward, class T>
	void stage_radix2_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;   // folds all offset arithmetic when unbatched
		std::size_t const sa    = Batched ? sa_ : 1;  // input element stride (user tile when fused)
		std::size_t const sb    = Batched ? sb_ : 1;  // output element stride
		std::size_t const half  = n_ / 2;
		std::size_t const tstep = n_ / (2 * ns);
		for(std::size_t block = 0; block != half; block += ns) {
			std::size_t const base = block * 2;
			for(std::size_t r = 0; r != ns; ++r) {
				TW const       w  = tw_[r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T const* const a1 = a0 + (half * sa);
				T* const       b0 = b + ((base + r) * sb);
				T* const       b1 = b0 + (ns * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const v0 = a0[j];
					T const v1 = fft_mul_dir<Backward>(w, a1[j]);
					b0[j]      = v0 + v1;
					b1[j]      = v0 - v1;
				}
			}
		}
	}

	// The multiply-by-(-/+ i) is expressed as a multiply by tw_[n/4] so it
	// stays generic over the element type and carries the correct sign.
	template<bool Batched, bool Backward, class T>
	void stage_radix4_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const q     = n_ / 4;
		std::size_t const tstep = n_ / (4 * ns);
		TW const          imu   = tw_[q];  // -i for forward, +i for backward (i.e. under conj-on-load for Backward)
		for(std::size_t block = 0; block != q; block += ns) {
			std::size_t const base = block * 4;
			for(std::size_t r = 0; r != ns; ++r) {
				TW const       w1 = tw_[r * tstep];
				TW const       w2 = tw_[2 * r * tstep];
				TW const       w3 = tw_[3 * r * tstep];
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
					T const x1 = fft_mul_dir<Backward>(w1, a1[j]);
					T const x2 = fft_mul_dir<Backward>(w2, a2[j]);
					T const x3 = fft_mul_dir<Backward>(w3, a3[j]);
					T const t0 = x0 + x2;
					T const t1 = x0 - x2;
					T const t2 = x1 + x3;
					T const t3 = fft_mul_dir<Backward>(imu, x1 - x3);
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
	template<bool Batched, bool Backward, class T>
	void stage_radix8_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const q     = n_ / 8;
		std::size_t const tstep = n_ / (8 * ns);
		TW const          imu   = tw_[2 * q];  // W8^2: -i for forward, +i for backward (i.e. under conj-on-load for Backward)
		TW const          w81   = tw_[q];
		TW const          w83   = tw_[3 * q];
		for(std::size_t block = 0; block != q; block += ns) {
			std::size_t const base = block * 8;
			for(std::size_t r = 0; r != ns; ++r) {
				TW const       w1 = tw_[r * tstep];
				TW const       w2 = tw_[2 * r * tstep];
				TW const       w3 = tw_[3 * r * tstep];
				TW const       w4 = tw_[4 * r * tstep];
				TW const       w5 = tw_[5 * r * tstep];
				TW const       w6 = tw_[6 * r * tstep];
				TW const       w7 = tw_[7 * r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T* const       b0 = b + ((base + r) * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const x0            = a0[j];
					T const x1            = fft_mul_dir<Backward>(w1, a0[(1 * q * sa) + j]);
					T const x2            = fft_mul_dir<Backward>(w2, a0[(2 * q * sa) + j]);
					T const x3            = fft_mul_dir<Backward>(w3, a0[(3 * q * sa) + j]);
					T const x4            = fft_mul_dir<Backward>(w4, a0[(4 * q * sa) + j]);
					T const x5            = fft_mul_dir<Backward>(w5, a0[(5 * q * sa) + j]);
					T const x6            = fft_mul_dir<Backward>(w6, a0[(6 * q * sa) + j]);
					T const x7            = fft_mul_dir<Backward>(w7, a0[(7 * q * sa) + j]);
					// radix-4 over the even legs (x0, x2, x4, x6)
					T const s0            = x0 + x4;
					T const s1            = x0 - x4;
					T const s2            = x2 + x6;
					T const s3            = fft_mul_dir<Backward>(imu, x2 - x6);
					T const e0            = s0 + s2;
					T const e1            = s1 + s3;
					T const e2            = s0 - s2;
					T const e3            = s1 - s3;
					// radix-4 over the odd legs (x1, x3, x5, x7), then W8^u twiddles
					T const u0            = x1 + x5;
					T const u1            = x1 - x5;
					T const u2            = x3 + x7;
					T const u3            = fft_mul_dir<Backward>(imu, x3 - x7);
					T const o0            = u0 + u2;
					T const o1            = fft_mul_dir<Backward>(w81, u1 + u3);
					T const o2            = fft_mul_dir<Backward>(imu, u0 - u2);
					T const o3            = fft_mul_dir<Backward>(w83, u1 - u3);
					b0[j]                 = e0 + o0;
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

	template<bool Batched, bool Backward, class T>
	void stage_radix3_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const n3    = n_ / 3;
		std::size_t const tstep = n_ / (3 * ns);
		TW const          w1c   = tw_[n3];      // W_3
		TW const          w2c   = tw_[2 * n3];  // W_3^2
		for(std::size_t block = 0; block != n3; block += ns) {
			std::size_t const base = block * 3;
			for(std::size_t r = 0; r != ns; ++r) {
				TW const       w1 = tw_[r * tstep];
				TW const       w2 = tw_[2 * r * tstep];
				T const* const a0 = a + ((block + r) * sa);
				T const* const a1 = a0 + (n3 * sa);
				T const* const a2 = a0 + (2 * n3 * sa);
				T* const       b0 = b + ((base + r) * sb);
				T* const       b1 = b0 + (ns * sb);
				T* const       b2 = b0 + (2 * ns * sb);
				for(std::size_t j = 0; j != m; ++j) {
					T const x0 = a0[j];
					T const x1 = fft_mul_dir<Backward>(w1, a1[j]);
					T const x2 = fft_mul_dir<Backward>(w2, a2[j]);
					b0[j]      = x0 + x1 + x2;
					b1[j]      = x0 + fft_mul_dir<Backward>(w1c, x1) + fft_mul_dir<Backward>(w2c, x2);
					b2[j]      = x0 + fft_mul_dir<Backward>(w2c, x1) + fft_mul_dir<Backward>(w1c, x2);
				}
			}
		}
	}

	template<bool Batched, bool Backward, class T>
	void stage_radix5_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t mm, std::size_t sa_, std::size_t sb_) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const n5    = n_ / 5;
		std::size_t const tstep = n_ / (5 * ns);
		TW const          w1c   = tw_[n5];
		TW const          w2c   = tw_[2 * n5];
		TW const          w3c   = tw_[3 * n5];
		TW const          w4c   = tw_[4 * n5];
		for(std::size_t block = 0; block != n5; block += ns) {
			std::size_t const base = block * 5;
			for(std::size_t r = 0; r != ns; ++r) {
				TW const       w1 = tw_[r * tstep];
				TW const       w2 = tw_[2 * r * tstep];
				TW const       w3 = tw_[3 * r * tstep];
				TW const       w4 = tw_[4 * r * tstep];
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
					T const x1 = fft_mul_dir<Backward>(w1, a1[j]);
					T const x2 = fft_mul_dir<Backward>(w2, a2[j]);
					T const x3 = fft_mul_dir<Backward>(w3, a3[j]);
					T const x4 = fft_mul_dir<Backward>(w4, a4[j]);
					b0[j]      = x0 + x1 + x2 + x3 + x4;
					b1[j]      = x0 + fft_mul_dir<Backward>(w1c, x1) + fft_mul_dir<Backward>(w2c, x2) + fft_mul_dir<Backward>(w3c, x3) + fft_mul_dir<Backward>(w4c, x4);
					b2[j]      = x0 + fft_mul_dir<Backward>(w2c, x1) + fft_mul_dir<Backward>(w4c, x2) + fft_mul_dir<Backward>(w1c, x3) + fft_mul_dir<Backward>(w3c, x4);
					b3[j]      = x0 + fft_mul_dir<Backward>(w3c, x1) + fft_mul_dir<Backward>(w1c, x2) + fft_mul_dir<Backward>(w4c, x3) + fft_mul_dir<Backward>(w2c, x4);
					b4[j]      = x0 + fft_mul_dir<Backward>(w4c, x1) + fft_mul_dir<Backward>(w3c, x2) + fft_mul_dir<Backward>(w2c, x3) + fft_mul_dir<Backward>(w1c, x4);
				}
			}
		}
	}

	// Direct radix-p stage for odd primes p <= fft_max_direct_radix, driven by
	// the precomputed p x p DFT matrix (no modulo in the inner loops).
	template<bool Batched, bool Backward, class T>
	void stage_generic_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t rr, TW const* wmat, std::size_t mm, std::size_t sa_, std::size_t sb_, T* arena) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const nr    = n_ / rr;
		std::size_t const tstep = n_ / (rr * ns);
		T* const          x     = xbuf_ptr(arena);
		for(std::size_t block = 0; block != nr; block += ns) {
			std::size_t const base = block * rr;
			for(std::size_t r = 0; r != ns; ++r) {
				T const* const asrc = a + ((block + r) * sa);
				std::copy_n(asrc, m, x);  // t == 0, twiddle == 1
				for(std::size_t t = 1; t != rr; ++t) {
					TW const       w  = tw_[t * r * tstep];
					T const* const at = asrc + (t * nr * sa);
					T* const       xt = x + (t * m);
					for(std::size_t j = 0; j != m; ++j) {
						xt[j] = fft_mul_dir<Backward>(w, at[j]);
					}
				}
				for(std::size_t u = 0; u != rr; ++u) {
					TW const* const wrow = wmat + (u * rr);
					T* const        dst  = b + ((base + r + (u * ns)) * sb);
					std::copy_n(x, m, dst);  // wrow[0] == 1
					for(std::size_t t = 1; t != rr; ++t) {
						TW const       wc = wrow[t];
						T const* const xt = x + (t * m);
						for(std::size_t j = 0; j != m; ++j) {
							dst[j] = dst[j] + fft_mul_dir<Backward>(wc, xt[j]);
						}
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
	template<bool Batched, bool Backward, class T>
	void stage_subplan_(T const* BOOST_MULTI_FFT_RESTRICT a, T* BOOST_MULTI_FFT_RESTRICT b, std::size_t ns, std::size_t rr, fft_engine const& sub, std::size_t mm, std::size_t sa_, std::size_t sb_, T* arena) const {
		std::size_t const m     = Batched ? mm : 1;
		std::size_t const sa    = Batched ? sa_ : 1;
		std::size_t const sb    = Batched ? sb_ : 1;
		std::size_t const nr    = n_ / rr;
		std::size_t const tstep = n_ / (rr * ns);
		std::size_t const m2    = ns * m;
		for(std::size_t block = 0; block != nr; block += ns) {
			T* const y = sub.buf_ptr(arena);
			for(std::size_t r = 0; r != ns; ++r) {
				T const* const asrc = a + ((block + r) * sa);
				T* const       y0   = y + (r * m);
				std::copy_n(asrc, m, y0);  // t == 0, twiddle == 1
				for(std::size_t t = 1; t != rr; ++t) {
					TW const       w  = tw_[t * r * tstep];
					T const* const at = asrc + (t * nr * sa);
					T* const       yt = y + (((t * ns) + r) * m);
					for(std::size_t j = 0; j != m; ++j) {
						yt[j] = fft_mul_dir<Backward>(w, at[j]);
					}
				}
			}
			T const* const z = sub.run(m2, Backward, arena);  // sub-DFTs of a backward transform are backward
			if(sb == m) {
				std::copy(z, z + (rr * ns * m), b + (block * rr * sb));
			} else {
				for(std::size_t idx = 0; idx != rr * ns; ++idx) {
					T const* const zr = z + (idx * m);
					T* const       br = b + (((block * rr) + idx) * sb);
					std::copy_n(zr, m, br);
				}
			}
		}
	}

	template<bool Batched, bool Backward, class T>
	void run_fused_impl_(T const* in, std::size_t si, T* out, std::size_t so, std::size_t m, T* arena) const {
		T const*          src  = in;
		T*                dst  = out_ptr(arena);
		T*                alt  = buf_ptr(arena);  // NOLINT(misc-const-correctness) written through after swap
		std::size_t       ns   = 1;
		std::size_t const last = stages_.size() - 1;
		for(std::size_t i = 0; i != stages_.size(); ++i) {
			stage_t const&    st = stages_[i];
			std::size_t const sa = (i == 0) ? si : m;
			T* const          d  = (i == last) ? out : dst;
			std::size_t const sb = (i == last) ? so : m;
			switch(st.kind) {
			case 0: stage_radix2_<Batched, Backward>(src, d, ns, m, sa, sb); break;
			case 1: stage_radix3_<Batched, Backward>(src, d, ns, m, sa, sb); break;
			case 2: stage_radix4_<Batched, Backward>(src, d, ns, m, sa, sb); break;
			case 3: stage_radix5_<Batched, Backward>(src, d, ns, m, sa, sb); break;
			case 4: stage_generic_<Batched, Backward>(src, d, ns, st.radix, wmat_.data() + st.aux, m, sa, sb, arena); break;
			case 6: stage_radix8_<Batched, Backward>(src, d, ns, m, sa, sb); break;
			default: stage_subplan_<Batched, Backward>(src, d, ns, st.radix, sub_[st.aux], m, sa, sb, arena); break;
			}
			src = d;
			std::swap(dst, alt);
			ns *= st.radix;
		}
	}

	template<bool Batched, bool Backward, class T>
	auto run_stages_(std::size_t m, T const* in, T* arena) const -> T const* {
		T const*    src = in;
		T*          dst = out_ptr(arena);
		T*          alt = buf_ptr(arena);  // NOLINT(misc-const-correctness) written through after swap
		std::size_t ns  = 1;
		for(stage_t const& st : stages_) {
			switch(st.kind) {
			case 0: stage_radix2_<Batched, Backward>(src, dst, ns, m, m, m); break;
			case 1: stage_radix3_<Batched, Backward>(src, dst, ns, m, m, m); break;
			case 2: stage_radix4_<Batched, Backward>(src, dst, ns, m, m, m); break;
			case 3: stage_radix5_<Batched, Backward>(src, dst, ns, m, m, m); break;
			case 4: stage_generic_<Batched, Backward>(src, dst, ns, st.radix, wmat_.data() + st.aux, m, m, m, arena); break;
			case 6: stage_radix8_<Batched, Backward>(src, dst, ns, m, m, m); break;
			default: stage_subplan_<Batched, Backward>(src, dst, ns, st.radix, sub_[st.aux], m, m, m, arena); break;
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
	template<bool Backward, class T>
	auto run_sixstep_(T const* in, T* arena, T* uout = nullptr) const -> T const* {
		auto const&       e1 = sub_[six_i1_];
		auto const&       e2 = sub_[six_i2_];
		std::size_t const n1 = six_n1_;
		std::size_t const n2 = six_n2_;

		T const* const z = e1.run(n2, Backward, in, arena);  // column FFTs: sub-DFTs of a backward transform are backward

		T* const                        yt = e2.buf_ptr(arena);
		constexpr std::size_t           tb = 32;  // 32 x 32 tiles staged through an L1 buffer, so both
		fft_tile_buffer<T, tb * tb> tile;         // the read and the write side stream contiguously (uninitialized for trivially-copyable T -- see fft_tile_buffer)
		for(std::size_t k10 = 0; k10 < n1; k10 += tb) {
			std::size_t const k1e = std::min(n1, k10 + tb);
			for(std::size_t j20 = 0; j20 < n2; j20 += tb) {
				std::size_t const j2e = std::min(n2, j20 + tb);
				for(std::size_t k1 = k10; k1 != k1e; ++k1) {
					std::size_t    idx = (k1 * j20) % n_;  // k1*j2 mod n, updated incrementally
					T const* const zr  = z + (k1 * n2);
					T* const       tr  = tile.data() + ((k1 - k10) * tb);
					for(std::size_t j2 = j20; j2 != j2e; ++j2) {
						tr[j2 - j20] = fft_mul_dir<Backward>(tw_[idx], zr[j2]);  // twiddle first, matching fft_mul(TW, T) convention
						idx += k1;
						if(idx >= n_) {
							idx -= n_;
						}
					}
				}
				for(std::size_t j2 = j20; j2 != j2e; ++j2) {
					T* const       yr = yt + (j2 * n1) + k10;
					T const* const tc = tile.data() + (j2 - j20);
					for(std::size_t k1 = 0; k1 != k1e - k10; ++k1) {
						yr[k1] = tc[k1 * tb];
					}
				}
			}
		}

		if(uout != nullptr && e2.can_fuse()) {  // final stage writes user memory directly
			e2.template run_fused_impl_<true, Backward>(e2.buf_ptr(arena), n1, uout, n1, n1, arena);
			return uout;
		}
		T const* const res = e2.run(n1, Backward, arena);  // row FFTs, batched over k1
		if(uout != nullptr) {
			std::copy(res, res + (n2 * n1), uout);
			return uout;
		}
		return res;
	}

	// Bluestein (Phase B, NOTES §10.5): the outer `Backward` does NOT change
	// the convolution sub-transform's own directions -- the mechanism is
	// fixed (canonical forward conv, then inverse conv), regardless of the
	// outer transform's direction. Only the chirp/postc conjugation and the
	// kernel-spectrum table selection depend on `Backward`. `conv` is a
	// single neutral sub-engine (collapsed fwd/bwd pair, see init_bluestein_)
	// run twice, forward then backward: `z` (the second run's default input
	// region, `conv.buf_ptr(arena)`) can therefore be the exact SAME memory
	// as `yf` (the first run's result) when the stage count is even -- safe
	// because the pointwise-product write `z[i] = f(yf[i])` only ever reads
	// and writes the SAME index, never a different one, so full aliasing
	// between `z` and `yf` is not a hazard (verified under ASan with both
	// odd- and even-stage-count conv_n_ values).
	template<bool Backward, class T>
	auto run_bluestein_(std::size_t m, T const* in, T* arena, T* out = nullptr) const -> T const* {
		auto const& conv = sub_[0];

		T* const y = conv.buf_ptr(arena);  // chirp-premultiplied input, zero-padded to conv_n_
		for(std::size_t k = 0; k != n_; ++k) {
			TW const c = chirp_[k];
			std::transform(in + (k * m), in + ((k + 1) * m), y + (k * m), [c](T const& v) { return fft_mul_dir<Backward>(c, v); });
		}
		std::fill(y + (n_ * m), y + (conv_n_ * m), T{});

		T const* const yf = conv.run(m, false, arena);  // canonical forward conv, always

		T* const        z   = conv.buf_ptr(arena);  // pointwise product with the precomputed kernel spectrum (may alias yf -- see comment above)
		TW const* const kft = Backward ? kernel_ft_bwd_.data() : kernel_ft_.data();
		for(std::size_t q = 0; q != conv_n_; ++q) {
			TW const kq = kft[q];
			// plain mul: table already carries the right direction. z may fully
			// alias yf here (see comment above); std::transform explicitly
			// permits the output range to equal the input range for a unary op.
			std::transform(yf + (q * m), yf + ((q + 1) * m), z + (q * m), [kq](T const& v) { return fft_mul(kq, v); });
		}

		T const* const zc = conv.run(m, true, arena);  // inverse conv, always

		T* const res = (out != nullptr) ? out : buf_ptr(arena);
		for(std::size_t k = 0; k != n_; ++k) {  // chirp-postmultiply (normalization fused into postc_)
			TW const pc = postc_[k];
			std::transform(zc + (k * m), zc + ((k + 1) * m), res + (k * m), [pc](T const& v) { return fft_mul_dir<Backward>(pc, v); });
		}
		return res;
	}
};

// --- N-D orchestration ------------------------------------------------------

// Transform one (possibly strided) 1-D fiber through the engine. `T` (the
// array's element type, deduced from `arena`) is independent of `TW` (the
// engine's own, fixed twiddle-table type).
template<class View1D, class T, class TW>
void fft_exec_fiber(View1D&& fib, fft_engine<TW> const& eng, bool backward, T* arena) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	if constexpr(std::is_pointer_v<std::decay_t<decltype(fib.base())>>) {
		if(fib.stride() == 1) {  // contiguous fiber: no gather, and the final pass writes back directly
			eng.run_contig_inplace(fib.base(), arena, backward);
			return;
		}
	}
	T* const b = eng.buf_ptr(arena);
	std::copy(fib.begin(), fib.end(), b);  // gather strided fiber
	T const* const res = eng.run(1, backward, arena);
	std::copy(res, res + eng.n_, fib.begin());  // scatter result back
}

// Transform every row-fiber of a rank-2 slab [batch][n] in vector batches:
// tiles of up to eng.mb_ fibers are gathered interleaved (batch index
// contiguous) and pushed through the batched stage kernels together.
template<class View2D, class T, class TW>
void fft_exec_slab(View2D&& slab, fft_engine<TW> const& eng, bool backward, T* arena) {  // NOLINT(cppcoreguidelines-missing-std-forward,readability-function-cognitive-complexity)
	using std::get;
	auto const yy = static_cast<std::size_t>(slab.size());
	auto const nn = eng.n_;
	if(yy == 0 || nn == 0) {
		return;
	}

	std::size_t const mb = std::min<std::size_t>(std::max<std::size_t>(eng.mb_, 1), yy);

	// Contiguous fibers transform faster one at a time straight from user
	// memory (no transpose gather) than through batched tiles.
	if constexpr(std::is_pointer_v<std::decay_t<decltype(slab.base())>>) {
		if(get<1>(slab.strides()) == 1) {
			auto const ylim = static_cast<std::ptrdiff_t>(yy);
			for(std::ptrdiff_t y = 0; y != ylim; ++y) {
				fft_exec_fiber(slab[y], eng, backward, arena);
			}
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
					fft_exec_fiber(slab[static_cast<std::ptrdiff_t>(y0)], eng, backward, arena);
					continue;
				}
				T* const tile0 = std::addressof(slab[static_cast<std::ptrdiff_t>(y0)][0]);
				eng.run_fused(tile0, sf, tile0, sf, mt, backward, arena);
			}
			return;
		}
	}
	auto const abs_       = [](auto s) { return s < 0 ? -s : s; };
	// Pick the gather loop order from the layout: move along whichever axis is
	// closer in memory in the inner copy loop.
	bool const fiber_near = abs_(get<1>(slab.strides())) <= abs_(get<0>(slab.strides()));

	auto&& cols = slab.rotated();  // cols[k][y] == slab[y][k]

	for(std::size_t y0 = 0; y0 < yy; y0 += mb) {
		std::size_t const mt = std::min(mb, yy - y0);
		if(mt == 1) {
			fft_exec_fiber(slab[static_cast<std::ptrdiff_t>(y0)], eng, backward, arena);
			continue;
		}
		T* const bp = eng.buf_ptr(arena);
		if(fiber_near) {  // fibers contiguous-ish: blocked-transpose gather, reads stream along k
			constexpr std::size_t kb = 64;
			for(std::size_t k0 = 0; k0 < nn; k0 += kb) {
				std::size_t const ke = std::min(nn, k0 + kb);
				for(std::size_t j = 0; j != mt; ++j) {
					auto it = slab[static_cast<std::ptrdiff_t>(y0 + j)].begin();
					for(std::size_t k = k0; k != ke; ++k) {
						bp[(k * mt) + j] = it[static_cast<std::ptrdiff_t>(k)];
					}
				}
			}
		} else {  // batch axis contiguous-ish: both reads and writes stream along j
			for(std::size_t k = 0; k != nn; ++k) {
				auto     it  = cols[static_cast<std::ptrdiff_t>(k)].begin() + static_cast<std::ptrdiff_t>(y0);
				T* const row = bp + (k * mt);
				std::copy_n(it, mt, row);
			}
		}

		T const* const res = eng.run(mt, backward, arena);

		if(fiber_near) {
			constexpr std::size_t kb = 64;
			for(std::size_t k0 = 0; k0 < nn; k0 += kb) {
				std::size_t const ke = std::min(nn, k0 + kb);
				for(std::size_t j = 0; j != mt; ++j) {
					auto it = slab[static_cast<std::ptrdiff_t>(y0 + j)].begin();
					for(std::size_t k = k0; k != ke; ++k) {
						it[static_cast<std::ptrdiff_t>(k)] = res[(k * mt) + j];
					}
				}
			}
		} else {
			for(std::size_t k = 0; k != nn; ++k) {
				auto           it  = cols[static_cast<std::ptrdiff_t>(k)].begin() + static_cast<std::ptrdiff_t>(y0);
				T const* const row = res + (k * mt);
				std::copy_n(row, mt, it);
			}
		}
	}
}

template<class Strides, std::size_t... Is>
auto fft_min_abs_mid_stride(Strides const& strs, std::index_sequence<Is...> /*unused*/) -> std::ptrdiff_t {
	using std::get;
	std::ptrdiff_t ret  = std::numeric_limits<std::ptrdiff_t>::max();
	auto const     acc_ = [&ret](std::ptrdiff_t s) {
        s   = (s < 0) ? -s : s;
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
// `last_backward`/`prev_backward` are independent (the last two axes may
// have different directions, e.g. `{forward, backward}` on a square shape --
// exactly the case that shares one engine across both axes in Phase B).
template<class ViewND, class T, class TW>
void fft_apply_last_pair(ViewND&& view, fft_engine<TW> const& last_eng, bool last_backward, fft_engine<TW> const& prev_eng, bool prev_backward, T* arena) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	constexpr auto rank = std::decay_t<ViewND>::dimensionality;
	if constexpr(rank == 2) {
		fft_apply_last(view, last_eng, last_backward, arena);            // fibers along axis 1
		fft_apply_last(view.rotated(), prev_eng, prev_backward, arena);  // fibers along axis 0, slab still hot
	} else {
		for(auto&& sub : view) {
			fft_apply_last_pair(sub, last_eng, last_backward, prev_eng, prev_backward, arena);
		}
	}
}

// Transform every fiber along the *last* axis of `view` through the engine.
// The rank-descent drops leading axes one at a time but keeps the leading axis
// of smallest stride alive (via transposed(), which swaps the first two axes),
// so that at rank 2 the batch axis is the one closest in memory.
template<class ViewND, class T, class TW>
void fft_apply_last(ViewND&& view, fft_engine<TW> const& eng, bool backward, T* arena) {  // NOLINT(cppcoreguidelines-missing-std-forward)
	constexpr auto rank = std::decay_t<ViewND>::dimensionality;
	if constexpr(rank == 1) {
		fft_exec_fiber(view, eng, backward, arena);
	} else if constexpr(rank == 2) {
		fft_exec_slab(view, eng, backward, arena);
	} else {
		using std::get;
		auto const strs = view.strides();
		auto const s0   = static_cast<std::ptrdiff_t>(get<0>(strs));
		auto const s0a  = (s0 < 0) ? -s0 : s0;
		if(s0a <= fft_min_abs_mid_stride(strs, std::make_index_sequence<static_cast<std::size_t>(rank) - 2>{})) {
			for(auto&& sub : view.transposed()) {
				fft_apply_last(sub, eng, backward, arena);
			}
		} else {
			for(auto&& sub : view) {
				fft_apply_last(sub, eng, backward, arena);
			}
		}
	}
}

// A Multi cursor (`.home()`) is base + strides with no extents. These helpers
// recover its rank and rebuild a full strided view once the extents (which the
// plan owns) are supplied -- the "extents (plan) + cursor (target)" split.
template<class Cursor>
using fft_cursor_strides_t = std::decay_t<decltype(std::declval<Cursor>().strides())>;

// A cursor exposes base()/strides() but, unlike an array/subarray, no sizes().
template<class C, class = void> struct fft_is_cursor_like : std::false_type {};
template<class C>
struct fft_is_cursor_like<C, std::void_t<decltype(std::declval<C>().base()), decltype(std::declval<C>().strides()), std::enable_if_t<!fft_is_multi_like<C>::value>>> : std::true_type {};

template<class Cursor>
inline constexpr std::ptrdiff_t fft_cursor_rank = static_cast<std::ptrdiff_t>(std::tuple_size_v<fft_cursor_strides_t<Cursor>>);

// Build layout_t<D> from extents and strides (all offsets 0, as for a home
// cursor): each level carries nelems = size*stride, which is the invariant
// layout_t uses to recover size() = nelems/stride and extent() bounds.
template<std::ptrdiff_t D>
auto fft_layout_from(std::array<std::size_t, static_cast<std::size_t>(D)> const& ext, std::array<std::ptrdiff_t, static_cast<std::size_t>(D)> const& str) -> multi::layout_t<D> {
	if constexpr(D == 0) {
		return multi::layout_t<0>{};
	} else {
		std::array<std::size_t, static_cast<std::size_t>(D) - 1>    sub_ext{};
		std::array<std::ptrdiff_t, static_cast<std::size_t>(D) - 1> sub_str{};
		std::copy(ext.begin() + 1, ext.end(), sub_ext.begin());
		std::copy(str.begin() + 1, str.end(), sub_str.begin());
		return multi::layout_t<D>{
			fft_layout_from<D - 1>(sub_ext, sub_str),
			str[0], 0,
			static_cast<std::ptrdiff_t>(ext[0]) * str[0]
		};  // NOLINT(cppcoreguidelines-pro-bounds-constant-array-index) [0] is valid for D>=1
	}
}

template<class Cursor, std::size_t... Is>
auto fft_strides_array(Cursor const& cur, std::index_sequence<Is...> /*unused*/)
	-> std::array<std::ptrdiff_t, sizeof...(Is)> {
	return {{static_cast<std::ptrdiff_t>(cur.template stride<static_cast<multi::dimensionality_type>(Is)>())...}};
}

// (cursor, extents) -> strided subarray sharing the cursor's memory.
template<class T, std::ptrdiff_t D, class Cursor>
auto fft_view_from_cursor(Cursor const& cur, std::array<std::size_t, static_cast<std::size_t>(D)> const& ext) {
	auto const str = fft_strides_array(cur, std::make_index_sequence<static_cast<std::size_t>(D)>{});
	using ptr_type = typename Cursor::element_ptr;
	return multi::subarray<T, D, ptr_type>{fft_layout_from<D>(ext, str), cur.base()};
}

// Raw, execute()-time-local scratch: allocates (never value-initializes) T
// storage for the plan's whole scratch arena. Every element is always fully
// written (by a gather step or a stage kernel) before it is ever read, so
// zero-initializing it first -- what a plain `std::vector<T>` would do -- is
// pure waste on every single `execute()` call.
//
// When `fft_skip_element_init<T>` holds (either the language already makes
// default-construction free, or the type is opted in through Multi's own
// `force_element_trivial_default_construction` customization point -- which
// std::complex<float/double> are), the constructor deliberately does NOT
// run `uninitialized_default_construct_n` either: std::complex is trivially
// copyable but NOT trivially default-constructible (its default constructor
// zero-initializes through defaulted arguments), so default-constructing the
// arena compiles to a full memset of the whole allocation on every call --
// verified in generated code -- silently reintroducing the very zero-fill
// this class exists to avoid (an earlier version of this file did exactly
// that, with a comment claiming it was free). Skipping construction for
// storage that is only ever stored-to-then-read is the same idiom
// multi::array itself uses (array.hpp); types without the opt-in still get
// proper lifetime starts.
// `Allocator` is the §10.4(c) GPU seam: any caller-supplied allocator
// satisfying the standard Allocator concept (allocate(n)/deallocate(p,n))
// for `T` directly can be threaded through here -- a fixed single-slot
// arena (allocate() always returns the same buffer, deallocate() a no-op)
// is the correct minimal fit for this class's own access pattern (exactly
// one allocate() followed by exactly one matching deallocate() per
// execute() call, never interleaved with any other request through the
// same allocator); a device allocator (Thrust/CUDA) is the same shape
// later. Deliberately does NOT rebind via allocator_traits: this class
// only ever allocates `T`, never some other node type, so requiring
// `Allocator` to be a rebindable class template (as e.g. std::vector's
// internal machinery does) would needlessly exclude simple, monomorphic,
// already-T-typed allocators. `Allocator::value_type` must already be `T`.
// Assumes allocate() returns a plain `T*` (true for std::allocator and
// std::pmr::polymorphic_allocator) -- fancy-pointer support is a separate,
// not-yet-needed extension.
template<class T, class Allocator = std::allocator<T>>
class fft_scratch_arena {
	static_assert(std::is_same_v<typename Allocator::value_type, T>, "Allocator::value_type must be T; fft_scratch_arena does not rebind");
	using alloc_traits = std::allocator_traits<Allocator>;

	Allocator   alloc_;
	T*          p_;
	std::size_t n_;

 public:
	// A plan needing zero scratch (e.g. a trivial n < 2 size) is legitimate;
	// skip the allocator call entirely rather than ask for a 0-element
	// allocation, which some standard libraries' allocate() flags as
	// suspicious under a stricter warning set than this file's own build
	// (e.g. GCC's -Walloc-zero, not part of this header's own tested flags
	// but part of this project's actual CI).
	explicit fft_scratch_arena(std::size_t n, Allocator const& alloc = Allocator{})
	: alloc_(alloc), p_(n == 0 ? nullptr : alloc_traits::allocate(alloc_, n)), n_(n) {
		if constexpr(!fft_skip_element_init<T>) {  // see class comment: for opted-in types (std::complex) this would memset the arena
			if(n_ != 0) {
				std::uninitialized_default_construct_n(p_, n_);
			}
		}
	}
	fft_scratch_arena(fft_scratch_arena const&)                    = delete;
	auto operator=(fft_scratch_arena const&) -> fft_scratch_arena& = delete;
	fft_scratch_arena(fft_scratch_arena&&)                         = delete;
	auto operator=(fft_scratch_arena&&) -> fft_scratch_arena&      = delete;
	~fft_scratch_arena() {
		if(n_ != 0) {
			if constexpr(!fft_skip_element_init<T>) {  // matches the constructor: only destroy what was constructed
				std::destroy_n(p_, n_);
			}
			alloc_traits::deallocate(alloc_, p_, n_);
		}
	}
	auto data() const -> T* { return p_; }
};

}  // end namespace detail

// Reusable multidimensional FFT plan: precomputes twiddle tables, stage
// factorizations, DFT matrices and scratch buffers for a given shape and
// direction, and applies them to any array/subarray of that shape (any
// strided layout) with `plan.execute(A)`, repeatedly, without
// re-allocation.
template<std::ptrdiff_t D, class TW = std::complex<double>>
class fft_plan {
	static_assert(D >= 1, "fft_plan requires at least one dimension");

	static constexpr std::size_t no_engine_ = static_cast<std::size_t>(-1);  // which_[a] sentinel for a `none` axis: never dereferenced

	// `engines_` container type: a plain `std::array` -- NOTES §10.1 item 9
	// (D-bounded, no heap allocation for this list). Unlike `fft_engine::
	// sub_` (Bluestein/six-step/large-prime sub-engines, which is data-
	// dependent on one length's factorization and NOT bounded by `D`; that
	// one stays a `std::vector`), the number of distinct TOP-LEVEL engines
	// is bounded by `D`. `fft_engine` has a default constructor (== the
	// existing, already-handled n<2 trivial state, not a new one -- see its
	// definition), so `std::array<fft_engine<TW>, D>` default-constructs
	// directly: slots `[0, distinct_count_)` get overwritten by assignment
	// as distinct lengths are discovered in the constructor below; the tail
	// `[distinct_count_, D)` stays at its cheap default (no heap tables).
	// `note_reach_`/`assign_offsets_`/`engine_count()` are all bounded to
	// `distinct_count_` explicitly, so a `none` axis (or a shared length)
	// still costs exactly nothing extra in scratch (§10.1 decision 3).
	using engines_container_ = std::array<detail::fft_engine<TW>, static_cast<std::size_t>(D)>;

	std::array<std::size_t, static_cast<std::size_t>(D)>    sizes_{};
	std::array<fft_direction, static_cast<std::size_t>(D)>  dirs_{};    // per-axis pass schedule (fft.NOTES.md §10)
	engines_container_                                      engines_{};  // one per distinct length (direction-neutral, Phase B), padded to exactly D; see engines_container_'s comment
	std::array<std::size_t, static_cast<std::size_t>(D)>    which_{};  // axis -> index into engines_ (always < distinct_count_), or no_engine_ for a `none` axis
	std::size_t                                             distinct_count_   = 0;  // live prefix length of engines_ (see engines_container_'s comment)
	std::size_t                                             scratch_elements_ = 0;  // total arena size for execute()

	// Engine serving axis `A` (compile-time axis index, resolved at plan
	// build). Caller must first confirm axis `A` is not `none` (dirs_[A] !=
	// fft_direction::none) -- which_[A] is a sentinel otherwise, never a
	// valid engines_ index; the assert catches any future call site that
	// forgets (the §10.5 guard).
	template<std::ptrdiff_t A>
	auto engine_() const -> detail::fft_engine<TW> const& {
		static_assert(A >= 0 && A < D, "axis out of range");
		assert(which_[static_cast<std::size_t>(A)] != no_engine_ && "engine_<A>() called for a `none` axis");
		return engines_[which_[static_cast<std::size_t>(A)]];
	}

	// Uniform recursive axis walk, made possible by per-axis directions:
	// every axis is one "transform the last axis of the current view if
	// active, then rotate and recurse" step, so `none`-skipping, the D == 1
	// case, and every partially-degraded combination fall out of ONE code
	// path instead of bespoke branches (an earlier three-way degraded-pair
	// branch hand-picked rotations per case and got one wrong -- see git
	// history; here the view is correctly positioned by construction).
	//
	// `view` is `arr` rotated K times (rotated() sends axis 0 to the back),
	// so its last axis is original axis K-1 -- or D-1 for K == 0. Walking
	// K = Start .. Stop therefore visits original axes in the order
	// D-1, 0, 1, ..., D-2 (order is free: 1-D passes along different axes
	// commute). Each axis is a distinct instantiation, bound to its engine
	// at compile time; rotated() preserves rank and type, so there is a
	// single View type per plan. `Stop` lets apply_ end the walk early at
	// axis D-3 when axes D-1/D-2 were already handled by the fused pair
	// pass. `T` (the array's element type) is deduced from `arena`,
	// independent of `TW`.
	template<std::ptrdiff_t K, std::ptrdiff_t Stop, class View, class T>
	void transform_axes_(View&& view, T* arena) const {  // NOLINT(cppcoreguidelines-missing-std-forward)
		constexpr std::ptrdiff_t axis = (K == 0) ? D - 1 : K - 1;
		if(dirs_[static_cast<std::size_t>(axis)] != fft_direction::none) {
			detail::fft_apply_last(view, engine_<axis>(), dirs_[static_cast<std::size_t>(axis)] == fft_direction::backward, arena);
		}
		if constexpr(K < Stop) {
			transform_axes_<K + 1, Stop>(view.rotated(), arena);
		}
	}

	template<class Extents, std::size_t... Is>
	static auto to_sizes_(Extents const& ext, std::index_sequence<Is...> /*unused*/) -> std::array<std::size_t, static_cast<std::size_t>(D)> {
		using std::get;
		return {{detail::fft_extent_size(get<Is>(ext))...}};
	}

	static auto broadcast_dirs_(int sign) -> std::array<fft_direction, static_cast<std::size_t>(D)> {
		std::array<fft_direction, static_cast<std::size_t>(D)> dirs{};
		dirs.fill(static_cast<fft_direction>(sign));
		return dirs;
	}

 public:
	// Per-axis direction constructor (fft.NOTES.md §10): `dirs[a] ==
	// fft_direction::none` leaves axis `a` completely untouched -- no engine
	// is built for it (exact scratch sizing; a `none` axis on a large/prime
	// length costs nothing), and it is never visited by apply_(). Engines
	// are still sign-aware in this phase (Phase A -- see fft.NOTES.md §10.3):
	// reuse is keyed on `(length, direction)`, so two same-length axes with
	// *different* directions get two engines; direction-neutral engines
	// (one engine shared regardless of direction) are Phase B.
	template<class Extents>
	explicit fft_plan(Extents const& extents, std::array<fft_direction, static_cast<std::size_t>(D)> const& dirs)
	: sizes_{to_sizes_(extents, std::make_index_sequence<static_cast<std::size_t>(D)>{})}, dirs_{dirs} {
		// `engines_` default-constructs (see engines_container_'s comment);
		// this loop overwrites its live prefix by assignment as distinct
		// lengths are discovered -- the tail past distinct_count_ stays at
		// its cheap default. Reuse is keyed on length ALONE (Phase B, NOTES
		// §10.5): e.g. a square {forward, backward} plan shares ONE engine
		// where Phase A built two.
		auto const rank = static_cast<std::size_t>(D);
		for(std::size_t a = 0; a != rank; ++a) {
			if(dirs_[a] == fft_direction::none) {
				which_.at(a) = no_engine_;
				continue;
			}
			auto const len = sizes_.at(a);
			auto const it  = std::find_if(engines_.begin(), engines_.begin() + static_cast<std::ptrdiff_t>(distinct_count_), [len](auto const& e) { return e.n_ == len; });
			if(it == engines_.begin() + static_cast<std::ptrdiff_t>(distinct_count_)) {
				engines_.at(distinct_count_) = detail::fft_engine<TW>{len};
				which_.at(a)                 = distinct_count_;
				++distinct_count_;
			} else {
				which_.at(a) = static_cast<std::size_t>(it - engines_.begin());
			}
		}

		// Compute every REAL engine's (and its whole sub_ tree's) peak scratch
		// requirement, then lay out one flat arena with disjoint, immutable
		// offsets -- see fft.NOTES.md §9.2/§10.4(a). For D >= 2, each
		// top-level engine is reachable from two entry scenarios (batched via
		// fft_exec_slab at its own mb_, and single-fiber via fft_exec_fiber at
		// m=1, which may itself divert to six-step); note_reach_ is
		// monotonic-max, so visiting both is safe and covers every real call
		// site. For D == 1, apply_() never batches (fft_apply_last's rank==1
		// case always goes through fft_exec_fiber at m=1) -- note_reach_(mb_)
		// would only reserve scratch for a path that never runs, up to mb_
		// (<=64) times larger than needed. Bounded to `distinct_count_`
		// (NOT engines_'s full physical D slots): the padding placeholder
		// engines past distinct_count_ must never see note_reach_/
		// assign_offsets_, or a `none`/shared axis would inflate scratch --
		// see engines_container_'s comment.
		for(std::size_t i = 0; i != distinct_count_; ++i) {
			auto& e = engines_[i];
			if constexpr(D >= 2) {
				e.note_reach_(e.mb_);
			}
			e.note_reach_(1);
		}
		std::size_t cursor = 0;
		for(std::size_t i = 0; i != distinct_count_; ++i) {
			engines_[i].assign_offsets_(cursor);
		}
		scratch_elements_ = cursor;
	}

	// Broadcast convenience: applies `sign` to every axis (the pre-§10 API).
	template<class Extents>
	explicit fft_plan(Extents const& extents, int sign = fft_forward)
	: fft_plan(extents, broadcast_dirs_(sign)) {}

	// Element count `execute()` will request from its allocator, for a
	// caller who wants to size their own scratch (e.g. a
	// std::pmr::monotonic_buffer_resource's byte size is this times
	// sizeof(T), plus alignment slack) rather than accept a fresh
	// std::allocator<T> per call. T-agnostic: the count is the same
	// regardless of which array element type a given execute() call uses.
	auto scratch_elements() const -> std::size_t { return scratch_elements_; }

	// Number of distinct top-level engines the plan built (harmless, genuinely
	// useful to callers). Direction-neutral engines (Phase B, NOTES §10.5)
	// share one engine per distinct AXIS LENGTH regardless of direction, so
	// e.g. a square {forward, backward} plan reports 1 here (2 in Phase A).
	auto engine_count() const -> std::size_t { return distinct_count_; }

 private:
	// The axis walk, shared by the cursor and array entry points. `view` is a
	// strided view of the planned shape; transformed in place. `T` (the
	// array's element type) is deduced from `arena`, independent of `TW`.
	//
	// One fast-path check, then the uniform recursion does everything else:
	// when BOTH of the last two axes are active, they are fused into a
	// single slab-by-slab pass (fft_apply_last_pair: both transforms run
	// while the rank-2 slab is cache-resident -- a measured win, see NOTES
	// §2.5) and the walk covers only the remaining axes 0 .. D-3. In every
	// other combination -- D == 1, any axis `none`, all axes `none` -- the
	// walk alone handles all D axes, skipping inactive ones; no per-case
	// rotation choices left to get wrong.
	template<class View, class T>
	void apply_(View&& view, T* arena) const {  // NOLINT(cppcoreguidelines-missing-std-forward)
		if constexpr(D >= 2) {
			if(dirs_[static_cast<std::size_t>(D - 1)] != fft_direction::none && dirs_[static_cast<std::size_t>(D - 2)] != fft_direction::none) {
				detail::fft_apply_last_pair(
				    view,
				    engine_<D - 1>(), dirs_[static_cast<std::size_t>(D - 1)] == fft_direction::backward,
				    engine_<D - 2>(), dirs_[static_cast<std::size_t>(D - 2)] == fft_direction::backward,
				    arena
				);
				if constexpr(D >= 3) {
					transform_axes_<1, D - 2>(view.rotated(), arena);  // axes 0 .. D-3
				}
				return;
			}
		}
		transform_axes_<0, D - 1>(view, arena);  // all axes: D-1, then 0 .. D-2
	}

 public:
	// `T`, the array's element type, is deduced fresh per call from
	// `Cursor::element` -- independent of `TW`, the plan's own (fixed,
	// construction-time) twiddle-table type (see fft.NOTES.md §9.2): one
	// plan, built once for a shape, can execute a complex<float> array today
	// and a complex<double> array tomorrow without rebuilding any tables.
	//
	// `Allocator` defaults to a fresh, stateless `std::allocator<T>` per call
	// (never plan-owned state -- that would reintroduce the concurrent-
	// execute() hazard §9.2 removed). A caller who wants to avoid paying for
	// allocation on every call -- e.g. the same pattern as this benchmark's
	// own repeated-execute() loop -- can pass any allocator satisfying the
	// standard Allocator concept instead, such as an arena/monotonic one
	// (`std::pmr::polymorphic_allocator<T>` backed by a
	// `std::pmr::monotonic_buffer_resource` the caller owns and reuses
	// across calls); this class does not implement or ship one itself.
	template<class Cursor, class Allocator = std::allocator<typename std::decay_t<Cursor>::element>, std::enable_if_t<detail::fft_is_cursor_like<std::decay_t<Cursor>>::value, int> = 0>  // NOLINT(modernize-use-constraints) C++17
	auto execute(Cursor const& home, Allocator alloc = Allocator{}) const -> Cursor {
		static_assert(detail::fft_cursor_rank<std::decay_t<Cursor>> == D, "cursor rank must match the plan");
		using T = typename std::decay_t<Cursor>::element;
		auto                                    view = detail::fft_view_from_cursor<T, D>(home, sizes_);
		detail::fft_scratch_arena<T, Allocator> arena(scratch_elements_, alloc);  // execute-time-local scratch; the plan itself owns none
		apply_(view, arena.data());
		return home;  // cursors are value types (base + strides); returned by value
	}

	// Checked convenience overload: the documented `plan.execute(A)` form,
	// taking the array/subarray itself instead of its bare cursor. Rank is
	// checked at compile time; the shape is validated against the planned
	// sizes with an assert (debug builds -- a cursor carries no sizes, so the
	// cursor overload above cannot check anything; this one can, and should).
	// SFINAE note: constrained on `fft_real<element>` being well-formed
	// rather than on a "looks like an array" trait -- shape/extents objects
	// (whose `element` is an index tuple with no `value_type`) drop out here
	// structurally, per the `fft_is_multi_like` footgun in fft.NOTES.md §10.5.
	template<
	    class MultiSubArray, class Allocator = std::allocator<typename std::decay_t<MultiSubArray>::element>,
	    class = detail::fft_real_t<typename std::decay_t<MultiSubArray>::element>,
	    std::enable_if_t<!detail::fft_is_cursor_like<std::decay_t<MultiSubArray>>::value, int> = 0>  // NOLINT(modernize-use-constraints) C++17
	auto execute(MultiSubArray&& arr, Allocator alloc = Allocator{}) const -> MultiSubArray&& {
		static_assert(static_cast<std::ptrdiff_t>(std::decay_t<MultiSubArray>::dimensionality) == D, "array rank must match the plan");
		assert(to_sizes_(arr.sizes(), std::make_index_sequence<static_cast<std::size_t>(D)>{}) == sizes_ && "array shape must match the planned sizes");
		execute(arr.home(), alloc);
		return std::forward<MultiSubArray>(arr);
	}
};

template<class MultiSubArray>
auto fft_inplace(MultiSubArray&& arr, int sign) -> MultiSubArray&& {
	using array_type = std::decay_t<MultiSubArray>;
	fft_plan<array_type::dimensionality, typename array_type::element> const plan{arr.sizes(), sign};
	return plan.execute(std::forward<MultiSubArray>(arr));
}

// Per-axis direction overload (fft.NOTES.md §10), e.g.
//   multi::fft_inplace({{forward, none, backward, forward}}, arr);
// `dirs` first, matching NOTES §10.5's deduction trick: the array parameter
// deduces MultiSubArray; the dirs parameter's type is then a *non-deduced*
// std::array<fft_direction, dimensionality-of-MultiSubArray>, so the braced
// list just initializes an already-known-size array (std::initializer_list
// would compile for any size, deliberately not used -- NOTES §10.1
// decision 2). This catches TOO MANY directions as a hard compile error
// (verified: "no matching function", not a SFINAE near-miss). It does NOT
// catch too FEW: std::array's own aggregate-init rules zero-pad missing
// trailing elements, and zero is fft_direction::none by construction, so
// e.g. `fft_inplace({{forward, backward}}, arr4d)` on a rank-4 array
// silently means `{forward, backward, none, none}`, not a compile error.
// Verified this can't be closed within std::array (N is a non-deduced
// context on both the dirs-argument and the array-argument side; tried and
// confirmed empirically). Accepted, documented gap (maintainer decision):
// closing it fully would need a custom fixed-arity wrapper type in place
// of std::array here, which wasn't judged worth the complexity -- get the
// direction count right.
template<class MultiSubArray>
auto fft_inplace(std::array<fft_direction, static_cast<std::size_t>(std::decay_t<MultiSubArray>::dimensionality)> const& dirs, MultiSubArray&& arr) -> MultiSubArray&& {
	using array_type = std::decay_t<MultiSubArray>;
	fft_plan<array_type::dimensionality, typename array_type::element> const plan{arr.sizes(), dirs};
	return plan.execute(std::forward<MultiSubArray>(arr));
}

template<class MultiSubArray>
auto fft_inplace_forward(MultiSubArray&& arr) -> MultiSubArray&& {
	return fft_inplace(std::forward<MultiSubArray>(arr), fft_forward);
}

template<class MultiSubArray>
auto fft_inplace_backward(MultiSubArray&& arr) -> MultiSubArray&& {
	return fft_inplace(std::forward<MultiSubArray>(arr), fft_backward);
}

}  // end namespace boost::multi

// NOLINTEND(altera-id-dependent-backward-branch,altera-unroll-loops,bugprone-easily-swappable-parameters,cppcoreguidelines-pro-bounds-pointer-arithmetic,misc-no-recursion,readability-function-cognitive-complexity,readability-identifier-length)

#endif  // BOOST_MULTI_ALGORITHMS_FFT_HPP
