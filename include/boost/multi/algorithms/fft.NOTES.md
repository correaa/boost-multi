# Generic in-place FFT for Multi — implementation notes

Audience: human developers and AI models working on `boost/multi/algorithms/fft.hpp`
or on the Multi core. This documents *what* was built, *how* it works, the
non-obvious **library idioms** discovered while writing it, and a concrete
**future-work** list.

- Implementation: [`fft.hpp`](./fft.hpp)
- Tests: [`../../../../test/algorithms_fft.cpp`](../../../../test/algorithms_fft.cpp) (auto-registered by the `test/*.cpp` glob)

---

## 1. What was built

A self-contained, header-only, **in-place multidimensional FFT** with a
**reusable plan** (in the FFTW sense: auxiliary tables and scratch allocations
are computed once and reused across repeated transforms):

```cpp
// reusable plan: tables + scratch built once, executed many times
multi::fft_plan<complex, 2> plan{multi::extents_t<2>{1024, 1024}, multi::fft_forward};
multi::fft_plan             plan2{A, multi::fft_forward};   // CTAD from a prototype array

// the plan holds the *extents*; it is applied on a *cursor* -- base + strides,
// the complement of extents -- obtained from any array/subarray with `.home()`:
plan(A.home());     // primitive form: cursor supplies base+strides, plan supplies shape
plan(A);            // sugar: checks A.extensions() == plan shape, then applies on A.home()
plan.execute(B);    // same, in place, any strided layout, repeatedly, no re-allocation

// one-shot convenience (builds a throw-away plan)
multi::fft_inplace(A, sign);   // also fft_inplace_forward / fft_inplace_backward
```

The plan/cursor split is deliberate: a plan is *shape + precomputed tables*, and
a cursor (`.home()`) is *where the data lives* (base pointer + strides), with no
shape of its own. `plan(cursor)` trusts the cursor to have the planned shape (a
cursor carries no sizes to check); `plan(array)` is the checked convenience that
extracts `array.home()`. This is also the GPU-facing shape: a cursor is exactly
what a device kernel receives (base + strides), so the same execution primitive
carries over (see section 8). Internally `plan(cursor)` rebuilds a strided view
from the plan's extents + the cursor via `layout_t`'s size = nelems/stride
invariant; that view then drives the unchanged orchestration.

Properties:

- **No external dependency** (no FFTW/cuFFT). Pure C++17 + standard library.
- **Any element type `T` obeying complex algebra** — closed under `+`, `-`, `*`,
  constructible as `T{re, im}`, value-initialized to zero (e.g. `std::complex<Real>`).
  Two customization points: `multi::fft_real<T>` (underlying real type, defaults
  to `T::value_type`) and `multi::fft_ops<T>` (kernel product, defaults to
  `operator*` — see §2.4 for why this exists).
- **Any rank D**; the N-D orchestration is rank-generic. A plan is constructible
  from any tuple-like extents (`extents_t<D>`, `A.sizes()`, `std::array<int,D>`).
- **Any size N in O(N log N)** — mixed-radix Stockham with radix-4/8/2/3/5
  kernels, table-driven direct kernels for primes ≤ 64, and **Bluestein's
  chirp-z algorithm** for larger prime factors (no size degenerates to O(N²)).
- **Any strided layout** Multi supports (rotated views, strided columns,
  sub-blocks); the same plan serves different layouts of the same shape.
- **FFTW sign/normalization convention**: `fft_forward == -1`, `fft_backward == +1`,
  transform is **unnormalized** (forward∘backward = N·identity).
- **Thread-safety**: a plan owns `mutable` scratch — concurrent `execute` on the
  *same* plan needs external synchronization (one plan copy per thread is fine).

Architecture: `fft_plan<T, D>` holds one `detail::fft_engine<T>` per **distinct
axis length** (a cube shares one engine across all three axes). An engine owns
the twiddle table, the stage factorization, the direct-prime DFT matrices, the
Bluestein/six-step sub-engines, and grow-only scratch buffers.

---

## 2. How it works

### 2.1 Numeric kernel — batched Stockham autosort, mixed radix

Everything below lives in `detail::fft_engine<T>` for one axis length `n`:

- **Precomputed twiddle table** `tw_[k] = exp(sign·2πi·k/n)` with the sign baked in.
- **Stage factorization**: the power-of-two part uses radix-4 stages plus a
  *single* radix-8 stage when the exponent is odd (measured: replacing a 4·2
  tail by one 8 saves a memory pass, but an all-8 plan loses to all-4 on
  register pressure — a radix-2 stage only ever appears for n == 2). Odd primes
  follow ascending; primes ≤ 64 use a direct kernel driven by a precomputed
  p×p DFT matrix (no modulo in inner loops); primes > 64 become nested
  **Bluestein sub-plans** (§2.2). Stockham is self-sorting for any factor order.
- **Every stage kernel is batched**: it transforms `m` interleaved fibers at
  once, laid out `buf[k*m + j]` with the batch index `j` contiguous, so inner
  loops auto-vectorize. The kernels are templated on `Batched`; the `m == 1`
  instantiation constant-folds all the `·m` offset arithmetic back to the
  scalar code (this mattered: leaving `m` runtime cost ~30% on 1-D).
- **Kernels take separate input/output element strides** (`sa`/`sb`, folded to
  1 when unbatched). This enables *fused* execution: when the batch axis is
  contiguous in user memory, the first stage reads the strided user tile and
  the last stage writes it back directly (`run_fused`), eliminating the
  gather and scatter passes entirely — safe because a Stockham first stage
  fully consumes its input, and layout validity guarantees batch extent ≤
  fiber stride (no tile self-overlap). This was worth ~1.3–1.5× on 3-D.
- Kernel src/dst pointers are `BOOST_MULTI_FFT_RESTRICT`-qualified (they never
  alias within one stage), removing runtime overlap checks and loop versioning
  (measured: ~7–10% on 2-D/3-D).
- Ping-pong between two grow-only buffers; no bit-reversal pass, no per-stage
  allocation, no modulo in inner loops. The multiply-by-(∓i) is `tw_[n/4]`,
  so kernels stay generic over `T` and sign-correct automatically.

### 2.2 Bluestein (chirp-z) for large primes

For prime `n > 64` the engine evaluates the DFT as a circular convolution of
length `M ≥ 2n−1`, where `M` is the *cheapest 5-smooth* (2ᵃ3ᵇ5ᶜ) candidate up
to the next power of two, scored by a per-stage cost model — e.g. 20480 =
2¹²·5 for n = 10007 (the smallest smooth 20250 = 2·3⁴·5³ and the power of two
32768 both lose; this selection took n = 10007 from 3.0× to ~1.6× of FFTW).
The convolution: pre-multiply by the chirp `c_j = exp(sign·iπj²/n)`,
convolve with the wrapped conjugate chirp via two nested engines
(signs `s` and `−s`; the kernel spectrum and the 1/M normalization are
precomputed into the plan), post-multiply. For a large prime *factor* p of a
composite n, the size-p sub-DFTs of that stage are batched through the nested
plan **all at once** (batch = ns·m), and its output layout coincides with the
stage's required output layout, so the result is copied back in one block.
Generic-`T` note: the conjugate chirp is built with `cos/−sin` directly — no
`conj(T)` is required of the element type.

### 2.3 Six-step decomposition for long fibers

Single fibers with `n ≥ 2¹³` (measured threshold) split as `n = n1·n2`
(balanced greedy split of the factor list). Viewing the fiber as a row-major
`[n1][n2]` grid:

1. length-n1 FFTs **batched over the contiguous j2 index** — the flat buffer
   already *is* the batched layout, so this pass needs no gather at all;
2. twiddle multiply `W_n^{j2·k1}` **fused into a tiled transpose** to `[n2][n1]`;
3. length-n2 FFTs batched over k1 — which lands directly in natural output order.

The twiddle-transpose stages 32×32 tiles through an L1 buffer so both sides
stream contiguously (power-of-two row strides otherwise alias cache sets).
Both FFT passes are wide-batch (vectorized) and cache-blocked. This was worth
1.3–2× on n ≥ 2¹⁸ and 2× on 720720 (it also auto-batches the direct
radix-7/11/13 stages). Contiguous fibers additionally skip the input gather
(the column pass reads user memory in place) *and* the output copy (the row
pass's final stage writes user memory directly).

### 2.4 The `fft_ops<T>` product — the single biggest performance fix

libstdc++'s `std::complex` `operator*` implements C-Annex-G semantics: an
inline fast path plus a NaN check that falls back to `__muldc3`. That branch
**blocks all auto-vectorization** of the kernels (measured: 2–3× on everything;
95 libcall sites in the benchmark binary). `fft_ops<std::complex<R>>::mul` uses
the plain 4-mul formula; generic `T` keeps `operator*`; users can specialize.
This recovers `-ffast-math`-level performance under strict FP flags.

### 2.5 N-D orchestration — idiomatic Multi

For `D >= 2` the **last two axes are transformed together, slab by slab**
(`fft_apply_last_pair`): both passes run while the rank-2 slab is still
cache-resident, which for 3-D replaces two full-array memory sweeps by one and
turns the second pass's strides slab-local (3-D pow2 went from ~1.4x to
~1.2-1.4x of FFTW). The remaining axes are reached by *static recursion* over
rotated views — every axis is a distinct instantiation bound to its engine
slot at compile time; there is no runtime axis parameter anywhere:

```cpp
detail::fft_apply_last_pair(arr(), engine_<D - 1>(), engine_<D - 2>());
if constexpr(D >= 3) { transform_middle_<1>(arr().rotated()); }

template<std::ptrdiff_t K, class View>       // view = arr rotated K times,
void transform_middle_(View&& view) const {  // so its last axis is axis K-1
    detail::fft_apply_last(view, engine_<K - 1>());
    if constexpr(K < D - 2) { transform_middle_<K + 1>(view.rotated()); }
}
```

Because `rotated()` preserves rank *and type*, the recursion instantiates one
View type at D-2 depths — the compile-time cost is negligible, and the
axis→engine map (`which_`) is fixed at plan construction.

`fft_apply_last` rank-descends to 2-D slabs, **keeping the smallest-stride
leading axis alive** (via `transposed()`, which swaps the first two axes) so
that at rank 2 the batch axis is the one closest in memory. A rank-2 slab is
transformed in tiles of up to `mb_` fibers (sized so 2·n·mb stays
cache-resident). Three layout-selected routes: **contiguous fibers** go one at
a time straight from user memory, final pass writing back directly (measured
faster than transpose-tiling them); **batch-axis-contiguous tiles** run fully
fused (first/last stage on user memory, no gather/scatter at all); anything
else is gathered interleaved with the copy loop order chosen from the strides.

---

## 3. Correctness

Verified two independent ways:

- Against a direct O(N²) reference DFT in `test/algorithms_fft.cpp`: radix and mixed sizes
  2…2048 (including all pow-2 stage combinations), Bluestein sizes 67, 101,
  134 = 2·67, 331, 1009; plus round-trips (1-D/2-D/3-D/4-D and the 65536
  six-step path), DC = sum, 2-D = composition of 1-D, 4-D = composition of
  2-D-of-2-D, strided column in place, plan reuse ≡ one-shot,
  plan-from-extents, same plan on contiguous and strided layouts of the same
  shape.
- Against **real FFTW** in the benchmark harness: every benchmarked size
  (pow-2, 5⁷, 10⁶, 720720, primes 1009/10007/100003, 2-D, 3-D) matches to
  relative error ~1e-15.

Compiles warning-clean under GCC `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion -Werror` and Clang `-Wall -Wextra -Wpedantic`.

---

## 4. Performance vs FFTW

Methodology: both libraries build their plan once (untimed) and recycle it;
only execution is timed. FFTW wisdom disabled, `FFTW_ESTIMATE`,
single-threaded, CPU cache flushed before every timed rep, GCC
`-O3 -march=native` **without** `-ffast-math`. Ratio = mine/FFTW (< 1 means
faster than FFTW); the range spans several independent runs (the machine shows
±15–20% run-to-run drift).

| case | mine / FFTW (exec, plan reused) |
|---|--:|
| 2-D 256² / 512² / 1024² (pow2) | **0.58–0.66× / 0.53–0.61× / 0.63–0.72×** (faster) |
| 2-D 1000² | 0.90–1.30× |
| 3-D 64³ / 128³ / 100³ | 1.37–1.43× / 1.12–1.37× / 1.31–1.65× |
| 1-D 256 / 1024 / 4096 | 1.05–1.13× / 1.16–1.21× / 1.28–1.47× |
| 1-D 2¹⁴ / 2¹⁶ / 2¹⁸ / 2²⁰ | 1.7–2.5× / 1.8–1.9× / 1.5–2.0× / 0.9–1.7× |
| 1-D 10⁶ (2⁶·5⁶) / 5⁷ | 1.3–1.9× / 0.9–1.8× |
| 1-D 720720 (2⁴·3²·5·7·11·13) | 2.5–2.9× (worst case; FFTW's codelet sweet spot) |
| 1-D primes 1009 / 10007 / 100003 | 1.2–1.3× / 1.6–1.7× / 1.3–1.8× |

One-shot (plan build + execute) additionally favors this implementation for
small/awkward sizes (≤ 4096, primes ≈ 1000, and 720720 where FFTW's ESTIMATE
planner takes ~1 s) because our planning is a cheap table fill.

Takeaways:

- **2-D is consistently faster than FFTW**; 3-D is within 1.1–1.65×;
  1-D is within 1.1–2× everywhere except 720720 (~2.5–2.9×).
- **No pathological sizes remain**: primes run via smooth-length Bluestein at
  1.2–1.8× (10007 was ~O(n²) at the start), 720720 went from ~19× to ~2.7×.
- The residual gap (mid-size 1-D, 3-D, highly-composite) is FFTW's
  hand-written SIMD codelets — including dedicated radix-7/11/13 codelets
  that explain the 720720 outlier — vs portable scalar-source C++.
- One-shot plan+exec on six-step/Bluestein sizes is dominated by first-touch of
  the sub-plan buffers — reuse the plan (that is what it is for).

Optimization history (exec-only, cumulative): naive DFT ~30–50× → twiddle
tables + Stockham radix-4 + per-axis reuse ~4× (pow2) → radix-3/5 fixed 5-heavy
sizes → batched kernels + `fft_ops` product (−2–3× everywhere) →
per-fiber no-gather contiguous path (−10–15% on 2-D/3-D), radix-8 tail rule,
Bluestein (primes now O(N log N)), six-step (−1.3–2× on long fibers), fused
strided first/last stages (3-D from ~2.2× to ~1.4–1.8×; 2-D to ~0.6×),
cost-model smooth convolution lengths, tiled six-step transpose, last-two-axes
slab pairing for D >= 3, `restrict`-qualified kernels.

Measured dead ends (kept out): all-radix-8 pow2 plans (register pressure beats
the saved pass at every size, even L1-resident n = 64), and routing small
cache-resident slabs through the transpose-tile batched path instead of
per-fiber (gather overhead eats the SIMD gain).

---

## 5. Multi library idioms discovered (the useful part for future work)

These are non-obvious things about Multi's core that made — or would make — this
code simpler. Several of my initial "missing feature" guesses were wrong; the
feature already existed under an idiom I didn't know.

1. **Rotate to iterate any axis.** `A.rotated()` cyclically permutes axes (axis 0
   goes to the back: sizes (2,3,4) → (3,4,2)) and returns a **same-rank view
   sharing memory**; `D` rotations return to the original orientation. This is
   how "do something along every axis" is expressed generically — transform the
   last axis, rotate, repeat.

2. **Subarray-pointers are the rebindable view handle.** A plain `subarray` is *not*
   rebindable — `v = v.rotated()` assigns **elements**, not the view. Instead:
   `auto p = &A();` yields a value-semantic pointer that carries the layout+base
   descriptor; `*p` recovers the view; `p = &(*p).rotated()` **rebinds**. Because
   `rotated()` preserves rank, the pointer type is stable, so a compile-time
   recursion over axes *can* collapse to a runtime `for` loop. (The plan's axis
   walk ultimately went the other way — static recursion, one instantiation per
   axis — so that each axis is bound to its engine at compile time; the pointer
   idiom remains the right tool when a runtime walk is genuinely wanted.)

3. **`transposed()` swaps the first two axes** (same-rank view). Combined with
   rank-descent (`for(auto&& sub : view)` drops axis 0) it gives enough control
   to steer *which* leading axis survives to the innermost level — used here to
   keep the smallest-stride axis as the vector batch axis. (Arbitrary axis
   permutations would be nicer; see future work.)

4. **A single fiber is first-class:** `A(2, multi::_, 5)` (fix all indices,
   leave one free) *is* the 1-D strided section along that axis — a writable
   rank-1 `subarray`; `fft_inplace(A(2, multi::_, 5))` works as-is. What the
   library does not yet offer is the *collection* of all such fibers along an
   axis as one flat iterable range (see future work); the orchestration here
   reaches every fiber via rotation + rank descent instead, which needs no
   runtime index enumeration and stays rank-generic. (Rank-generic placeholder
   indexing was prototyped: the placeholder slot is a compile-time property, so
   it needs pack-splicing + an odometer — longer than the descent, and
   fiber-at-a-time execution forfeits the slab-level batching/fusion: measured
   2× slower on 3-D.)

5. **Single-axis size/extent accessors:** `get<d>(A.extents())` gives axis `d`'s
   `index_extension`; `get<d>(A.sizes())` gives its **length** directly; `get<d>(A.strides())`
   its stride. `d` is a compile-time index. (No runtime `A.extension(d)`; rotate so
   the axis is last and use `get<D-1>`.)

6. **`elements()` vs `flatted()` vs `flattened()`** — three different "flatten"s:
   - `A.elements()` — a flat range over **all scalars** (odometer over all D axes);
     works for any layout. Great for element-wise fill/reduce.
   - `A.flatted()` — merges the outer two axes into one **assuming contiguity**. Its
     validity assert is commented out, so on a non-contiguous (e.g. rotated) view it
     **silently fabricates an invalid layout → memory corruption**. Prefer `flattened()`.
   - `A.flattened()` — merges the outer two axes using a general **bi-strided
     (`bilayout`/`bistride`)** layout, so it works on non-contiguous views too. It is
     the *right mechanism*, but currently **not usable for rank-generic code**:
     its result type reports `dimensionality` as a function (not `static constexpr`),
     its `bistride` iterator lacks `operator==` (so `for(auto&& row : F)` doesn't
     compile), and it doesn't chain (`flattened()` on a `bilayout` has no `flatten`).

7. **Type/name facts:** `element` (member type) is the scalar type; `dimensionality`
   is a `static constexpr`; `num_elements()` is the total count. `extents_t<D>` is the
   current type name — **`extensions_t<D>` is a backward-compatibility alias**; prefer
   `extents_t`. Multi's default size type is **signed** (`ssize_t`); `size_t` (signed)
   is deprecated in favor of `ssize_t`.

8. **1-D subarrays expose `stride()` and `base()`** — exploited here: a
   unit-stride fiber with a raw-pointer base is handed to the kernel directly
   (`eng.run_contig_inplace(fib.base())`), skipping gather and scatter. Guard
   such fast paths with `std::is_pointer_v<decltype(fib.base())>` so
   fancy-pointer (GPU-ish) arrays fall back to the iterator gather.
   Note: `reversed()` returns a **const** view even from a mutable array, so
   negative-stride writable views cannot arise through the public API.

9. **`multi::inplace_array`** is a stack-storage, **compile-time-shaped** array (backed
   by C arrays). Use it only when the shape is a compile-time constant and small
   enough to want cache/stack residency — not for runtime-sized buffers like FFT fibers.

10. **Non-performance-portable `std::complex` arithmetic.** Not a Multi idiom but
    load-bearing for any numeric kernel in this codebase: `std::complex::operator*`
    under strict FP compiles to a branch + `__muldc3` libcall (Annex G), which kills
    auto-vectorization. Route hot products through a customization point
    (`fft_ops<T>` here) rather than requiring users to pass `-ffast-math`.

---

## 6. Future work (roughly by value)

- **Compile-time codelets for small n.** The weak spot in 3-D is the scalar
  per-fiber pass over n = 64…1024 contiguous fibers. Dispatching common small
  sizes to templated instantiations (`run_fixed<64>()`, chosen once at plan
  time) lets the compiler fully unroll stage loops and constant-fold offsets —
  an FFTW codelet minus the intrinsics. Estimated 1.3–2× on that path.
- **Per-stage packed twiddle tables.** Kernels load `tw_[r*tstep]` — strided,
  one cache line per butterfly in early stages. Repacking each stage's
  twiddles sequentially makes those loads stream; trivial memory cost for the
  small-n engines that dominate 2-D/3-D (cap for large-N 1-D).
- **Plan-time autotuning.** All routing constants (radix-8 tail rule, `mb_`,
  six-step threshold, per-fiber vs tiles) were measured on one machine. An
  optional `fft_measure`-style plan flag timing 2–3 candidate strategies would
  make the performance portable — FFTW's actual planner advantage.
- **Small-prime codelets (Winograd/Rader style) for 7, 11, 13.** The one
  remaining outlier (720720 at ~2.7×) spends its time in the O(p²)-per-group
  direct generic stages; minimal-multiplication small-prime kernels are how
  FFTW wins there.
- **Explicit SIMD kernels — with the right expectations.** Measured state
  (GCC 15, `-O3 -march=native`, strict FP): 30 loops in this header already
  auto-vectorize, yet compiling with `-fno-tree-vectorize` costs only ~7–10% —
  so the residual gap vs FFTW is *not* "loops don't vectorize". It is (a) the
  interleaved complex (AoS) layout, whose vectorized product needs re/im
  permutes that halve the effective FLOP rate (FFTW's codelets are scheduled
  around this), and (b) the single-fiber scalar path, where butterflies are
  independent across `r` but strided, so vectorizing across them needs data
  movement compilers won't invent. `std::execution::unseq` does not help the
  kernels: butterflies are 4-in/4-out (not a `std::transform` shape), zip/iota
  iterators degrade to input-category so libstdc++'s PSTL runs them serially,
  and unseq itself lowers to `#pragma omp simd` only under `-fopenmp(-simd)`.
  Its one real payload — asserting no aliasing — is already captured by the
  `BOOST_MULTI_FFT_RESTRICT` qualifiers on the kernel src/dst (measured −7%
  2-D, −10% 3-D). The standard-C++ path that would genuinely close the gap is
  C++26 `std::simd` (P1928; unlike `std::experimental::simd` it has the
  permutes complex products need): an `fft_ops`-style butterfly specialization
  over `std::simd<double>`, keeping the generic kernels as fallback.
- **Rader's algorithm** as an alternative to Bluestein for primes just above 64
  (Rader needs an (n−1)-point convolution vs Bluestein's ≥ (2n−1)-point; it's
  ~2× cheaper per transform but needs a primitive root and only works for primes).
- **Parallelism.** Fibers, tiles, and slabs along an axis are independent; the
  engine's scratch is the only shared state, so parallel execution needs
  per-thread engines (plans are copyable — one per thread already works today).
- **r2c / c2r and normalization options** (real-input transforms; optional 1/N
  or 1/√N scaling) for parity with common FFT APIs.
- **Plan ergonomics**: axis subsets (transform only axes {0,2}), out-of-place
  `plan(in, out)`, a `shared`/thread-safe execute mode, and possibly a
  layout-bound tier (`plan.bind(A.layout())`) precomputing per-axis strategy —
  the current plan is deliberately extents-only, with layout-dependent routing
  decided cheaply at execute time.
- **Depends on Multi core: a rank-generic `fibers(d)`** — a flat, iterable,
  layout-agnostic range of 1-D fibers along axis `d` (the rank-1 analog of
  `elements()`). A *single* fiber is already expressible as
  `A(i, multi::_, k)`; what's missing is the enumerated range of all of them.
  ~80% exists as `flattened()`/`bistride` (see §5.6); finishing it (static
  `dimensionality`, iterator comparisons, chaining) would collapse
  `fft_apply_last` into `for(auto&& fib : A.fibers(d)) …`. An arbitrary-axis
  `transposed<i, j>()` would likewise simplify the batch-axis steering.

---

## 7. Design boundary (why the kernel is "raw")

The N-D orchestration is deliberately idiomatic Multi (rotations, `transposed()`,
sub-array iteration, iterator gather/scatter, stride introspection) because that
is where genericity over rank and layout lives. The stage kernels are deliberately
raw index arithmetic on contiguous scratch because that is the numeric hot path —
expressing butterflies through strided iterators would block the tight,
auto-vectorizable batched loops that make this competitive with FFTW. The
boundary between the two is the gather/scatter (or, on the fast paths, a raw
`base()` pointer handed across) — FFTW draws the same line.

---

## 8. CUDA adaptation plan (not yet implemented)

The architecture is already CUDA-shaped: Stockham (no bit-reversal, uniform
stages) is what GPU FFT libraries use; the batched layout `buf[k*m + j]`
(batch contiguous) is exactly warp-coalesced when `j` maps to `threadIdx.x`;
the fused strided stages' (pointer, stride, width) interface is
layout-agnostic; and plans already separate "build tables once" (host) from
"execute many" (upload once, launch many). Incremental steps:

1. **Extract butterfly bodies** into `BOOST_MULTI_HD` inline functions (the
   per-(block, r, j) work) shared by the host loops and device kernels. Pure
   refactor; CPU codegen must not regress (re-benchmark).
2. **`fft_memory_space<Ptr>` trait** dispatching host (raw `T*`, current
   paths) vs device (thrust/Multi-CUDA fancy pointers), specialized in an
   opt-in adaptor header so the core stays CUDA-free. The current
   host-iterator gather fallback must *never* run on device pointers. The
   entry point is already the right one: `plan(cursor)` accepts any pointer
   type in `cursor.base()`, so a device cursor flows in unchanged and the
   trait picks the device orchestration.
3. **Device table mirror**: keep host `std::vector` tables; add a lazily
   uploaded POD `engine_view` (raw device pointers + sizes) per engine.
4. **Device stage launchers**: one `__global__` kernel per stage kind around
   the shared butterfly body; thread → (j, r, block); stream-ordered launches
   replace the host stage loop. Grids are natively 3-D, so an extra
   (outer-slab, stride) index runs a whole 3-D middle axis in one launch.
5. **Granularity flip**: same rotation/rank-descent orchestration, but the
   device branch batches the whole axis (m = all fibers; `mb_` is a host
   cache concern) and skips slab pairing (a CPU cache optimization). Needs a
   cap-and-tile fallback for the 2·n·m device scratch on huge arrays.
6. **Bluestein**: pointwise chirp/kernel multiplies as elementwise kernels;
   sub-engine recursion stays host-side (it only orchestrates launches).
   **Six-step**: skip on device (its cache-blocking rationale is CPU-only).
7. **Tests**: mirror `algorithms_fft.cpp` on Multi-thrust device arrays (the
   test already compiles as CUDA in the nvcc lane); `thrust::complex` should
   work unchanged through the `T{re,im}`/`+`/`-`/`*` contract and `fft_ops`.
8. **Benchmark vs cuFFT** with the `adaptors/cufft` harness — the goal is
   *generic* device FFT (custom element types, arbitrary strided layouts,
   header-only), not beating cuFFT's tuned codelets for `float/double`.
