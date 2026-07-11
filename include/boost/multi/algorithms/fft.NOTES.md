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
  This is current-design-only: the §9 redesign removes every `mutable` member
  from the engine hierarchy, which retires this caveat entirely — see §9.2.

Architecture: `fft_plan<T, D>` holds one `detail::fft_engine<T>` per **distinct
axis length** (a cube shares one engine across all three axes). An engine owns
the twiddle table, the stage factorization, the direct-prime DFT matrices, the
Bluestein/six-step sub-engines, and grow-only scratch buffers. This paragraph
is the original design, kept for context; see §9.2 (landed) for the current
`fft_plan<D, TW>` / externalized-scratch shape.

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
// (current shape, post-§10: one fused-pair fast path when both of the
// last two axes are active, then ONE uniform recursive walk for the rest;
// with per-axis directions the walk alone handles every other case)
detail::fft_apply_last_pair(view, engine_<D - 1>(), engine_<D - 2>(), arena);
if constexpr(D >= 3) { transform_axes_<1, D - 2>(view.rotated(), arena); }
// ... or, when the pair doesn't apply: transform_axes_<0, D - 1>(view, arena);

template<std::ptrdiff_t K, std::ptrdiff_t Stop, class View, class T>  // view = arr rotated K times;
void transform_axes_(View&& view, T* arena) const {                   // last axis = K-1 (D-1 for K==0)
    constexpr std::ptrdiff_t axis = (K == 0) ? D - 1 : K - 1;
    if(dirs_[axis] != fft_direction::none) { detail::fft_apply_last(view, engine_<axis>(), arena); }
    if constexpr(K < Stop) { transform_axes_<K + 1, Stop>(view.rotated(), arena); }
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
  decided cheaply at execute time. Per-axis mixed direction (forward on some
  axes, backward or skipped on others) is a real future direction, but not a
  drop-in extension of the current single-`sign`-per-plan design: the engine
  dedup key (currently `n_ == len`, correct only because every engine in a
  plan shares one sign — see §9) must widen to `(len, sign)` once sign is
  per-axis, or it would silently reuse a wrong-direction engine — a
  correctness bug, not just a missed optimization. "Skip this axis" is a
  third, distinct state, not representable as a trivial engine: it must be a
  bypass in the orchestration itself (`apply_`/`transform_axes_`/
  `fft_apply_last_pair`), since even a no-op engine would still pay for
  gathering/scattering that axis's data. `fft_apply_last_pair` in particular
  assumes its two axes get uniform treatment for the cache-locality win
  (§2.5); a mixed pair (different sign, or one skipped) needs a fallback to
  two independent single-axis passes for that one case.
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

---

## 9. Plan internals: redundancy audit and the T-decoupling redesign

This section is design analysis and an agreed-upon direction, not necessarily
fully reflected in the current class body — the plan/engine split is being
actively restructured toward it. It came out of asking, plainly: does
`fft_plan` need to store *this* particular piece of state, or is it already
recoverable from something else the plan holds anyway?

### 9.1 Redundancy audit of `fft_plan`'s own members

- **`sign_` is genuinely redundant.** It's written once at construction and
  read only inside `init_()` (same construction call) to seed `engines_`.
  Every `fft_engine` already stores its own `sign_` (needed there — see
  below), and every engine in one plan is built with the *same* sign, so
  `engines_.front().sign_` already holds exactly what `fft_plan::sign_` holds.
  Its only remaining job is to answer the public `sign()` accessor, which can
  just delegate to an engine instead of keeping a second copy.
- **`sizes_` is not redundant in the same way**, even though
  `engines_[which_[a]].n_ == sizes_[a]` also holds by construction. The
  difference is *when* it's read: `sign_` was construction-only, but
  `sizes_` is read on every single `execute()` — `fft_view_from_cursor(home,
  sizes_)` needs the whole shape to reconstruct a view from a bare cursor
  (cursors carry no size of their own), and the checked array overload's
  `matches_()` validates the caller's shape against it. Deriving that array
  by gathering `engines_[which_[a]].n_` for every `a`, on every call, would
  trade a flat, cheap, direct read for repeated indirection through
  (possibly large) engine objects, for a member that's a few bytes and never
  changes. Worth keeping as-is.
- **Reusing one engine per distinct axis length (not per axis) is correct
  and worth keeping.** A cubic 3-D plan shares a single engine across all
  three axes rather than building three identical ones — real savings
  (twiddle table, direct-prime matrices, any Bluestein/six-step sub-tree,
  each is genuine O(N)-ish construction work) for the common case of
  square/cubic transforms, at the cost of one linear scan over at most `D`
  engines during construction only. Free: `D` is always tiny (array rank),
  and the scan never runs on the execute path (`which_[a]` is a flat O(1)
  index by then).
- **The engine list should stay a `std::vector` with linear scan, not a
  `std::map`.** At `N ≤ D` (tiny), a tree's O(log N) has no room to beat a
  contiguous scan on constant factors alone, and a node-based container adds
  cache-unfriendly pointer-chasing a flat array doesn't have. The lookup is
  also construction-time-only (see above), so even a hypothetically "faster"
  map would save nanoseconds against the microseconds-to-milliseconds of
  actual engine construction it's guarding.

### 9.2 The bigger move: decoupling the plan from the array's element type

Prompted by a specific requirement: **the plan should not own the ping-pong
scratch buffers (`buf_`/`out_`/`xbuf_`), only their required sizes**;
allocation happens at `execute()` time, and if that allocation is a real
concern, the fix is a fast (e.g. monotonic/arena) allocator, not baking the
memory into the plan. The consequence, once buffers are out of the picture,
is that the *only* T-typed state left in `fft_engine`/`fft_plan` is the
twiddle machinery (`tw_`, `wmat_`, `chirp_`, `postc_`, `kernel_ft_`) — so the
plan need not be templated on the array's element type `T` at all, only on
whatever type the twiddle tables use.

Shape this implies:

- **`fft_plan<D, TW = std::complex<double>>`, not `fft_plan<T, D>`.** `TW` is
  the twiddle/table element type, a plan-level choice independent of any
  array it's later applied to. **Implemented as `<D, TW>`, not `<TW, D>`** —
  C++ requires every template parameter after one with a default to also
  have one, and `D` (the rank) has no sensible default, so a default for
  `TW` is only expressible if `D` comes first. This reorders the existing
  `fft_plan<T, D>` spelling (a discussed, deliberate API break — see the
  landed implementation's commit for the two calling files updated).
- **`execute()` becomes a template on the array's element type, deduced per
  call.** Implemented via `Cursor::element` (every Multi cursor already
  exposes this nested typedef), not by switching the public call convention
  to take the array/subarray directly — `plan.execute(arr.home())` keeps
  working unchanged; only the *implementation* deduces `T` where it
  previously used the class's own (now removed) `T`. One plan, built
  once for a shape, can then execute against a `complex<float>` array today
  and a `complex<double>` array tomorrow without rebuilding any tables.
- **Buffers move from persistent engine members to execute-time-local
  state**, sized from an element-count query the engine exposes (e.g.
  `scratch_elements()` returning `n_ * mb_`, plus whatever Bluestein/six-step
  sub-engines need) — the plan reports *how many* `T`s are needed without
  ever storing a `T`.
- **The stage kernels need a genuine cross-type multiply**: today's
  `fft_ops<T>::mul(T, T) -> T` generalizes to `fft_ops<T, TW>::mul(T const&,
  TW const&) -> T`. This is where a real, explicit tradeoff surfaces rather
  than staying implicit: if `TW` matches `T`'s own precision, this costs
  nothing extra (today's behavior). If `TW` is deliberately wider than `T`
  (double twiddle, float data — the likely default), every multiply promotes
  the float operand up, multiplies in double, narrows the result once — this
  is *better* accuracy than the earlier "compute wide, store narrow" twiddle
  proposal (the twiddle value itself is never prematurely rounded down at
  all now, only the per-multiply result rounds once), but it is not free:
  inside the batched SIMD loop, widening a register of floats to double and
  narrowing back is real work, and a double op fills half the same-width
  vector register a float op would — a genuine accuracy-for-throughput
  trade, not a strict win, and one `TW` conveniently exposes as a dial rather
  than a hidden default (picking `TW` to match `T`'s own precision opts back
  into today's zero-overhead behavior).
- **Decision (1), settled: `TW` defaults to `std::complex<double>`, but only
  for ergonomics** — so a plan declaration doesn't force spelling out the
  twiddle type for the common case. This is explicitly *not* a claim that
  double is fundamentally the right twiddle precision; `TW` is a fully free
  parameter and every choice is equally "correct" for its own tradeoff (§2.4).
  The default just avoids boilerplate, it doesn't privilege double.
- **Decision (2), settled in the common case, open at the edge: the
  allocator is passed explicitly to `execute()`.** No default is assumed by
  the plan. The alternative — the *plan itself* holding a default allocator
  to fall back on when the caller passes none — is explicitly parked as
  "ergonomic but dangerous," unresolved. Two concrete reasons it's genuinely
  tricky, not just a style call: (a) a `std::allocator<T>`-shaped default
  can't live in the plan at all without reintroducing the very `T`-dependency
  this redesign removes — the plan would need a *type-erased* resource (e.g.
  `std::pmr::memory_resource*`) rebound to a `T`-typed
  `std::pmr::polymorphic_allocator<T>` only at `execute()` time, to stay
  T-agnostic; (b) even type-erased, a *shared* default resource reintroduces
  the exact concurrent-`execute()` hazard that motivated pulling the scratch
  buffers out of the engine in the first place (mutable shared state needs
  external synchronization, or one instance per thread) — so a plan-owned
  default wouldn't just be a convenience shortcut, it would quietly bring
  back a thread-safety caveat this whole redesign was trying to shed.
  *Update: settled in §10.4(b)* — there IS a defaulted overload, but its
  default is a fresh stateless `std::allocator<T>` constructed per call,
  never plan-owned state, which sidesteps both hazards above.
- **Corollary, verified: this removes every `mutable` member in the engine,
  not just the obvious ones.** Grepping the current file, `mutable` appears
  in exactly three places — `buf_`, `out_`, `xbuf_` — all three the scratch
  this redesign externalizes. Everything else (`tw_`, `wmat_`, `stages_`,
  the Bluestein state, the six-step state, and `sub_`, the nested sub-engine
  tree) is set once at construction and read-only after. Since `sub_` is
  instances of the same template, this holds *recursively*: the entire
  engine hierarchy — top-level plus every nested Bluestein/six-step
  sub-engine — becomes genuinely immutable post-construction, with zero
  remaining mutable state to race on. That directly retires the
  thread-safety caveat in §1: concurrent `execute()` on the *same* plan from
  multiple threads becomes safe automatically, as long as each call supplies
  its own scratch/allocator — no more "one plan copy per thread" needed.

This composes cleanly with the CUDA plan in §8, and arguably more directly
than that section originally assumed: a plan that isn't tied to a fixed `T`,
whose `execute()` deduces the array type per call and receives its scratch
from outside, maps naturally onto "same plan, host float array today, device
double array tomorrow" — the memory-space dispatch in §8 step 2 and this
redesign are pulling in the same direction.

## 10. Partial / mixed-direction FFTs — settled design + implementation plan

> Session-by-session execution ordering for this section and §9.2 (with
> gates, commit checkpoints, and a definition of done) lives in the
> companion file `fft.PLAN.md`.

Feature request (2026-07): per-axis transform directions, e.g. on a 4-D array

    multi::fft_inplace({{forward, none, backward, forward}}, inout);

where `none` means "leave that axis completely untouched." A plan with `none`
on some axes *is* a batched lower-dimensional FFT (`{none, forward}` on a 2-D
array = "FFT each row") — the same unification FFTW's guru interface gets
from loop-dimensions vs transform-dimensions, obtained here by simply
skipping passes. This section records the decisions (all settled, discussed
and agreed with the maintainer) and a step-by-step plan detailed enough for
another model/developer to execute without re-deriving the rationale.

### 10.1 Settled decisions and why

1. **Direction is a per-axis, three-valued property**: `forward`, `none`,
   `backward`. Suggested representation: `enum class fft_direction : int
   { forward = -1, none = 0, backward = +1 }` — values chosen so `forward`/
   `backward` interconvert trivially with the existing `int` sign convention
   (`fft_forward == -1`, `fft_backward == +1`, FFTW-compatible) and `none`
   is falsy.

2. **Runtime values, compile-time arity.** The spec is a
   `std::array<fft_direction, D>` (D entries enforced by the type), NOT
   template parameters (`fft_inplace<forward, none, ...>`). Rationale,
   settled after explicit challenge: direction is consumed once per *pass*
   (one branch per axis per O(N log N) execution — unmeasurable), so
   lifting it into the type buys no codegen; it would infect `fft_plan`'s
   type (different combos = different types: no containers, no runtime
   selection), break the natural callers of this exact feature (solvers
   flipping forward↔backward per phase, rank-generic code building the list
   in a loop), and risk 3^D instantiations of the call path in a
   header-only library. The *only* statically checkable property of a
   direction spec is its arity, and `std::array<_, D>` mostly checks that —
   **update, verified while implementing (§10.3 step 4): only one-sided.**
   `{{f, none}}` against a rank-4 `std::array<fft_direction, 4>` (too FEW
   entries) is legal aggregate-init, zero-padding the rest, and zero is
   `fft_direction::none` — silently *not* a compile error. Only too MANY
   entries fails to compile (confirmed: "no matching function", overload
   resolution rejects it outright). A from-scratch attempt to close this
   (deduce the direction-array's length independently and `static_assert`
   it against the array's rank) confirmed a braced-init-list is a
   non-deduced context for `std::array<T, N>`'s `N` too — so this can't be
   closed without replacing `std::array` here with a custom fixed-arity
   wrapper (a constructor template constrained to exactly `D` arguments).
   Maintainer decision: accept and document, not worth the complexity.
   A constexpr template sugar forwarding to the runtime API can be added
   later, non-breaking, if ever wanted.

3. **Directions are baked into the plan at construction, not passed to
   `execute()`.** Constructor takes extents + the direction array; `execute`
   keeps its current lean signature. Rationale (this reverses an initial
   execute-time proposal, on these grounds):
   - *Immutability/thread-safety*: execute-time direction would force the
     plan to either eagerly build engines for every axis — wasteful in
     exactly the partial-FFT scenario (e.g. batched 1-D over the rows of a
     matrix whose column count is a large prime: a full Bluestein engine,
     chirp + convolution twiddles, for an axis never transformed) — or build
     them lazily at execute, i.e. mutate the engine list, reintroducing the
     very mutable-shared-state / shared-plan-concurrency hazard §9.2 just
     eliminated. Direction-at-construction keeps the engine set exact,
     fixed, and immutable.
   - *Plan reuse across directions is low-value here*: unlike FFTW (plan =
     expensive measured search), our construction is O(n) trig per distinct
     axis length — negligible next to O(N^D) data. The forward/backward
     roundtrip case is served fine by two plans.
   - *Bonuses*: exact scratch sizing (a skipped axis can't inflate the
     buffer requirement — relevant to §9.2's `scratch_elements()`), no new
     runtime-mismatch error class at execute, and the plan fully describes
     its transform (printable, cacheable by key).

4. **Engines stay direction-neutral; direction lives only in the plan's
   pass schedule.** Target state: tables are stored for one canonical sign
   (forward); the backward kernel conjugates twiddles *on load* (one sign
   flip in register, vectorizable, free); kernels are templated on sign, and
   the per-pass dispatch picks the instantiation. Engine reuse then stays
   keyed on size alone, so two same-size axes with *opposite* directions
   share one engine — this is precisely the answer to the concern recorded
   earlier that mixed directions would "play against" size-keyed engine
   sharing. Explicit anti-goal: do NOT bake per-direction conjugated tables
   into engines "since the plan knows the direction anyway" — that forfeits
   the sharing for no measurable kernel speedup.

5. **Normalization**: stays unnormalized (current convention). With mixed
   forward/backward there is no coherent single 1/N convention; document
   that `{forward, backward}` on two axes is *not* an identity on either
   axis, and that forward-then-backward on the same axis scales by that
   axis's length.

6. **Semantics guarantees**: elements are *never written* for `none` axes'
   passes (the skip is total, so untransformed data is bit-identical, not
   approximately equal — testable); an all-`none` plan is a valid no-op.

7. **API surface**: new constructor overload
   `fft_plan(extents, std::array<fft_direction, D>)`; the existing
   `(extents, int sign)` constructor stays as the broadcast convenience
   (sign applied to every axis). One-shot convenience gains a
   directions-array overload. **Naming (maintainer decision)**: keep the
   existing `multi::fft_inplace` family name for the wrapper — `do_fft`
   appeared in discussion only as a placeholder example and must NOT be
   used as the actual name.

8. **The plan stays parameterized on a complex (twiddle) type that is
   agnostic of what it executes on** — i.e. this feature builds on, and
   must not regress, the §9.2 decoupling (landed): `fft_plan<D, TW>` where `TW` is
   the table element type chosen at plan construction, while the array
   element type `T` is deduced per `execute()` call. Nothing in the
   direction schedule touches `T` or `TW`; `dirs_` is plain data. An
   implementer working before §9.2 has fully landed should still keep every
   piece of this feature (`fft_direction`, `dirs_`, pass skipping,
   engine pruning) independent of the array element type.

9. **Exploit the compile-time dimension `D` to bound allocations.** The
   plan is already a template on `D`, and the number of distinct engines is
   bounded by `D` (fewer still when axes share a length or are `none`). So
   the engine list does not need heap allocation at all: a fixed-capacity
   inline container of at most `D` engines (e.g. `std::array` of
   engine-sized slots plus a count — an `inplace_vector`-style member)
   replaces the current `std::vector`, making plan construction
   allocation-free at the top level and keeping engines contiguous for the
   (construction-only) linear-scan reuse lookup of §9.1. Note the engines
   themselves still own O(n) tables internally; the bound is on the *list*,
   not the tables.

### 10.2 Current-code obstacles the implementer must know

(Verified against the file as of this writing; §9's restructuring is in
flight, so re-audit before starting.)

- **Engines are sign-aware today**: `fft_engine` stores `sign_`; `tw_` is
  built with sign-scaled theta; `wmat_` (direct-prime DFT matrices) is
  derived from `tw_`; Bluestein's `chirp_`/`postc_` use sign-scaled theta
  and `kernel_ft_` is the FFT of the chirp; sub-engines are constructed
  with `sign_` (and Bluestein's convolution pair with `sign_`/`-sign_`).
  Migrating to direction-neutral engines (decision 4) touches all of these.
- **Bluestein subtlety**: the backward chirp is the conjugate of the forward
  chirp, but `kernel_ft_` is a *transform of* the chirp, and
  FFT(conj(x)) = conj(FFT(x)) *index-reversed* — so "conjugate on load" for
  the precomputed `kernel_ft_` is not a plain elementwise conj. If this
  proves fiddly, the sanctioned fallback is: neutral tables for the smooth
  Stockham path (where conj-on-load is straightforward), but keep
  Bluestein engines direction-keyed (share on `(n, direction)` there only).
- **The execute path pairs the last two axes** (`fft_apply_last_pair`
  transforms axes D-1 and D-2 together, slab by slab, for cache locality)
  when both are active; every other combination is handled by ONE uniform
  recursion, `transform_axes_<K, Stop>` ("transform the last axis of the
  current view if active, rotate, recurse", visiting axes D-1, 0, 1, ...,
  D-2). History behind that shape: the first Phase A implementation
  instead used a three-way degraded-pair branch with hand-picked rotations
  per case, and picked one wrong -- `view.rotated()` (sends axis 0 to the
  back → new last axis is 0) where `view.unrotated()` (sends the LAST axis
  to the front → new last axis is D-2) was required. The two coincide only
  at D == 2, so every 2-D test passed while D >= 3 was wrong --
  out-of-bounds wrong, for non-cubic shapes. Caught in review (fixed, with
  non-cubic 3-D regression tests), then the whole branchy dispatch was
  replaced by the uniform walk, in which the view is correctly positioned
  by construction and no per-case rotation choice exists at all. Lessons
  that survive the rewrite: verify rotation semantics empirically on a
  probe before relying on them, and test every dispatch path at D >= 3
  with a NON-CUBIC shape (D == 2 and cubic shapes both mask axis mix-ups).
- **1-D FFTs along different axes commute**, so skipping/mixing needs no
  new orchestration math — pass order stays a free (cache-driven) choice.

### 10.3 Step-by-step implementation plan

Phase A first (correct, minimal diff), Phase B after (the target state);
each phase leaves the tree green under the full strict-flags test build.

**Phase A — feature lands, engines stay sign-aware — DONE** (fft.PLAN.md
Session 2; see that file's DONE note for the one design gap found and
accepted while implementing it):

1. Add `enum class fft_direction` (§10.1 item 1) next to the existing
   `fft_forward`/`fft_backward` constants, plus tiny helpers
   (`to_sign(fft_direction) -> int`, validity as "is not none").
2. `fft_plan` gains member `std::array<fft_direction, D> dirs_` (the pass
   schedule) and the new constructor. The existing `(extents, sign)`
   constructor delegates by broadcasting. Engine-construction loop: skip
   axes with `dirs_[a] == none` (sentinel value in `which_[a]`, e.g.
   `size_t(-1)`, never dereferenced for none axes); engine-reuse lookup
   keyed on `(length, direction)` for now (Phase A keeps sign inside the
   engine), i.e. the `find_if` predicate tests `e.n_ == len && e.sign_ ==
   to_sign(dirs_[a])`.
3. `apply_()` consults `dirs_`: skip `none` passes; degrade the last-pair
   optimization per §10.2; `transform_middle_` skips none axes (runtime
   branch per axis — per-pass cost, irrelevant). D == 1 case: none = no-op.
4. `fft_inplace(dirs, arr)` overload (name settled per §10.1 item 7 — not
   `do_fft`); header-comment API block updated.
5. Tests (extend `test/algorithms_fft.cpp`, same strict `-Werror` build):
   - all-`none` plan is a bit-identical no-op;
   - `none` axes bit-identical while other axes transform (memcmp-style
     equality on untouched fibers, not tolerance-based);
   - composability: `{f, none}` then `{none, f}` ≈ `{f, f}` (tolerance);
   - batched equivalence: `{none, f}` on a 2-D array ≈ looping the existing
     1-D plan over rows;
   - mixed roundtrip: `{f, none, b}` then `{b, none, f}` ≈ n₀·n₂ × original;
   - reference-DFT cross-check of a mixed spec on small sizes including a
     prime length (exercises Bluestein backward) and smooth lengths;
   - same-size opposite directions (square 2-D, `{f, b}`) — exercises the
     engine keying; verify against reference DFT.

**Phase B — direction-neutral engines (the §10.1-item-4 target) — DONE**
(fft.PLAN.md Session 3; see that file's DONE note for the two decision
points — Bluestein's fwd/bwd sub-engine pair collapsed to one, and the
`kernel_ft_bwd_` closed-form table — and the ASan/parity verification of
the resulting scratch aliasing):

6. Remove `sign_` from `fft_engine`; build `tw_`/`wmat_` with canonical
   (forward) sign; template the Stockham stage kernels on a `Sign` (or
   `bool Backward`) parameter that conjugates twiddles on load; per-pass
   dispatch in `fft_apply_last`/`fft_apply_last_pair` selects the
   instantiation from the plan's `dirs_`. Direct-prime `wmat_` path: same
   conj-on-load treatment. **Landed as designed** — see §10.5's updated
   anchor map for the resulting shape (`fft_mul_dir<Backward>`, the
   `<Batched, Backward, T>` stage kernels, the runtime `bool backward`
   public entries).
7. Bluestein: attempt the neutral form (conjugate chirp on load; derive the
   backward convolution from the forward `kernel_ft_` via the
   conjugate/index-reversal identity, or store the one extra table if that's
   cleaner); if it degrades clarity, take the sanctioned fallback of §10.2
   and leave Bluestein engines direction-keyed. **Landed via the second
   table** (`kernel_ft_bwd_`, the closed-form conjugate/index-reversal —
   no direction-keyed fallback needed); the fwd/bwd conv sub-engine pair
   also collapsed to one neutral engine (a further win beyond what this
   step asked for — see fft.PLAN.md Session 3 DONE note for the aliasing
   argument this required).
8. Engine reuse key drops back to size alone. Re-run the full test battery;
   the same-size-opposite-direction test from step 5 now also proves the
   sharing (can assert `engines_.size() == 1` for square `{f, b}` via a
   test-only observer or just by the numerics). **Landed**: a public
   `engine_count()` accessor was added (harmless, useful to callers too),
   and the existing test updated in place to assert `engine_count() == 1`.
9. Update §1's thread-safety text and this file if Phase B changes any of
   the §9.2 immutability conclusions (it shouldn't — it only *removes*
   state from engines). **Confirmed**: no immutability conclusion changed;
   engines are, if anything, more clearly immutable post-construction now
   that `sign_` (the last piece of per-engine direction state) is gone.

**Ordering vs the §9 T-decoupling**: orthogonal state (direction schedule
vs TW/T split and buffer externalization) — can land before or after §9's
restructuring; if §9 lands first, step 2's "skip none axes" also feeds the
exact `scratch_elements()` max (decision 3, bonuses). Whichever lands
second must re-run the other's tests.

**Optional step (either phase)**: replace the `std::vector` engine list
with the `D`-bounded inline-capacity container of §10.1 item 9 —
independent of the direction feature proper, but this is the natural moment
since the engine-construction loop is being touched anyway.

**Optional follow-up benchmark** (not part of the feature): batched 1-D via
`{none, f}` vs FFTW's advanced (`fftw_plan_many_dft`) interface, same
methodology as `benchmark/algorithms_fft.cpp`.

### 10.4 Standing requirements restated (maintainer: "even if redundant")

These repeat §9.2/§8 on purpose, so an implementer reading only §10 cannot
miss them; where a point *updates* an earlier section, that is called out.

- **(a) No execution buffers in the plan.** The plan/engines hold no
  ping-pong or scratch storage (`buf_`/`out_`/`xbuf_` in the old layout) —
  only the required *sizes* (element counts). All scratch is materialized
  at `execute()` time. This is the §9.2 externalization; the direction
  feature must not reintroduce any of it (and prunes it further: `none`
  axes contribute nothing to the scratch max).
- **(b) `execute()` takes allocator parameters — and now also gets a
  defaulted overload.** This *settles* §9.2's "decision (2)", which had
  left the no-allocator-passed case open: the maintainer has since decided
  there should be an overload with defaults. The safe shape: the defaulted
  overload constructs a fresh stateless `std::allocator<T>` (for `T`
  deduced from the executed array) *per call* — never a plan-owned
  allocator or shared resource, which would reintroduce both the
  T-dependency and the concurrent-execute hazard §9.2 documents. Callers
  with allocation pressure pass their own (arena/monotonic/pool, or a
  device allocator — see (c)).

  **Landed** (`execute()`'s `Allocator` template parameter, defaulted;
  `fft_scratch_arena` retemplated on it, no rebinding — see the commit for
  the rationale). Recorded here for whoever benchmarks against the task-1
  per-call-allocation overhead: two patterns were evaluated for a caller
  wanting to avoid paying for allocation on every `execute()` call, both
  verified empirically rather than assumed —
  - A hand-rolled **LIFO stack allocator** (push on `allocate`, pop on a
    matching `deallocate`) is the correct minimal fit for
    `fft_scratch_arena`'s own access pattern (exactly one `allocate()` then
    one matching `deallocate()` per call, never interleaved with anything
    else through the same allocator instance). Deliberately not a
    single-slot "always return the same buffer" allocator — that would
    silently *alias* two logically distinct live allocations under any
    nested-in-time usage instead of catching the misuse.
  - **`std::pmr::unsynchronized_pool_resource`** (or `synchronized_` for
    the thread-safe form) genuinely reclaims on `deallocate()` via
    per-size-class free lists: measured 100 allocate/deallocate cycles of
    the same size costing only 3 upstream allocations (one-time warm-up),
    then fully stable. More general than the stack allocator (any
    deallocation order, not just LIFO) at more per-call overhead — a
    reasonable default recommendation for callers who'd rather not write a
    custom allocator.
  - **`std::pmr::monotonic_buffer_resource` does NOT give this property**,
    despite the "reuse" intuition its name suggests — its `deallocate()`
    is a documented no-op (it only ever grows, releasing everything at
    once on destruction). Measured: 20 allocate/deallocate cycles of the
    same size into a pool sized for exactly one triggered 5 separate
    upstream allocations. Do not recommend this one for repeated-execute()
    reuse.
- **(c) Keep the code shaped for a GPU (CUDA + Thrust) implementation.**
  §8 has the adaptation plan; the concrete implications for anyone touching
  the plan/engine/execute structure now:
  - the allocator-on-execute design of (b) is the GPU entry point for
    memory: a `thrust::device_allocator`-style allocator (or one wrapping a
    CUDA stream-ordered pool) must be passable where `std::allocator` is,
    so scratch acquisition must go through the allocator abstraction only —
    no raw `new`/`malloc`/`std::vector` for execute-time scratch;
  - keep the §7 design boundary intact: N-D orchestration in idiomatic
    Multi (memory-space-agnostic), numeric kernels behind a dispatch seam
    where device kernels can be substituted per §8 step 2 (dispatch on the
    pointer/cursor's memory space);
  - twiddle tables (`TW`-typed, per §10.1 item 8) are host-built today;
    nothing in this feature may assume they are addressable from kernel
    inner loops in any way that would block a later device-side copy
    (i.e. keep table *access* funneled through the engine, not scattered
    raw pointer arithmetic in new code);
  - the direction schedule (`dirs_`) is trivially-copyable plain data by
    construction — keep it that way (it must be shippable to a device or
    captured by a lambda without host references).

### 10.5 Code tricks and operational details for the executor

Concrete facts verified against the current file, plus the non-obvious
tricks. **Update, post-§9.2 (both scratch-externalization and T/TW split
have now landed)**: the anchor map below predates that work and is stale in
one important way beyond line numbers — the *shapes* changed, not just
positions. Corrected facts, still **re-grep every anchor before relying on
it** (§10 itself hasn't landed yet, so more churn is expected):

- **`fft_plan<D, TW = std::complex<double>>`, not `fft_plan<T, D>`** (order
  reordered from the plan originally described here — see §9.2 item on
  `fft_plan<D, TW>`). Members: `sizes_` (`std::array<size_t, D>`, used by
  `fft_view_from_cursor` on every execute — must keep entries for ALL axes
  including `none` ones), `engines_` (`std::vector<detail::fft_engine<TW>>`,
  one per distinct `(length, direction)` pair — Phase A landed), `which_`
  (axis → engine index, or the `no_engine_` sentinel for a `none` axis),
  `dirs_` (the per-axis schedule — landed). `execute()` is now itself where
  `T` (the array's element type) is deduced, via `typename
  std::decay_t<Cursor>::element` — `apply_`/`transform_axes_` are
  templates on `T`, taking the execute-time arena as `T* arena`.
  `execute()` also takes an `Allocator` template parameter (default
  `std::allocator<T>`, deduced/defaulted from `Cursor`), threaded to
  `fft_scratch_arena<T, Allocator>` — landed, see §10.4(b).
  `scratch_elements()` is a public accessor now (needed by any caller
  sizing their own arena).
- **`fft_engine<TW>`, not `fft_engine<T>`.** Every data-touching method
  (`run`, `run_fused*`, `run_stages_`, `run_sixstep_`, `run_bluestein_`, all
  `stage_*_` kernels) is now `template<..., class T>`, with `T` deduced from
  the data-pointer arguments, independent of the enclosing `TW`. Any new
  direction-aware dispatch parameter (§10's `Backward`) slots in alongside
  `Batched`/`T` the same way. `engine_<A>()` (compile-time axis accessor,
  `static_assert`s the range) — now also asserts axis `A` is not `none`
  (which_[A] != no_engine_) at runtime in debug builds; callers must still
  check `dirs_[A]` before calling.
- `apply_()` (post-simplification shape): one fast-path check — both of
  the last two axes active → `fft_apply_last_pair` + `transform_axes_<1,
  D-2>` over axes 0..D-3; otherwise `transform_axes_<0, D-1>` walks ALL
  axes (order D-1, 0, 1, ..., D-2), skipping `none` per axis. There is no
  D == 1 special case and no degraded-pair branch anymore; direction
  dispatch for Phase B's `Backward` selection slots naturally into the
  walk (one place) plus the pair call (one place).
- `fft_engine<TW>::fft_engine(std::size_t nn, int sign)`; stage kernels
  `stage_radix2_/4_/8_/3_/5_/generic_` dispatched by a runtime
  `switch(st.kind)` inside `run_stages_<Batched, T>` and the fused variant
  `run_fused_impl_<Batched, T>`.
- **`fft_ops<T, TW = T>` customization point (two type parameters now)**:
  `mul(TW const& w, T const& x) -> T`, twiddle first. Generic default widens
  `x` to `TW`, multiplies, narrows the result once; the
  `fft_ops<std::complex<R1>, std::complex<R2>>` partial specialization keeps
  the explicit 4-mul/2-add form for ALL complex pairings, same-type and
  mixed alike — an earlier same-type-only (`<complex<R>, complex<R>>`)
  version routed the mixed T ≠ TW case through the generic default's
  `operator*`, i.e. a `__muldc3` libcall in every twiddle multiply of the
  mixed path (found and fixed in a review pass; verified by counting
  libcalls in generated assembly). `fft_mul(w, x)` free-function convention is
  twiddle-first everywhere in the smooth-path kernels *except* one spot in
  `run_sixstep_`'s twiddle-transpose step, which had data-first and was
  fixed to match during the T/TW split — a real (if harmless pre-split)
  bug, worth knowing about if you're conjugating/direction-templating that
  call.
- Tests: `test/algorithms_fft.cpp`, Boost lightweight_test (`BOOST_TEST`,
  `return boost::report_errors();`), helpers `dft_reference(in, sign)`
  (1-D, reference direct DFT), `max_abs_diff`, `tol = 1e-9`. The 2-D
  reference pattern (apply `dft_reference` per row, then per column of the
  running result, ~lines 140–165) is exactly the pattern to imitate for
  mixed-direction references. Picked up by the `test/*.cpp` CMake glob.

**The uniform-conjugation invariant (Phase B's key enabler, verified):**
every direction-dependent value in every smooth-path kernel is *loaded from
a table* — the per-stage twiddles, the fixed roots `w1c`/`w2c`/… in
radix-3/5 (`tw_[n3]`, `tw_[n5]`, …), the ±i constant `imu` in radix-4/8
(`tw_[q]`, `tw_[2*q]` — the comments even say "-i for forward, +i for
backward"), and the direct-prime `wmat_` (built purely from `tw_`). There
are NO direction-dependent literals or sign-flipped constants in kernel
bodies. Therefore: conjugate every table load uniformly and the entire
smooth path is direction-correct. Corollary pitfall: do not "optimize" by
special-casing `imu` or the fixed roots out of the conjugation — they need
it exactly as much as the loop twiddles.

**The fused conjugate-multiply trick (zero extra instructions):** don't
materialize `conj(w)` then multiply. Add a sibling to the (now two-type)
`fft_ops<T, TW>::mul` — `w` is the TW-typed table operand, `x` the T-typed
datum, exactly as `mul` has them:

    conj_mul(w, x) == mul(conj(w), x)   // -> T; conjugates ONLY the table operand
    // fft_ops<complex<R1>, complex<R2>> specialization — same 4 mul + 2 add
    // as mul, two signs flipped, same promoted-type widening:
    { (wr*xr) + (wi*xi),
      (wr*xi) - (wi*xr) }

(generic fallback: `mul(conj(w), x)` with ADL `conj`). Then a tiny
compile-time selector, e.g. `fft_mul_dir<bool Backward>(w, x)` → `mul` or
`conj_mul`, and the kernel diff is mechanical: every `fft_mul(table_value,
datum)` becomes `fft_mul_dir<Backward>(table_value, datum)`. Convention to
keep: the *table* operand is always the first argument (already true
everywhere today) — conjugation must apply to that operand only.

**Plumbing the sign down — landed as designed.** `bool Backward` sits
alongside `Batched` on `run_stages_`/`run_fused_impl_`/`stage_subplan_` and
the stage kernels (`<Batched, Backward, T>`); the runtime `switch(st.kind)`
is untouched. `run_sixstep_`/`run_bluestein_` are also templated on
`Backward` (not just the stage kernels) so their own direction-dependent
table loads (the six-step transpose twiddle; Bluestein's chirp/postc/
kernel-table selection) get the same one-instantiation-per-pass treatment.
The engine's public entries (`run`, `run_fused`, `run_contig_inplace`) take
a runtime `bool backward` and dispatch ONCE per invocation to the
`<..., Backward>` instantiation.

**Bluestein specifics (Phase B) — landed.** The constructor now builds
*one* neutral convolution sub-engine (`sub_.emplace_back(conv_n_)`, no
sign), run twice from `run_bluestein_`: forward (`<false>`) to produce
`yf`, then backward (`<true>`) on the pointwise product to produce `zc` —
regardless of the OUTER transform's own direction, which only affects
chirp/postc conjugation and kernel-table selection, never the conv
mechanism itself. The precomputed spectrum obeys `kernel_ft_backward[k] ==
conj(kernel_ft_forward[(N-k) mod N])` (FFT of a conjugated sequence =
conjugated, index-reversed FFT) — the index reversal is why plain
conj-on-load doesn't work for `kernel_ft_`; landed as a second table
(`kernel_ft_bwd_`), computed by this closed form at construction (no extra
engine run needed) rather than the direction-keyed-Bluestein-engines
fallback. `chirp_`/`postc_` are plain elementwise conjugates (no reversal),
so `fft_mul_dir<Backward>` handles them directly, no second table.
**Aliasing consequence of the one-engine collapse** (verified under ASan,
both stage-count parities — n=101 gives 4 stages/even, n=331 gives 5/odd):
the second run's default input region, `conv.buf_ptr(arena)`, can be the
exact SAME memory as `yf` (the first run's result) when the engine's stage
count is even. Safe regardless: the pointwise-product write `z[i] =
f(yf[i])` only ever reads and writes the SAME index, so full aliasing
between `z` and `yf` across the whole array is not a hazard (documented at
`run_bluestein_`'s definition in fft.hpp).

**Wrapper deduction trick (why the argument order is dirs-first, and how
`{{...}}` compiles):** a braced-init-list is a non-deduced context, so
`std::array<fft_direction, DD>` can never deduce `DD` from `{{f, none}}`.
Make the dirs parameter's type *dependent on the array argument's rank*
(e.g. `std::array<fft_direction, rank_of<Arr>>` where `Arr` deduces from
the second parameter): deduction succeeds from `arr` alone, the dirs type
is then fixed, and the braced list just initializes it. Do NOT fall back
to `std::initializer_list<fft_direction>` — that compiles everywhere but
demotes the arity check to runtime, violating §10.1 decision 2. (For the
`fft_plan` constructor there is no issue: `D` is the class parameter.)

**C++17 constraints and naming collisions:** the library is C++17 — no
`using enum` for ergonomics. `enum class fft_direction { forward = -1,
none = 0, backward = +1 }` does not collide with the existing
`inline constexpr int fft_forward/fft_backward` (different scopes), but
users must then spell `multi::fft_direction::forward` etc.; if shorter
spellings are wanted, add distinct `inline constexpr fft_direction`
constants (NOT named `fft_forward`/`fft_backward` — those names are taken
by the `int` constants and must stay for API stability).

**Known trait footgun (relevant when adding overloads):**
`detail::fft_is_multi_like` (line ~142) is satisfied by `multi::extents_t`
too, not just arrays — it already cost one debugging session (a vestigial
SFINAE guard rejecting legitimate extents arguments had to be removed). If
new overloads need to distinguish "directions array" / "extents" / "multi
array", dispatch on something structural (e.g. the element type being
`fft_direction`) rather than trusting that trait.

**Build/verify commands (as used throughout this project):**

    # strict test build (must stay green after every step):
    g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
        -Wsign-conversion -Werror -Iinclude \
        test/algorithms_fft.cpp -o /tmp/fft_test.x && /tmp/fft_test.x

    # benchmark (only on an idle, unthrottled machine; see the
    # COMPILATION_INSTRUCTIONS comment at the top of the file):
    #   benchmark/algorithms_fft.cpp

Bit-identity tests for `none` axes must use exact equality (`==` on
elements / memcmp semantics), not `tol` — the guarantee in §10.1 item 6 is
"never written", not "numerically close".

## 11. Algorithm-ification survey — raw loops → named algorithms (2026-07-10)

Maintainer-requested survey (this section is the deliverable; the implementer
— Sonnet — executes from it). Task: replace raw loops with *named* standard
algorithms where a good fit exists, so that (a) the code reads conceptually
and (b) execution policies can slot in later. Constraints set by the
maintainer, all binding:

- `std::for_each` is allowed but is the **last resort** — prefer algorithms
  that name the operation (`transform`, `copy_n`, `reverse_copy`, `find_if`).
- No SIMD intrinsics, no `std::simd` (except, far future, as a localized
  last resort). Vectorization must come from the compiler. The preference
  ladder (recorded 2026-07-10): stride-1 layouts first, algorithms that can
  later take `unseq` second, other compiler hints (`__restrict`, no
  `-ffast-math`) third.
- No behavior change: every rewrite in Tier A below is an order-preserving
  elementwise operation, so outputs must be **bit-identical** before/after
  (see §11.7). Anything that would reassociate a floating-point reduction
  (e.g. `transform_reduce` on stage_generic_'s t-sum) is out of scope
  without explicit maintainer sign-off.
- No perf regressions (protocol in §11.7). Anchors below are as of commit
  9808b00df — **re-grep before editing**, the file drifts.

### 11.1 The C++17 policy trap (read first)

`std::execution::unseq` is **C++20** (P1001R2), not C++17. C++17's
`<execution>` has only `seq`/`par`/`par_unseq`, and the parallel ones imply
threading (maintainer: not now) plus a TBB link dependency on libstdc++.
Therefore: land the algorithm *shapes* now with **no policy argument** —
policy-free `std::transform`/`std::copy_n` inline to code identical to the
raw loops (verify per §11.7), and the conceptual win is immediate. Do NOT
add `par`/`par_unseq` anywhere.

**`unseq` itself tried and rejected (2026-07-11), on three independent
grounds — do not re-attempt without new evidence on all three:**

1. **Doesn't exist on libc++ at all**, in `-std=c++17` OR `-std=c++20`
   (`clang++ -stdlib=libc++`, LLVM 18, current as of this test): "no member
   named 'unseq' in namespace std::execution" -- a hard compile error, not
   a missing-symbol link error. libc++ has never implemented
   `<execution>`'s parallel-algorithm overloads. Shipping `unseq` in this
   header would break every clang+libc++ consumer outright (a real,
   commonly-chosen toolchain -- e.g. most macOS clang, many Linux users who
   opt in), not degrade gracefully.
2. **Requires linking `-ltbb` even for `unseq` alone on libstdc++** --
   verified: `std::transform(std::execution::unseq, ...)` compiles but
   fails to LINK ("undefined reference to
   tbb::detail::r1::execution_slot") without `-ltbb`, even though `unseq`
   itself does no threading. libstdc++ dispatches every execution-policy
   overload (including the purely-vectorization-hint ones) through the
   same PSTL backend, which needs TBB's symbols regardless. This is a new,
   mandatory external link dependency for a currently zero-dependency
   header-only library.
3. **Measured SLOWER, not faster, on libstdc++** (the one toolchain where
   it even runs) -- three scenarios, `-O3 -march=native`, idle machine, 3
   reps each averaged over 500-3000 iterations:
   - Bluestein A1 loops (run_bluestein_'s three `std::transform`s) at
     `m == 1` (the common 1-D case): unseq ~40 -> ~46 us/rep, **+15%
     slower**.
   - Same loops at `m == 64` (Bluestein as a sub-transform of a batched
     2-D `{none, forward}` plan, n=1009 prime, batch=64 -- the case with
     genuine per-row vector width to exploit): ~2469 -> ~2934 us/rep,
     **+19% slower**.
   - `fft_exec_slab`'s A4 `std::copy_n` calls (mt=64 batch-near
     gather/scatter): no measurable difference either way (noise-level) --
     a plain memory copy has nothing left for a vectorization hint to add.
   Plausible cause: these ranges are short (`m` <= ~64) and the plain,
   policy-free overloads already auto-vectorize fully under `-O3
   -march=native` (verified by the Tier A bit-identity+perf-neutral
   checks) -- so `unseq` adds only the policy-dispatch/PSTL-backend
   indirection cost with no vectorization gain left to capture.

   **Full-suite confirmation (2026-07-11, maintainer requested "try
   everywhere, GCC-only, ignore portability"):** patched ALL 12 Tier A
   call sites (A1, A2, A3, A4, A5, A7 -- every `std::transform`/`copy_n`
   this section landed) with `std::execution::unseq` and re-ran the full
   benchmark suite (1D/2D/3D + both `many` sweeps) against the unpatched
   baseline, same machine, same idle/AC protocol, no drift warning either
   run. Mean delta (mine MFLOPS, unseq vs baseline): 1D **-4.9%**, 2D
   **-1.3%**, 3D **+0.9%** (noise), many(h32) **+0.1%** (noise),
   many(h256) **-0.6%** (noise). Never a clear win anywhere; 1D's clear
   loss lines up exactly with the isolated Bluestein numbers above (1D has
   the most prime/Bluestein-forcing sizes in the sweep). Confirms the
   single-site findings generalize -- this was not a fluke of the two
   sites originally tested.

Conclusion: `unseq` is a dead end for this file on all three axes
(portability, dependency-freedom, and actual speed) with the current
compilers/stdlibs. Compiler auto-vectorization of the plain, policy-free
Tier A algorithms remains the only viable lever per the maintainer's
SIMD policy (memory: `fft-simd-policy`) -- look for stride-1/layout wins
(§11.6 W1) or restructure loops to expose more to the optimizer, not
execution policies.

### 11.2 What is already algorithmic (no work)

`std::copy` (fiber gather/scatter in `fft_exec_fiber`, `run`'s n<2 path,
`run_contig_inplace`'s tail, `stage_subplan_`'s contiguous fast path,
`run_sixstep_`'s uout copy), `std::fill` (Bluestein zero-pad, bootstrap),
`std::find_if` (plan ctor engine reuse), `std::sort`/`min`/`max`/`clamp`
(construction). The survey's overall finding: the raw loops that remain are
mostly the ones C++17's std *cannot* express — which is exactly what
motivates the wishlist in §11.6.

### 11.3 Tier A — implement now (bit-identical, C++17-clean)

Hot-path items first. For each: anchor grep key → current shape → change.

**A1. `run_bluestein_`'s three elementwise loops** (grep
`chirp-premultiplied input`, `pointwise product with the precomputed`,
`chirp-postmultiply`; ~lines 1234-1261). Each inner j-loop is a
`std::transform` with a bound scalar; keep the outer k/q loop:

    // premultiply (per row k):
    std::transform(in + (k * m), in + ((k + 1) * m), y + (k * m),
        [c](T const& v) { return fft_mul_dir<Backward>(c, v); });
    // pointwise product (per row q) -- z may FULLY alias yf (documented
    // above the loop); std::transform explicitly permits result == first
    // for unary ops, and the buffers are otherwise disjoint (never partial
    // overlap: same arena offsets or different regions entirely):
    std::transform(yf + (q * m), yf + ((q + 1) * m), z + (q * m),
        [kq](T const& v) { return fft_mul(kq, v); });
    // postmultiply (per row k): same shape as premultiply, with postc_[k].

These are the Bluestein hot path: perf-check per §11.7 (the 5-smooth
benchmark does NOT cover Bluestein — ad-hoc timing needed). m == 1 gives
length-1 transforms; confirm inlining leaves the scalar path unchanged.

**A2. `stage_subplan_` copies** (grep `y0[j] = asrc[j]` and the scatter
`br[j] = zr[j]`; ~lines 1082-1084, 1098-1104): `std::copy_n(asrc, m, y0)`
for the t == 0 gather row; `std::copy_n(zr, m, br)` inside the scatter's
outer idx loop. (The twiddle-gather loop `yt[j] = fft_mul_dir(...)` is the
same zip shape as the butterflies — Tier C, stays.)

**A3. `stage_generic_`'s two copy loops** (grep `t == 0, twiddle == 1` and
`wrow[0] == 1`; ~lines 1034-1036, 1048-1050): both are `std::copy_n(src, m,
dst)`. Only the copies — the twiddle gather and the accumulation loop stay
raw (§11.5 C2).

**A4. `fft_exec_slab` batch-near gather/scatter inner loops** (grep
`batch axis contiguous-ish`; ~lines 1352-1359 and the scatter twin
~1375-1383): the inner j-loops copy through a Multi strided iterator —
`std::copy_n(it, mt, row)` (gather) and `std::copy_n(row, mt, it)`
(scatter). The iterator is random-access; elementwise semantics identical.
The `fiber_near` kb-blocked variants are NOT in this tier (§11.5 C5).

**A5. `init_bluestein_` cold loops** (construction; clarity-only):
- `postc_` (grep `branch-free product, same as the kernels`):
  `std::transform(chirp_.begin(), chirp_.end(), postc_.begin(),
  [inv_m](TW const& c) { return fft_mul(inv_m, c); });`
- `kernel_ft_bwd_` (grep `INDEX-REVERSED spectrum, so`): the k = 0 element
  is its own mirror; the rest is a reversed conjugate copy —

      kernel_ft_bwd_.resize(conv_n_);
      using std::conj;
      kernel_ft_bwd_[0] = conj(kernel_ft_[0]);
      std::transform(kernel_ft_.rbegin(), std::prev(kernel_ft_.rend()),
          std::next(kernel_ft_bwd_.begin()),
          [](TW const& v) { return conj(v); });

  (rbegin() is kft[N-1] → kbwd[1]; N-1 elements. The n = 101 BACKWARD test
  is the correctness gate for this one.)

**A6. `sub_index_`** (grep `auto sub_index_`): linear scan →
`std::find_if(sub_.begin(), sub_.end(), [rr](fft_engine const& e) { return
e.n_ == rr; })` — matches the plan ctor's existing idiom.

**A7. `fft_layout_from`'s element loop** (grep `sub_ext.at(i)`): two
`std::copy` calls over the tail of `ext`/`str` — but note the `.at()` there
is doing bounds documentation, not real work; drop to `std::copy(ext.begin()
+ 1, ext.end(), sub_ext.begin())` and same for `str`. Cold; cosmetic.

### 11.4 Tier B — propose, measurement- or maintainer-gated

**B1. Bluestein wrapped-kernel mirror** (grep `y[conv_n_ - j] = dj`): after
the chirp loop fills `chirp_` (that loop keeps its sequential `jsq` scan —
the incremental mod-2n update is deliberate overflow avoidance), the
wrapped kernel `y` becomes two named steps:

    std::transform(chirp_.begin(), chirp_.end(), y,
        [](TW const& c) { using std::conj; return conj(c); });
    std::reverse_copy(y + 1, y + n_, y + (conv_n_ - (n_ - 1)));

(no overlap: conv_n_ >= 2n-1 guarantees the mirrored tail starts at or
after y + n_). Reads as the math ("kernel = conj-chirp plus mirrored
tail"). Cold path; do it if the split survives review as clearer.

**B2. Slab blocked-transpose gather/scatter as Multi assignment**: the
`fiber_near` gather is conceptually `scratch_view.rotated() = slab_block`
(an `array_ref<T, 2>` over `bp` with shape [nn][mt]). Blocked on wishlist
item W1: Multi's elementwise assignment today has no cache tiling, and the
hand-written kb = 64 blocking exists because it measured faster. Only adopt
behind a benchmark comparison (2-D sweep + the many sweep).

### 11.5 Tier C — stays raw, with the reason on record

- **C1. Butterfly inner j-loops** (radix-2/3/4/5/8): P-input/P-output zip
  transform. C++17 has no zip iterator; a `for_each` over j would rename
  the loop without naming the *operation* (last-resort clause). The loops
  already vectorize (contiguous j, `__restrict`, no `-ffast-math`
  needed). Revisit when W3/W4 exist, or C++23 `views::zip` + C++26
  parallel range algorithms arrive.
- **C2. `stage_generic_` twiddle-gather + accumulation**: the gather is a
  row-scaling (BLAS dgmm shape), the accumulation is a small complex GEMM
  (OUT[u][j] = Σ_t W[u][t]·X[t][j], p ≤ 64, m ≤ 64). `transform_reduce`
  would reassociate the t-sum (forbidden). BLAS dispatch (multi has
  adaptors) loses at these sizes to call overhead and forfeits generic-T.
  Name the shapes in a comment; keep the loops.
- **C3. Six-step fused twiddle-transpose**: the `idx` recurrence is a
  sequential scan (incremental mod-n avoids a per-element div/mod — it
  exploits tw[(k1·j2) % n] being an outer product of phases), and the
  32×32 tiling is measured. The std-expressible version would split one
  fused pass into two and pay div/mod. Wishlist W2 is the algorithmic
  form.
- **C4. Construction scans**: trial-division factorization, `next_smooth_`'s
  argmin (a `min_element` over a transformed iota — not expressible in
  C++17 without materializing), the chirp `jsq` scan. Cold, inherently
  sequential or clearest as written.
- **C5. `fiber_near` kb-blocked gather/scatter**: same tiled-transpose-copy
  shape as C3 minus the twiddle; W1's territory.

### 11.6 Wishlist — truly multidimensional algorithms (for the maintainer)

None of these exist in std C++17/20/23 as parallel-algorithm shapes; all
have prior art elsewhere. Each would subsume concrete loops in this file:

- **W1. Policy-aware copy/transform between strided multidimensional
  views** — `multi::copy(policy, src_view, dst_view)` choosing loop order
  from strides and cache-tiling when the views are relatively transposed.
  Prior art: Kokkos `deep_copy`, HPTT, MKL `mkl_zomatcopy`, cuBLAS `geam`.
  Subsumes: A4's loops, B2, C5, stage_subplan_'s strided scatter. This is
  the single highest-value ask — the slab path it would own is exactly
  where the new `fftw_plan_many_dft` benchmark shows the largest gap.
- **W2. Indexed transform / multidimensional tabulate** — elementwise op
  receiving the multi-index: `multi::tabulate(view, f(i, j, ...))` and the
  fused transposed variant `B = f(indices, A_transposed)`. Covers: twiddle
  table, `wmat_` (a gather `tw_[(t·u%rr)·wr]`), chirp, and — fused with
  W1's tiling — the six-step twiddle-transpose (C3). `thrust::tabulate` is
  the 1-D special case; nothing multidimensional exists anywhere in C++.
- **W3. N-ary zip transform / "apply a small linear operator along one
  axis, batched over the rest"** — the butterfly shape (P same-shape input
  views → P output views, elementwise over the batch domain), and its
  runtime-size sibling (stage_generic_'s GEMM). Prior art:
  `thrust::zip_iterator`, numpy `einsum`/`apply_along_axis`, BLAS
  `gemm_strided_batched`, FFTW codelets.
- **W4. Execution-policy `for_each` over an index domain** —
  `multi::for_each(policy, extensions, f(i, j))`: the Kokkos
  `MDRangePolicy` analog. The (block, r) spaces of every stage kernel are
  independent 2-D iteration domains; this is the natural policy carrier
  that needs no kernel restructuring.
- **W5. Broadcasted views as algorithm inputs** — Multi already has two
  candidate primitives, and (discussed with the maintainer 2026-07-10)
  neither is zippable as-is; the refinement below is the agreed direction.
  - `.broadcasted()` (array_ref.hpp ~3249) builds
    `layout_t<2>(inner, 0, 0, 1)` — stride 0, nelems 1, unbounded leading
    extent. Degenerate as an algorithm operand: `layout_t`'s own
    `size() == nelems/stride` invariant divides by zero (the in-source
    TODO "introduce a broadcasted_layout?" acknowledges this), shape
    equality against a real [n][m] view cannot hold, the trip count can
    never come from it, and pointer-stepping iterators collapse at
    stride 0 (`begin() == end()` along the broadcast axis) unless
    iterators carry an explicit index.
  - `.repeated(n)` (array_ref.hpp ~3257) has the right *semantics* —
    finite extent restores shape equality and a usable trip count, and
    every FFT call site knows the batch width, so the unbounded form is
    never actually needed: `chirp.repeated(m)` is [m][n] with row j ==
    chirp, zippable against `in.rotated()` into `y.rotated()` with no
    rotation of the repeated view itself. BUT it is a lambda-backed
    function view (`f ^ extensions`, element access through
    `invoke_square`), not a strided view: hoisting the per-row constant
    out of the inner loop (which the hand-written kernels get by
    construction — `chirp_[k]` loaded once per row into a register) then
    depends on the optimizer seeing through the lambda layer, and a
    lambda view has no mapping onto the Thrust iterator model (§11.8),
    where a stride-0 strided view maps directly onto a
    constant/permutation iterator, trivially coalesced.
  - **Agreed refinement — the primitive to build**: realize the
    `broadcasted_layout` TODO as *finite* `repeated(n)` semantics on a
    strided view — stride 0 with a definite extent n (i.e. repeated =
    broadcasted + finite extent, as ONE strided primitive, not a
    lambda). That single form is zippable by any stride-aware W1/W3
    algorithm, hoistable by construction (a stride-0 inner axis is a
    visible loop invariant, no lambda in the way), and GPU-mappable.
    With it, all three Bluestein loops and stage_generic_'s row-scaling
    become one-liners of the form `Y = mul(chirp.repeated(m), IN)`.
  - Either way W5 stays contingent on W1/W3: a broadcast view without an
    N-ary elementwise transform to feed it into has no consumer (C++17's
    binary `std::transform` over Multi views iterates ROWS, so its op
    would have to return a whole row value — an allocation per row).

### 11.7 Verification protocol (for the implementer)

1. Usual gates: strict `-O2 -Werror` (g++ AND clang++), `-O3 -Walloc-zero`,
   ASan+UBSan, full test suite, plus `-Wfloat-equal -Wcast-align=strict`
   (both bit the CI recently — keep them in the local gate).
2. **Bit-identity harness** (the real gate for this task): build the test
   battery at HEAD and with the changes; every Tier-A rewrite is
   order-preserving elementwise, so transformed arrays must be
   byte-identical across builds for identical inputs. Minimum coverage:
   Bluestein n = 101/331/1009 forward AND backward, six-step n = 8192
   roundtrip, the non-cubic 3-D mixed case, a 2-D slab shape that takes
   the batch-near gather (column-major-ish strides). A temporary test-side
   memcmp harness is fine; do not commit it.
3. **Perf**: the 5-smooth benchmark does NOT exercise Bluestein — for A1,
   time an ad-hoc loop (n = 1009, ~2000 reps, idle machine, AC) before vs
   after; A4 is covered by the 2-D sweep and the many sweep (smoke run,
   idle-protocol only for published numbers). m == 1 shapes: confirm the
   1-D benchmark spot sizes are unchanged (length-1 transform/copy_n must
   inline away).
4. Commit message names this section (e.g. "fft: raw loops → named
   algorithms (NOTES §11 Tier A)"); note any Tier-A item deliberately
   skipped and why.

### 11.8 Thrust correspondence (why this aligns with the §8 CUDA plan)

Every Tier-A shape has a direct Thrust name, so expressing them as
algorithms now makes the §8/§10.4(c) port mechanical rather than a
rewrite: A1 → `thrust::transform` (broadcast scalar per row; or one flat
`transform` with a `permutation_iterator(chirp, counting/m)` index map),
A2/A3/A4 → `thrust::copy_n`, A5 → `thrust::transform` (+
`reverse_iterator`), tabulates (W2's 1-D cases) → `thrust::tabulate`,
butterflies (C1) → `for_each_n` over a `zip_iterator` — and the dgmm-like
row scalings have a cuBLAS analog (`cublasZdgmm`). The wishlist items W1-W4
are likewise the exact shapes a device backend wants (coalescing-aware
tiled copy, MD-range launch).

### 11.9 FMA customization point — tried and reverted (2026-07-11)

Maintainer asked whether `fft_ops` should also customize `mul_add(w, x, y)
-> w*x + y` (an accumulate-fused counterpart to `mul`), for
`stage_generic_`'s accumulation loop (`dst[j] = dst[j] + fft_mul_dir(wc,
xt[j])`, the one genuine hot AXPY/GEMV-shaped site among the Tier A
rewrites, run O(p^2) times per direct-radix stage for primes <= 64).
Implemented, measured, reverted — net loss both times, for two distinct
reasons worth recording so this isn't re-attempted blind:

- **First attempt** — naive nested `fma(wr, xr, fma(-wi, xi, y))`: chains
  TWO fma latencies onto the loop-carried accumulator `y`, instead of the
  original code's ONE (the complex product was independent of `y`, computed
  ahead of/parallel with the accumulator dependency; only the final add
  touched it). Measured **~75% SLOWER** (n=61 direct-prime stage, m=1: 3.7
  -> 6.6 us/rep; m=64 batched: 237 -> 419 us/rep). Classic FMA-chaining
  anti-pattern: fusing accumulation into a MULTI-fma chain can serialize
  work that was previously pipelined.
- **Second attempt** — corrected shallow-chain form: ONE fma per component
  computes the product *independent of y* (`fma(wr, xr, -(wi*xi))`), then a
  single plain add folds in the accumulator last (same chain depth as the
  original). Still **~13-16% SLOWER** (m=1: 3.7 -> 4.3 us/rep; m=64: 237 ->
  269 us/rep) -- root cause this time confirmed via assembly diff (`-S`,
  packed `ymm` instruction counts): explicit `std::fma()` calls measurably
  REDUCE the compiler's auto-vectorization of this loop (packed
  mul/add/sub/fma instruction count 299 -> 259 comparing before/after at
  identical `-O3 -march=native`), so it falls back to more scalar work.
  GCC's own automatic FP contraction (already confirmed present: 811
  vfmadd/vfmsub/vfnmadd instructions with NO code change, §11.1) picks
  fusion opportunities THROUGH vectorization; forcing `fma()` explicitly
  short-circuits that choice and loses more from reduced vectorization than
  it gains from fused rounding.
- Both attempts passed correctness (existing `dft_reference`-tolerance
  tests, not bit-identity -- maintainer explicitly waived that for this
  experiment) and the full strict/ASan/UBSan gate; reverted purely on
  measured performance. Removed entirely rather than left dead
  (`fft_ops::mul_add`/`conj_mul_add`, `detail::fft_mul_add_dir` all
  deleted) -- no unused speculative API surface kept around.

**Conclusion, consistent with §11.1's `unseq` finding**: for THIS file's
loop shapes (short-to-medium ranges, `m` <= ~64, already relying on the
compiler's own auto-vectorization + auto-contraction under strict `-O3
-march=native`), manually forcing a lower-level numeric strategy
(execution policies, explicit `fma()`) tends to fight the optimizer rather
than help it. The compiler already has more context (the whole loop, target
ISA, vector width) than a call-site-local `fma()` hint does. Matches the
maintainer's SIMD policy (memory: `fft-simd-policy`) directly: compiler-
driven vectorization wins here, manual numeric-strategy overrides don't.

### 11.10 `std::execution::par` — genuine win, same fatal blockers (2026-07-11)

Maintainer asked to try `std::execution::par` (real threading, not just a
vectorization hint) "in algorithms that could benefit from it." Unlike
§11.1/§11.9's negative results, this needed picking a DIFFERENT kind of
site: every Tier A execute()-path loop is short (`m` <= ~64 per call),
where thread-dispatch overhead would dominate even worse than `unseq`'s
policy-dispatch overhead did. The one genuinely good candidate: the
twiddle-table build loop (`fft_engine`'s constructor, `tw_.resize(nn); for
k in [0,nn): tw_[k] = {cos(theta), sin(theta)};`) -- large (up to millions
of elements for big 1-D sizes), expensive-per-element (real `cos`+`sin`),
embarrassingly parallel (no cross-k dependency), and CONSTRUCTION-time only
(runs once per plan, not once per `execute()`).

Patched (scratch copy only, `/tmp`, never touched the real header) to
`std::for_each(std::execution::par, tw_.begin(), tw_.end(), [step,
tw_base](TW& v) { auto const k = static_cast<std::size_t>(&v - tw_base);
... })` (index recovered via pointer difference -- C++17 has no
`views::iota`, and materializing an index vector just to feed `transform`
would add unfair allocation overhead to the comparison). Measured plan
CONSTRUCTION time (not execute()) on this 12-core machine, idle/AC:

| n | sequential | `par` | speedup |
|---|---|---|---|
| 65,536 | 894 us | 312 us | 2.9x |
| 262,144 | 3,410 us | 931 us | 3.7x |
| 1,048,576 | 15,264 us | 3,925 us | 3.9x |
| 2,097,152 | 39,763 us | 18,679 us | 2.1x (six-step sub-engine construction, unparallelized, starts dominating at this size) |

Correctness confirmed (full test suite, tolerance-based). **This IS a real
win** -- a genuinely different result from §11.1/§11.9, because this site
has the profile parallelism actually wants (large N, real per-element
work, one-shot) rather than fighting the compiler's own vectorization
choices on tiny per-call ranges.

**Not adopted, for the same reasons as §11.1's `unseq` rejection --
unchanged by this result:** `std::execution::par` is the SAME `<execution>`
facility with the SAME two blockers: doesn't exist on libc++ at all (hard
compile error, not degraded-but-working), and requires linking `-ltbb` on
libstdc++ even to use it. A genuine speedup doesn't override "breaks every
clang+libc++ consumer outright" for a header-only, zero-dependency
library. Also worth weighing even if portability were solved: this is
CONSTRUCTION-time, not execute()-time -- the library's whole design
(§9.2, "plan once, execute many times, cheaply") already treats
construction cost as off the hot path; a 2-4x win here only matters for
callers who build many large plans or build large plans on a latency-
sensitive first call, not for the steady-state repeated-execute() pattern
the benchmarks in this file otherwise measure. If `<execution>` portability
is ever solved (e.g. a future std baseline bump, or a build-time opt-in
`-DBOOST_MULTI_FFT_ALLOW_PAR` gated behind `__cpp_lib_execution` AND an
explicit TBB-link acknowledgment from the consumer), this specific site --
and its likely sibling, Bluestein's `chirp_`/wrapped-kernel construction
loops, also O(n) trig, also construction-time, not separately measured
here -- would be the ones to revisit first.

### 11.11 `std::execution::par` INSIDE execute() — tried, catastrophically worse (2026-07-11)

Follow-up to §11.10 (which only tried `par` at plan CONSTRUCTION time):
does `par` help any of the Tier A EXECUTE()-path sites themselves? Patched
A1 (Bluestein's three `std::transform`s) and A4 (slab `copy_n`
gather/scatter) with `std::execution::par` (scratch copy only, `/tmp`, real
header untouched) and re-ran the exact same microbenchmarks as §11.1's
`unseq` test, for direct comparison:

| site | baseline | `unseq` (§11.1) | `par` |
|---|---|---|---|
| Bluestein, m=1 (n=1009) | ~40 us/rep | ~46 us/rep (+15%) | **373 us/rep (9.3x slower)** |
| Bluestein, m=64 batched | ~2469 us/rep | ~2934 us/rep (+19%) | **23637 us/rep (9.6x slower)** |
| slab `copy_n`, mt=64 | ~68 us/rep | ~67 us/rep (noise) | **1814 us/rep (~27x slower)** |

Correctness confirmed (full test suite). As predicted before measuring (and
now confirmed rather than assumed): real thread-pool dispatch/
synchronization overhead per `par` call is enormous relative to the actual
work in these ranges (a few dozen complex multiplies, `m` <= 64) --
dramatically worse than `unseq`'s policy-dispatch-only overhead, because
`par` actually spins up/synchronizes worker threads per call, not just
picks a vectorization strategy. Every Tier A execute()-path site is a
per-call, short-range loop (`m` <= ~64) -- the exact opposite of what
threading wants (large N, amortized dispatch cost). No further per-site
exploration inside execute() is warranted: the mechanism (real threading
overhead on tiny per-call ranges) generalizes to every remaining
unconverted Tier A/Tier C site too, all of which are the same
`m`-bounded shape or smaller.

**Where `par` DOES help (§11.10, unchanged): construction-time-only, large-N
loops** (the twiddle table; likely Bluestein's chirp/wrapped-kernel
construction, not separately measured) -- fundamentally different profile
from anything reachable inside `execute()`'s per-call hot path. Combined
picture for `<execution>` policies in this file: `unseq` is a wash-to-
slight-loss everywhere (§11.1); `par` is a large win at construction time
and a catastrophic loss at execute time. Both remain unshippable regardless
(libc++ non-support + libstdc++'s `-ltbb` requirement, §11.1) -- this
section is about WHERE the mechanism would help IF that were solved, for
if/when it ever is.

### 11.12 `-ffast-math` — tried, net regression, confirms the existing ban (2026-07-11)

Maintainer asked to build with `-ffast-math` and compare against the
baseline, purely to check -- this is the flag the project's own benchmark
header already documents as deliberately never used ("the fft_ops
customization point exists specifically to get vectorized performance
under strict IEEE semantics... relaxing that here would make the
comparison less representative of how the library actually ships").
Result confirms that stance was correct, empirically:

- **Correctness**: essentially unaffected on representative sizes (smooth
  n=1024, Bluestein prime n=1009, six-step n=8192) -- max relative error
  vs a reference DFT stayed at the 1e-12 to 1e-11 level under both strict
  IEEE and `-ffast-math`, no meaningful precision loss detected by this
  check. (Not exhaustive -- adversarial inputs, e.g. near-cancellation
  cases, weren't probed.)
- **Speed**: full benchmark suite (1D/2D/3D + both `many` sweeps),
  `-ffast-math` added to the existing `-O3 -march=native -mtune=native
  -funroll-loops -fno-math-errno` build, same idle/AC machine, clean run
  (no calibration-drift warning). Mean delta vs the no-`-ffast-math`
  baseline: 1D **-19.8%**, 2D **-11.1%**, 3D **-9.3%**, many(h32) **-10.1%**,
  many(h256) **-9.1%** -- a NET REGRESSION on every sweep, not an
  improvement. Per-size range: mostly losses (worst case -59.5% on a 1D
  size), with a handful of small wins scattered in 2D/3D/many (up to
  +17.8%) -- i.e. unpredictable and size-dependent, the downside
  materially larger than the upside.
- Consistent with §11.9's FMA finding and §11.1's general pattern this
  session: this codebase's kernels are already well-matched to the
  compiler's DEFAULT (strict-but-contracting) auto-vectorization for these
  loop shapes, and further "help" via aggressive reassociation
  (`-fassociative-math`/`-freciprocal-math`, part of `-ffast-math`) tends
  to change the vectorizer's instruction-selection choices for the worse
  here, the same mechanism (not just a coincidence) as explicit `fma()`
  reducing packed-vector instruction counts in §11.9.

**Conclusion: stays banned**, now on THREE independent grounds instead of
one -- the existing "keeps the benchmark representative" argument, the
already-known accuracy-relaxation risk `-ffast-math` carries in general,
and now a measured, repeatable, net SPEED regression on this specific
codebase. Not a close call.

### 11.13 Split-radix — implemented, correctness-verified, reverted on speed (2026-07-11)

Maintainer-requested "mixed-radix" implementation, scoped (after
clarification) to the "surgical split-radix" option: a real recursive
N/2 + N/4 + N/4 decomposition (Yavne 1968 / Duhamel-Hollmann 1984 combine)
as a NEW top-level engine mode, parallel to how Bluestein and six-step
already work -- chosen at construction time for pure powers of two in
`[16, fft_sixstep_min)`, via recursive delegation to sub-engines rather
than a `stages_` list entry (split-radix's asymmetric per-level split
doesn't factor into "a list of uniform radices" the way radix-4/8/2 do --
see the discussion that preceded implementation).

**Design** (for anyone revisiting this): `init_splitradix_()` builds three
sub-engines -- one length-n/2 ("even" stream E) via the normal
`sub_index_` dedup, and TWO length-n/4 ("odd" streams O1 from x[4m+1], O3
from x[4m+3]) as DISTINCT `sub_` entries (NOT deduped), because unlike
Bluestein's single-neutral-engine collapse (§10.5, two SEQUENTIAL calls),
split-radix's combine step needs O1[k] and O3[k] alive SIMULTANEOUSLY --
same aliasing hazard six-step's n1==n2 case already guards against, same
fix. `run_splitradix_` gathers the three strided sub-sequences, runs each
sub-engine, then combines via:

    X[k]             = E[k] + U + V
    X[k + n/2]       = E[k] - (U + V)
    X[k + n/4]       = E[k + n/4] - i(U - V)
    X[k + n/4 + n/2] = E[k + n/4] + i(U - V)

for k in [0, n/4), U = W_n^k·O1[k], V = W_n^{3k}·O3[k]. Uses `tw_[k]`,
`tw_[3k]` (never needs a modulo: 3k < n_ always for k < n/4), and `tw_[n/4]`
for the "±i" combine -- the same table entry and `fft_mul_dir` trick
`stage_radix4_`/`stage_radix8_`'s `imu` already use, so the existing
uniform-conjugation invariant (§10.5) extends unchanged to backward.

**Bug found and fixed during implementation** (real lesson, not just
process): `tw_[n/4]` evaluates to `-i` (forward), not `+i` -- the
initial code computed `neg_i_diff = fft_mul_dir<Backward>(tw_[quarter],
diff)` intending it to MEAN `+i*diff`, then wrote `ekq - neg_i_diff` /
`ekq + neg_i_diff`, which is backwards. Symptom was a clean SWAP of
`X[k+n/4]` and `X[k+n/4+n/2]` (each held the other's correct value) --
caught immediately via a standalone n=16 vs `dft_reference` debug dump,
not by guessing from the failing test suite's aggregate output. Fix:
either flip the +/- at the call sites, or rename to make the sign explicit
(`neg_i_diff`) and swap the following two lines accordingly (done).

**Correctness, fully verified after the fix**: every power of two in
scope (16 through 4096) forward AND backward against `dft_reference`
(1e-14 to 1e-15 relative error, scaling sanely with size), a batched
(m>1, via a `{none,forward}` 2-D plan) check at 5 fiber sizes, and --
notably -- Bluestein's own n=1009 case (whose internal `conv_n_=2048` now
ALSO recurses through split-radix, confirmed via a standalone probe: this
is why the bit-identity harness showed exactly ONE divergence, at n=1009,
after this landed -- an EXPECTED consequence of an internal algorithm
swap, not a bug, and the tolerance-based Bluestein tests already
confirmed it stayed correct). Full gate green: g++/clang++ strict, `-O3
-Walloc-zero`, ASan+UBSan (both the full suite and the standalone
exhaustive sweep) -- no aliasing bugs surfaced, confirming the
distinct-sub-engine scratch design was right.

**Benchmarked and REVERTED -- 50-80% SLOWER, not faster:**

| size | before | after (split-radix) | delta |
|---|---|---|---|
| 1D n=128 | 1910 MFLOPS | 548 | -71.3% |
| 1D n=1024 | 4729 | 1024 | -78.3% |
| 1D n=4096 | 5990 | 1228 | -79.5% |
| 2D n=128 | 7196 | 2987 | -58.5% |
| 2D n=1024 | 7393 | 2517 | -66.0% |
| 3D n=64 | 8014 | 3478 | -56.6% |
| 3D n=256 | 7507 | 3182 | -57.6% |

Uniformly, dramatically worse across every size and dimensionality tested
(idle/AC machine, clean run, no drift warning). **Root cause**: split-
radix's real ~20% fewer-FLOPs algebraic advantage is completely swamped by
the cost of the RECURSIVE DELEGATION strategy chosen to implement it.
Every recursion level (and split-radix recurses `log4(n)` levels deep,
since each half/quarter sub-engine also qualifies and recurses further)
pays its own STRIDED gather (stride-2 for the even stream, stride-4 for
the two odd streams) plus a combine pass -- real extra memory traffic at
EVERY level, reintroducing exactly what the existing Stockham autosort
engine was carefully designed to AVOID (no gather/scatter between internal
stages, just ping-pong buffer swaps within one flat, cache-friendly
iteration). This was flagged as the likely integration cost BEFORE
implementation began (see the design discussion this section's intro
refers to: "a genuinely different (recursive, or specially-iterativized)
engine structure... not a stage-list addition") -- confirmed, and the
magnitude (worse than every other rejected optimization this session,
including `std::execution::par` misused inside `execute()`) settles it.

**Reverted completely** (all of: member state, constructor branch,
`note_reach_` branch, `run()` dispatch, `build_tw_`/`init_splitradix_`
helpers, `run_splitradix_` itself) -- confirmed via `git diff` showing
zero difference against the pre-experiment commit, and the bit-identity
harness confirming exact restoration.

**What WOULD be needed for this to pay off** (for whoever revisits): a
genuinely ITERATIVE split-radix, integrated into the SAME flat
`stages_`-list-plus-ping-pong-buffer machinery the rest of this engine
uses -- i.e. expressing the N/2+N/4+N/4 split as one or more STAGE KINDS
processed in-place within the existing autosort loop (the way `stage_
radix8_` is already "two fused radix-4 sub-butterflies plus a combine,"
just with an asymmetric split instead of a uniform one), not as recursive
sub-engine delegation. That is a substantially larger design task --
closer in scope to Phase B's direction-neutral rework than to anything
attempted this session -- and wasn't attempted here.

### 11.14 `perf` profiling: where the cycles actually go (2026-07-11)

Per `fft.PROFILE-TASK.md`: measurement only, no product-code changes.
Harnesses (`prof_mine.cpp`/`prof_fftw.cpp`, both in the session
scratchpad, never the repo) build a plan once, warm up once, then loop
`execute()` (mine) or `fftw_execute()` (FFTW, `FFTW_MEASURE`) back-to-back
with no cache flush and no reload between reps -- deliberate, for perf
sample density. **This means the cyc/point ratios below are NOT the same
measurement as the official flushed-cache benchmark's `%-of-FFTW`
figures** (`benchmark/algorithms_fft.cpp`, `fft_bench_*_nowisdom.dat`);
they answer a different, narrower question -- "given hot caches and a
resident plan, where do cycles go within one binary's own execution" --
and should not be quoted as a re-measurement of the committed 55-68%
figure. Build: exactly the flags the task specifies (`-O3 -march=native
-mtune=native -funroll-loops -fno-math-errno -DNDEBUG -g
-fno-omit-frame-pointer`, no `-fno-inline`). Machine: AC power, idle,
`perf_event_paranoid=1` (user-set), package temp nominal, checked before
and after the run.

**Per-case counters** (`perf stat -d -d`, cycles/point = total cycles ÷
total output points across all reps):

| case | n / shape | mine cyc/pt | mine IPC | mine L1-miss% | mine LLC-miss% | FFTW cyc/pt | FFTW IPC | FFTW L1-miss% | FFTW LLC-miss% | ratio (mine/FFTW) |
|---|---|---|---|---|---|---|---|---|---|---|
| P1/F1 | 1D n=1024 | 26.60 | 3.23 | 8.8% | 3.7% | 7.23 | 2.71 | 11.5% | 4.4% | 3.68x |
| P2/F2 | 1D n=4096 | 32.22 | 3.11 | 19.1% | 0.2% | 11.87 | 2.50 | 23.9% | 0.0% | 2.71x |
| P3/F3 | 1D n=1,048,576 (six-step) | 211.86 | 0.74 | 26.0% | 29.7% | 188.24 | 1.86 | 11.8% | 26.6% | 1.13x |
| P4/F4 | many [256][1024], `{none,forward}` | 26.66 | 3.19 | 9.5% | 0.9% | 9.17 | 2.12 | 14.4% | 3.4% | 2.91x |
| P5/F5 | 1D n=1009 (Bluestein) | 166.93 | 2.86 | 15.5% | 1.2% | 89.72 | 3.02 | 12.2% | 0.1% | 1.86x |

Branch-miss rates were <1% for every case, both binaries -- not a factor
anywhere, omitted from the table.

**The headline pattern, P1/P2/P4**: our IPC is HIGHER than FFTW's (3.1-3.2
vs 2.1-2.7) yet we burn 2.7-3.7x more cycles per point. Higher IPC with
more total cycles means more total INSTRUCTIONS retired per output point
-- this is not a stall/cache problem for these three cases, it's an
instruction-count problem. L1/LLC-miss rates are unremarkable (single
digits to ~20%, in FFTW's own range too). This is the opposite of what a
"we're bandwidth-bound, need bigger radices" story would show (that
predicts LOW IPC + high miss rates); instead it's exactly what "FFTW's
codelets are algebraically leaner per point" predicts.

**P3 (six-step, huge n) is qualitatively different**: IPC collapses to
0.74 (FFTW: 1.86), L1-miss% nearly doubles FFTW's (26.0% vs 11.8%) -- this
one case IS memory/stall-bound. Only 1.13x FFTW's cyc/point in THIS
harness (hot-cache, back-to-back reps) -- consistent with, but not a
re-derivation of, the official benchmark's much worse flushed-cache figure
for this region; the gap between "1.13x here" and "3-4x slower there" is
itself informative (see verdict below).

**P5 (Bluestein) sits in between**: 1.86x, IPC close to FFTW's (2.86 vs
3.02) but LLC-miss% is 1.2% vs FFTW's 0.1% -- some real memory cost from
the chirp gather/convolution, but instruction count still likely the
larger term (radix-8-dominated fiber crunch, same story as P1/P2/P4's
`fft_mul_dir` cost, doubled since the conv runs forward+backward).

**Per-case self-time breakdown** (`perf record --call-graph dwarf` +
`perf report`, self-time %, non-cumulative):

- **P1/P2** (n=1024, 4096): 97.5-99.0% of all self-time is inside the one
  fused `run_fused_impl_<...>` / `stage_radix4_` inline blob (everything
  inlines into `execute()` per the task's build flags, so gather/scatter,
  twiddle load, and arithmetic are NOT separable symbols -- they're all
  the same inlined function). Within that blob, `fft_mul_dir`/`fft_ops::mul`
  (the twiddle complex-multiply itself) accounts for ~34% of self-time in
  both cases. L1-miss% for these cases (8.8%, 19.1%) is unremarkable, so
  this 34% reads as arithmetic cost (a complex multiply is 4 mul + 2
  add/sub, or 3 mul + 5 add/sub with the Karatsuba trick), not a load-miss
  hotspot on `tw_[r*tstep]` -- no distinct "twiddle load" symbol separates
  from the arithmetic to test that in isolation, but the miss-rate
  evidence argues against it being memory-bound.
- **P4** (batched many): 98.77% in the same fused kernel as P1/P2 (this
  is literally the same `stage_radix4_` template instantiation, invoked
  once per fiber). The actual per-fiber gather/scatter (`fft_exec_slab`
  itself, `fft_exec_fiber`, excluding the inlined kernel call) is **0.76%
  + 0.08% = 0.84%** of total self-time. Batched-contiguous overhead is
  negligible; P4's 2.91x gap to FFTW is the same instruction-count story
  as P1/P2, not a batching-specific inefficiency.
- **P3** (six-step): `run_stages_<true,false>`/`run_fused_impl_<true,false>`
  (the column-FFT pass) 17.65%+17.60% (same code, nested self-time,
  effectively ~17.6% of wall time), `run_sixstep_` itself (the tiled
  transpose + twiddle combine) **14.32%**, and **43.77% in unresolved
  kernel-space (`[k]`) samples** -- confirmed via a separate `strace -f -c`
  run to be `mmap`/`munmap` calls, one pair essentially per `execute()`
  call (124 `munmap` + 145 `mmap`, ~99.85% of all syscall time). Root
  cause, confirmed with a standalone probe: `plan.scratch_elements()` for
  n=1,048,576 is 6,291,456 elements = **100.7 MB** -- an allocation this
  large falls straight through `std::pmr::unsynchronized_pool_resource`
  (used both in this harness and in the official benchmark) to a raw
  `mmap`/`munmap` pair every call. The pool resource, sized for small/
  medium repeated allocations, provides **zero mitigation** at this scale.
  This is a distinct finding from anything in candidates #2-#5 below: it's
  an allocator problem, not an algorithm problem, and it's currently
  eating a plausible ~40%+ of wall time in the worst-measured region.
- **P5** (Bluestein): near-exact 50/50 split, `run_stages_<false,false>`
  21.39% vs `run_stages_<false,true>` 21.14% (plus matching pairs of
  `stage_radix8_<false,false/true>` at ~11.4% each and `run_bluestein_`
  itself at 14.28%+3.52%). This symmetry is EXPECTED, not a bug: per
  Phase B (§10.5), Bluestein's internal convolution is a single neutral
  sub-engine that always runs forward-then-backward regardless of the
  outer transform's own direction, so the two nearly-identical `false`/
  `true` template instantiations doing equal work is exactly the designed
  behavior.

**FFTW codelets/strategy per size**: F1 (n=1024) and F2 (n=4096) show
FFTW's execution spread across many small anonymous JIT-generated
codelets with no exported symbol names (perf sees raw addresses only) --
a rough diversity proxy (distinct addresses at >=0.1% self-time) counts
**353 for F1, 203 for F2, 318 for F5** -- consistent with FFTW picking a
plan built from many small, size-specialized generated kernels rather
than one generic looped routine (the qualitative basis for "FFTW's
codelets are algebraically leaner per point," matching the IPC/cyc-point
pattern above). **F3 (n=1,048,576)**, by contrast, spends its time in
FFTW's own NAMED library routines, not anonymous codelets -- top self-time
symbols:

    28.00%  fftw_cpy2d
     5.95%  fftw_cpy2d_pair
     1.30%  fftw_dft_zerotens
     1.24%  fftw_transpose
     0.55%  fftw_rdft_zerotens
     0.21%  fftw_twiddle_awake
     0.18%  fftw_tile2d
     0.07%  fftw_choose_radix

i.e. ~35%+ of FFTW's OWN time on this size is in transpose/copy machinery
(`fftw_cpy2d`/`fftw_cpy2d_pair`/`fftw_transpose`/`fftw_tile2d`) --
confirming huge-N FFTs are inherently transpose/copy-dominated for FFTW
too, not a defect unique to our six-step path. FFTW simply has more
highly-tuned copy/transpose primitives (tiled, `fftw_tile2d`-named) and,
per this profiling run, no allocator tax comparable to ours (FFTW uses
`fftw_malloc` once at plan-build time via `FFTW_MEASURE`, not per-call).

**Verdict, candidate by candidate:**

- **#2, per-stage sequential twiddle tables**: **NOTHING / WEAK-NO.**
  L1-miss% for P1/P2/P4 (8.8-19.1%) is unremarkable and comparable to or
  lower than FFTW's own miss rates on the same sizes -- no visible
  cache-miss hotspot to attribute to strided `tw_[r*tstep]` reads. The
  aggressive inlining (per the task's no-`-fno-inline` build) also means
  we cannot cleanly isolate a "twiddle load" instruction stream from the
  surrounding arithmetic to test this more precisely; what we CAN say is
  the data gives no positive signal for it, and the self-time cost that
  IS visible (`fft_mul_dir`, ~34%) reads as arithmetic, not load-latency.
- **#3, radix-16 kernel / fewer passes**: **SUPPORTED, but re-scoped.**
  P1/P2/P4 show HIGHER IPC than FFTW with 2.7-3.7x more cycles/point --
  classic signature of "more total instructions per point," which fewer,
  larger-radix passes directly reduces. This is NOT the "low IPC + high
  LLC-miss" bandwidth-bound signature the task framed as the trigger
  condition for #3 -- so the mechanism is "fewer arithmetic instructions
  per output point" (matching FFTW's many-small-specialized-codelet
  strategy), not "fewer memory passes." Still points at #3 as the right
  lever for the smooth/batched cases, just via instruction-count
  reduction rather than bandwidth relief.
- **#4, six-step rework**: **WEAKENED, and superseded by a cheaper prior
  fix.** P3's transpose+combine (`run_sixstep_`, 14.32%) plus the
  column/row-FFT arithmetic (~17.6% each) do NOT dominate the way the
  task's trigger condition ("supported iff P3's transpose dominates")
  anticipated -- the single largest attributable cost is **43.77% in
  kernel-space `mmap`/`munmap`**, an allocator artifact of one 100.7 MB
  one-shot allocation defeating the pool resource, not an algorithmic
  defect in the transpose scheme itself. FFTW's own F3 profile confirms
  huge-N is inherently transpose/copy-heavy for FFTW too (~35%+ of ITS
  self-time in `fftw_cpy2d`/`fftw_transpose`/`fftw_tile2d`), so a
  six-step rework competes against an already transpose-bound reference,
  not a transpose-free one. Re-measuring after fixing the allocator (see
  recommendation) is a prerequisite before this candidate's real
  remaining upside can even be estimated.
- **#5, batched-contiguous heuristic**: **WEAKENED.** P4's actual
  gather/scatter overhead (`fft_exec_slab`/`fft_exec_fiber`, excluding the
  inlined per-fiber kernel) is 0.84% of self-time -- essentially all of
  P4's cost is the SAME `stage_radix4_` arithmetic as the plain 1D case
  (P1), and P4's 2.91x gap to FFTW tracks P1's 3.68x gap closely. There is
  no batching-specific inefficiency to fix; whatever fixes #3 for 1D
  fixes P4 too, for free.

**Recommended next task**: fix the P3 allocator problem BEFORE attempting
any six-step algorithmic rework. Replace the per-`execute()`-call
`std::pmr::unsynchronized_pool_resource` scratch allocation (or the
caller-owned equivalent in the official benchmark) with a persistent,
reused arena -- e.g. a caller-held buffer backing a
`std::pmr::monotonic_buffer_resource`, sized once to `scratch_elements()`
and reused across calls -- for the huge-n path specifically. This is
justified purely by the numbers above: 43.77% of P3's wall-time is
currently kernel-space `mmap`/`munmap`, confirmed by both `perf` sampling
and `strace -c` syscall-time attribution to be essentially one alloc/free
pair per call. That is larger than either of `run_sixstep_`'s own two
components (17.6% column-FFT, 14.3% transpose+combine) individually, and
plausibly larger than #4's entire remaining algorithmic upside. Re-run
this same profiling case (P3 only) after the allocator fix; only then
does the six-step-transpose-vs-arithmetic split become measurable enough
to decide whether #4 is still worth attempting.

**Caveats**: F4's `perf record` call-graph was not captured (only `perf
stat` counters) -- the FFTW-side self-time breakdown for the batched case
is not available; this does not affect the P4 verdict above, which rests
on OUR OWN self-time split. The GHz-derived clock-rate column was
computed but dropped from the table above due to a units bug in the
parsing script (cosmetic; IPC and miss-rate fields, which the verdicts
depend on, parsed correctly and were spot-checked against the raw `perf
stat` text). Machine was idle/AC/cool for the full run; no drift or
throttling observed between cases.

### 11.15 P3 allocator fix, measured: monotonic arena vs per-call pool (2026-07-11)

Direct follow-up to §11.14's recommendation. Same P3 harness (1-D
n=1,048,576, six-step, 120 reps + 1 warm-up), same build flags, same idle/
AC/cool machine -- only the allocator changed. Added `run_1d_arena()` to
the scratchpad harness: instead of a fresh `std::pmr::unsynchronized_
pool_resource` per call, a `std::vector<std::byte>` arena is allocated
**once**, outside the loop, sized to `plan.scratch_elements() *
sizeof(complex) + 4096` (alignment slack), backing a
`std::pmr::monotonic_buffer_resource`; `mbr.release()` is called after
every `execute()` to rewind the bump pointer to the front of the same
buffer without freeing it (per the mechanism discussed with the
maintainer: monotonic + release() never touches the upstream resource
once the initial buffer is large enough, unlike a pool, which caps what
it pools internally and delegates anything above that -- and a 100.7 MB
one-shot request is always above that cap regardless of pool tuning).

**`strace -f -c` confirms the mechanism directly**: pool version, 120
reps, showed 124 `munmap` + 145 `mmap` (essentially one pair per call,
99.85% of syscall time, per §11.14). Arena version, same 120 reps: **4
`munmap` + 25 `mmap` total for the entire process** (process/library
startup plus the one-time arena allocation) -- i.e. the per-call
allocator syscalls are gone, not reduced.

**`perf stat -d -d` result — this is not a marginal win:**

| metric | pool (old, §11.14) | monotonic arena (new) | change |
|---|---|---|---|
| wall time | 6.237s | 3.067s | **2.03x faster** |
| cycles/point | 211.86 | 104.72 | **2.02x fewer** |
| instructions/point | 156.90 | 67.67 | **2.32x fewer** |
| sys time | 2.771s | 0.051s | 54x less |
| page faults (whole run) | 1,991,049 | 32,917 | 60x fewer |
| IPC | 0.74 | 0.65 | slightly lower |
| L1-dcache-miss% | 26.03% | 60.77% | up (see below) |
| LLC-miss% | 29.67% | 23.74% | down |

`perf record` self-time confirms it lands where predicted: kernel-space
(`[k]`) samples drop from **43.77% to 2.48%** of total self-time. The
remaining self-time redistributes across the same three symbols as
before, now as a larger share of a much smaller total: column-FFT
(`run_stages_<true,false>`/`run_fused_impl_<true,false>`) ~35.7%,
`run_sixstep_` (transpose+combine) 25.29%.

**Why instructions/point dropped too, not just cycles**: this wasn't
predicted going in -- the pool resource's own free-list/chunk-search
bookkeeping costs real retired instructions in addition to the syscalls,
and (bigger effect) every fresh `mmap` hands back zero-filled virtual
pages that fault in one at a time on first touch: 1,991,049 page faults
across the pool run (~16,600 per call, close to 100MB/4KiB=25,600 pages,
partly amortized) vs 32,917 for the WHOLE arena run (the buffer is
touched once; every subsequent `execute()` reuses already-resident
pages). Minor-fault handling is real instructions +a trap, not just
kernel `sys` time, which is why `user` time also dropped (3.465s ->
3.014s) even though the FFT computation itself is unchanged.

**Why L1-miss% went UP (26.03% -> 60.77%) despite everything getting
faster**: this is a ratio, and the denominator changed more than the
numerator. Total L1-dcache-loads fell from 4.24B to 1.53B (fewer
instructions overall, per above) while L1-dcache-load-misses fell only
from 1.10B to 0.93B -- the genuine FFT-kernel cache misses (transpose,
twiddle, six-step gather) are largely fixed costs of the algorithm on
this size and didn't shrink nearly as much as the allocator-induced load
volume around them did. Read as: the pool version's L1-miss% was
ARTIFICIALLY LOW because it was diluted by a flood of cheap, mostly-
resident-page pool-bookkeeping loads; the arena version's 60.77% is
closer to the true miss rate of the underlying six-step computation
itself.

**Effect on the §11.14 candidate verdicts**: strengthens rather than
overturns them. #4 (six-step rework) remains WEAKENED for now, but the
reasoning sharpens: with the allocator artifact removed, P3's REAL
per-point cost is roughly half what §11.14's raw numbers suggested, and
the still-standing self-time split (~61% column/row-FFT arithmetic vs
~25% transpose+combine) means any six-step rework is now competing
against a smaller, more arithmetic-dominated baseline than it looked to
be facing before this fix -- worth re-deriving the "25-35% of FFTW"
figure from the official flushed-cache benchmark (§11's methodology)
with this allocator fix applied to the actual `execute()` scratch
strategy (not just this harness) before scoping any six-step work
further.

**Scope note**: this experiment only touched the scratchpad harness's
allocator usage, per the standing rule -- `fft.hpp` and the official
benchmark are unchanged. Applying this fix for real requires either (a)
`fft_plan` growing an option to own/reuse a persistent scratch arena
across `execute()` calls (a real, if small, API/lifetime design task --
`execute()` is currently `const` and stateless about scratch reuse by
design), or (b) documenting the monotonic-arena-plus-`release()` pattern
above as the recommended caller-side idiom for repeated huge-n calls,
which requires no product-code change at all. Not decided here; flagging
both options for the maintainer.

**Decision (2026-07-11): option (b).** Rejected (a) on design grounds, not
just convenience: `fft_plan` owning persistent scratch would make
`execute()` mutable and bind the plan to one allocator/type at
construction, defeating the point of a `const`, allocator-per-call plan
that's meant to be shared across callers/threads/allocator contexts. The
arena-plus-`release()` idiom is the documented fix; see §11.16 for it
applied to the official benchmark.

### 11.16 Official benchmark re-run with the arena fix, plus a new batched-2D case (2026-07-11)

Unlike §11.14/§11.15 (measurement-only, scratchpad-only), this section
covers real changes to tracked files, requested directly: (1) apply
§11.15's monotonic-arena fix to `benchmark/algorithms_fft.cpp` itself,
replacing its per-plan `std::pmr::unsynchronized_pool_resource` (whose
file-header comment claimed -- incorrectly, per §11.14/§11.15's direct
measurement -- to "genuinely reclaim on deallocate()"); (2) add a new
`sweep_many3d()` benchmark case for `{none, forward, forward}` on a 3-D
array (a full 2-D FFT per batch layer) against FFTW's rank-2
`fftw_plan_many_dft`; (3) add a matching correctness test in
`test/algorithms_fft.cpp` for the same direction combination (previously
covered: `{none,forward,none}` and `{forward,forward,none}` on a 3-D
array, §11's regression tests; NOT previously covered: both trailing axes
active with the leading axis `none` -- the shape `sweep_many3d` now
benchmarks). All three: `arena_alloc<Plan>` (a small struct pairing a
`std::vector<std::byte>` sized to `plan.scratch_elements()` with a
`std::pmr::monotonic_buffer_resource`, `.reset()` calling `.release()`)
replaces the pool in `calibrate()`, `sweep<D>()`, and `sweep_many()`;
`sweep_many3d()` follows `sweep_many()`'s structure with `fftw_n[2] =
{n,n}`, `fftw_plan_many_dft(2, fftw_n, depth, ..., 1, n*n, ..., 1, n*n,
...)` (rank 2, contiguous n x n layers, `idist = odist = n*n`, matching
the row-major `(depth, n, n)` layout exactly).

**Correctness first**: the new `test/algorithms_fft.cpp` case (c)
verifies `{none, forward, forward}` on the existing non-cubic `(2,5,4)`
regression array (chosen there specifically because it catches axis
mix-ups that a cubic shape would hide) via a separable two-pass reference
(axis 2 direct, axis 1 via `.rotated()`, order-independent since a 2-D
DFT is separable) -- passes at the existing `tol`. Full test binary
(`test/algorithms_fft.cpp`) still green after both changes.

**Confirmed the pool resource's own file-header claim was wrong,
directly in the official flushed-cache benchmark** (not just the
profiling harness): re-running `sweep<1>` end to end, the huge-N tail
(single-fiber sizes at or above `fft_sixstep_min = 8192`) drops
substantially:

| n | old ratio (pool) | new ratio (arena) |
|---|---|---|
| 390,625 | 2.977 | **1.010** |
| 524,288 | 3.372 | 1.765 |
| 531,441 | 2.810 | 1.379 |
| 1,048,576 | 3.560 | 1.938 |
| 1,259,712 | 3.292 | 1.680 |
| 1,594,323 | 2.714 | 1.487 |
| 1,600,000 | 3.863 | 2.029 |
| 1,953,125 | 2.843 | 1.407 |
| 2,097,152 | 3.408 | 1.916 |

Every size at or above `fft_sixstep_min` improved, several by close to
2x, matching §11.15's profiling-harness prediction almost exactly (n=
390,625 landed at 1.010 -- essentially matching FFTW). Below
`fft_sixstep_min` (n <= 65,536), the ratio moves both up and down by
single-digit-to-~20% between runs with no consistent direction -- ordinary
run-to-run noise (this is one benchmark run, not the multi-run average
the methodology comment recommends for publication-grade numbers), not a
regression, and consistent with the arena fix mattering only for
allocations large enough to hit the mmap threshold.

**2-D and 3-D barely moved, and that's expected, not a discrepancy**:
`fft_sixstep_min` gates on a single transform axis's length, not total
array size, and the tested 2-D sweep tops out at side 2000, 3-D at side
300 -- neither ever reaches 8192 on any axis, so neither sweep exercises
the six-step path (or its allocator tax) at all. Their post-fix numbers
move within the same run-to-run noise band as 1-D's sub-8192 region, for
the same reason. This is a real (if narrow) prior gap in the benchmark's
size ranges: the 2-D/3-D sweeps as configured cannot currently measure
whether the allocator fix (or a future six-step rework) helps
multi-dimensional huge transforms, only single-axis ones -- worth a
note for whoever extends this sweep's size ranges next, not something
fixed here.

**New `{none, forward, forward}` batched-2D benchmark** (depth=32, one
run, `_nowisdom`):

| n (layer side) | N_total | mine mflops | FFTW mflops | ratio |
|---|---|---|---|---|
| 8 | 2,048 | 1853 | 5602 | 3.023 |
| 9 | 2,592 | 2607 | 4461 | 1.711 |
| 16 | 8,192 | 5377 | 10775 | 2.004 |
| 20 | 12,800 | 4881 | 12152 | 2.490 |
| 25 | 20,000 | 4362 | 9830 | 2.253 |
| 27 | 23,328 | 5174 | 6280 | 1.214 |
| 32 | 32,768 | 6877 | 13922 | 2.024 |
| 64 | 131,072 | 7556 | 11257 | 1.490 |
| 81 | 209,952 | 6278 | 9038 | 1.440 |
| 100 | 320,000 | 5549 | 10711 | 1.930 |
| 125 | 500,000 | 4774 | 7690 | 1.611 |
| 128 | 524,288 | 6616 | 10148 | 1.534 |
| 243 | 1,889,568 | 5974 | 5762 | **0.964** |
| 256 | 2,097,152 | 7722 | 9161 | 1.186 |

Broadly in the same 1.2-2.5x-of-FFTW band as the existing 1-D `{none,
forward}` batched sweep (§11's "many" cases, 55-61%-ish region) -- no
sign of a batching-specific penalty for the 2-D case either, consistent
with §11.14's P4 finding that gather/scatter overhead is negligible
regardless of how many axes are actually transformed per batch element.
n=243 (a pure `3^5`, one `stage_radix3_`-only layer) landed at 0.964 --
we beat FFTW there, in this one run; plausible (radix-3 is a
comparatively simple, well-matched kernel) but not something to
generalize from a single sample -- worth a repeat run before reading
anything into it. None of these sizes reach `fft_sixstep_min` either
(max n=256), so this table says nothing about the allocator fix; it's a
pure "does the per-axis direction feature cost anything for a 2-axis
batch" measurement, and the answer is no more than the existing 1-D
batched case already showed.

**Committed**: `benchmark/algorithms_fft.cpp` (arena fix +
`sweep_many3d`), `test/algorithms_fft.cpp` (new correctness case),
`fft_bench_{1d,2d,3d}_nowisdom.dat` and matching `.png` plots
(regenerated from this run), and the rebuilt `algorithms_fft_nowisdom.x`
binary (tracked in-repo per existing convention). The new
`fft_bench_many3d_h32_nowisdom.dat` and the pre-existing, never-tracked
`fft_bench_many_h{32,256}_nowisdom.dat` follow the repo's established
convention of NOT tracking "many"-family sweep outputs -- left as local
artifacts, not committed.

### 11.17 Does cross-axis gather/transpose cap candidate #3's payoff in 2-D/3-D? Measured: no (2026-07-11)

Follow-up question, not covered by §11.14: candidate #3 (radix-16 kernel)
was scoped against P1/P2/P4, all effectively 1-D (P4's `{none,forward}`
transforms exactly one axis). A genuine multi-axis transform (both/all
axes forward) does more than repeat that per-axis kernel -- it also has
to move data BETWEEN axis passes, since only one axis at a time is
contiguous in a row-major layout. Before committing to a radix-16
prototype scoped only from 1-D evidence, this measures whether that
cross-axis data movement is a separate, non-trivial cost that would cap
radix-16's payoff in the 2-D/3-D cases the maintainer is most worried
about.

**Method**: measurement only, scratchpad harness (`prof_2d3d.cpp`), same
build flags and machine-quiet protocol as §11.14, persistent monotonic
arena per §11.15/§11.16 (no allocator noise in these numbers). Two cases,
both full transforms (every axis `forward`, not a batched/degenerate
case): **M1** 2-D, n=1024x1024, 600 reps; **M2** 3-D, n=256^3, 40 reps.
`perf record --call-graph dwarf`, self-time split between the per-axis
kernel (`run_fused_impl_`/`stage_radix4_`, wherever the compiler inlined
it) and the explicitly-separate gather/scatter functions
(`fft_exec_slab`, `fft_exec_fiber` -- the functions the original
§11.14 task named as the gather/scatter suspects).

| case | kernel self-time (all `run_fused_impl_` instantiations) | `fft_exec_slab`+`fft_exec_fiber` self-time | IPC | L1-miss% |
|---|---|---|---|---|
| M1 (2-D, 1024x1024) | 98.93% | 0.40% | 2.20 | 20.9% |
| M2 (3-D, 256^3) | 95.90% | 1.26% | 1.66 | 18.2% |

**Verdict: cross-axis gather/transpose is NOT a hidden cost center in
either case** -- consistent with P4's finding (§11.14, gather/scatter
0.84% for a single-axis batch), not a new, worse number for genuine
multi-axis transforms. Mechanism, from reading `fft.hpp`
(`fft_apply_last_pair`, `include/boost/multi/algorithms/fft.hpp:1399-1406`):
the last two axes of any transform are fused into ONE slab-by-slab pass
("slab still hot") rather than a separate transpose-then-transform step,
so there is no explicit bulk-transpose call for `perf` to attribute time
to in either M1 (2 axes, fully covered by the fused pair) or M2 (the
third axis's separate `transform_axes_` recursion still resolves to the
same low gather/scatter share).

**The one real difference from 1-D, and why it doesn't change the
verdict**: IPC drops with dimensionality (1-D P1/P2/P4's per-axis kernel:
~3.1-3.2; M1: 2.20; M2: 1.66) and L1-miss% climbs (single digits/teens in
1-D vs 18-21% here). This is NOT a separate transpose cost hiding
outside the measured gather/scatter functions -- it's the strided access
pattern for non-innermost axes, which is fused directly into the same
kernel call (no separate copy loop exists to attribute it to
separately). Read correctly, this STRENGTHENS the case for candidate #3
rather than complicating it: fewer, larger radix-16 passes mean fewer
strided sweeps over non-innermost axes too, so radix-16's benefit should
compound with dimensionality, not get capped by an untouched transpose
bottleneck.

**Answer to the maintainer's question**: no, there is no 2-D/3-D-specific
ceiling on candidate #3. The 2-D/3-D full-transform cost is, to within
~1-4%, the same per-axis kernel arithmetic that dominates the 1-D case,
repeated per axis with no separate gather/transpose tax. A radix-16
prototype scoped from the 1-D evidence should transfer to 2-D/3-D
essentially intact; no additional design work (e.g. reworking
`fft_apply_last_pair` or adding an explicit transpose stage) is indicated
before starting that prototype.
