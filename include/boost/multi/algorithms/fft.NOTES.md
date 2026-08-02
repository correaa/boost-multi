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

### 11.18 Radix-16 kernel -- implemented, correctness-verified, reverted on speed (2026-07-11)

Candidate #3, the lever §11.14/§11.17 pointed at (P1/P2/P4 showed higher
IPC than FFTW but 2.7-3.7x more cycles/point -- an instruction-count, not
bandwidth, problem; §11.17 confirmed no 2-D/3-D-specific transpose
bottleneck would cap it either). Implemented as a NEW flat stage kind
(`stage_radix16_`, `kind=7`), following `stage_radix8_`'s own precedent
exactly: two internal radix-8 sub-DFTs (even legs `x0,x2,...,x14`, odd
legs `x1,x3,...,x15`, each itself two radix-4 sub-DFTs plus a W8
combine, byte-for-byte the same butterfly network `stage_radix8_` already
uses) plus a final W16 combine. `W16^2 == W8^1` and `W16^6 == W8^3`
(already computed for the internal radix-8s) and `W16^4` is the same
global `tw_[n_/4]` "+/-i" constant `stage_radix4_`/`stage_radix8_` use --
all reused, not re-derived. Critically, this stays a FLAT stage in the
existing `stages_` list plus ping-pong buffer -- no recursive sub-engine,
no extra gather/scatter -- explicitly avoiding split-radix's (§11.13)
mistake.

**Factorization**: the power-of-two part's greedy chunking was extended
to prefer radix-16 (4 bits/stage) over radix-4 (2 bits/stage), with a
radix-8+radix-4 tail absorbing a stray 1-bit remainder (borrowed from the
radix-16 count, mirroring how the existing scheme already borrows 3 bits
for a trailing radix-8 rather than ever emitting a lone radix-2), plain
radix-4/radix-8 tails for 2-/3-bit remainders, and radix-2 only for
n == 2 itself. Verified independently (a standalone script replicating
the arithmetic, not by reading the C++ back) that this actually selects
radix-16 broadly: n=1024 -> `[16,16,4]`, n=4096 -> `[16,16,16]`,
n=1,048,576 -> five `16`s, n=2048 (Bluestein's own internal `conv_n_` for
n=1009) -> `[16,16,8]`.

**Correctness, fully verified**: standalone exhaustive sweep (535 sizes:
every multiple of 16 from 16 to 8176, all pure powers of two through
4096, several odd-mixed composites), forward AND backward against
`dft_reference`, all passing at 1e-9 relative tolerance; a batched
(m>1, `{none,forward}` 2-D plan) check at 8 radix-16-triggering fiber
sizes; Bluestein n=1009 (internal `conv_n_=2048`, now recursing through
radix-16) both directions; a six-step n=1,048,576 forward-then-backward
roundtrip (`== n * id`) to 1e-21 relative residual. Separately, a
10-size ASan+UBSan smoke test (one representative size per distinct
factorization shape, not a full re-sweep -- the O(n^2) reference DFT
dominates sweep time regardless of sanitizer, so re-running all 535
sizes under ASan was correctly judged not worth the wall-clock cost)
found no memory-safety issues. Full test suite green under g++, clang++,
and g++ with `-fsanitize=address,undefined`.

**Benchmarked and REVERTED -- regression, not improvement, and it lands
specifically on the sizes that actually use radix-16:**

| size | old (radix-4/8) ratio to FFTW | new (radix-16) ratio | delta |
|---|---|---|---|
| 1D n=1024 | 1.142 | 1.401 | +22.7% |
| 1D n=4096 | 1.428 | 1.858 | +30.1% |
| 1D n=16384 | 1.771 | 2.471 | +39.5% |
| 1D n=65536 | 1.792 | 2.319 | +29.4% |
| 2D n=256 | 1.391 | 1.926 | +38.5% |
| 2D n=1024 | 1.505 | 1.789 | +18.9% |
| 3D n=16 | 1.392 | 2.505 | +80.0% |
| 3D n=64 | 1.389 | 1.720 | +23.8% |
| 3D n=256 | 1.153 | 1.611 | +39.7% |

Full official (flushed-cache) benchmark, all three dimensionalities,
against the currently-committed (arena-fixed, §11.16) baseline: sizes
whose factorization actually uses radix-16 stages are worse across the
board, several by 20-80%; sizes that barely touch it (odd-factor-heavy,
where the power-of-two part is small or absent) stay flat or move within
ordinary run-to-run noise. That pattern -- degradation tracking radix-16
USAGE, not size or dimensionality in general -- is itself the evidence:
this is the kernel's own cost, not a confound.

**Root cause, matching the risk flagged before implementation**: register
pressure, the same failure mode that already beat an all-radix-8 scheme
(the standing code comment) and split-radix (§11.13). `stage_radix16_`
holds 16 twiddle-multiplied inputs plus ~20 live intermediate combine
terms (`E0..E7`, `O0..O7`, `t1..t7`) at peak inside the innermost loop --
roughly double `stage_radix8_`'s own live-value count, which itself
already sits at the edge of what wins on this architecture. The ~20-30%
extra instructions-per-point that IPC/cycle profiling (§11.14) showed
FFTW avoiding is real, but a radix-16 kernel built this way pays it back
and more in spill/reload traffic -- fewer passes, but each pass costs
more than a radix-4/8 pass did, net negative.

**Reverted completely**: `stage_radix16_` itself, the `kind=7` dispatch
arm in both `run_fused_impl_`/`run_stages_`'s switches, the `case 16`
in the radix-to-kind switch, and the factorization greedy-chunking
change -- confirmed via `git diff` showing zero difference against the
pre-experiment commit, matching split-radix's (§11.13) revert protocol
exactly.

**What WOULD be needed for this to pay off**: not a bigger prototype of
the same shape -- the register-pressure ceiling is now confirmed twice
(radix-8-over-4, and this) at increasing severity, so a THIRD attempt at
"one bigger uniform-radix stage, more live values per butterfly" is not
indicated. A different angle on candidate #3 -- e.g. restructuring the
radix-8 (or even radix-4) kernel's OWN instruction count without adding
more simultaneous live values (fewer temporaries via algebraic
re-association, or explicit narrower SIMD-width batching so the
"batch" dimension `m` absorbs register pressure instead of the radix
itself) -- is the more promising direction if this candidate is revisited,
but that's a different, not-yet-scoped design task, not a variant of what
was tried here.

### 11.19 Re-scoping the target: how close are we to FFTW_ESTIMATE (nowisdom)? (2026-07-11)

After §11.9-§11.18 (unseq/par/fma/-ffast-math/split-radix/radix-16 all
tried and reverted; the one real win, §11.15/§11.16's allocator fix, is
in), the FFTW_MEASURE comparison this whole file tracks is a strong
baseline -- FFTW's own runtime codelet search picks close to its best
possible execution strategy for each size. A more modest, and possibly
more useful, question: how do we compare to FFTW_ESTIMATE (wisdom
disabled), which skips that search and uses generic heuristics instead --
a much weaker target, and one worth knowing the current distance to now
that the easy algorithmic wins are exhausted.

**Method**: the existing `-DUSE_ESTIMATE -DDISABLE_WISDOM` build mode
(already supported by `benchmark/algorithms_fft.cpp`, already had
tracked-but-stale `_estimate.dat`/`.png` outputs from before this
session's allocator fix) re-run against the CURRENT code (post §11.16's
arena fix, radix-16 fully reverted per §11.18) -- same methodology
(flushed cache, interleaved timing, plan-recycled), same idle/AC/cool
machine. No product-code changes in this section; measurement only,
against an already-existing build configuration.

**Result: much closer, and outright ahead in large parts of the
parameter space.**

| dimensionality | sizes tested | outright wins (ratio < 1.0) |
|---|---|---|
| 1-D | 45 | 8 |
| 2-D | 28 | 8 |
| 3-D | 20 | 6 |

Full 1-D (`n`, `N_total`, ratio to FFTW_ESTIMATE, lower is better):

| n | N | ratio | | n | N | ratio |
|---|---|---|---|---|---|---|
| 125 | 125 | 0.917 W | | 20250 | 20,250 | 1.394 |
| 128 | 128 | 1.168 | | 24000 | 24,000 | 1.606 |
| 144 | 144 | 0.734 W | | 27000 | 27,000 | 1.324 |
| 180 | 180 | 0.893 W | | 32768 | 32,768 | 1.729 |
| 200 | 200 | 1.396 | | 59049 | 59,049 | 1.120 |
| 243 | 243 | 0.787 W | | 65536 | 65,536 | 1.681 |
| 256 | 256 | 0.951 W | | 78125 | 78,125 | 1.128 |
| 512 | 512 | 1.179 | | 131072 | 131,072 | 1.466 |
| 625 | 625 | 1.099 | | 172800 | 172,800 | 1.261 |
| 729 | 729 | 0.986 W | | 177147 | 177,147 | 1.159 |
| 1024 | 1,024 | 1.143 | | 230400 | 230,400 | 1.282 |
| 1080 | 1,080 | 0.937 W | | 250000 | 250,000 | 1.485 |
| 1296 | 1,296 | 0.860 W | | 262144 | 262,144 | 1.576 |
| 1600 | 1,600 | 1.162 | | 390625 | 390,625 | 1.184 |
| 2048 | 2,048 | 1.239 | | 524288 | 524,288 | 1.318 |
| 2187 | 2,187 | 1.099 | | 531441 | 531,441 | 1.420 |
| 3125 | 3,125 | 1.741 | | 1048576 | 1,048,576 | 1.368 |
| 4096 | 4,096 | 1.432 | | 1259712 | 1,259,712 | 1.288 |
| 6561 | 6,561 | 1.783 | | 1594323 | 1,594,323 | 1.379 |
| 8192 | 8,192 | 1.666 | | 1600000 | 1,600,000 | 1.522 |
| 15625 | 15,625 | 1.274 | | 1953125 | 1,953,125 | 1.199 |
| 16384 | 16,384 | 1.635 | | 2097152 | 2,097,152 | 1.351 |
| 19683 | 19,683 | 1.213 | | | | |

("W" = outright win, ratio < 1.0.)

Full 2-D and 3-D (n = side length):

| 2-D n | ratio | | 2-D n | ratio | | 3-D n | ratio | | 3-D n | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 24 | 0.760 W | | 216 | 1.226 | | 8 | 3.731 | | 100 | 1.404 |
| 25 | 1.586 | | 243 | 1.082 | | 9 | 1.889 | | 125 | 1.100 |
| 27 | 0.683 W | | 250 | 1.892 | | 15 | 1.997 | | 128 | 1.255 |
| 32 | 1.465 | | 256 | 0.674 W | | 16 | 1.606 | | 144 | 0.865 W |
| 40 | 0.997 W | | 320 | 1.174 | | 20 | 2.313 | | 216 | 0.967 W |
| 60 | 1.421 | | 375 | 1.523 | | 25 | 2.195 | | 243 | 0.769 W |
| 64 | 1.502 | | 405 | 1.174 | | 27 | 0.870 W | | 250 | 1.323 |
| 75 | 1.434 | | 486 | 1.306 | | 32 | 1.815 | | 256 | 0.557 W |
| 81 | 1.236 | | 512 | 0.784 W | | 64 | 1.237 | | 300 | 1.078 |
| 100 | 1.914 | | 625 | 1.308 | | 81 | 0.926 W | | | |
| 125 | 1.612 | | 729 | 1.183 | | 90 | 1.057 | | | |
| 128 | 1.666 | | 1024 | 0.763 W | | | | | | |
| | | | 1215 | 1.360 | | | | | | |
| | | | 1350 | 1.261 | | | | | | |
| | | | 1600 | 0.969 W | | | | | | |
| | | | 2000 | 0.900 W | | | | | | |

**Pattern**: wins cluster in two distinct places -- smaller/odd-composite
1-D sizes (125-1296, where FFTW_ESTIMATE's generic heuristic doesn't
have much room to beat a straightforward mixed-radix engine), and LARGE
2-D/3-D sizes (2-D n>=1024, 3-D n>=144 -- up to 2x faster at 3-D n=256).
The remaining, concentrated gap is a mid-range, heavily power-of-two 1-D
band (n~2048-65536, ratios 1.2-1.8x) -- the same region §11.14 flagged as
instruction-count-bound and where radix-16 (§11.18) tried and failed to
help.

**Implication**: the practical target has effectively moved. Against
FFTW_MEASURE, `multi::fft_plan` sits at roughly 55-80% of FFTW across
most of the space (per the committed `_nowisdom.dat` files) with a few
much worse regions (now improved by §11.15/16's allocator fix). Against
FFTW_ESTIMATE, we're already competitive or ahead across most sizes
tested, with one concentrated remaining gap rather than a broad one. Since
radix-8-over-4, split-radix, and radix-16 have all now lost to register
pressure trying to shrink that band's instruction count (§11.18's closing
note), the next lever for it -- if pursued -- is reducing the EXISTING
radix-4/8 kernels' own temporary count (algebraic re-association) or
spending the batch dimension `m` to absorb register pressure, not another
bigger-uniform-radix attempt.

**Committed**: `fft_bench_{1d,2d,3d}_estimate.dat` and matching `.png`
plots (regenerated from this run, replacing the stale pre-allocator-fix
versions); no product code changed.

### 11.20 Redundant full-array-sweep hypothesis for 2-D/3-D: refuted (2026-07-11)

Follow-up question to §11.17: that section ruled out a dominant gather/
scatter/transpose FUNCTION for 2-D/3-D (`fft_exec_slab`/`fft_exec_fiber`
self-time <1.5%), but didn't address a different mechanism -- TOTAL
MEMORY TRAFFIC. `fft_apply_last_pair` (`fft.hpp:1399-1406`) fuses the
last two axes of any transform into one hot-slab pass, but for D >= 3
the remaining axis (axis 0 in 3-D) gets its OWN separate full sweep over
the entire array. For an array that doesn't fit in cache, that second
sweep re-reads/re-writes everything from RAM after the first sweep
already touched every element once -- plausibly ~2x the memory traffic
an ideal single fused cache-blocked pass (the multi-axis analogue of
six-step) would pay. Hypothesis: this costs us disproportionately more
than FFTW once the array exceeds the 12 MiB L3 cache on this machine,
and that gap should widen with size.

**Method**: no new measurement needed -- the already-committed
`fft_bench_{2d,3d}_nowisdom.dat` (§11.16/current) already span sizes
from comfortably-cached to far-exceeding-L3 (this machine: L1d 32 KiB/
core, L2 256 KiB/core, L3 12 MiB shared). Read directly.

| 3-D n | footprint | ratio to FFTW |
|---|---|---|
| 32 | 512 KB (fits L2/L3) | 1.815 |
| 64 | 4 MB (fits L3) | 1.389 |
| 128 | 32 MB (2.7x over L3) | 1.420 |
| 256 | 268 MB (22x over L3) | **1.153** |

| 2-D n | footprint | ratio to FFTW |
|---|---|---|
| 32 | 16 KB | 1.502 |
| 64 | 64 KB | 1.350 |
| 128 | 256 KB (~fits L2) | 1.731 |
| 256 | 1 MB (fits L3) | 1.391 |
| 512 | 4 MB (fits L3) | 1.538 |
| 1024 | 16 MB (over L3) | 1.505 |
| 2000 | 64 MB (far over L3) | 1.727 |

**Verdict: refuted.** The hypothesis predicts the ratio gets WORSE past
the L3 boundary. It does the opposite in 3-D: n=32 (comfortably cached)
is the WORST point in the whole sweep (1.815x), and n=256 (22x over L3,
by far the largest and most memory-pressured case) is the BEST (1.153x,
close to FFTW parity). 2-D shows no clean trend at all -- ratio bounces
between 1.35 and 1.73 with no monotonic relationship to whether the
array fits in cache. If anything the pattern runs backwards from the
hypothesis: the worst relative performance is at SMALL sizes, consistent
with fixed per-call overhead (dispatch, small-batch inefficiency)
dominating there rather than large sizes exposing an extra-sweep memory
tax. Either the axis-0 sweep isn't costing us disproportionately versus
FFTW, or FFTW pays a comparable cache-miss penalty at these sizes too,
so the RATIO stays flat/improves even as absolute cycles/point rise for
both libraries.

**Net effect on candidate set for 2-D/3-D specifically**: nothing left.
§11.17 ruled out a dominant gather/transpose function; this section
rules out disproportionate memory bandwidth from the multi-axis walk's
structure. Combined with §11.18's radix-16 failure (register pressure,
dimensionality-independent since it's the same shared per-axis kernel),
there is no evidence-backed, 2-D/3-D-specific lever remaining. 2-D/3-D
track 1-D's residual gap because they run the same kernel, not because
of anything specific to multi-axis walking. Closing it further requires
the same not-yet-scoped 1-D kernel work §11.18 closed with (reducing the
existing radix-4/8 kernels' own temporary count without adding live
values) -- not new 2-D/3-D-specific design work.

### 11.21 First-stage twiddle-skip fast path -- implemented, correctness-verified, reverted: real cycle win, net regression under flushed cache (2026-07-11)

A genuinely different, low-risk candidate: `run_fused_impl_`'s stage loop
always starts with `ns = 1` for the first stage, and inside
`stage_radix{2,3,4,5,8}_`, `ns == 1` forces the inner `r`-loop to only
ever take `r == 0` -- meaning `w1`/`w2`/... `== tw_[0] == 1` for every
single block of the FIRST stage (the whole first pass over the array).
Those twiddle multiplies are providably no-ops, but `tw_` is a runtime-
built table, so the compiler can't fold them away on its own. Added an
`if(ns == 1) { ...fast path, twiddle multiplies elided... return; }`
branch, checked once per function call (not per iteration), to each of
the five direct kernels -- no register-pressure change, no algorithm
change, unlike every previous candidate in this series.

**Correctness, fully verified**: 523-size sweep (dense n=2..512, plus
larger explicit sizes) forward+backward, 18 batched fiber sizes, three
Bluestein sizes (101, 1009, 2), and the n=1,048,576 six-step roundtrip --
all passing. ASan+UBSan clean. g++ and clang++ both green.

**Preliminary check (misleading in hindsight)**: a hot, back-to-back
(no cache flush) `perf stat` comparison showed a clean, consistent win --
P1 (n=1024): cycles -9.6%, instructions -16.7%; P2 (n=4096): cycles
-12.1%, instructions -14.2%, wall time -9.7%; P4 (batched): cycles
-11.9%, instructions -16.8%, wall time -14.2%. No regression anywhere in
this harness. This is exactly the kind of result that looked like it
should ship.

**Full official (flushed-cache) benchmark told a different story.**
2-D and 3-D mostly improved (2-D: outright wins 8->11/28, most sizes
better by 5-20%; 3-D: wins 6->8/20, similar pattern) -- consistent with
the preliminary check. **1-D showed severe, unambiguous regressions** at
several mid-range sizes: n=256 +124%, n=1024 +116%, n=512 +88%, n=2048
+54%, n=1296 +56%, n=144 +57% (ratio to FFTW_ESTIMATE, worse = higher).
Outright wins dropped 8->4/45. Not noise -- these are 2-4x the size of
any other single-run fluctuation seen in this file's benchmarking.

**Root cause, confirmed directly, not just inferred**: the `if(ns==1)`
branch duplicates the ENTIRE block/element loop body inside each kernel
function -- and since every stage of every `execute()` call runs through
the SAME compiled function (the switch in `run_fused_impl_` calls
`stage_radix4_` for every stage regardless of that stage's `ns`), this
roughly doubles the function's code size for ALL calls, not just the
`ns==1` ones. A targeted flushed-cache-per-call comparison at n=1024
(mirroring the official benchmark's own methodology, not the hot-loop
preliminary check) showed: instruction count flat (15.037B -> 15.014B,
as expected -- the removed multiplies are a small fraction of a
`flush_cache()`-dominated call), but **L1-icache-load-misses up 24%**
(3.88M -> 4.80M) and **iTLB-load-misses up 19%** (106K -> 127K). Under a
COLD icache (the flushed-benchmark's actual regime, and every real
caller's likely regime too -- a plan is typically not called in a tight
tofrom-nowhere loop), the extra code has to be re-fetched from L2/L3/RAM
on every single call, and that cost outweighs the saved arithmetic. The
preliminary hot-loop check missed this entirely because repeated calls
let the icache warm up once and stay warm across all reps -- exactly the
regime this file's own methodology notes (§11's intro, and repeated
throughout) warn isn't representative of steady-state, cache-cold
callers, which is why the OFFICIAL benchmark flushes caches before every
timed call in the first place.

**Why 2-D/3-D didn't regress the same way**: plausibly because their
per-call compute is much larger relative to the fixed cold-icache-fetch
cost (more total array elements processed per call), so the same
absolute icache-refetch tax is a smaller fraction of a much bigger call,
while 1-D's benchmark sizes have comparatively small per-call work and
very high rep counts, making the fixed per-call icache cost a much larger
relative fraction of each call's time. Not measured further; the
official benchmark's per-dimensionality asymmetry is consistent with
this but wasn't independently isolated.

**Reverted completely**: all five `if(ns==1)` fast-path branches (the
diff was saved and reverse-applied via `git apply -R`, then confirmed
byte-identical to the pre-experiment commit via `git diff`). Full test
suite green after revert.

**What WOULD be needed for this to pay off**: the same optimization
without doubling code size -- e.g. a single shared loop body with the
"skip the multiply when `ns==1`" decision folded into a per-r branch
INSIDE the existing loop (checked `ns` times total per call instead of
duplicating the whole body), accepting a small, predictable per-iteration
branch in exchange for not bloating the function; or, more in the spirit
of "no register pressure, no code growth," rely on `fft_mul_dir`'s own
generality being just as cheap as a copy for a genuine `1.0 + 0.0i`
multiply on this architecture (worth directly measuring the codegen
difference between "multiply by a runtime-provably-1 complex value" and
"copy" before assuming the win requires a branch at all). Not attempted
here; this is the second candidate this session (after §11.13's
recursive split-radix and §11.18's radix-16) to look like a clean win in
isolation and lose once measured against the benchmark's own, more
representative, cold-cache methodology -- a pattern worth remembering
before trusting any future preliminary micro-benchmark that doesn't
flush caches.

### 11.22 `-i`/`+i` multiply shortcut (`imu`) -- implemented, correctness-verified, reverted: worse than §11.21, and by a different mechanism (2026-07-11)

A different-in-kind candidate from everything else in this series: the
`imu` constant (`tw_[n/4]`, used in `stage_radix4_`'s combine once per
element and `stage_radix8_`'s combine three times per element) is
provably exactly `-i` (forward) or `+i` (backward) by construction --
never a general twiddle -- yet every use went through the full generic
4-multiply/2-add complex product (`fft_ops::mul`/`conj_mul`). Multiplying
by `±i` is a component swap and sign flip, not a product. Added
`fft_ops::mul_neg_i`/`mul_pos_i` (generic fallback = the existing product,
for user-specialized element types; `std::complex` specialization = the
branchless swap-negate) and a matching `fft_mul_i_dir<Backward>`
dispatcher preserving the same conjugation convention as `fft_mul_dir`,
then swapped the four `imu` call sites (one in `stage_radix4_`, three in
`stage_radix8_`) to use it. Unlike §11.21, this adds NO branch and NO
code duplication -- if anything the replacement code is simpler/smaller.

**Correctness, fully verified**: same 523-size exhaustive sweep (dense
n=2..512 plus larger sizes) forward+backward, 18 batched fiber sizes,
three Bluestein sizes, six-step n=1,048,576 roundtrip -- all passing.
ASan+UBSan clean. g++ and clang++ both green.

**Learned from §11.21: went straight to the full official flushed-cache
benchmark, skipped trusting a hot-loop preliminary check.** Good thing --
the result is a clear, severe regression, WORSE than §11.21's and hitting
ALL THREE dimensionalities this time (§11.21 only hurt 1-D): 1-D n=4096
+178%, n=1024 +158%, n=256 +108%; 2-D n=256 +114%, n=1024 +112%, n=64
+94%; 3-D n=64 +92%, n=256 +73%, n=128 +45%. Outright wins collapsed:
1-D 8->4/45, 2-D 8->2/28, 3-D 6->5/20.

**Root cause: plausible but NOT confirmed to the same depth as §11.21**
(no asm-level or vectorization-report investigation was completed before
reverting, given the severity of the result). Working hypothesis: the
original code computed w1/w2/w3/imu as four structurally IDENTICAL
generic-multiply expressions in a row inside the same loop body, which
plausibly let the compiler's auto-vectorizer treat them as a uniform,
schedulable/vectorizable sequence (this file has no manual SIMD --
fft-simd-policy in project memory -- so it depends entirely on the
compiler recognizing and exploiting exactly this kind of uniformity).
Replacing ONE of the four with a differently-shaped operation (swap-
negate instead of multiply) breaks that uniformity, and losing whatever
auto-vectorization or instruction-scheduling pattern the compiler was
applying to the uniform version costs more than the removed arithmetic
saves -- consistent with the severity being similar across all three
dimensionalities (unlike §11.21's code-bloat mechanism, which hit 1-D
specifically because of its higher call-count/lower-per-call-work
profile). Should be confirmed with `-fopt-info-vec-missed` or a
disassembly diff before anyone revisits this shape of change.

**Reverted completely**: all `fft_ops::mul_neg_i`/`mul_pos_i`,
`fft_mul_i_dir`, and the four call-site swaps, via `git apply -R` against
the saved diff, confirmed byte-identical to the pre-experiment commit via
`git diff`. Full test suite green after revert.

**Running tally for this session's "reduce the kernel's own work, without
adding registers/passes/code-size" attempts**: split-radix (extra passes),
radix-16 (register pressure), first-stage twiddle-skip (icache bloat),
and this `±i` shortcut (vectorization/scheduling, unconfirmed mechanism)
have now ALL failed, each for a different micro-architectural reason.
Four independent negative results at this level of the design space is a
strong signal that the compiler's existing auto-vectorized codegen for
the uniform, generic kernel shape is already close to a local optimum
that resists further ad-hoc algebraic simplification -- not proof no
lever exists, but enough evidence that the next attempt at THIS level
should not be another micro-optimization guess without first getting a
vectorization report or disassembly to justify it. The two substantive
options genuinely still open, neither tried this session: (a) SIMD
intrinsics, explicit control instead of hoping the auto-vectorizer keeps
cooperating -- the standing policy explicitly reserves this as a last
resort, and it's now backed by four repeated demonstrations of auto-
vectorization fragility to any manual tweaking; (b) a persistent,
non-`<execution>`-dependent thread pool for large-transform execute()
calls specifically -- construction-time threading was a genuine 2-4x win
(§11.10) blocked only by `<execution>`'s libc++/TBB portability issues,
never by the underlying idea; a hand-rolled pool sidesteps that blocker
but is a substantially bigger design task (persistent state, lifetime,
thread-safety) than anything attempted in this series.

**Follow-up: is §11.21/§11.22's regression an `-march=native`-specific
artifact?** Checked before concluding this series, since both reverted
changes plausibly interact with the compiler's auto-vectorizer, and this
machine's `-march=native` resolves to `-march=skylake` (confirmed via
`gcc -march=native -dM -E`: AVX2/FMA/BMI2, no AVX-512) -- a specific
tuning model, not a neutral baseline. Built all three variants (baseline,
§11.21 twiddle-skip, §11.22 `±i` shortcut) under three targets:
`-march=native -mtune=native`, `-march=x86-64-v3 -mtune=generic` (same
ISA level, generic scheduling model instead of Skylake-specific), and
`-march=x86-64 -mtune=generic` (SSE2 baseline, no AVX2/FMA at all), using
a flushed-cache-per-call `perf stat` probe (cycles/instructions/icache-
misses, mirroring §11.21's own confirmatory methodology) at n=256/1024/
4096.

**Result: inconclusive, and not worth further investment.** At n=1024,
§11.21's icache-miss increase reproduced under ALL THREE targets (not
just native) -- the code-bloat mechanism isn't an `-march=native`
artifact. At n=256 and n=4096, the signal was within run-to-run noise for
both variants, and §11.22's severe regression (clearly visible in the
full official benchmark, §11.22) did not reproduce at all in this
single-size, non-interleaved probe -- meaning the probe itself isn't an
adequate stand-in for the official benchmark's methodology (interleaved
FFTW timing, `reps_for()`-scaled rep counts, a multi-minute sweep with
its own drift characteristics) for this question. Getting a real answer
would require re-running the FULL official benchmark under each
alternative `-march` for both variants -- a multi-run investment
significantly larger than this check, on top of two changes that have
already failed once each. Judged not worth it: nothing so far suggests
either change would behave differently under a different target, and the
one data point that DID reproduce cleanly (§11.21 at n=1024) reproduced
the SAME failure, not a different one. Not pursued further; flagged here
so it isn't re-asked without this context.

### 11.23 Mixed-precision twiddle table (TW=complex<float> on complex<double> data): no speedup (2026-07-11)

Not part of the main optimization series above -- a follow-up question
after adding the `vec3` custom-element-type test (T independent of TW,
fft.hpp's own design point): does a NARROWER twiddle table (`TW =
complex<float>`) speed up a transform on `T = complex<double>` data,
versus the normal `TW = T = complex<double>` case? Motivation: the
twiddle table `tw_` is O(n) and, for large n, doesn't fit cache -- a
half-size table (float instead of double) means half the bytes to
stream on every load, a real bandwidth argument, IF twiddle loads are
actually a bottleneck for the sizes in question.

**Method**: scratchpad harness (`mixed_tw_bench.cpp`), same methodology
as the official benchmark (`benchmark/algorithms_fft.cpp`): plan built
once outside the timed region, cache flushed before every individual
timed call, interleaved timing between the two plans (`fft_plan<3,
complex<double>>` vs `fft_plan<3, complex<float>>`, same
`complex<double>` array both times, same persistent monotonic-arena
allocator per §11.15/16). 3-D sizes n=16..256 (side length), the same
range the official 3-D sweep covers. `max_rel_err` computed against the
pure-double result as an accuracy sanity check, not the main question.

| n (3D side) | N total | double (ms) | mixed (ms) | ratio (mixed/double) | max rel. error |
|---|---|---|---|---|---|
| 16 | 4,096 | 0.041 | 0.041 | 1.00 | 1.3e-7 |
| 24 | 13,824 | 0.131 | 0.138 | 1.05 | 4.1e-7 |
| 32 | 32,768 | 0.301 | 0.302 | 1.00 | 2.5e-7 |
| 48 | 110,592 | 1.038 | 1.010 | 0.97 | 4.3e-7 |
| 64 | 262,144 | 2.589 | 2.929 | 1.13 | 3.6e-7 |
| 96 | 884,736 | 12.746 | 12.789 | 1.00 | 5.1e-7 |
| 128 | 2,097,152 | 28.85 | 29.37 | 1.02 | 5.0e-7 |
| 192 | 7,077,888 | 111.35 | 114.23 | 1.03 | 6.2e-7 |
| 256 | 16,777,216 | 269.68 | 284.00 | 1.05 | 5.9e-7 |

(n=128/192/256 re-run at 10 reps instead of the sweep's initial 3, to
check the largest-size numbers weren't noise -- the direction held, the
magnitude came down from the noisier first pass, e.g. n=256 1.136 ->
1.053.)

**Verdict: no speedup anywhere in the tested range; a small, consistent
slowdown that grows with size** (up to ~5% at n=256, the largest case
tested). Accuracy is exactly as expected (~1e-7 relative error, float-
precision twiddle quantization) -- correctness is fine, the performance
hypothesis just doesn't pay off. Mechanism: `fft_ops<complex<double>,
complex<float>>`'s generic formula (fft.hpp's `fft_ops<complex<R1>,
complex<R2>>` specialization) promotes the narrower operand up to
`std::common_type_t<R1,R2>` (== `double` here) BEFORE multiplying, and
narrows only the final result once -- so using `complex<float>`
twiddles does not reduce the arithmetic work at all; every multiply
still happens at double precision. Only the table's own memory footprint
shrinks, and per this data, that's not the bottleneck at these sizes
(consistent with §11.14's finding that the smooth/batched kernels are
instruction-count-bound, not memory-bound) -- so there is no bandwidth
win to collect, while the extra widen/narrow conversion at every twiddle
load is a small but real fixed cost that shows up instead. The
T-independent-of-TW mechanism itself works correctly (this is a real,
supported use case -- see the `vec3` test and its README-style
walkthrough); it's specifically "narrower TW as a speed trick" that
doesn't work, on this machine, in this size range. Not investigated
further (e.g. whether a size regime exists where the twiddle table
alone exceeds L3 while everything else still fits, which might change
the balance) -- flagged here rather than assumed either way.

### 11.24 Experimental packed contiguous batches: useful for active 2-D, not a solution for generalized-many (2026-07-12)

The current slab executor has a deliberately cheap fast path for
contiguous fibers: invoke the 1-D engine directly once per fiber.  This
is the right thing for a genuinely independent batch of 1-D transforms,
but it means a 2-D/3-D transform pays the dispatch/setup cost of the
1-D engine separately for every row.  The experiment here asks whether
the existing blocked gather/run/scatter path can do better by packing a
contiguous fiber batch into one rank-2 scratch block before calling the
engine.

**Implementation.** `BOOST_MULTI_FFT_EXPERIMENT_PACK_CONTIGUOUS_BATCHES`
suppresses that direct contiguous-fiber early return for non-six-step
engines only.  It therefore reuses the generic blocked pack ->
`eng.run(m)` -> scatter path already used for non-contiguous fibers.
Six-step transforms retain their direct path.  The macro is off by
default: this is a schedule experiment, not a proposed default
behavior.  The complete strict `test/algorithms_fft.cpp` build and run
passed both with and without the macro.

**Benchmark.** Full official sweeps, in-place for both Multi and FFTW;
FFTW used `FFTW_ESTIMATE` with wisdom disabled, plans were built outside
the timed region, input restoration was also outside the timed region,
and the cache was flushed before each timed call.  The direct baseline
was taken on a quiet machine; the packed run was usable but less clean
(its calibration changed 0.2434 -> 0.2343 ms, 3.7%).  Consequently the
numbers below compare the geometric mean of Multi/FFTW *within each
run*; the packed/direct wall-time ratio is informative but not treated
as a precise cross-run speedup.

| sweep | direct Multi/FFTW | packed Multi/FFTW | packed/direct time | sizes packed wins |
|---|---:|---:|---:|---:|
| 1-D | 1.219 | 1.234 | 1.194 | 2/45 |
| 2-D | 1.193 | 1.075 | 1.056 | 12/28 |
| 3-D | 1.427 | 1.260 | 0.966 | 12/20 |
| generalized-many, h=256, `{none, forward}` | 2.075 | 2.031 | 1.065 | 6/14 |
| generalized-many 3-D, h=32, `{none, forward, forward}` | 1.806 | 1.522 | 0.911 | 10/14 |

The 1-D control behaves as expected: packing is not beneficial and the
small difference in its within-run ratio also bounds the amount of
ambient-run skew.  The active 2-D/3-D cases do benefit in their ratio
to FFTW, especially the 3-D generalized-many case.  The h=256
generalized-many case -- the primary target -- improves only about 2%
geometrically, with regressions concentrated at several powers of two
(and a large regression at n=2048).  It is therefore not credible to
enable this blanket per-axis packing rule by default.

**Conclusion and next experiment.** The evidence supports amortizing
work over a full active rank-2 tile, not blindly packing each
contiguous batch on every axis.  The promising follow-up is a fused
rank-2 scratch schedule: pack a tile once, perform its row transform,
transpose or hand off in scratch for its column transform, then scatter
once.  That removes the repeated per-axis gather/scatter traffic which
this experiment leaves in place.  It needs a scratch-layout/cost-model
extension and a careful memory bound, followed by the same strict test
gate and a paired, idle-machine benchmark before considering it for the
default schedule.

### 11.25 Direct O(p^2) kernel for small power-of-two sizes (n=16, 32) -- implemented, correctness-verified, reverted: 6-8x regression (2026-07-16)

Motivation: at n=32 the 2-D sweep shows multi::fft_plan running at roughly
half FFTW's speed (ratio ~2.0), the worst gap in the whole size range.
FFTW's advantage at sizes this small is generally attributed to its
generated, fully-unrolled codelets, which do the whole transform in one
pass with compile-time-constant twiddles and no intermediate memory
traffic. `fft.hpp` instead always splits the power-of-two part of `n`
into radix-4/8 stages -- for n=32 that is `[4, 8]`, two stages with a
full scratch read/write and twiddle-multiply pass between them.

**Implementation.** `fft.hpp` already has a direct, table-driven O(p^2)
kernel (`stage_generic_`, `kind==4`, driven by a precomputed p x p DFT
matrix) used today only for prime factors up to `fft_max_direct_radix`
(64). `BOOST_MULTI_FFT_EXPERIMENT_DIRECT_POW2_KERNEL` extends that same
existing, already-tested machinery to the power-of-two part of `n` when
it is exactly 16 or 32: instead of pushing `[4,4]` or `[4,8]` stages, it
pushes a single composite factor (16 or 32) which falls through to the
same direct-kernel `default:` case a prime of that size would use. No
new kernel code, only a change to which sizes reach the existing path.
Capped at 32, deliberately *not* `fft_max_direct_radix` (64): n=64 is a
common six-step sub-transform size (e.g. n=16384 = 256*64, used by this
file's own `calibrate()` self-check), invoked once per row: enabling the
direct kernel at 64 caused a ~4x regression in that self-check alone,
diagnosed and confirmed via `calibrate()`'s before/after drift readings
before any real sweep was run, and is not otherwise reported here.  The
complete strict `test/algorithms_fft.cpp` build
(`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion
-Werror`) and an `-fsanitize=address,undefined` build both passed, capped
at 32.

**Benchmark.** 2-D and generalized-many-3-D (h=32) sweeps, in-place,
`FFTW_ESTIMATE` with wisdom disabled, cold-cache, on an idle machine
(calibration drift 0.3-5.7% across the three back-to-back direct/packed/
pow2direct runs used for the numbers below).

| sweep | direct Multi/FFTW | pow2direct Multi/FFTW | pow2direct/direct time | sizes pow2direct wins |
|---|---:|---:|---:|---:|
| 2-D | 1.206 | 1.339 | 1.109 | 14/28 |
| generalized-many 3-D, h=32 | 1.821 | 2.408 | 1.292 | 7/14 |

The aggregate numbers already show a net regression, but the effect is
concentrated and severe exactly at n=16 and n=32 (the only sizes the
macro touches) rather than spread thin: n=32 is 6.9x slower in 2-D and
8.1x slower in many-3-D; n=16 is 5.2x slower in many-3-D. n=64 (outside
the macro's range) is unchanged in both sweeps, confirming the
regression is caused by the direct-kernel substitution and not some
unrelated run-to-run variation.

**Root cause.** Arithmetic, not overhead. O(32^2) = 1024 multiply-adds
for the direct matrix kernel vs. the existing 2-stage radix-4/radix-8
decomposition at roughly O(32 log2 32) ~ 160 operations -- about a 6x
arithmetic blow-up, matching the observed ~6-8x wall-time regression
closely enough that no other mechanism needs to be invoked. The
motivating premise (removing a memory/twiddle pass would offset the
extra arithmetic at small n, the same overhead-bound argument that
partly justified [[fft-flushed-cache-methodology|this file's other cold-cache experiments]])
does not hold here: the arithmetic cost increase is too large to be
masked by saving one pass.

**Conclusion.** Reusing the existing O(p^2) direct kernel for small
power-of-two sizes is not a viable path to closing the n=32 gap with
FFTW -- ruled out decisively, not just "not beneficial." If FFTW's
real advantage at these sizes is architectural (a true O(p log p)
fully-unrolled codelet with no intermediate memory pass and
compile-time-constant twiddles), closing that gap requires building an
actual codelet-style kernel, which is a substantially larger,
from-scratch effort than routing existing machinery differently -- not
attempted here.

### 11.26 Fused rank-2 scratch schedule -- implemented, correctness-verified, reverted: no viable mechanism under this benchmark's methodology (2026-07-17)

Follow-up to §11.24's proposed next step: a genuinely fused D==2 schedule
(row-FFT axis 1 into scratch, transpose in scratch, batched axis-0 FFT,
scatter once) instead of the default two independent full-array axis passes
(`fft_apply_last_pair`).

**Design correction made before implementation.** The initial framing (a
small, cache-resident "tile" processed independently per axis) does not
work: axis-1's FFT needs every element of a row to produce any output value,
and axis-0's FFT needs every element of a column -- neither transform is
decomposable into a small partial block. The only structurally valid fused
schedule needs a full `O(n0*n1)` scratch buffer (one complete extra copy of
the last-pair slab), not a small tile. Working through the concrete gather
step further showed the fused schedule does not reduce total bytes moved
either: the default schedule already does the *contiguous* axis-1 pass
optimally (1 read + 1 write, in place, no gather); the fused schedule
replaces that with 1 read (user memory) + 1 write (scratch), then still pays
the same unavoidably-strided axis-0 gather/scatter either way. Total
traffic is the same 4 array-sized touches, just 2-to-user-memory + 2-to-
scratch instead of 4-to-user-memory. The only remaining hypothesized win:
the axis-0 gather reads from a buffer *just written*, potentially still
warm in cache, versus user memory that may have been evicted by the time
the second pass runs.

**Implementation.** `BOOST_MULTI_FFT_EXPERIMENT_FUSED_PAIR_SCHEDULE`, gated
at `fft_apply_last_pair`'s `rank==2` base case (the same site reached for
every D>=2 plan's last-pair, including D>=3's recursion into it). New
`fft_plan`-level (not engine-level) scratch region `fused_off_`, sized
`n0*n1` from the last two axes' engine lengths, computed once in the
constructor and folded into the existing single-arena `scratch_elements_`
-- no plan-owned buffer, `execute()` stays allocator-per-call. New function
`fft_exec_fused_pair` does the row-FFT-into-scratch / gather-from-scratch /
batched-axis-0 / scatter-to-user-memory sequence; falls back to the default
two-pass schedule when axis 1 isn't contiguous. As a byproduct, the
six-step decomposition's existing tiled transpose-with-twiddle loop
(`run_sixstep_`) was extracted into a shared `fft_transpose_tile_` helper
(`WithTwiddle` compile-time switch) so the new schedule's plain (no-twiddle)
transpose reuses it instead of duplicating the loop -- this refactor alone
is behavior-preserving (verified below) and does not depend on the macro.

The complete strict `test/algorithms_fft.cpp` build
(`-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror`)
and an `-fsanitize=address,undefined` build both passed, in all four
combinations (macro on/off x plain/ASan). A `.text` size check confirmed
the transpose-helper refactor alone is not a code-size regression for the
default (macro-off) build (496755 -> 495839 bytes, a small *reduction*,
consistent with the compiler sharing one out-of-line transpose
instantiation instead of duplicating it inline at every six-step call
site) and the new schedule adds only ~1.5KB (~0.3%) when the macro is on.

**Benchmark.** 2-D, generalized-many-3-D (h=32), and a 1-D control (this
macro's code path is structurally unreachable from D==1, included as a
noise-floor check), in-place, `FFTW_ESTIMATE` with wisdom disabled,
cold-cache, idle machine (calibration drift 3-6% across the two runs used
below; two earlier attempts were discarded for elevated drift from
background browser load).

| sweep | direct Multi/FFTW | fusedpair Multi/FFTW | fusedpair/direct time | sizes fusedpair wins |
|---|---:|---:|---:|---:|
| 1-D (control, unreachable) | 1.208 | 1.201 | 0.929 | 32/45 |
| 2-D | 1.220 | 1.372 | 1.118 | 1/28 |
| generalized-many 3-D, h=32 | 1.839 | 1.959 | 1.044 | 2/14 |

The 1-D control's ~7% divergence, despite the macro's code being provably
unreachable from that path, sets a noise floor from incidental cross-binary
codegen/layout differences alone. The 2-D (+12%) and many-3-D (+4% overall,
but only 2/14 sizes actually faster) regressions are well outside that
floor and consistent across almost every individual size, not concentrated
at a few outliers the way §11.25's regression was.

**Root cause.** The benchmark this project uses to evaluate every `fft.hpp`
change (`benchmark/algorithms_fft.cpp`) deliberately flushes a 64MB buffer
before *every single timed call*, precisely to simulate a cold-cache real
caller (see [[fft-flushed-cache-methodology]]). The fused schedule's one
remaining hypothesized win -- reading the axis-0 gather from a buffer still
warm from just being written -- is exactly the kind of warmth this
methodology is designed to erase: after the flush, both user memory and the
new scratch buffer start equally cold. So the schedule paid the cost of a
more complex code path and an extra `O(n)` scratch touch while structurally
forfeiting the only benefit it could have collected under this benchmark.
This is a different flavor of the same lesson as the twiddle-skip case
(§11.21/[[fft-flushed-cache-methodology]]): there, a hot-loop check hid a
real cold-cache cost; here, the intended benefit itself depends on warmth
that a cold-cache benchmark cannot show by construction.

**Conclusion.** Reverted (code only; this write-up and the transpose-helper
refactor's lesson -- that extraction was and remains behavior-preserving --
are the only lasting artifacts). §11.24's open thread is now closed: the
fused rank-2 schedule does not have a viable mechanism for improvement
under this project's cold-cache benchmark methodology, and there is no
indication a warm-cache scenario is what real callers of this library
experience either. Any future attempt in this direction should identify a
mechanism that does not depend on cache warmth carrying over between an
array's two axis passes before implementing anything.

### 11.27 Feasibility check for a size-32 flat codelet: naive shape loses 2.4-2.6x in the batched case that matters (2026-07-17)

Follow-up to §11.25 (which ruled out reusing the existing O(p^2) direct
kernel for n=16/32). The remaining, not-yet-tried mechanism for closing the
n=32 gap with FFTW (roughly 2x at that size) is a genuine FFTW-style
codelet: one flat, fully-unrolled O(p log p) pass with no intermediate
memory round-trip. Before attempting a real engine-integrated
implementation, this was scoped as a cheap, throwaway feasibility check
(prototype code lives outside the repo, in the session scratchpad --
nothing here touches `fft.hpp`), because this codebase has already failed
twice at "one bigger single-stage kernel" (all-radix-8 rejected outright;
`stage_radix16_`, §11.18, reverted after a 20-80% regression, root-caused to
register pressure) and a true codelet is architecturally further in that
same direction.

**What was built.** A from-scratch radix-16 reconstruction (`stage_radix16_`
was never committed -- git history confirms only its NOTES write-up landed
-- so there was nothing to recover; this rebuilds the documented shape: two
internal radix-8-style sub-DFTs on even/odd legs + a W16 combine, using
plain scalar `cx{re,im}` arithmetic with the same hand-expanded multiply
form `fft_ops` uses in production, to avoid `std::complex::operator*`'s
Annex-G branch biasing the comparison) as a calibration point, then a size-32
one-shot prototype built the same way one level deeper (two radix-16
sub-DFTs + W32 combine). Both verified correct against a naive DFT reference
(max abs error < 1e-9, forward direction, m=1 and m=3 batch, multiple random
trials).

**Stack-usage proxy (`-fstack-usage`), same production flags:** inconclusive.
The radix-16 reconstruction showed only 88-264 bytes -- smaller than
`stage_radix8_`'s own already-in-production 568 bytes, and far short of the
"~20 live values, roughly double radix8" the real (unrecoverable) failed
attempt was described as having. Either this reconstruction's algebra
happens to need fewer live values than whatever was originally tried, or
stack-usage genuinely doesn't capture the failure mode that hurt the
original (register pressure that hurts instruction scheduling without
literally spilling to the stack). The size-32 prototype did show a real
escalation (1184-1216 bytes, roughly 2x `stage_radix5_`'s 968), but this
number alone was not treated as decisive either way, precisely because the
radix-16 calibration point didn't reproduce the expected signal.

**Timing (hot-loop, not flushed-cache -- a fast filter only, per
[[fft-flushed-cache-methodology]]):** the picture reversed completely
between the unbatched and batched case.
- m=1 (unbatched): the flat codelet ran ~10-11% *faster* than the full
  `fft_plan<1>::execute()` path for n=32 (consistent across repeated runs).
- m=64 (batched, matching this size's actual `batch_width_()` clamp and
  what the 2-D/many3d benchmarks actually exercise): the flat codelet ran
  **2.4-2.6x slower**, not faster.

**Root cause, confirmed via `-fopt-info-vec-missed` (the "hard evidence"
[[fft-register-pressure-pattern]] asks for), not just inferred:** the
prototype's batch (`m`) loop calls a function taking 32 scalar-by-value `cx`
inputs and 32 scalar-by-reference outputs per batch element -- `gcc`
reported `couldn't vectorize loop` for exactly this loop. Production
kernels (`stage_radix4_`/`stage_radix8_`) never hit this: their batch loop
is written directly as a raw, contiguous array-indexed loop with every
twiddle constant hoisted *outside* the `j` loop, so the compiler sees a
uniform SIMD-izable stream over `j`. A naive flat codelet, structured as "one
function call per batch element," is invisible to the vectorizer as a
batch loop at all -- a *different* failure mode than the register-pressure
spilling §11.18 hit, discovered specifically because this check measured
the batched case instead of stopping at the (misleadingly encouraging)
unbatched number.

**Conclusion.** The naive, first-attempt codelet shape is a decisive loss
for the regime that actually matters (batched, as the real benchmarks
exercise) -- not adopted, no engine integration attempted. A codelet that
preserved the vectorization-friendly shape (batch loop as the primitive,
raw array indexing, twiddles hoisted outside) is still theoretically
possible, but restructuring for it means every one of the ~30 named
temporaries in the butterfly network (e0..e15, o0..o15, plus twiddled
intermediates) becomes a batch-width-wide vector quantity instead of a
scalar -- a substantially bigger rewrite than this feasibility check
attempted, and one with a real, not-yet-evidenced risk of reintroducing
`stage_radix16_`'s own register-pressure failure from a different angle
(vector-width-many live values instead of scalar-many). Not scoped further
in this pass; a future attempt should prototype that SIMD-preserving shape
specifically and re-run this same batched vec-info + timing check before
considering engine integration, rather than assuming the unbatched result
generalizes.

### 11.28 Size-32 codelet, both shapes prototyped + one integrated and benchmarked: reverted, 2-D n=32 regresses ~70% under cold cache (2026-07-17)

Full successor to §11.27, executed from a written plan. Two codelet shapes
were prototyped standalone (scratchpad, using the real `fft_mul_dir`
customization point so the multiply algebra matches production), then one
was integrated behind `BOOST_MULTI_FFT_EXPERIMENT_CODELET32` (off by
default) and run through the official flushed-cache benchmark. Both compute
exactly the default `[4,8]` stage pair (radix-4 ns=1, then radix-8 ns=4) in
one pass, twiddles from `tw_` via `fft_mul_dir<Backward>`, unity multiplies
structurally omitted.

**Phase-1 path trace (verified against source).** A 2-D 32x32 transform runs
its ROW pass as 32 separate m=1 fibers (`fft_exec_fiber` ->
`run_contig_inplace` -> fused `run_fused_impl_`) and its COLUMN pass as one
batched m=32 call (`run_fused`). So the codelet's real workload is
m=1-dominated (32x the fiber count of the single batched column call). Two
integration requirements found and confirmed: (T1) a single-stage `[32]`
plan makes `can_fuse()` (`stages_.size()>=2`) and `run_contig_inplace`'s
`>=2` gate both false, so without a gated fix the codelet silently falls off
the fused fast path onto slower gather/copy paths -- correct results, but
defeating the whole point; (T2) the fused in-place path passes `a==b`, so
the codelet (unlike every other stage kernel, where ping-pong guarantees
distinct per-stage buffers) must NOT `restrict`-qualify its user pointers --
safe because it reads all 32 inputs into local scratch before writing any
output. No size-32 sub-engine arises in any benchmark sweep (5-smooth sizes
=> no Bluestein/large-prime; six-step only for n>=8192 with balanced sides
>=~90), so only a genuine size-32 axis routes through the codelet.

**Two prototype shapes (Phase 2).**
- *Variant A (monolithic):* one `for j` batch loop, the whole 32-network in
  its body, 32-element local scratch. Vec-info: the batch loop does NOT
  vectorize (two consecutive inner loops); a structural inner loop
  vectorizes instead.
- *Variant B (layered, j-tiled):* radix-4 layer then radix-8 layer, each an
  innermost `jj` loop over a raw `[32][JT]` L1 tile. Vec-info: BOTH batch
  loops vectorize (layer 1 at 32-byte, layer 2 at 16-byte). This is the
  shape §11.27 concluded was needed.

Both correct to ~1e-13 vs naive DFT (forward+backward, m in {1,3,32,64,100},
interleaved and strided, and -- for the integrated one -- in-place a==b).

**Hot-loop timing (Phase 2, the misleading signal).** ratio = codelet/prod:
| m  | Variant A | Variant B |
|----|----------:|----------:|
| 1  | 0.55 (+45% faster) | 1.09 (9% slower) |
| 32 | 1.36 (36% slower)  | 0.78 (+22% faster) |
| 64 | 1.22 (22% slower)  | 0.80 (+20% faster) |
Variant A wins unbatched (as §11.27 found, larger here from fusing both
stages); Variant B wins batched (its vectorization paying off). Since the
2-D path is m=1-dominated, a hot-loop composite (32x row + 1x column)
predicted **Variant A ~19% faster** for the real 2-D transform and Variant B
~break-even -- so, against the plan's vectorization-centric design, VARIANT A
was integrated and benchmarked. (Explicit caveat recorded at the time: the
m=1 prod number carried per-call plan overhead the in-2D row pass does not,
so A's advantage was an upper bound; and hot-loop != flushed-cache.)

**Official flushed-cache benchmark (Phase 5), 2 runs, idle machine, n=32
rows (codelet vs base, +delta = slower):**
| sweep | run 1 | run 2 |
|---|---:|---:|
| 2-D (primary target) | +67.8% | +71.1% |
| 3-D | +10.2% | +7.6% |
| many h=32 | +40.4% | +47.8% |
| many h=256 | -11.7% | -10.7% |
| many3d h=32 (primary target) | -8.6% | -9.2% |

**Decisive reversal, and the mechanism.** The hot-loop composite predicted
A ~19% FASTER on 2-D; the real benchmark shows it ~70% SLOWER. The sign of
the effect tracks batch width: the codelet WINS the wide-batch cases (many
h=256, many3d h=32 -- one wide `run_fused` call amortizes the fetch of the
fully-unrolled kernel over 64+ fibers, and avoiding the inter-stage arena
round-trip saves real cold-cache memory traffic) and LOSES badly the
m=1-dominated cases (2-D, many h=32 -- the row pass makes 32 separate cold
codelet calls per transform; the fully-unrolled, non-vectorized kernel's
large code footprint is re-fetched cold each time, and its scalar per-element
work is slow). This is the §11.21/[[fft-flushed-cache-methodology]] lesson a
third time: a hot-loop win (icache warm, kept hot across reps) hid a
cold-cache icache/code-size cost that the flushed benchmark -- and real
callers -- actually pay. The m=1 dominance that made the composite favor A is
exactly what makes A worst under cold cache: 32 cold big-codelet fetches per
2-D transform.

**Decision.** G3 required the 2-D n=32 primary target to improve >=10%; it
regresses ~70%. Reverted (`fft.hpp` + benchmark back to HEAD, confirmed
diff-clean; this write-up is the only artifact). Both open threads from
§11.27 are now closed with evidence: the naive (A) shape loses under cold
cache in the case that matters, and the SIMD-preserving (B) shape, while it
does vectorize, was only ~break-even on the m=1-dominated 2-D path in
hot-loop and was not integrated (its batched wins don't help a workload
whose dominant cost is m=1 fibers). The structural obstacle is not the
kernel's arithmetic or vectorization but the schedule: n=32's 2-D cost is
dominated by 32 independent m=1 row transforms, and no single-fiber codelet
(vectorized or not) changes that the row pass pays a cold per-fiber cost 32
times. A future attempt would have to change the SCHEDULE (batch the row
fibers too), which is the packed-contiguous-batches direction already
reverted in §11.24 -- i.e. the codelet and the batching schedule would have
to succeed together, neither alone. Not pursued further.

### 11.29 Explicit AVX2 SIMD pilot (stage_radix4_, complex&lt;double&gt;): correct and bit-exact, but no measurable win -- likely memory-bound, not compute-bound (2026-07-17)

Prompted by this session's own diagnostic profiling (the "many"/batched sweeps'
steady-state ratio to FFTW is ~1.5-2.1x, clearly worse than the ~1.0-1.4x the
single-transform sweeps show, previously attributed to a batching-schedule gap
-- see the session's live diagnosis before this section). The project's
standing policy ([[fft-simd-policy]]) reserves manual SIMD intrinsics as a
"localized last resort"; this was that resort, attempted deliberately (not a
speculative try) after ~10 auto-vectorization-preserving kernel-shape
experiments (§11.13, §11.18, §11.24, §11.25, §11.26, §11.27, §11.28) all
failed or came back mixed, suggesting the auto-vectorizer itself had
plateaued below FFTW's hand-tuned-codelet throughput.

**Implementation.** An explicit AVX2 fast path inside `stage_radix4_`'s
existing `for j` batch loop only (structure otherwise untouched: block/r
loops, twiddle hoisting, `BOOST_MULTI_FFT_RESTRICT`), gated
`if constexpr(Batched && T==TW==std::complex<double>)`, `#if
defined(__AVX2__)` (no CMake/build changes -- code does not exist at all in a
non-AVX2 TU), with a `BOOST_MULTI_FFT_DISABLE_AVX2` escape hatch. 2 packed
`complex<double>` per `__m256d`, via the standard shuffle+`addsub_pd`
complex-multiply technique (the same pattern FFTW's own SIMD codelets use).
Deliberately `mul_pd`+`addsub_pd`, **not** fused `vfmadd` -- same operation
order as the scalar `fft_ops` path, so every lane's rounding is bit-identical
to scalar, not just close. `fft_ops`/`fft_mul_dir` (the customization point
`vec3` and mixed-precision executions rely on, `test/algorithms_fft.cpp:83-112,
511-548`) were not touched at all -- the SIMD path is a sibling branch inside
one kernel's body, invisible to (not instantiated for) every other `T`.

**Correctness**: strict build (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion
-Wsign-conversion -Werror`) clean in all four combinations (with/without
`-mavx2`, g++/clang++), `-fsanitize=address,undefined` clean in both build
modes (in particular confirms the unaligned `loadu`/`storeu` intrinsics never
read/write out of bounds). **New verification technique**: because the design
avoids FMA, built the test binary twice -- once with the SIMD path active,
once with `-mavx2 -DBOOST_MULTI_FFT_DISABLE_AVX2` forcing the scalar path --
ran both on identical seeded random inputs across every radix-4-exercising
1-D size (pure powers of 4, mixed composites) and batched "many" cases
including odd `howmany` (5,7,63,65, exercising the scalar remainder loop
after the SIMD pairs): 465,144 `complex<double>` output values, **bit-for-bit
identical** between the two builds -- stronger evidence than a tolerance
check, and confirms the intrinsics compute exactly what the scalar path
computes, not just approximately.

**Benchmark: no measurable win.** Two full runs each of `many h=32`, `many
h=256` (the sweeps that most cleanly showed the diagnosed gap),
`1-D`/`2-D`/`3-D`/`many3d h=32`, idle machine, calibration clean throughout.
Per-size AVX2/baseline ratio clusters at 0.98-1.02x **regardless of whether
the size's factorization uses radix-4** -- e.g. `many h=256` run 2: n=512
improved 7% (0.928) but n=128 and n=4096 (both radix-4-heavy) regressed 9%
and 3% (1.091, 1.026); no size showed a repeatable win across both runs.
Geomean ratio to FFTW barely moved anywhere (`many h=256`: 2.056 -> 2.025,
well within run-to-run noise per the two-run comparison). This is
indistinguishable from noise, not a signal tracking radix-4 usage the way a
real kernel-level win should (contrast §11.18's regression, which tracked
radix-16 usage cleanly in the same kind of per-size table).

**Decision.** Per this project's standing rule (a kernel change is judged by
the full benchmark, not by whether the technique is sound), reverted --
`fft.hpp` confirmed diff-clean against HEAD. Correctness and the bit-exact
technique are the reusable artifacts; the code itself is not kept.

**Working hypothesis for why (not confirmed -- no hardware performance
counters available in this environment; `perf` is access-restricted here).**
The "many" sweeps time one `execute()` call over the whole `howmany*n` block
after a single cold-cache flush -- e.g. h=256, n=1024 is 4MB, larger than
typical L1/L2. FFT has low arithmetic intensity (O(n log n) flops over O(n)
data -- few flops per byte moved), so once the working set exceeds cache,
wall time is plausibly dominated by memory bandwidth/latency, not ALU
throughput -- in which case doubling arithmetic throughput via SIMD has
nothing to buy. This would also retroactively explain the earlier diagnostic
finding that FFTW itself shows near-zero batch benefit above n~256 (this
session, live): if FFTW is *also* memory-bound at that point, its edge over
Multi is not raw per-element arithmetic speed (which this experiment now
suggests isn't the bottleneck for either library at these sizes) but
something about total memory traffic or access pattern -- fewer
passes/bytes moved per transform, not faster per-element compute. If this
holds, it reframes essentially the whole line of kernel-arithmetic
experiments in this file (§11.13 onward, now including this one): none of
them could have closed the gap, because the gap was never primarily an
arithmetic-throughput problem at the sizes these benchmarks probe. **Not
verified** -- would need `perf stat` (cache-miss/memory-bandwidth counters)
on a machine where hardware counters are available, or a controlled
in-cache-vs-out-of-cache working-set-size sweep, to confirm before treating
this as established rather than a plausible explanation for an otherwise
surprising null result.

**Follow-up: FMA variant, same null result.** Before accepting the
memory-bound hypothesis, re-tried with `_mm256_fmaddsub_pd` (fused
multiply-add: `x*wr_wr` and the `-/+ (x_sw*wi_wi)` addsub fused into one
instruction, replacing the separate `mul_pd`+`addsub_pd` pair) -- one fewer
instruction per twiddle multiply, in case the plain-AVX2 variant's lack of
effect was itself an artifact of leaving throughput on the table rather than
evidence of a memory-bound regime. Gated `#if defined(__AVX2__) &&
defined(__FMA__)` (confirmed `-mavx2` alone does NOT define `__FMA__`;
needs `-mfma` explicitly, though `-march=native` implies both). Fusing the
multiply-add means one rounding step instead of two, so results are no
longer bit-identical to the scalar path (a real, expected IEEE difference,
not a bug) -- quantified directly: 465,144 values compared against the
bit-exact reference dump, max absolute difference 2.8e-14, max relative
difference 1.4e-13, consistent with a few-ULP fusion effect and nothing
alarming. Full correctness gate repeated and clean (strict build all three
relevant flag combinations, ASan/UBSan, g++/clang++).

Benchmark (2 full runs, `many h=256` and the other sweeps, idle machine,
clean calibration throughout): **same null result as the non-FMA variant,
and the same tell that it's noise** -- the sign of the radix-4-vs-non-radix-4
differential flips between runs (run 1: radix-4 sizes geomean 0.992 vs
non-radix-4 control 0.962, i.e. radix-4 sizes did WORSE than the unaffected
control; run 2: radix-4 0.993 vs control 1.015, reversed), and individual
sizes flip sign run-to-run too (e.g. n=1024: 1.002 then 0.968; n=2048: 1.010
then 1.049, consistently regressing both times). Reverted, same as the
non-FMA variant -- `fft.hpp` confirmed diff-clean against HEAD again.

This strengthens rather than weakens the memory-bound hypothesis above:
shaving one instruction per twiddle multiply via fusion made no
measurable difference either, consistent with arithmetic instruction count
not being the limiting resource at these sizes under this benchmark's
cold-cache methodology. Still not confirmed via direct hardware-counter
measurement (unavailable in this environment) -- but two independent
SIMD variants (plain and fused) both landing on the same null result, with
the same "control sizes move as much as treatment sizes" tell both times,
is stronger evidence than either alone.

### 11.30 Closing diagnosis: the gap is memory-bound (stage count / bytes moved), not compute-bound -- constexpr twiddles ruled out, SIMD ruled out (2026-07-17)

Synthesis of the last several sections of this session (§11.24-§11.29), prompted
directly by §11.29's surprising null result: two independent explicit-SIMD
variants of `stage_radix4_` (plain AVX2 and FMA), both numerically verified,
both produced *no* measurable benchmark improvement. That result only makes
sense if raw arithmetic throughput was never the bottleneck -- this section
tests that directly, analytically and against measured data, without touching
`fft.hpp`.

**Twiddle-table traffic (the "would `constexpr` twiddles help" question):
ruled out, no experiment needed.** For every size in the `many h=256` sweep,
twiddle-table bytes touched (`tw_`, size `n`) are 250x-3000x smaller than the
data bytes touched (size `n * howmany`) per `execute()` call -- e.g. n=32:
twiddle traffic is 0.1% of total; n=4096: 0.03%. Even *completely* eliminating
twiddle-table loads (which `constexpr` values would not fully do anyway --
floating-point immediates still cost a `.rodata` load on x86-64, there is no
"free" way to materialize an arbitrary double in a register) caps out at a
fraction of a percent of total memory traffic. Not the lever, and cheap enough
to rule out by direct calculation rather than by building anything.

**Memory-pass count (bytes moved per `execute()` call): the dominant term,
and it quantitatively predicts measured time.** `fft_engine`'s factorization
(`fft.hpp:469-494`) reproduced in Python for every `many h=256` size: current
scheme does 1 full read + 1 full write of the whole `[howmany][n]` batch *per
stage* (matching `run_stages_`/`run_fused_impl_`'s ping-pong exactly), so
total bytes moved = `stages * 2 * howmany * n * sizeof(complex<double>)`.

| n | factorization | stages | bytes moved | predicted @ 20 GB/s | measured | pred/meas |
|---|---|---:|---:|---:|---:|---:|
| 256 | `[4,4,4,4]` | 4 | 8.39 MB | 0.419 ms | 0.353 ms | 1.19 |
| 512 | `[4,4,4,8]` | 4 | 16.78 MB | 0.839 ms | 0.807 ms | 1.04 |
| 1024 | `[4,4,4,4,4]` | 5 | 41.94 MB | 2.097 ms | 1.726 ms | 1.22 |
| 2048 | `[4,4,4,4,8]` | 5 | 83.89 MB | 4.194 ms | 4.073 ms | 1.03 |
| 4096 | `[4,4,4,4,4,4]` | 6 | 201.33 MB | 10.066 ms | 8.756 ms | 1.15 |

A single fitted bandwidth constant (20 GB/s, a plausible single-thread
DDR4/DDR5 figure, not independently measured -- `perf`'s memory-bandwidth
counters are access-restricted in this environment) predicts measured time
within 3-22% consistently across a 16x range of `n` spanning 4 to 6 stages
with genuinely different factorizations. A model driven purely by stage count
would not track measured time this closely across structurally different
factorizations if compute time (not memory traffic) were dominant -- this is
real, if not airtight, quantitative support, not just a plausible story.

**Synthesis with FFTW.** FFTW's codelet library (generated via `genfft`,
twiddles baked in as compile-time literals, unrolled up to much larger direct
sizes than this file's radix-8 ceiling) very likely does substantially fewer,
bigger passes per transform for the same `n` -- e.g. 2 large-codelet passes
where this library does 5 small-stage passes for n=1024. Under a
bandwidth-bound regime, halving stage count roughly halves bytes moved and
roughly halves wall time, which is the right order of magnitude for the
observed ~2-2.3x gap at n=1024-4096. `FFTW_ESTIMATE` with wisdom disabled
uses static heuristics, no runtime search -- so this is not FFTW being
"smarter," it is FFTW's *code* containing structurally less memory traffic,
baked in at code-generation time.

**Why this reframes §11.13-§11.29.** If correct, every kernel-arithmetic
experiment in this file's history -- split-radix, all-radix-8, twiddle-skip,
the `+-i` shortcut, mixed-precision twiddles, both explicit-SIMD variants --
was attacking a dimension (per-element compute speed) that was never the
dominant cost at the sizes these benchmarks probe. The two experiments that
DID attack the right dimension (fewer/bigger passes) both failed for
*implementation-specific* reasons compatible with the theory rather than
refuting it: `stage_radix16_` (§11.18, -20 to -80%) grew live-value count
per pass and hit a register-pressure ceiling; the size-32 codelet (§11.28,
2-D n=32 +67-71%) was correct and fast in isolation but was deployed into a
schedule (32 independent cold m=1 row calls per 2-D transform) where a large
unrolled kernel's own code-refetch cost dominated. Neither result contradicts
"fewer passes should help" -- both show that *how* you get fewer passes
matters as much as whether you do.

**Where this leaves the remaining lever.** The only mechanism left standing
with a real quantitative story is reducing stage count (bytes moved), and
the target is narrow: a scheme needs to fuse multiple current stages into
fewer passes *without* growing per-pass live-value count the way a bigger
flat radix kernel does, and *without* being deployed into an m=1-dominated
calling pattern the way the size-32 codelet was. Neither prior attempt found
that combination; this session did not attempt a third. Any future attempt
should design explicitly against both known failure modes from the start
(measure live-value count / register pressure before benchmarking, per
[[fft-register-pressure-pattern]]; check which calling pattern -- batched or
m=1-dominated -- the target size/context actually uses, per §11.28's lesson)
rather than discovering either after the fact.

### 11.31 Construction-time stride-aware `fft_plan`: investigated, not implemented -- evidence against it already in this file (2026-07-17)

Maintainer question following §11.30: could `fft_plan` learn the array's
stride/layout at *construction* time (it currently only knows extents/
directions -- `execute()` re-derives layout from the runtime `Cursor` fresh
on every call, via `fft_view_from_cursor`/`fft_layout_from`, `fft.hpp:
1470-1501`), instead of re-discovering contiguity via runtime branches
(`fib.stride()==1` in `fft_exec_fiber`; `get<0>/get<1>(slab.strides())==1`
in `fft_exec_slab`, `fft.hpp:1270-1393`) on every `execute()` call? Given
§11.30 identified memory traffic as the dominant cost, the question was
whether hoisting this decision reduces bytes moved or only removes branch
overhead.

**Investigated (read-only, no code written) rather than implemented,
because the evidence against it is already in this file's own history:**

1. The contiguity fast paths that actually change bytes moved (skipping a
   full gather+scatter pass when an axis is unit-stride) are already taken
   whenever the data warrants it -- the runtime check that selects them is
   already measured negligible: §11.17 (`fft.hpp` `fft_exec_slab`/
   `fft_exec_fiber` self-time, 0.40-1.26% of total wall time in full 2-D/3-D
   transforms). Hoisting an already-negligible per-call branch removes
   overhead that isn't measurably there.
2. The remaining runtime stride-dependent choices (`fiber_near` blocking
   order, `fft_min_abs_mid_stride`-driven axis reordering) affect *how* a
   strided gather/scatter is laid out, not *whether* one happens or how much
   data it touches -- no bytes-moved effect to hoist.
3. **Direct precedent already measured in this file**: §11.26 (this
   session) implemented and reverted a fused rank-2 schedule built on
   exactly the reasoning "reschedule *when* a strided gather happens to
   reduce traffic" -- its own finding was that total bytes moved does not
   change, because the gather is intrinsic to non-unit-stride data, not to
   when or how the decision to perform it is made. Construction-time vs.
   execute-time stride knowledge is a variation on "when," not a new
   mechanism -- §11.26 already answered the underlying question.
4. The dominant memory-traffic term §11.30 quantified (engine-internal stage
   ping-pong, `stages * 2 * howmany * n * sizeof(T)`) lives entirely inside
   `run_stages_`/`run_fused_impl_`, a layer this idea doesn't touch at all --
   even a perfect implementation of stride-aware dispatch couldn't reach the
   cost that actually dominates.

**Conclusion**: not implemented. Recorded as a "stop before building" case
distinguishable from most of this file's other reverted experiments (which
were plausible until measured) -- here, three independent already-measured
facts about this exact codebase point the same way before writing any code,
and one of them (§11.26) is a directly analogous already-completed
experiment for the underlying "does rescheduling around strides reduce
bytes moved" question, not merely a suggestive precedent. If a future
session revisits this, the open question is not "would hoisting the check
help" (answered: no) but whether there's a *different* mechanism construction-
time layout knowledge could unlock beyond re-timing an already-cheap
dispatch decision -- not identified in this pass.

### 11.32 A benchmark gap, not just a code gap: Multi's genuinely-batched path was never measured, and it beats FFTW_ESTIMATE broadly once it is (2026-07-17)

Direct follow-up to §11.31's trace: while confirming the call path for a
construction-time-known layout, it became clear that **no existing sweep in
this file's benchmark suite exercises Multi's genuinely batched (`m>1`,
fused, no gather/scatter) execution path in isolation.**

- `sweep_many` (`benchmark/algorithms_fft.cpp`) uses `{none, forward}` on a
  `(howmany, n)` row-major array -- the transformed axis is the *last*
  (contiguous) axis, which always routes through the m=1-per-fiber path
  (`fft_exec_fiber`/`run_contig_inplace`), never the batched fused path.
  Confirmed independently by this session's earlier diagnostic profiling,
  which measured "Multi batch benefit ~1.00x" for exactly this pattern --
  no internal batching happens at all, despite the name.
- The 2-D/3-D/many3d sweeps *do* reach the genuinely batched path (the
  "column pass" of any transform where both of the last two axes are
  active, `fft_apply_last_pair`), but only ever mixed with an m=1 row pass
  in the same reported number -- never isolated.

**The isolating pattern, traced and confirmed against source
(`fft.hpp:1645-1454`, `apply_`/`transform_axes_`/`fft_apply_last`/
`fft_exec_slab`):** `fft_plan<2>{(n, howmany), {forward, none}}` -- transform
the *non-last* axis, leave the last axis `none`. `apply_()`'s fused-pair
guard requires both `D-1` and `D-2` non-`none`, so it's skipped;
`transform_axes_` visits axis `D-1` (`none`, zero-cost no-op) then rotates
and calls `fft_apply_last` on axis 0 -- the same call shape
`fft_apply_last_pair` uses for its own column pass. In the rotated view,
`get<0>(strides)==1 && get<1>(strides)==howmany>1 && can_fuse()` routes into
`fft_exec_slab`'s fused batch-axis-contiguous branch (`fft.hpp:1325-1336`):
genuinely batched, `mt = min(mb_, howmany)`, no gather/scatter pass at all.
This is also directly FFTW's native "strided batch" advanced-interface
pattern (`istride=howmany, idist=1`), so it's a fair, apples-to-apples
comparison FFTW itself is built to make efficient too.

**Implementation.** Added `sweep_many_strided` to
`benchmark/algorithms_fft.cpp`, mirroring `sweep_many`'s scaffolding exactly
(same methodology: flushed cache, interleaved timing, plan-recycled arena
reuse) but with the array axes swapped (`(n, howmany)` instead of `(howmany,
n)`, `{forward, none}` instead of `{none, forward}`) and FFTW's istride/idist
swapped to match (`istride=howmany, idist=1` instead of `istride=1,
idist=n`). No `fft.hpp` changes -- purely a missing benchmark, not a missing
feature.

**Result: Multi substantially outperforms FFTW_ESTIMATE in this pattern,
for most sizes.**

| sweep | geomean (mine/FFTW) | Multi wins |
|---|---:|---:|
| many-strided, howmany=32 | 0.825 (Multi ~18% faster on average) | 10/14 |
| many-strided, howmany=256 | 0.541 (Multi ~85% faster on average) | 11/14 |

Sharp, consistent split by size:
- **n <= 128** (32, 64, 128 specifically -- the small power-of-two-ish
  sizes): Multi is slower, 1.2-1.4x. This includes n=32, the exact size
  targeted by the whole §11.25/§11.27/§11.28/§11.29 codelet/SIMD line.
- **n >= 243**: Multi is faster, often dramatically -- up to 3.4x faster at
  n=4096, howmany=256 (ratio 0.29). The advantage grows with batch depth
  (h=256 beats h=32 substantially at the same n), consistent with deeper
  batching amortizing a fixed per-call cost more effectively.

**Why this matters beyond one more data point.** Every prior "Multi trails
FFTW by ~1.2-2x" statement made earlier in this session's live diagnosis was
built entirely on sweeps that either measure single m=1-dominated transforms,
or -- as this section discovered -- a "many" sweep that (despite its name)
never actually exercises batching at all. The one code path that *does*
genuinely batch turns out to be one of Multi's strongest areas, not a
weakness, for most of the size range. This is a real correction to the
overall performance picture this session had been operating under, not a
footnote: the earlier ~2x-gap framing that motivated the codelet/SIMD
experiments (§11.25 onward) was accurate for the contexts those experiments
actually targeted (m=1-dominated row passes, and the mislabeled "many"
sweep) but should not be read as characterizing Multi's performance broadly.

**What this does and doesn't imply for the small-n gap.** The motivating
case for a size-32 codelet still exists -- Multi is measurably slower
exactly where one would apply it (n=32, both howmany values) -- but the gap
there (1.28-1.42x, likely fixed per-call/dispatch overhead given it
shrinks as `howmany` work-per-call grows relative to it, not the
~2x-and-worse arithmetic-throughput framing that originally motivated
§11.25/§11.27-29) is smaller and differently-located than previously
assumed. Any future codelet attempt should be scoped and measured against
*this* sweep (`many-strided`, isolated batched, both `howmany` depths)
specifically, not the row-dominated 2-D composite that misled the Variant-A
integration decision in §11.28.

### 11.33 A third axis pattern, {forward,none,forward}: the gap doesn't matter, the m=1 pass does (2026-07-17)

Follow-up to §11.32, requested directly: benchmark `{forward, none,
forward}` on a `(n, depth, n)` array -- two transformed axes (0 and 2)
separated by an untouched middle axis, unlike `sweep_many3d`'s adjacent
`{none, forward, forward}` pair (which hits the fused-pair fast path) or
`sweep_many_strided`'s single active axis. `apply_()`'s fused-pair guard
needs both `D-1` and `D-2` non-`none`; here `dirs_[D-2]` (axis 1) is `none`,
so the guard fails and the two active axes are handled by two *separate*
passes: axis 2 (last, contiguous) via the ordinary m=1-per-fiber path, axis
0 (via a once-rotated view) via whatever contiguity its rank-3 recursion
finds -- traced to still land on the same fused batch-contiguous mechanism
`sweep_many_strided` isolates (batched over axis 2's stride-1 extent), just
with an extra outer `depth` loop layered on top of both passes.

**Implementation.** Added `sweep_gap3d` to `benchmark/algorithms_fft.cpp`.
FFTW comparison needed the *guru* interface (`fftw_plan_guru_dft`), not the
simpler advanced interface `sweep_many`/`sweep_many3d`/`sweep_many_strided`
use -- the two transformed dimensions aren't a simple contiguous embedding
here (`dims = {n, stride=depth*n}` for axis 0, `{n, stride=1}` for axis 2;
`howmany_dims = {depth, stride=n}` for the gap axis). Verified independently
before trusting any timing: both Multi's plan output and the hand-derived
guru-interface parameters checked against a naive DFT reference on a small
case (n=8, depth=3) -- both correct to ~2e-14.

**Result: Multi is mostly slower here (geomean 1.66, 2/14 wins), and closely
tracks `sweep_many3d`'s numbers at the same sizes/depth** (e.g. n=8: 3.9-4.1x
slower in both; n=256: ~0.70x, a real win, in both) despite the structural
difference (gap vs. adjacent). This is itself the finding: **axis adjacency
is not what determines the outcome.** What both `gap3d` and `many3d` share,
and what `many_strided` lacks, is an m=1-per-fiber pass on the last axis --
`many_strided` deliberately places its one active axis in the *non-last*
position specifically to avoid that pass entirely, which is why it alone
shows the dramatic win. `gap3d` and `many3d` both keep axis 2 (or the
trailing pair) active and last, paying that pass either way, gap or no gap.

**Refines §11.32's diagnosis.** The dividing line isn't "batched vs. not
batched" or "adjacent vs. gapped" -- it's specifically whether *any* active
axis is last/contiguous (forcing an m=1-per-fiber pass into the composite).
Any transform configuration containing such a pass inherits its cost
regardless of what else the configuration also does well; only a
configuration that avoids it entirely (like `many_strided`'s single
non-last active axis) gets to show the batched path's real strength
undiluted.

### 11.34 Variant B's size-32 codelet, retested in its actual isolated-batched context: still a regression, not a win (2026-07-18)

§11.28's own hot-loop numbers showed "Variant B" (the layered, JT=16-tiled
size-32 codelet from the SIMD-preserving-codelet experiment,
`scratchpad/codelet32/proto.hpp`) winning +22%/+20% at batch widths m=32/64
-- but it was never integrated, because Variant A (monolithic) was chosen
instead to fit the then-current 2-D benchmark's m=1-dominated row pass. With
§11.32's `sweep_many_strided` now isolating Multi's genuinely-batched,
gather/scatter-free fused path (exactly the context Variant B's numbers came
from), this was the natural next thing to check.

**Integration** (`fft.hpp`, all under `BOOST_MULTI_FFT_EXPERIMENT_CODELET32B`,
off by default):
- New `stage_codelet32_<Batched,Backward,T>` kernel: Variant B's code
  verbatim, ported from the scratchpad's standalone `cx`/`fft_mul_dir` shape
  to the engine's `T`/`TW`-generic stage-kernel signature
  `(a, b, ns, mm, sa_, sb_)` (matching `stage_radix4_`/`stage_radix8_` etc.;
  `ns` is always 1 here and unused beyond an `assert`). Uses the existing
  `fft_tile_buffer<T, 32*16>` for its [32][JT=16] scratch tile -- no new
  buffer-management machinery.
- New `kind = 7`; constructor override replaces the ordinary `{4,8}`
  two-stage factorization of `nn==32` with a single `{32}`-factor, kind-7
  stage, so the codelet is reachable as the SOLE stage of a plan, not
  composed with anything else.
- `can_fuse()` narrowly relaxed: a lone kind-7 stage now also returns `true`
  (previously `stages_.size() >= 2` was required for the fused, no-gather-
  scatter path). Justified because a single codelet32 stage has the same
  aliasing-safety property multi-stage plans rely on for their first stage
  (`ns==1`, same in-place-safe kernel shape) -- this is the "single-stage
  case" extension flagged as needed in the original integration plan.
  Deliberately scoped to kind==7 only; does NOT relax the rule for other
  single-stage plans (e.g. small direct-kernel primes). `run_contig_inplace`
  updated to call `can_fuse()` instead of duplicating the `>= 2` check, so
  both call sites stay consistent by construction.

**Correctness gate: full green.** Strict `-O2 -Wall -Wextra -Wpedantic
-Wshadow -Wconversion -Wsign-conversion -Werror` build and the existing
`test/algorithms_fft.cpp` suite (including the `vec3` generic-type coverage)
passed for g++ and clang++, macro on and off (4 builds). `-fsanitize=address,undefined`
clean for both macro states (2 more builds). 6/6 green.

**Benchmark: consistent regression, not noise.** Built two benchmark
binaries (`-DDISABLE_WISDOM -DUSE_ESTIMATE`, otherwise identical), macro off
vs. on, and ran a focused n=32 check across `sweep_many_strided` (both the
target context, h=32 and h=256) plus `sweep_many`/`sweep_many3d`/`sweep_gap3d`
at n=32 as a regression check. Two full runs each (idle machine, calibration
drift <2% both times):

| pattern                    | baseline ratio (run1, run2) | codelet32B ratio (run1, run2) |
|-----------------------------|------------------------------|--------------------------------|
| many_strided h=32 (target)  | 1.392, 1.310                 | 1.597, 1.655                   |
| many_strided h=256 (target) | 1.364, 1.294                 | 1.744, 1.280                   |
| many h=32 (regression)      | 1.866                        | 2.100                          |
| many3d h=32 (regression)    | 2.726                        | 2.862                          |
| gap3d h=32 (regression)     | 2.269                        | 2.463                          |

(ratio = mine/FFTW, lower is better.) The target pattern (`many_strided`,
h=32) got repeatably *worse* with the codelet in both runs (+15%, +26%
relative to its own baseline); h=256 was worse in one run and roughly flat
in the other. Every other pattern checked also got mildly worse, never
better. **Reverted** (`fft.hpp` back to clean HEAD; `git status --short`
confirms byte-clean); no adoption.

**Why the isolated hot-loop win didn't translate.** The hot-loop comparison
in §11.28 measured only the codelet's own arithmetic/copy loop, in cache,
with no surrounding pipeline. In the real engine, `stage_codelet32_` adds an
extra tile copy in *and* out (`a`/`b` &lt;-&gt; the `[32][16]` tile) on top of
the single memory pass an ordinary two-stage `{4,8}` pipeline already does --
i.e. it trades two ordinary stage passes (each already reading and writing
the ping-pong buffers once) for one pass plus a tile-buffer round trip, which
is *more*, not less, memory traffic overall for this exact size. This lines
up with [[fft-simd-policy]]'s working hypothesis (memory-bandwidth-bound, not
compute-bound, once the batched working set leaves cache): a kernel-shape
change that adds memory traffic to save arithmetic loses here even when the
arithmetic-only slice of the work vectorizes better, because arithmetic was
never the bottleneck. Two independent codelet variants (A in §11.28's
integrated 2-D benchmark, B here in its own ideal isolated-batched context)
have now both regressed on contact with a real flushed-cache benchmark --
closes out the size-32 codelet line of investigation; no further variant is
planned without new evidence that memory traffic, not arithmetic, is the
lever being pulled.

### 11.35 Per-stage packed twiddle tables (§6's listed idea) -- implemented, correctness-verified, net wash to mild regression, not adopted (2026-07-24)

Maintainer-requested prototype of §6's "Per-stage packed twiddle tables"
suggestion, the one item in that list never actually attempted in any §11
session (unlike compile-time codelets, SIMD, mixed-precision twiddles, all
tried and reverted). Also asked, separately: whether compiler flags
(specifically `-ffast-math`, maintainer explicitly waiving IEEE conformance
this time) change the picture now that the access pattern is different.

**Hypothesis.** Radix-4/8/3/5 stage kernels load `tw_[k*r*tstep]` for several
`k` per butterfly -- for early/middle stages (`tstep` large), consecutive `r`
land on different cache lines (`tstep >= 4` complex-doubles per line means
each load touches a fresh 64B line for 16B of payload). Repacking each
stage's actually-visited twiddles into a small sequential buffer, built once
at plan-construction time, should turn that into streaming reads.

**Implementation** (`fft.hpp`, gated by `BOOST_MULTI_FFT_EXPERIMENT_PACKED_TWIDDLE`,
off by default -- zero diff when undefined): scoped to radix-4 stages only
(the dominant kernel: the power-of-two factorizer emits almost nothing else,
see the constructor comment at `fft_engine::fft_engine`). New member
`twp_`; constructor packs `(w1,w2,w3)` per `r` contiguously into it for every
radix-4 stage, offset stored in that stage's `st.aux` (unused by kind 2
otherwise) under a new kind 7. New `stage_radix4_packed_`, a byte-identical
twin of `stage_radix4_` except its three twiddle loads come from `twp_` at
that offset instead of the strided `tw_` computation. `run_stages_`/
`run_fused_impl_` gain a `case 7` next to `case 2`, both macro-gated.

**Correctness gate: full green.** Strict `-O2 -Wall -Wextra -Wpedantic
-Wshadow -Wconversion -Wsign-conversion -Werror` (g++ and clang++, macro on
and off -- clang caught an `-Wunused-but-set-variable` on the macro-off path
for the `ns_build` tracker, fixed by wrapping its declaration in the same
`#if` as its only use) and `-fsanitize=address,undefined` (same 4 combinations).
6/6 green.

**Benchmark: net wash, not the hoped-for win.** Used a trimmed copy of
`benchmark/algorithms_fft.cpp` (`sweep_many_strided` only, howmany 32/256,
n in {64..4096}, `-DDISABLE_WISDOM -DUSE_ESTIMATE`) -- the correct isolating
context per §11.32/§11.34 (genuinely-batched, gather-free, radix-4-dominated
for these sizes). Two runs per configuration to separate signal from noise
(run-to-run swings up to ~8% were observed even for the SAME binary, e.g.
baseline h32 n=256: 0.683 -> 0.736).

| n (h32) | baseline (avg of 2) | packed (avg of 2) | packed/baseline |
|---:|---:|---:|---:|
| 64 | 1.496 | 1.228 | 0.82 (packing wins) |
| 128 | 1.401 | 1.410 | 1.01 |
| 256 | 0.710 | 0.685 | 0.97 |
| 512 | 0.765 | 0.790 | 1.03 |
| 1024 | 0.566 | 0.603 | 1.07 |
| 2048 | 0.576 | 0.718 | 1.25 (packing loses, badly) |
| 4096 | 0.615 | 0.606 | 0.99 |

(ratio columns are mine/FFTW; the packed/baseline column is the one that
matters here, <1 = packing helped.) h256 showed the same shape, smaller
swings (0.96-1.15 range, no size above ~8% either direction). Geomean across
all 14 (size x howmany) points: **~1.003 -- statistically a wash**, built
from one real win (n=64, both howmany, -18%/-8%) roughly canceling one real
loss (n=2048/h32, +25%), with everything else inside the run-to-run noise
band measured above.

**Why the theory didn't pay off.** `twp_` is a strict *addition* to the bytes
touched per execution, not a replacement -- nothing stops reading `tw_`
itself (the radix-4 kernel's `imu = tw_[q]` scalar still reads it, and every
OTHER stage kind in the same plan still reads `tw_` directly). For these
problem sizes `tw_` is small enough (n=4096 complex<double> = 64KB) to mostly
stay resident in L2 across the repeated executions this benchmark's
methodology measures, so the strided-access penalty the hypothesis targeted
was likely already cheaper in practice than assumed -- consistent with
[[fft-flushed-cache-methodology]] and the §11.30/§11.34 pattern that adding
memory traffic to avoid a supposedly-expensive access pattern tends to lose
once the "expensive" pattern turns out to be cache-resident anyway. The one
consistent real win (smallest n, both batch depths) is the case with the
FEWEST stages and thus the SMALLEST added `twp_` footprint relative to what
it saves -- suggestive that a much narrower version (pack only the
first/largest-`tstep` stage, not every radix-4 stage in the plan) might
isolate the win without the added-footprint cost at larger n, but that is a
different, untested experiment, not this one.

**`-ffast-math` re-check, maintainer having waived IEEE conformance this
session: still a net regression, unchanged verdict.** Same trimmed sweep,
baseline (no packing) with vs without `-ffast-math` added to the existing
`-O3 -march=native -mtune=native -funroll-loops -fno-math-errno` flags:
every single size was flat-or-worse (h32: n=64 +47%, n=128 +41%, n=512 +17%,
n=1024/2048 +5-10%; the only two flat/better points were n=256 and n=4096,
within noise). Reconfirms §11.12 on a fresh code path -- the ban was never
about precision policy (which the maintainer has now explicitly waived
twice), it is a measured, repeatable speed regression on this codebase's
loop shapes, for a mechanism (aggressive reassociation defeating the
vectorizer's default instruction selection) that has nothing to do with
what's being packed or not.

**Disposition: reverted**, following this file's standing practice
(§11.34) of not leaving unadopted experimental code behind even under a
default-off macro -- `git diff` on `fft.hpp` confirmed clean after revert.
If a future session wants to chase the "smallest-n only" thread above, this
section is the starting point; re-derive rather than uncomment, since the
scoped-to-first-stage-only variant is a different offset-computation shape,
not a smaller diff of this one.

### 11.36 Parallelism: the file-header thread-safety claim, actually measured -- a real, zero-code-change ~5x lever (2026-07-24)

§6's future-work list named parallelism as unexplored; every §11 experiment
to this point is single-threaded. `fft.hpp`'s own file-header comment
already claims "concurrent `execute()` calls on the SAME plan object from
multiple threads are safe with no external synchronization needed" (a plan
owns no scratch; `execute()` allocates its arena locally per call) -- but
that claim had never been benchmarked, only reasoned about. This session
did, standalone, no `fft.hpp` changes (this is a usage pattern, not a
library change).

**Setup.** A standalone prototype (`std::thread` pool, one shared `const
fft_plan`, one persistent `arena_alloc` monotonic-buffer arena PER THREAD --
same idiom the official benchmark suite already uses for single-threaded
runs) against an embarrassingly-parallel workload: many independent
same-size 1-D transforms, split across threads. Reference: FFTW's own
`fftw_plan_with_nthreads` on the equivalent single batched
`fftw_plan_many_dft` call (the fair comparison -- FFTW threading its own
batch internally, not a hand-rolled loop). Machine: 6-core/12-thread
(i7-8700). Two workload scales per size (10x apart) to check the result
wasn't a short-run timing artifact -- both agreed.

| n, count | Multi 1thr | Multi 6thr (x) | Multi 12thr (x) | FFTW 1thr | FFTW 6thr | FFTW 12thr |
|---|---:|---:|---:|---:|---:|---:|
| 1024, 40000 | 153k/s | 775k/s (5.05x) | 811k/s (5.29x) | 325k/s | 945k/s | 937k/s |
| 4096, 10000 | 31k/s | 163k/s (5.19x) | 143k/s (4.55x) | 52k/s | 205k/s | 203k/s |
| 256, 160000 | 674k/s | 3011k/s (4.47x) | 3209k/s (4.76x) | 1458k/s | 3627k/s | 3533k/s |

**Result: real, substantial, and free.** Multi scales ~2x at 2 threads,
~3.5-3.7x at 4, peaks at **~4.5-5.3x at 6 threads** (matching this machine's
6 physical cores -- hyperthreading past that gives little or nothing more,
consistent with a memory-bandwidth-bound workload not benefiting from extra
logical threads sharing the same execution units/cache). FFTW shows the
same shape and roughly the same per-thread-count ratio to Multi as the
existing single-threaded gap (~15-20%, unchanged by threading) -- i.e.
**threading is a genuinely orthogonal lever**: it doesn't close the
per-transform gap §11.30 diagnosed as memory-bound, it multiplies whatever
throughput was already there by however many cores are actually available,
for free, using a capability the library already has.

**Why this is unlike every other §11 experiment.** Every compute-kernel
attempt (SIMD, codelets, split-radix, packed twiddles, ...) required
`fft.hpp` changes and risked regressions from added complexity or memory
traffic, for gains that kept evaporating on contact with the flushed-cache
benchmark. This lever requires **zero library changes** -- the thread-safety
was already designed in (§9.2's plan/scratch decoupling, specifically to
make this possible) and just needed a caller to actually use it and someone
to measure that it works. The only remaining gap is exposure: no example or
benchmark in this repo currently demonstrates the pattern, so a user reading
`fft.hpp` would have to derive the `arena_alloc`-per-thread idiom themselves
from the file-header comment and the benchmark suite's existing (single-
threaded) arena-reuse pattern.

**Not done in this session**: promoting the prototype into a permanent
benchmark/example, or documenting the pattern in the header/NOTES beyond
this record. Candidate follow-ups if this is pursued further: (a) add a
`sweep_parallel`-style entry to `benchmark/algorithms_fft.cpp` so the
scaling is tracked over time like every other sweep here; (b) a short
runnable example (not just prose) showing the shared-plan/per-thread-arena
pattern, since it is one indirection removed from the direct API surface
(`execute(home, alloc)` takes an allocator, not a thread pool).

### 11.37 Flattening adjacent untouched batch axes in `fft_apply_last` -- implemented, size-gated, ADOPTED (2026-07-25)

Maintainer-requested continuation of single-threaded improvement work,
targeting specifically the weakest measured N-D pattern: 3-D `{forward,
none, forward}` ("gap3d") was up to **5.6x** slower than FFTW (existing
data, `fft_bench_gap3d_h32_estimate.dat`), far worse than 2-D `{forward,
forward}` (0.80-1.25x) or 3-D `{forward,forward,forward}` (0.58-0.97x,
often a Multi win).

**Diagnosis.** 2-D is already near-optimal: the fused-pair path does both
axes in one full-array pass. 3-D-all-active needs a *minimum* of 2 passes,
not a missed optimization -- axis 0's transform needs every value along
that axis for a fixed (j,k), which doesn't exist until the fused-pair pass
over axes 1,2 has completed for every index; a third axis cannot be folded
into the same slab-resident pass (a mathematical/data-dependency argument,
not something that needed measuring -- directly analogous to §11.26's
"no viable fusion" finding for the rank-2 case, re-derived here from the
dependency structure for rank 3). The gap case (`{forward,none,forward}`)
is genuinely different: when the fused-pair guard fails (the middle axis is
`none`), `fft_apply_last`'s rank>2 branch reaches `fft_exec_slab` (the
efficient batched kernel, §11.32's strongest-measured mechanism) only after
peeling the OTHER untouched axis one index at a time through a plain C++
loop -- confirmed by tracing `sweep_gap3d`'s own file comment ("an extra
outer depth loop multiplying call count for BOTH passes"). This yields many
small batched calls instead of one wide one, even though for a plain
(C-order-adjacent) array the two peeled axes could be merged into a single
batch dimension with no data movement at all.

**Implementation** (`fft.hpp`, `fft_apply_last`'s rank>2 branch): Multi
already has the exact primitive needed -- `subarray::is_flattable()` /
`.flatted()` -- checking precisely "these two leading axes are adjacent in
memory" (outer axis's stride equals the inner sub-layout's own reported
element count) and merging them into one axis, one rank lower. Added a
check before the existing transposed-vs-plain peel: if flattable, recurse
on the flattened (one-rank-lower) view instead of looping. Recursing
(rather than a one-shot flatten) means a D>3 case with several
consecutively-flattable untouched axes collapses all the way to rank 2 in
one step when possible.

**First cut: real, but a genuine split, not a clean win.** Unconditional
flattening (`sweep_gap3d`, `{8,16,20,25,27,32,64,81,100,128,243,256}`,
depth=32, two runs to rule out noise -- confirmed reproducible, not noise):
n <= 32 gained (up to **-20%**, e.g. n=8: ratio 4.06 -> 3.23), but n >= 64
*regressed* (up to **+21%**, e.g. n=64: 1.41 -> 1.71). Net geomean ~1.02 --
a wash tilted slightly negative. Root cause of the large-n regression not
fully isolated (candidate: some size-dependent routing choice the
un-flattened peel path was incidentally getting right, bypassed once
flattening applies unconditionally) -- flagged as an open question, not
resolved, since the gate below made it moot for shipping purposes.

**Fix: gate on `eng.n_`, matching the observed crossover exactly.** Added
`eng.n_ <= 32` alongside `is_flattable()` (the crossover in the data above
is exactly there: n=32 flat/neutral, n=64 the first clear regression) --
same convention this file already uses for other empirically-tuned,
size-dependent routing thresholds (e.g.
`BOOST_MULTI_FFT_DISABLE_PACK_CONTIGUOUS_BATCHES`'s `nn>=48`). Re-measured,
two runs, averaged against the two ungated baseline runs:

| n | baseline | gated | change |
|---:|---:|---:|---:|
| 8 | 4.06 | 2.94 | **-27%** |
| 16 | 1.97 | 1.79 | -9% |
| 20 | 2.72 | 2.57 | -6% |
| 25 | 2.80 | 2.67 | -5% |
| 27 | 1.04 | 0.99 | -5% |
| 32 | 2.17 | 2.19 | flat |
| 64 | 1.41 | 1.47 | +4% (noise) |
| 81 | 0.82 | 0.78 | -5% (better) |
| 100 | 0.78 | 0.74 | -5% (better) |
| 128 | 1.27 | 1.37 | +7% (noise) |
| 243 | 0.73 | 0.73 | flat |
| 256 | 0.70 | 0.70 | flat |

Every remaining delta at n >= 64 is inside the ~5-8% run-to-run noise band
this session already established (repeat-run variance on this exact
sweep); every n <= 32 point is a real, well-above-noise win. Geomean across
all 12 points: **~0.955** -- a genuine ~4.5% net improvement, no downside
anywhere. Cross-checked against `sweep<3>` (`{forward,forward,forward}`,
which also reaches this same code path for its axis-0 pass): same shape,
n=8 improved (4.58 -> 3.58, **-22%**), n>32 flat-to-slightly-better, no
regression.

**Correctness gate: full green.** Strict `-O2 -Wall -Wextra -Wpedantic
-Wshadow -Wconversion -Wsign-conversion -Werror` and
`-fsanitize=address,undefined`, g++ and clang++, both the default (fix
enabled) and `BOOST_MULTI_FFT_DISABLE_FLATTEN_BATCH_AXES`-disabled states.
8/8 green. `test/algorithms_fft.cpp` already exercises 3-D mixed-direction
(`none`-containing) plans, so the flattened path is covered by the existing
suite, not just the benchmark.

**Disposition: ADOPTED, default-on**, with a disable escape hatch
(`BOOST_MULTI_FFT_DISABLE_FLATTEN_BATCH_AXES`) following the same
convention as `BOOST_MULTI_FFT_DISABLE_PACK_CONTIGUOUS_BATCHES` -- unlike
every other experiment in this file, this one is a clean measured win with
no offsetting regression once gated, so it ships rather than reverts.
Open thread for a future session: isolate why unconditional flattening
regressed at n >= 64 specifically (not done here, since the size gate made
it unnecessary for shipping) -- understanding that mechanism might allow
widening the `n_ <= 32` threshold instead of just working around it.

### 11.38 Software prefetch for the strided single-fiber gather/scatter -- tried, no measurable effect, reverted (2026-07-25)

Direct follow-up to §11.37, looking for further N-D headroom. Two related
ideas were considered and ruled out by reasoning alone before writing any
code (worth recording so they aren't re-attempted blind):

- **Applying §11.37's flatten trick inside `fft_apply_last_pair`'s own
  rank>2 loop** (used by the fused-pair mechanism itself, i.e. 3-D
  all-active and `{none,forward,forward}`): this loop peels its outer axis
  one slice at a time specifically so that, at the rank==2 base case, BOTH
  halves of the fused pair run on the SAME 2-D slab while it is still
  cache-resident ("slab still hot", the function's own comment). Batching
  the outer axis across many slices before the second half runs would
  separate the two halves in time for any given slab, destroying exactly
  that warmth -- the same mechanism §11.26 already measured (a warmth-
  dependent fusion has no viable win under this project's cold-cache
  benchmark methodology), just pulling in the opposite direction here. Not
  prototyped; the dependency argument is sufficient on its own.

**What was tried: software prefetch, a mechanism distinct from every prior
SIMD/`-ffast-math`/codelet attempt.** `fft_exec_fiber`'s strided gather/
scatter (`std::copy(fib.begin(), fib.end(), b)` and its scatter
counterpart -- reached whenever a single non-contiguous fiber is
processed, i.e. `stride() != 1`, the contiguous case already returns
earlier via `run_contig_inplace`) is a plain strided memory walk, one fresh
cache line touched per element once the stride exceeds a line's element
count -- the same cost shape as the early radix-stage twiddle loads §6/§11
already diagnose, just for the gather itself instead. `__builtin_prefetch`
a few elements ahead is a pure memory-system hint: no arithmetic, no
vectorization, not subject to [[fft-simd-policy]]'s SIMD/intrinsics ban (a
genuinely different lever than every previously-banned idea in this file).
Implemented behind `BOOST_MULTI_FFT_EXPERIMENT_PREFETCH_GATHER` (an
explicit loop replacing `std::copy`, since a strided iterator gives
`std::copy` no hook to interleave prefetches; `pf_dist = 8` elements
ahead), verified correct (strict `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion -Werror` and `-fsanitize=address,undefined`,
g++ and clang++, macro on/off -- 8/8 green).

**Benchmark: no measurable effect, in either direction.** `sweep<2>`,
`sweep<3>`, `sweep_many3d`, `sweep_gap3d` (same sizes as §11.37), macro on
vs off. One run showed an eye-catching outlier (2-D n=1024: ratio 0.979 ->
0.582, a seeming 40% win) -- isolated, no similar effect at neighboring
sizes (n=729, n=1215 both flat), which on its own is a reason for
suspicion rather than excitement. A focused 2-D-only re-run, twice more,
did NOT reproduce it (n=1024 stayed in the 0.978-1.019 band across all 4
runs) -- confirmed a one-off measurement fluke (background load, thermal,
or scheduling noise on a single data point), not a real effect. Looking at
the full 2-D sweep across two clean repeat runs: on/off deltas are the
same size as off-vs-off run-to-run noise at these sizes (e.g. n=24 alone
swung 0.657 -> 0.807 between two runs of the SAME unmodified binary) --
no consistent direction, no signal above that noise floor, at any size in
any of the four sweeps.

**Why no effect, plausibly.** `fft_exec_fiber`'s single-fiber (m=1,
non-contiguous) path is a narrower target than it first appears -- most of
the N-D sweeps here route through the batched `fft_exec_slab` path instead
(already cache-blocked, `kb=64` tiles, see its own gather/scatter loops),
so the code actually modified sees limited traffic in these particular
sweeps. It's also plausible the CPU's own hardware prefetcher already
handles a plain constant-stride walk adequately without a software hint --
consistent with [[fft-flushed-cache-methodology]]'s standing diagnosis
that this codebase's gap vs FFTW is memory-bound in TOTAL BYTES MOVED
(stage count, pass count), not in how efficiently a given strided access
is hidden once it happens.

**Disposition: reverted**, `git diff` on `fft.hpp` confirmed clean of the
macro afterward. Consistent with this file's practice of not leaving
unadopted experimental code behind. If a future session revisits
prefetching, do it against `fft_exec_slab`'s own (already-tiled) gather/
scatter instead -- that is the path actually carrying the bulk of traffic
in the sweeps that matter (2-D/3-D composite sizes), unlike the narrower
single-fiber path tried here.

### 11.39 §11.37's open thread, resolved: `mb_`'s cache budget didn't actually adapt with size -- root cause found (cachegrind), fixed, ADOPTED (2026-07-26)

Maintainer follow-up, motivated by a concrete real-workload grid (2-D and
3-D 32-128, three direction patterns, plus a first-ever 4-D case) that
landed mostly ABOVE §11.37's `eng.n_ <= 32` gate -- meaning that fix helped
almost none of it. This made §11.37's own open thread ("isolate why
unconditional flattening regressed at n_ >= 64") directly load-bearing
instead of a nice-to-have.

**No perf access** (`perf_event_paranoid=4`, no sudo requested/used) --
used `valgrind --cachegrind` instead (software-simulated counters, no
kernel permissions needed) for all measurement in this section.

**Baseline grid, measured fresh** (2-D/3-D 32/64/128, three 3-D direction
patterns, plus 4-D 32x128x128x128 `{none,fwd,fwd,fwd}` -- correctness-
verified against FFTW to 3.5e-15 before trusting any of its numbers, a
brand new pattern never exercised before): ratios 1.4-2.9x across the
board -- expected, since this grid is ALL pure powers of two, the single
hardest region for Multi (§11.25-11.29/11.34's exhausted codelet-gap
territory), not new evidence against those closed threads.

**Diagnosis, this time with real counters instead of a hypothesis.**
Cachegrind on a single gap3d execute() (n=64, depth=32), comparing the
shipped gated flatten (§11.37, inert at n_=64) against unconditional
flattening:

| | gated (no flatten @ 64) | unconditional flatten |
|---|---:|---:|
| I refs | 93,776,391 | 89,800,731 (-4.2%) |
| D1 miss rate | 23.2% | 28.2% (+5.0pp) |
| D1 write-miss rate | 39.7% | 50.6% (+10.9pp) |
| LL (L3) miss rate | 0.0% | 0.0% (flat) |

Flat LL confirms this is an L1-locality problem, not a bandwidth one. A
first hypothesis (the flatten check runs BEFORE the existing
`transposed()`-vs-plain smallest-stride heuristic, possibly short-
circuiting a better arrangement) was tested directly (reordered to check
flattenability AFTER that heuristic's choice) and made **no measurable
difference** -- ruled out, not the mechanism.

**Actual mechanism, found by instrumenting `run_fused_impl_` to print its
own `m` parameter** (a call counter, not a profiler, but decisive): gated,
96 total kernel calls -- 64 at `m=32`, 32 at `m=64` (the two gap3d passes
happen to batch at different widths, an accident of which axis each pass's
smallest-stride heuristic picks as outer vs batch). Unconditional flatten:
64 calls, **all** at `m=64` -- same total work (64*32+32*64 = 64*64 =
4096, conserved either way), just redistributed into uniformly bigger
calls. At `n=64`, one ping-pong buffer is `n*m*sizeof(complex<double>)` =
`64*64*16 = 64KB` at `m=64` -- **double the 32KB L1** -- versus `64*32*16 =
32KB`, exactly at the L1 boundary, at `m=32`. Flattening doesn't add work --
it removes the "coincidentally smaller, L1-friendlier" calls the
non-flattened path happened to have and replaces them with uniformly
worse ones.

**Root cause traced one level further: `mb_`'s own sizing.**
`batch_width_(nn)` targets a 4MB (`1<<22`) budget -- L2/L3-class, not L1 --
then clamps to a flat cap of 64. For every `nn` in this grid's whole range
(32-2048), the 4MB-budget computation itself always exceeds 64, so the
clamp saturates and `mb_` is **not actually nn-adaptive at all** over this
range -- it's a flat 64, coincidentally fine at small nn (buffer fits L1)
and provably not at nn=64 (buffer is 2x L1).

**Two candidate fixes measured, one overshoots.** (a) An L1-proportional
budget (`1<<15` instead of `1<<22`, so `mb_` shrinks as nn grows, e.g.
mb_=4 at nn=256): tested with cachegrind and the real sweep -- WORSE than
the flat 32 cap at large nn (gap3d n=243: ratio 0.71 -> 1.01; many3d
n=243: 0.70 -> 1.07). Too-small `m` reintroduces per-call dispatch
overhead, the opposite failure mode -- there is a real sweet spot, not a
monotonic "smaller is better" curve. (b) A flat, size-gated cap (64 for
nn<=32 -- unchanged, already-good regime -- 32 above it): matched or beat
option (a) everywhere it mattered, simplest to reason about, adopted.

**Also tested: does flatten's own gate still need to stay at n_<=32 once
the mb_ fix lands, or can it widen/go unconditional?** Compared (mb_ fix)
+ (flatten gated at 32, unchanged) against (mb_ fix) + (flatten
unconditional): differences were small and inconsistently signed across
sizes (noise-level) -- no reliable additional win from widening flatten.
**Flatten's gate stays exactly as shipped in §11.37; only the `mb_` cap
changed.**

**Result, full sweep, gap3d and many3d** (single confirmatory run against
the actual shipped file, not a scratch copy):

| n | gap3d before | gap3d after | many3d before | many3d after |
|---:|---:|---:|---:|---:|
| 64 | 1.473 | 1.370 | 1.980 | **1.629** |
| 81 | 0.780 | 0.795 | 0.802 | 0.808 |
| 100 | 0.738 | 0.765 | 1.379 | 1.369 |
| 128 | 1.366 | 1.332 | 1.900 | **1.640** |
| 243 | 0.732 | 0.729 | 0.842 | 0.784 |
| 256 | 0.697 | 0.676 | 0.749 | 0.674 |

n<=32 unaffected (same cap=64 either way; small deltas there are pure
run-to-run noise, confirmed by the logic being byte-identical). n=81/100
mixed, small (noise-level or a minor real effect at the
`pack_contiguous`/nn>=48 threshold boundary -- not investigated further).
n=64/128/243/256 -- the maintainer's actual stated range -- show real,
repeatable wins, many3d especially (-18% and -14% at n=64/128).

**Correctness gate: full green** (strict `-Wall -Wextra -Wpedantic
-Wshadow -Wconversion -Wsign-conversion -Werror` + `-fsanitize=address,undefined`,
g++/clang++) plus, since this changes a plan-construction sizing path used
by every plan in the library (not just FFT-specific benchmarks), the
**entire project ctest suite** (101/101 -- array core, BLAS, FFTW adaptor,
MPI, thrust/OMP) re-run and green.

**Disposition: ADOPTED**, `batch_width_`'s cap changed from a flat 64 to
`(nn > 32) ? 32 : 64` directly in `fft.hpp` (no macro -- this is a plan-
sizing constant, not a routing branch with a meaningful A/B toggle).
Closes §11.37's open thread: the n_>=64 unconditional-flatten regression
was never really about flattening itself, it was `mb_` handing that (and
every other) code path an L1-hostile batch width once nn grew past the
one size the flat-64 cap happened to suit.

### 11.40 A separate, unrelated `array_ref.hpp` fix found while investigating a downstream CI failure (2026-07-26)

Not an fft.hpp item, recorded here only because it surfaced mid-session:
a downstream project (inq) hit a GCC `-Werror=maybe-uninitialized` on
`array_iterator`'s converting copy constructor, tracing to
`subarray_ptr`'s defaulted default constructor leaving `offset_` (a plain
integral `difference_type`) genuinely uninitialized -- the existing
comment on that constructor only flags `base_` as intentionally left that
way; `offset_` had the identical problem, unmentioned. Some GCC version/
flag combination (not reproducing in this project's own local strict
builds) correctly-or-near-correctly detects this through
`array_iterator`'s inlined default/converting constructors. Fixed with a
zero-cost in-class default member initializer (`= 0`) on `offset_` only --
`base_` stays as-is (a "fancy pointer" in some contexts, where forced
initialization isn't free or generically possible; `offset_` is always a
plain integer, safe and free to initialize). Verified: full strict gate
(both compilers) and the entire ctest suite (101/101) green.

### 11.41 The `nn >= 48` packing threshold was stale after §11.39 -- removing it is both simpler and ~8% faster (2026-07-30)

Follow-up to §11.39, from a simple observation: `fft_exec_slab`'s
`pack_contiguous = nn >= 48 && mb > 1` was measured back when
`batch_width_` capped `mb_` at a flat 64. §11.39 changed that cap to 32
above `nn == 32`. Since the packed tile is `mb * nn` elements, halving
`mb_` halves the tile -- so the cost the threshold was protecting against
(a tile too large to stay L1-resident, losing more to misses than the
batched kernels win back) had already been removed. A constant tuned
against a since-changed constant is worth re-measuring, not trusting.

**Measured, four threshold values**, on 2-D, 3-D, gap-3-D, many-3-D and
both `many` sweeps (geomean of per-sweep geomeans; idle machine):

| threshold | overall |
|---|---:|
| `nn >= 96` | 1.414 |
| `nn >= 48` (previous) | 1.354 |
| `nn >= 32` | 1.336 |
| none (`mb > 1`) | **1.241** |

Monotonic in the same direction throughout, and the endpoint wins by a
wide margin: **~8% overall** against the previous threshold. Repeat runs
put the noise floor far below that (per-variant overall geomean reproduced
to 0.4% and 0.0% across two runs each), so this is signal, not drift.
Per sweep, dropping the threshold entirely: gap-3-D **-17%**, many-3-D
**-16%**, 3-D **-10%**, many(h=256) -5%, 2-D -2%; only many(h=32) went the
other way, +2%.

**Cross-checked on a real target grid** (2-D/3-D/4-D, 32-128, the
direction patterns a maintainer workload actually uses) with a useful
property: the change can only affect `nn < 48` -- above that, both
versions compile to the same path. Splitting the grid on exactly that line
separates effect from noise cleanly:

- `nn < 48` (path genuinely changed): **-25.9%**
- `nn >= 48` (path byte-identical): +2.8% -- necessarily noise, and a
  useful incidental calibration of how much run-to-run spread these
  power-of-two sizes carry when the machine is not fully idle.

**Disposition: ADOPTED.** `pack_contiguous` is now just `mb > 1` -- pack
whenever there is more than one fiber to batch. This is a rare case where
the simplification and the speedup are the same change: a tuned magic
constant disappears, and the code gets faster. The
`BOOST_MULTI_FFT_DISABLE_PACK_CONTIGUOUS_BATCHES` escape hatch is kept for
A/B work. Correctness: strict gate (`-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion -Werror`) and `-fsanitize=address,undefined`,
g++ and clang++, macro on and off, plus the full project ctest suite
(101/101) -- all green.

**Standing lesson**, worth applying beyond this instance: several constants
in this file were co-tuned against each other on one machine at one time
(§6 lists plan-time autotuning as future work precisely because of this).
When one of them moves, the others are suspect -- §11.39 moved `mb_` and
this section found the next domino. `fft_sixstep_min`, the radix-8 tail
rule, and `fft_max_direct_radix` have not been re-measured since and are
the obvious remaining candidates.

**Immediate application of that lesson, and a negative result worth
recording**: `mb_`'s own cap was itself tuned alongside the (now removed)
threshold, so it was re-swept against the new always-pack baseline --
16 / 24 / 32 (current) / 48 / 64 giving overall 1.235 / 1.210 / 1.234 /
1.247 / 1.291. The shape is sensible (a shallow U: too small forfeits
batching, too large overflows L1, and 64 is worst -- independently
reconfirming §11.39) and 24 came out best, ~1.6% ahead of the shipped 32
with all four repeat runs cleanly ordered (both 24-runs below both
32-runs, no overlap).

**Not adopted.** 1.6% against a 1.3% own-run spread is inside the band
where this file has repeatedly been wrong before (§11.35's packed twiddles
looked like a win at similar magnitude and were a wash), and every number
here comes from one machine -- exactly the portability caveat §6 raises
against hand-tuned routing constants. Changing a constant to a
non-power-of-two on that evidence would be trading a known-good value for
a marginally-better-on-this-box one. Recorded so a future session with a
second machine, or with the `fft_measure`-style plan-time autotuning §6
wants, can settle it with evidence this one does not have.

### 11.42 Two more co-tuned constants re-swept: `fft_sixstep_min` confirmed, the batch-width BUDGET was the stale one (~2%, and it was never really adaptive) (2026-07-30)

Direct application of §11.41's closing lesson, working through the
constants it named as unverified.

**`fft_sixstep_min` (2^13): confirmed correct, no change.** Swept the
exponent 11..15 on a large-n 1-D sweep (28 sizes, 4096..2^21):
geomean 1.419 / 1.418 / **1.416** / 1.427 / 1.536. 11-13 are a plateau
(differences well inside noise) and 13 sits at the bottom of it, so the
shipped value stands. The tail is informative though: at 2^15 the
non-power-of-two sizes just above the threshold blow up badly (15625:
1.25 -> 2.17, 20250: 1.32 -> 2.32, 24000: 1.60 -> 2.42), which is a
sharper demonstration than the original tuning note that six-step is
what keeps awkward large sizes competitive at all.

**The batch-width budget: swept, adopted at 2^18, then REVERTED at 2^22
after the harness was found incomplete.** `batch_width_` computes
`budget = 2^22 / (2*sizeof(TW)*nn)` then clamps to a cap; with a 4MB
budget that division exceeds the cap for most `nn` in range, so `mb_` is
really decided by the cap. Sweeping the exponent over
2-D/3-D/gap-3-D/many-3-D/many(h=32,h=256) showed a clean U with a minimum
at 2^18 = 256KB (overall 1.276/1.246/**1.220**/1.235/1.231/1.237/1.249 for
2^16..2^22), reproduced to 0.3%, and it was adopted on that basis.

**That was wrong, and the way it was wrong is the point.** The harness
contained the sweeps the change was aimed at and omitted the one it was
most likely to hurt. `many_strided` -- the genuinely-batched, gather-free
path, and the case where Multi is furthest AHEAD of FFTW -- depends
directly on `mb_` being wide, and a 256KB budget starves it: at h=256 it
went 0.548 -> **0.619**, 13% worse, on Multi's flagship result. Re-running
the sweep with `many_strided` included inverts the ordering completely:

| budget | 2^18 | 2^20 | 2^21 | 2^22 (shipped) |
|---|---:|---:|---:|---:|
| overall (with many_strided) | 1.0430 | 1.0249 | **1.0169** | 1.0222 |

2^18 is now **+2.0% WORSE** than the original. 2^21 is nominally best but
only -0.35% against a 0.23-0.34% repeat spread -- inside noise by the
standard §11.41 set -- so **2^22 stands unchanged**.

**Lesson, and it generalizes past this file:** a tuning harness must
include the workloads a change is most likely to *hurt*, not only the ones
it targets. Trimming the benchmark for iteration speed silently selected
for a favourable answer here. Every constant sweep in §11.41-11.43 used a
trimmed harness; §11.41's (packing threshold) and §11.43's (stride
conflicts) were separately confirmed against the full suite, but this one
was not until after adoption. The rule going forward: no routing constant
changes on trimmed-harness evidence alone.

**Knobs.** While sweeping, the four routing constants gained
`#ifndef`-guarded macro overrides (`BOOST_MULTI_FFT_MAX_DIRECT_RADIX`,
`BOOST_MULTI_FFT_SIXSTEP_MIN_LOG2`, `BOOST_MULTI_FFT_BATCH_WIDTH_CAP`,
`BOOST_MULTI_FFT_BATCH_BUDGET_LOG2`) with unchanged defaults. This is
worth keeping independently of any one experiment: every one of these was
tuned on a single machine (§6's standing caveat), and until the
`fft_measure`-style plan-time autotuning §6 wants exists, a compile-time
override is the only way a user on different cache geometry can correct
them without patching the header.

### 11.43 Power-of-two batch strides thrash cache sets in the outer-fused path -- routing those to the per-fiber path is worth ~2.4% (2026-07-30)

Found by looking at per-SIZE numbers instead of per-sweep geomeans, which
is what had been hiding it. In `many(h=256)` the batched 1-D sweep is not
uniformly bad -- it splits sharply by factorization:

| n | 81 | 125 | 243 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ratio | 1.00 | 0.87 | 0.79 | 2.17 | **2.80** | 2.48 | 2.35 |

The 3- and 5-smooth sizes are *wins*, while the powers of two next to them
are 2.2-2.8x losses. Size alone does not explain that (243 and 256 do
essentially the same amount of work), so it is not an arithmetic or
stage-count effect.

**Mechanism.** With a contiguous fiber axis, `fft_exec_slab` takes the
outer-fused path, which walks the batch directly in user memory using the
caller's fiber stride (`run_fused_outer` -> `run_fused_impl_`'s `ja`).
When that stride is a large power of two, successive batch elements are
separated by a power-of-two byte offset and map onto the same cache sets:
the tile then thrashes one set rather than using the cache. n=512 is
512*16 = 8KB apart, n=1024 is 16KB, and so on -- exactly the sizes that
regress. Non-power-of-two strides scatter across sets and are unaffected,
which is why 243 and 125 are fine.

**Fix, and a false start worth recording.** The first attempt let the
conflicting strides fall through to the *gather* path (blocked transpose
into scratch). That made things WORSE, not better -- h=32 n=2048 went
2.23 -> 2.70 -- because the gather pays for a transpose copy that buys
nothing when each fiber is already contiguous. Re-routing them instead to
the plain **per-fiber** path (one fiber at a time, straight from user
memory, no tile and no copy) is what works. Both alternatives had to be
measured; reasoning picked the wrong one.

**Result** (guard: fiber stride in bytes >= 8192 and a multiple of 8192;
8KB is the smallest stride at which the effect was actually observed --
at 4KB, n=256, packing still won):

| sweep | before | after |
|---|---:|---:|
| many(h=32) | 1.444 | **1.357** (-6.0%) |
| many(h=256) | 1.508 | **1.425** (-5.5%) |
| gap-3-D | 1.148 | 1.122 (-2.3%) |
| 2-D | 0.945 | 0.931 (-1.5%) |
| many-3-D | 1.302 | 1.285 (-1.3%) |
| 3-D | 1.083 | 1.110 (+2.5%) |
| **overall** | **1.2219** | **1.1927 (-2.4%)** |

Per size where the guard fires: h=256 n=512 **-22.6%**, n=1024 -13.1%;
h=32 n=1024 **-16.1%**, n=512 -7.5%. Reproducible: two runs of the new
code gave 1.1926 and 1.1928 (0.02% apart) against 1.2200/1.2239 before.

**Methodological note.** Individual sizes that the guard does NOT touch
moved by up to +-15% between runs even though their code path is
identical. Per-size numbers from a single run are therefore not
trustworthy on their own here; the geomeans (11 sizes) and the repeat runs
are what make the -2.4% believable. This also means §11.41's split-by-code-path
trick -- comparing only sizes the change cannot affect -- is the reliable
way to calibrate noise in this suite.

### 11.44 Single-stage plans were never batched at all -- the small-n outliers explained and mostly fixed (2026-07-30)

Found immediately after the readability pass, which is the point worth
recording: naming `stage_kind` and pulling the stride folding into one
helper made `fft_exec_slab`'s routing legible enough that the gap was
visible by reading it.

**The deficiency.** `can_fuse()` requires `stages_.size() >= 2` (a distinct
last stage must exist, so the first can safely alias the output). The
factorizer emits exactly ONE stage for n in {2, 3, 4, 5, 8} and for every
prime <= 64 -- n=8 in particular is a lone radix-8 stage under the
radix-8-tail rule. For those engines `can_fuse_outer()` is false, and the
contiguous-fiber branch fell through to the **per-fiber** loop: each fiber
transformed separately at m == 1, fully scalar, no batching whatsoever.
This is why every sweep's n=8 point was a 3-5x outlier (many4d n=8 was
5.5x, the single worst number in the whole suite) while n=9 and n=16 --
two stages, therefore fusable -- were fine.

**The fix is a routing change, not a kernel change.** The gather path
below already batches any engine: it gathers `mt` fibers into the
[frequency][batch] scratch, calls `eng.run(mt, ...)` -- which handles
single-stage, Bluestein and six-step engines alike -- and scatters back.
It was simply unreachable for contiguous fibers, because the
`!can_fuse_outer()` case was claimed by the per-fiber branch first. Letting
non-fusable engines fall through to it costs a gather/scatter pair and buys
a vectorized m == mt inner loop.

**Measured** (back-to-back A/B, two runs each, same binary pair):

| at n=8 | before | after |
|---|---:|---:|
| many-3-D | 4.072 | **3.346 (-17.8%)** |
| many-4-D | 5.537 | **4.955 (-10.5%)** |
| gap-3-D | 3.174 | 3.018 (-4.9%) |
| 3-D | 4.129 | 3.982 (-3.5%) |

Consistently negative across four independent sweeps, which is what makes
it signal rather than noise. Per sweep overall: many-4-D -3.6%, many-3-D
-1.8%, gap-3-D -1.6%, 2-D -0.6%, 3-D and many(h=32) flat; overall -0.9%.

**Noise calibration, for free.** many(h=256) moved +1.9% -- but its size
list starts at n=32 and every entry factors into >= 2 stages, so this
change *cannot* reach it. That +1.9% is therefore a pure noise
measurement on an unchanged code path, and a useful reminder that
per-sweep moves under ~2% in this harness mean nothing on their own. It is
the same split-by-code-path check §11.41 used.

**Still open.** n=8 remains the worst point in every sweep (3.0-5.0x) even
after the fix -- batching helps but does not close a gap that is really
about per-call overhead dominating a 8-point transform. The obvious next
step is the opposite of this one: make n=8 fusable at all, by having the
factorizer emit {4, 2} instead of {8} when the caller is going to batch.
That trades one wider stage for two narrower ones and would need measuring
against the radix-8 tail rule it contradicts (§6), so it is left as a
lead, not a change.

### 11.45 §8 step 1 ("extract butterfly bodies, pure refactor") is NOT free -- it costs 39% on CPU (2026-07-30)

§8's CUDA plan opens with: *"Extract butterfly bodies into `BOOST_MULTI_HD`
inline functions (the per-(block, r, j) work) shared by the host loops and
device kernels. **Pure refactor**; CPU codegen must not regress
(re-benchmark)."* Implemented it. The caveat in parentheses turns out to be
the whole story, and "pure refactor" is wrong.

**What was built.** Five free functions at namespace scope (deliberately
not `fft_engine` members, so device code need not instantiate the engine):
`fft_butterfly2/3/4/5/8`, each taking the twiddles plus already-loaded
inputs and producing the outputs, with all addressing left in the caller.
That is the correct separation on paper -- arithmetic is identical on host
and device, while the loop nest is exactly where CPU cache blocking and GPU
thread mapping diverge.

**Three calling conventions, all lossy.** Measured by counting packed-SIMD
instructions (`vmulpd|vaddpd|vsubpd|vfmadd|vfnmadd|vfmsub`) in the
optimized benchmark object, against 2498 for the inline kernels:

| convention | packed SIMD |
|---|---:|
| inline kernels (shipped) | **2498** |
| return `std::array<T, R>` | 1458 (-42%) |
| outputs as `T&` out-params | 1828 (-27%) |
| inputs by value, outputs `T&` | 2114 (-15%) |

**Wall clock, which is far worse than the proxy suggested.** Best variant
(2114, "only" -15% on the instruction count), back-to-back A/B, two runs
each: **+39.1% overall**, and not marginal anywhere -- gap-3-D +59.7%,
many-3-D +58.5%, 2-D +56.2%, 3-D +42.6%, many(h=256) +40.8%. Run spreads
0.02% and 0.62%, so this is not noise. Reverted; the restored file
re-measures at exactly 2498.

**Mechanism.** The stage kernels get their vectorization from
`BOOST_MULTI_FFT_RESTRICT` on the `a`/`b` pointers: the compiler knows
loads and stores cannot alias, so the `j` loop vectorizes. Passing
individual *elements* across a function boundary launders that away --
inside the butterfly there are only references, with no restrict
relationship the optimizer can see. Loading at the call site (inputs by
value) recovers part of it, which is why that variant scored best, but not
all: the stores still cross the boundary. This is the same class of failure
as §11.9 (explicit `fma()` reduced vectorization) and §11.1 (`unseq`) --
anything that puts a call boundary between the restrict pointers and the
arithmetic costs more than it saves.

**What this means for the GPU port** -- and it is a real constraint, not a
detail:

- **The host and device paths cannot share butterfly *functions*** without
  giving up ~40% of CPU performance. §8 step 1 as written is not viable.
- The remaining options are all worse than the plan assumed: duplicate the
  arithmetic (host keeps today's inline kernels, device gets its own copy,
  and the two must be kept in sync by tests rather than by construction);
  share via macros (preserves inlining, but macro-defined numerics are
  their own maintenance problem); or share a *loop-level* function that
  takes the restrict pointers and the whole `j` range -- which keeps CPU
  codegen but is useless to a device kernel, since there one thread wants
  ONE butterfly, not a loop over `m`.
- The third option is the right one, and it is now **measured, not
  assumed** (this was first written as a hypothesis; the numbers below
  were added after testing it).

**Stage-level extraction: verified free.** Hoisted `stage_radix4_` out of
`fft_engine` into a namespace-scope `fft_stage_radix4(tw, n, a, b, ns,
strides...)` -- same loop, same `BOOST_MULTI_FFT_RESTRICT` pointers, tables
passed in as parameters instead of read off `this`. The member became a
one-line forwarder.

| variant | packed SIMD | wall clock |
|---|---:|---:|
| inline kernels (shipped) | 2498 | baseline |
| **stage-level free function** | **2582** | **-1.5%** |
| butterfly-level (§ above) | 2114 | +39.1% |

Vectorization is fully preserved -- marginally *better*, in fact -- and
wall clock is neutral-to-slightly-faster (many(h=32) -4.8%, many(h=256)
-5.0%, everything else within noise; two runs 0.30% apart). The boundary
is harmless precisely because the `j` loop and the `restrict` pointers stay
together inside one function, which is the property the butterfly-level
split destroyed.

**So the port has a viable shape after all**: the stage is the shareable
unit. A device implementation mirrors the same signature (tables +
restrict pointers + extents in, one pass out) and supplies its own thread
mapping, while the host keeps today's vectorized loop. What cannot be
shared is only the innermost arithmetic, which will have to be duplicated
between host and device and kept in sync by tests rather than by
construction.

The experimental hoist was reverted rather than shipped, since converting
one of seven stages leaves an inconsistent file; the finding is what
matters, and doing all seven is a coherent follow-up that now has evidence
behind it.

Recorded so the next attempt does not start by re-doing the butterfly
extraction and re-discovering the 39%. §8 step 1 should be struck and
replaced with the stage-level framing above.

### 11.46 Plan-time autotuning (`fft_measure`): implemented twice, both measurably WORSE, not shipped (2026-07-31)

§6 lists plan-time autotuning as "FFTW's actual planner advantage" and it is
the one item on that list never attempted. This session made the case for it
concrete: §11.41-11.43 showed that constants here go stale when a neighbour
moves, that a value winning on one access pattern can lose on another
(§11.42's revert), and that the `mb_` cap's optimum sat inside noise on this
box (§11.41) -- i.e. problems hand-tuning structurally cannot fix. Built it.
It does not work yet, and the reasons are worth more than the code was.

**Design and API.** `enum class fft_planning { estimate, measure }`, a
defaulted third constructor argument so existing call sites are untouched,
plus a `batch_width(axis)` accessor so a caller can see what was chosen.
The tuning coordinate is the batch-width **cap** (one number, shared by all
engines including sub-engines, threaded through `fft_engine`'s constructor)
rather than per-engine widths: it is a 1-D search, it moves all engines
coherently, and it is exactly the constant §11.39 proved matters.

**Attempt 1 -- per-engine microbenchmark: +1.3% worse.** Timed
`engine.run(m, ...)` on a scratch buffer of the engine's own length, keeping
the width with the best time per fiber. Correct (bit-identical output; `mb_`
changes tiling, not arithmetic) and it did pick distinct widths -- n=32 ->
16, n=64 -> 8 against defaults of 64 and 32. But it measures an isolated,
**hot-cache** 1-D run, and the real path is cold-cache N-D. It therefore
systematically preferred narrow widths, which lose once each call starts
cold. This is [[fft-flushed-cache-methodology]] exactly, and it was cited
earlier in this same session (§11.35) before being walked into again.

**Attempt 2 -- whole plan, real shape, flushed cache: +1.8% worse.** Probe
the actual `execute()` on a C-order scratch array of the planned shape, with
a 32MB flush between timings, five candidate caps, min-of-reps, and a margin
below which the default is kept. Now measuring the right thing, and much
more conservative (it mostly kept the default). Still behind.

**Attempt 3 -- stricter acceptance (7 reps, 12% margin): +2.4%, WORSE than
attempt 2.** This is the informative one. A stricter margin should mean fewer
deviations and therefore convergence *toward* `estimate`, so getting further
away falsifies "the margin is the only mechanism". The likely confound is the
probe itself: measure mode transiently allocates ~96MB (probe array +
flusher) per plan construction, and the benchmark builds one plan per size,
so the process's page mappings going into the timed region differ from the
`estimate` run's. A tuner that perturbs the state it is about to measure is
measuring itself.

**The deeper problem, and why no amount of margin fixes it here.** This
machine is the box every default in this file was fitted on, this session
included. Measure mode has nothing to find; every deviation it makes is
noise-driven, and noise can only cost. **The feature cannot demonstrate value
on the machine whose constants were tuned to it** -- that is not a bug in the
implementation, it is a property of the experiment, and it means this box can
only ever produce evidence against.

**Disposition: reverted, not shipped.** Shipping a mode that measurably costs
1.8-2.4% when enabled, justified only by theory, would repeat §11.42's
mistake (adopting on a harness that could not see the downside).

**Follow-up: the mechanism was then validated separately, and it is sound --
the margin was the bug.** "Does the tuner need different hardware to be
proven?" turned out to conflate two questions, only one of which needs it:
*can a short plan-build measurement rank caps correctly* (testable here) vs
*is the shipped default already best here* (yes, hence no gain). Testing the
first: compile-time cap in {8,16,32,64}, and for each, a SHORT measurement
(3 flushed reps, what a tuner can afford) against a LONG one (40 reps,
ground truth), three passes:

| shape | short winner | long winner | per-pass agreement | short spread |
|---|---|---|---|---:|
| 3-D 64^3 | cap 32 | cap 32 | **3/3** | 5.1-13.9% |
| many-3-D 32x64^2 | cap 16 | cap 16 | 2/3 | 2.3-7.4% |

So the short probe **reliably rejects a clearly-wrong cap** -- for 3-D 64^3
cap 8 is +26% and cap 64 is +14% against the winner, and it caught that every
pass. It **cannot resolve near-ties**: in many-3-D caps 16 and 32 differ by
1-4%, inside the 2-7% spread of a 3-rep measurement, so it coin-flips.

That is exactly why attempts 1-3 lost. On this machine every candidate sits
within a few percent of the default, so the tuner was never detecting a real
difference -- it was flipping coins among near-ties, and each wrong flip
costs. It is not that measurement cannot work; it is that it was being asked
a question below its resolution.

**This supplies the number §11.46 was missing.** The acceptance margin must
exceed the short-measurement spread, measured here at **3.5-13.9%**. The 3%
margin of attempt 2 was far too loose and 12% (attempt 3) still marginal; a
defensible threshold is **>=15%**, or more reps to shrink the spread at the
cost of plan-build time. At >=15% on this box the tuner would essentially
never fire (harmless), while still correcting a default that is badly wrong
elsewhere -- which is the actual use case.

What a future attempt needs, in order:

1. **A >=15% acceptance margin** (or enough reps to justify a tighter one),
   now quantified rather than guessed.
2. **A probe that does not disturb the process** -- reuse one static buffer
   across all plans rather than allocating ~96MB per construction, and size
   the flush to the machine's LLC instead of a fixed 32MB. Attempt 3's
   inversion (stricter margin, WORSE result) is unexplained without this.
3. **A second machine with different cache geometry** -- no longer needed to
   validate the mechanism, only to demonstrate an actual win. This box is
   32KB L1d / 256KB L2 per core; anything with a materially different L2
   (AMD Zen 512KB-1MB, Ice Lake+ 48KB L1d, Sapphire Rapids 2MB L2, Apple
   M-series 128KB L1d) would be a real test.
4. Only then, tune. Candidate coordinates in priority order: the batch-width
   cap (§11.39), the packing/stride-conflict thresholds (§11.43), the flatten
   gate (§11.37).

The API shape (`fft_planning` argument, `batch_width()` accessor, cap
threaded through `fft_engine`) is worth reusing verbatim -- it was the easy
part and it is right. The measurement is the hard part and remains unsolved.

### 11.48 The stage hoist was adopted and then REVERTED: it trades 1-D for batched 1-D, and the harness could not see it (2026-08-01)

§11.47 shipped the six-kernel stage hoist on the strength of -7.0% on
many(h=32), -5.5% on many(h=256) and -1.6% overall, measured with
`bench_tune4.cpp`. **That harness contains no `sweep<1>`.** A focused 1-D
A/B afterwards:

| | members | hoisted |
|---|---:|---:|
| 1-D geomean (47 sizes) | **1.1961** | **1.2983 (+8.5%)** |
| 1-D wins | 13/47, 11/47 | 7/47, 6/47 |

Reproducible (member spread 0.1%, hoisted 2.0%), and brutal per size:
n=512 +73%, n=729 +67%, n=625 +61%, n=243 +56%, n=1024 +56%. Sizes across
every radix family, so it is not one kernel.

So the hoist is not the free groundwork §11.45 predicted -- it **trades**:
the batched paths gain 5-7%, the single-fiber (m == 1) path loses 8.5%.
Plausibly the m == 1 path is latency-bound rather than throughput-bound,
where the extra template parameter's instantiation growth (text +11%) and
the forwarding layer cost more than the `restrict`-qualified table pointer
wins. Not diagnosed further.

**Reverted.** A refactor justified as "neutral, for the GPU port" that
costs 8.5% on the largest sweep in the suite does not meet its own bar.

**This is the SECOND time this session the same mistake shipped**, and it
is worth naming as a pattern rather than an incident:

- §11.42: batch budget adopted on a harness omitting `many_strided` -- the
  one sweep a narrow `mb_` starves. Cost 13% there. Reverted.
- §11.47/this: stage hoist adopted on a harness omitting `sweep<1>` -- the
  one sweep the change hurts. Cost 8.5%. Reverted.

Both times the trimmed harness was built for iteration speed, and both
times trimming silently selected for a favourable answer, because the
sweeps dropped were the ones least like the change's target. The rule
§11.42 wrote down ("a tuning harness must include the workloads a change
is most likely to hurt") was correct and was not followed here, because
the hoist was framed as a refactor rather than a tuning change and so felt
exempt. It was not. **Any change touching the stage kernels must be
measured against the FULL suite, refactor or not.**

**For the GPU port**: the stage-level interface is still the right target
(§11.45's finding stands -- the butterfly is not shareable, the stage is),
but it cannot be reached by hoisting the existing kernels wholesale. A
future attempt has to keep the m == 1 path unaffected -- e.g. hoist only
the `Batched == true` instantiation, or give the device its own stage
implementations against the same signature without moving the host ones.

---

## §12 GPU porting design notes

> **Review note**: the GPU-specific statements in this section (kernel launch
> strategy, shared-memory limits, synchronisation model) should be reviewed by
> Fable before any implementation begins.

### 12.1 Plan internals must become allocator-aware for GPU

For GPU execution, the GPU kernels only access the twiddle tables — the O(n) bulk
data.  Control structures (`stages_`, `sub_`, engine metadata) live on the host and
orchestrate kernel launches; they can stay as `std::vector` on the host.  Only the
table vectors need to reside in device memory:

- `tw_` — twiddle table, size n
- `wmat_` — concatenated DFT matrices for generic radices, size O(n) in the worst case
- `chirp_`, `postc_`, `kernel_ft_`, `kernel_ft_bwd_` — Bluestein tables, size O(conv_n)

Switching these from `std::vector<TW>` to `multi::array<TW, 1, Allocator>` (with
`fft_engine` / `fft_plan` gaining an `Allocator` template parameter) would allow
them to live in device memory by passing a device allocator to the plan constructor.
`multi::array` is exactly the right container here since it is already
allocator-aware.  The remaining `std::vector` members (stages, sub-engines) are
unaffected.

**Blocker**: the construction code uses `push_back` / `emplace_back` / `resize` +
offset tracking to build tables incrementally during factorisation.  `multi::array`
does not have `push_back`.  The fix requires a two-pass construction: first walk the
factorisation to compute the final sizes, then allocate and fill.

### 12.2 Engine count is already bounded; sub-engines are the remaining dynamic allocation

`fft_plan::engines_` is already `std::array<fft_engine<TW>, D>` (line 1606) — D
slots, a compile-time constant, no heap allocation.  Currently `distinct_count_` (≤ D)
records how many slots are actually used and gates iteration over them.

Applying the same sentinel convention as `sub_` and `stages_`: initialise all D
slots to `fft_engine(1)` and let real engines overwrite from the front.
`fft_engine(1)` is a natural trivial sentinel — its constructor hits `if(n_ < 2)
return` immediately (no twiddle tables built) and any `apply_()` call on it also
returns immediately.  With this invariant `distinct_count_` becomes unnecessary: all
D slots can be iterated unconditionally with no data-dependent branch.

The remaining dynamic allocation in the engine tree is `fft_engine::sub_`, a
`std::vector<fft_engine>` holding nested engines:

- Bluestein: 1 sub-engine (the convolution sub-plan)
- Six-step: 2 sub-engines (the n1 and n2 sub-transforms)
- Generic prime stages: 1 sub-engine per stage (rare; usually 0–1 in practice)

In practice `sub_.size()` is 0, 1, or 2 for the vast majority of transform lengths;
it is bounded by the number of stages which is at most log₂(n) ≤ 30.  Replacing
`sub_` with `std::array<fft_engine, MaxSub>` (e.g. `MaxSub = 4` for almost all
cases, `MaxSub = 32` for a conservative bound) would eliminate this last per-engine
heap allocation.

`stages_` (one entry per radix stage, also ≤ log₂(n)) and the local factorisation
scratch `fac` (a `std::vector<std::size_t>` inside the constructor) are similarly
bounded and could become `std::array<stage_t, 32>` / `std::array<std::size_t, 32>`.

For all of these fixed-size arrays the preferred sentinel convention is: **all slots
are initialised to `1`**; valid entries overwrite from the front.  `1` is the
multiplicative identity, so a stage with `radix == 1` is a no-op and an engine with
`n_ == 1` is a length-1 DFT (also a no-op).  The primary benefit is **branchless,
vectorisable iteration**: the whole array can be processed unconditionally with no
data-dependent branch, no early exit, and no separate count variable.

**CPU impact**: these changes do not meaningfully speed up `execute()` on CPU.  The
hot path is the butterfly arithmetic; the stage loop is ≤ 30 iterations already in
L1 cache regardless of storage type.  Moving `stages_` from a heap-pointed vector to
an in-struct array gives at best a rounding-error cache-locality improvement.  The
motivation is GPU porting, not CPU performance.

### 11.49 The fused-pair pass silently cancelled §11.37's batch flattening for rank >= 3 -- gating it on slab size is worth 4-10% at small n (2026-08-01)

Two optimizations, adopted a week apart, have been undoing each other ever
since the second one landed:

- §2.5's **fused-pair pass**: when the last two axes are both active,
  `apply_()` sends them through `fft_apply_last_pair`, which descends to
  rank-2 slabs and transforms both axes while the slab is cache-hot.
  Halves the full-array sweeps. A real, measured win.
- §11.37's **batch-axis flattening**: `fft_apply_last` merges adjacent
  untouched leading axes into ONE wide batch before calling
  `fft_exec_slab`. Gated on `eng.n_ <= 32`, "where per-call overhead
  dominates". Also a real, measured win.

They are mutually exclusive for rank >= 3. The pair pass's descent hands
`fft_exec_slab` one slab at a time -- `m == sizes_[D-2]` -- and never
reaches the flattening code at all. On `32 x 8 x 8 x 8 {none,f,f,f}` that
is **m == 8 against m == 2048**. Both were benchmarked against a tree
containing the other, so both looked like wins; the interaction is
invisible unless the two paths are compared directly at one shape.

Decomposition of `many4d` n=8 (hot loop, 16384 elements) that exposed it:

| pass | ms |
|---|---:|
| `{n,f,f,f}` all three axes | 0.171 |
| `{n,n,f,f}` the fused pair | 0.129 |
| axis 3 alone (flattens, m == 2048) | 0.044 |
| axis 2 alone (m == 8 slabs) | 0.072 |
| axis 1 alone (m == 64) | 0.041 |

The two axes cost **less run separately (0.116) than fused (0.129)**.
Fusion is a net loss at this size: an 8 x 8 slab is 1 KiB and stays
resident either way, so the cache saving is nil while the lost batch width
is pure cost.

**Fix**: `fft_plan::fuse_last_pair_()` -- fuse only when the rank-2 slab is
at least `fft_fuse_pair_min_slab` (1024) ELEMENTS. Stated in elements, not
a side length, so a rectangular slab is judged by what actually occupies
the cache. Rank 2 is exempt: no descent, so no batch width to lose.
Threshold swept at 1 / 256 / 1024 / 4096 -- 1024 is optimal in BOTH
directions (4096 also unfuses n=32, costing ~7% there).

**A methodology failure worth recording, because it nearly shipped a
number.** The first validation was the usual three separate suite runs
(base, gate, base again). It reported "-3.4% overall" -- and it was
worthless:

- `many_strided_h32`, a sweep the gate provably cannot reach, moved 21%.
- Two **identical** binaries disagreed by 18% on `many_h32` (1.689 vs 1.392).
- Calibration drifted 0.31 ms -> 0.14 ms across the session; the machine
  got ~2x faster as background load settled.

Separate processes with one sample each cannot resolve a 5% effect under
that drift, no matter how many sweeps they contain. This is a different
failure from §11.42/§11.48 (which had the wrong workloads); here the
workloads were right and the *sampling* was wrong.

Replaced with an interleaved same-process A/B: both variants in one
binary, alternating A,B,A,B so drift is shared and cancels; 21 samples per
variant reported as the MEDIAN; the same 64 MiB cache flush before every
timed call; pinned to one core; and -- the part that makes it checkable --
**five shapes where the gate must do nothing, measured alongside**.

| shape | delta |
|---|---:|
| *control* 2d 32x32 (rank 2, exempt) | -0.0% |
| *control* 2d 128x128 (rank 2, exempt) | +1.8% |
| *control* 3d 32^3 (slab == 1024, still fused) | -0.8% |
| *control* many3d 32x128x128 (still fused) | +0.2% |
| *control* many4d 32x32^3 (still fused) | +0.1% |
| gap3d 32x8x8 `{f,n,f}` (pair path never taken) | -0.3% |
| 3d 16^3 | **-3.5%** |
| many4d 32x16^3 | **-5.2%** |
| 3d 8^3 | **-7.4%** |
| many4d 32x8^3 | **-7.6%** |
| many3d 32x8x8 | **-9.4%** |
| many3d 32x16x16 | **-9.8%** |

Noise floor +-2% from the controls; every active shape clears it. The
`{f,n,f}` row is a free second control -- with axis D-2 `none` the pair
condition is false, so that shape never fused and must read zero. It does.

**ADOPTED.** Clean under g++/clang++ `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion -Werror`, C++17 and C++20, and
address+UB sanitizers.

Scope, stated honestly: this lands at n=8 and n=16, BELOW the 32..128
range that matters to the maintainer. It fixes the worst absolute ratios
in the suite (n=8 is 3.8-4.7 against FFTW on every rank >= 3 sweep) but
does not touch n=128.

**The open lead it exposed.** Even unfused, the middle axis is weak: for
`32 x 8 x 8 x 8` along axis 2 the walk reaches `m == 8`, against `m == 2048`
for axis 3 (0.072 ms vs 0.044 ms above). The batch axes are three separate
axes (strides 1, 64, 512) and only ONE can be the batch, because
`fft_exec_slab` takes a single batch stride. No permutation fixes this --
axes 3 and 1 together (offsets 0..7, 64..71, ...) are not a uniform stride.
The fix would be a rank-3 slab entry point with TWO batch axes, which is
also the "fuse all the loops" shape the §8/§12 GPU port wants. Not
attempted here.

### 11.50 §11.30's "memory-bound" diagnosis is REFUTED: the power-of-two gap is instruction count, and it is ~2.5x (2026-08-01)

§11.30 (2026-07-17) closed the power-of-two investigation with "the gap is
memory-bound (stage count / bytes moved), not compute-bound -- constexpr
twiddles ruled out, SIMD ruled out", and §11.29 had found explicit AVX2
"no measurable win -- likely memory-bound". Four batching changes have
landed since (§11.39 `mb_` budget, §11.41 packing threshold, §11.43 stride
guard, §11.44 single-stage batching), so the diagnosis was re-tested.

**The experiment.** Fix the transform length, sweep ONLY the working set
(vary the batch count), no cache flush -- so the small cases genuinely live
in L1 and the large ones genuinely stream from DRAM. If memory-bound, the
Multi/FFTW ratio should collapse toward 1 as the working set shrinks into
cache; if compute-bound, it is flat.

Batched 1-D, in-place, contiguous, FFTW_ESTIMATE, i7-8700 (L1d 32K, L2
256K, L3 12M):

| n | 16 KiB (L1) | 128 KiB (L2) | 1 MiB (L3) | 8 MiB (L3) | 128 MiB (DRAM) |
|---|---:|---:|---:|---:|---:|
| 32 | 2.71 | 3.11 | 2.94 | 2.84 | 2.12 |
| 64 | 2.58 | 2.84 | 2.75 | 2.47 | 1.94 |
| 128 | 2.95 | 2.78 | 2.52 | 2.42 | 1.95 |

The ratio does not collapse in cache -- it is WIDEST there and *narrows*
toward DRAM, because DRAM bandwidth caps FFTW and not us. Multi's own
throughput is nearly flat (12.5 -> 7.5 GFLOPS across a 8000x working-set
range) while FFTW's falls 34 -> 15.7. **FFTW is the one that becomes
memory-bound; we are compute-bound everywhere.** At 34 GFLOPS FFTW is at
~50% of this machine's AVX2 peak, which is what a tuned FFT achieves;
Multi at ~12 GFLOPS is at ~18%.

**Confirmed independently by instruction count** (callgrind, n=128,
1 MiB working set, plan-build and allocator excluded):

| | Ir |
|---|---:|
| Multi `stage_radix8_` | 20.08 M |
| Multi `run_fused_impl_` (two radix-4 stages + gather/scatter) | 17.61 M |
| **Multi total** | **37.70 M** |
| **FFTW codelets** | **15.23 M** |

**2.48x more instructions against a 2.5x time ratio -- IPC is the same.**
Not stalls, not misses: we simply execute 2.5x the instructions per
butterfly. Note the auto-vectorizer IS working (`-fopt-info-vec` reports
32-byte AVX2 on all seven batch loops); it is producing correct vector
code that is nonetheless 2.5x the instruction count of a hand-tuned
codelet.

**Per-kernel cost** (per element, per stage; 65536 elements x 20 reps):

| kernel | instr/element | share of Ir at n=32 / n=128 |
|---|---:|---|
| radix-4 | ~7.1 | -- |
| radix-8 | **15.2** | 60% / 49% |

Radix-8 costs 2.1x the instructions of radix-4 for 1.5x the work
(log2 8 vs log2 4) -- **43% less efficient per unit of work** -- and n=64
(4.4.4, the only one of the three with no radix-8 stage) has the best
ratio of the three, 2.06.

**A hypothesis this suggested, tested, and REFUTED.** If we are compute-
bound, the factorization comment's trade ("replacing a 4*2 tail by one 8
saves a whole memory pass") is backwards, and 2^odd should use a radix-2
tail instead: n=32 as 4.4.2, n=128 as 4.4.4.2. Implemented behind
`BOOST_MULTI_FFT_ODD_EXPONENT_TAIL` and measured:

| n | working set | radix-8 tail | radix-2 tail |
|---|---|---:|---:|
| 32 | 1 MiB | **10042 MFLOPS** | 7961 (-21%) |
| 128 | 1 MiB | **9926 MFLOPS** | 7567 (-24%) |
| 128 | 16 KiB (L1) | 11252 | 11222 (tie) |
| 64 | any | unchanged (control: no radix-8 either way) |

Reverted. The extra memory pass costs more than the instructions it saves
-- EXCEPT in L1, where the two tie exactly, which is the same story from
the other side: with no memory traffic the instruction counts are
comparable, and out of L1 the extra pass decides it. So the shipped
radix-8 tail is correct, and stage count still matters for absolute
speed even though it does not explain the FFTW gap. n=64 unchanged is a
clean control that the switch did only what it claimed.

**What this changes.** The power-of-two direction was closed four times
(§11.25, §11.27, §11.28, §11.34 -- size-32 codelets) and finally written
off by §11.30 for a reason that is now measurably wrong. The available
headroom is not the ~10% those attempts were chasing: it is ~2.5x, it is
entirely in instructions-per-butterfly, and it sits in the maintainer's
stated target range (n=32..128 is where FFTW's codelets are strongest).
`stage_radix8_` is the single largest line item, at half of all
instructions for the two sizes that use it.

**What it does NOT license.** Standing policy is compiler
auto-vectorization only -- no intrinsics or `std::simd` except as a
localized last resort. The auto-vectorizer is not failing to fire; it is
firing and losing 2.5x anyway. So closing this gap means either
hand-written kernels (a policy decision for the maintainer, not one to
take unilaterally) or finding what makes the generated vector code so much
denser than FFTW's. NOT attempted here -- §11.45's 39% and §11.48's 8.5%
are recent reminders of what "obviously neutral" refactors of these
kernels actually cost.

### 11.51 Why the generated vector code is 2.5x denser: it is 128-bit, and radix-8 will not go to 256-bit at any price (2026-08-01)

§11.50 established the gap is instruction count, not memory. This is the
mechanism, and it argues AGAINST the hand-written-kernel conclusion that
§11.50 seemed to point at.

**The one-line answer.** FFTW's hot codelet at runtime is
`fftw_codelet_n1fv_64_avx`: every one of its 1412 FP operations is
**256-bit packed**. Our `stage_radix8_<true,false,complex<double>>` runs
all 71 of its FP operations at **128-bit**. GCC did vectorize it, but across the complex
number's own (re,im) pair, not across the batch. Same instruction pattern,
half the width. That is the 2x, and the rest is loop overhead.

It is not uniform across our kernels: `run_fused_impl_<true,false>`, which
inlines the radix-4 stages, is a MIX -- 132 ops at 256-bit, 144 at
128-bit, and 170 outright SCALAR. So radix-4 reaches full width only
partly, and radix-8 never. That is the 7.1 vs 15.2 instr/element split
§11.50 measured, and it also says the headroom is not confined to
radix-8: 170 scalar FP ops in the fused path are their own target.

(CORRECTION, same day: this section first reported these as raw
ymm/xmm register counts -- 607/81 for FFTW, 0/172 for radix-8, 311 ymm
for run_fused_impl_. That metric is WRONG and was overstating our
position: scalar double arithmetic on x86-64 also uses xmm registers
(`vmulsd`, `vaddsd`), so an "xmm" count conflates 128-bit SIMD with plain
scalar code. The numbers above classify by the instruction's actual width
(`pd` on ymm = 4 doubles, `pd` on xmm = 2, `sd` = 1). The qualitative
conclusion is unchanged and the scalar ops it exposed are new
information.)

**Four things tried to move it. None did.**

| attempt | result |
|---|---|
| `-mprefer-vector-width=256` / `=512` | 0 ymm (identical text) |
| `-fno-tree-slp-vectorize` (stop SLP taking the complex pair) | 0 ymm |
| `-fvect-cost-model=unlimited`, `-funroll-loops` | 0 ymm |
| `#pragma omp simd` on the j loop + `-fopenmp-simd` | 0 ymm (text grew to 420) |

**A hypothesis, tested and refuted.** The batch strides `ja`/`jb` are
runtime values, so perhaps GCC cannot prove contiguity. Traced, and the
strides are worth recording:

| layout | `ja` (in) | `jb` (out) |
|---|---:|---:|
| element-contiguous user array, n=128 (2-D/3-D/many) | 1 | **128** |
| element-contiguous user array, n=32 | 1 | **32** |
| batch-contiguous user array (the `many_strided` shape) | 1 | **1** |

`run_fused_impl_` sets `jb = (i == last) ? jb_ : 1`, so only the FINAL
stage inherits the user's batch stride. For an element-contiguous array
that stage's stores are scattered 2048 bytes apart, and AVX2 has no
scatter instruction at all -- so for that call vectorising across j is not
merely unprofitable, it is impossible.

That is a real constraint and worth knowing. It is NOT the binding one:
specialising the radix-8 inner loop on `ja == 1 && jb == 1` (generic
lambda over `integral_constant`, so the unit case sees literal 1s) still
produced **0 ymm**, shrank the text 43% (387 -> 222) -- and made things
WORSE, executed instructions 20.08M -> 23.82M (+19%), time neutral to
slightly down. Reverted.

**So the remaining explanation is register pressure, now with a
mechanism.** A 256-bit radix-8 across the batch needs 8 ymm live for
x0..x7 plus 7 broadcast twiddles = 15 of AVX2's 16 registers before any
temporary. Radix-4 needs 4 + 3 = 7 and fits, which is why it gets 256-bit
and radix-8 does not. This is the same wall as the reverted radix-8-over-4,
radix-16 and split-radix attempts -- but those recorded "register
pressure" as a conclusion, and this is the first time it has been tied to
a specific register budget and a measured width.

**Why this argues against writing intrinsics for radix-8.** The register
budget is a property of AVX2, not of GCC's cost model. Hand-writing the
same 8-point butterfly at 256-bit hits the same 16 registers and spills
too -- and the spilling is already visible in the current 128-bit code
(twiddles reloaded from `0x58(%rsp)`, `0x68(%rsp)`, ... inside the inner
loop). An intrinsics rewrite would be a large, policy-relaxing change
aimed at a wall that is not the compiler's.

**What might actually work, in rough order of promise:**

1. Keep radix-8 but cut its live values: the kernel already decomposes into
   two radix-4 sub-butterflies. Scheduling those so only one is live at a
   time trades a couple of reloads for a halved register footprint, and
   would be a source change inside the existing policy.
2. Make the final stage write batch-contiguous scratch and transpose
   separately, so no stage ever sees a scattered `jb`. Costs one extra
   pass, and §11.50's radix-2 experiment is a warning about what an extra
   pass costs -- but that pass would be a pure blocked copy, not another
   butterfly.
3. AVX-512 has 32 registers AND scatter, which removes both walls at once.
   Not available on this machine (i7-8700).

**Not attempted here.** Every one of these has the shape that §11.45
(+39%) and §11.48 (+8.5%) punished: a confident restructuring of these
kernels. Each needs its own interleaved A/B before it goes anywhere near
a commit.

### 11.52 §11.51's chosen fix (re-schedule the radix-8 sub-butterflies) -- implemented, no effect, and two measurement lessons (2026-08-01)

§11.51 recommended cutting radix-8's live set by writing the two radix-4
sub-butterflies so their values never overlap: load the even legs
(x0,x2,x4,x6), reduce them to e0..e3, and only THEN load the odd legs.
Peak live drops from 8 legs + 7 twiddles to 4 + 4 and 3/4 of the twiddles,
which should fit AVX2's 16 registers. Pure reordering -- every output is
the same expression tree, so bit-identical.

**Implemented. No effect whatsoever:** 387 -> 390 instructions, still 0
ops at 256-bit, identical 128-bit count. GCC builds its own schedule from
the data-flow graph and hoists the loads back; source order does not
constrain it. Reverted.

To actually force the split one would have to break the compiler's freedom
-- e.g. two passes over `j` with e0..e3 staged through a small L1 buffer --
which adds a scratch round-trip. Not attempted; on the evidence below it
is no longer clearly the right target anyway.

**Lesson 1: the xmm/ymm metric §11.51 was built on was wrong.** Scalar
double arithmetic uses xmm registers too, so counting "xmm" lumps 128-bit
SIMD together with scalar code. Corrected classification (by the
instruction's real width) is in §11.51; the headline survived, but it had
been HIDING something: `run_fused_impl_` contains 170 genuinely SCALAR FP
operations. A metric that flatters the code in one direction can conceal a
problem in another.

**Lesson 2: standalone micro-probes of these kernels are not
representative.** Minimal reproductions of the radix-4 and radix-8 batch
loops (unit strides, `__restrict__`, in their own TU) compile to FULLY
SCALAR code -- 0 packed ops of any width. They do not reproduce even the
128-bit vectorization the real kernels get, so any conclusion drawn from
them about vector width would have been an artifact. The register-pressure
story in §11.51 is therefore still only supported by the real kernels
(radix-8 at 0 x 256-bit vs run_fused_impl_ at 132), not by the micro-probe
comparison, and should be treated as unproven.

**Where this leaves the direction.** The three candidates §11.51 listed are
untouched, but the ranking should change: the 170 scalar FP ops in
`run_fused_impl_` are a target nobody had seen, and unlike the radix-8
widening they do not depend on the unproven register-pressure argument.

### 11.53 Two fibers per iteration: partial 256-bit achieved, and it bought NOTHING -- which undercuts the width theory itself (2026-08-01)

§11.51/§11.52 argued the gap is width: FFTW's codelet is 1412 FP ops all
at 256-bit, ours are 128-bit. The natural test is to hand the vectorizer
two independent butterflies in one basic block, so SLP has something to
widen. With unit batch strides `a[j]` and `a[j+1]` are adjacent, so the
pair is a single 256-bit load.

Prototyped in `stage_radix4_`: a paired loop stepping j by 2, guarded on
`ja == 1 && jb == 1`, with the original loop kept as the tail and the
strided fallback.

**It worked, and it did not matter.**

| | before | after |
|---|---:|---:|
| radix-4 FP ops at 256-bit | 0 (in an inlined mix of 132/144/170) | **63** of 169 |
| executed Ir, whole transform | 37.70 M | 37.43 M (**-0.7%**) |
| n=128, 1 MiB working set | 9926 MFLOPS | 10177 (+2.5%) |
| n=32, 1 MiB working set | 10042 MFLOPS | 9748 (**-2.9%**) |

37% of radix-4's arithmetic moved to full width and the transform got
0.7% fewer instructions and no reliable time change. Correct under
sanitizers. Reverted.

**This is the most important negative result of the three.** §11.50 proved
we execute 2.5x FFTW's instructions at identical IPC; §11.51 found our
arithmetic is 128-bit where FFTW's is 256-bit and concluded width was the
lever. This says width is NOT the lever, or at least not on its own:
widening a third of the arithmetic in the kernel that runs 2 of 3 stages
changed essentially nothing.

The likely reading is that the 2.5x is not in the arithmetic at all. The
FP operations are a minority of the instruction stream -- the earlier
static mix of `stage_radix8_` was 107 scalar `mov`, 70 address-arithmetic
(`shl`/`add`/`lea`/`imul`) and 95 vector moves against 77 arithmetic ops.
Halving the count of the 20% does little; the loads, stores, address
computation and (visible in the disassembly) twiddle reloads from stack
are the other 80%. FFTW's codelets are straight-line with compile-time
constant offsets, which removes the address arithmetic entirely rather
than vectorising it.

**So the next attempt should target instruction COUNT, not width** -- and
specifically the addressing. Everything in these kernels is indexed
`a0[(k * q * sa) + j * ja]` with runtime `q`, `sa`, `ja`; FFTW's generated
codelets have all of that folded into constants. That reframes the
size-32-codelet idea (§11.25/27/28/34, four reverts) as being about
constant-folded ADDRESSING rather than about unrolled arithmetic, which is
not what those attempts were testing.

Recorded, not attempted. Note also that three consecutive hypotheses in
§11.50, §11.52 and §11.53 have now been refuted by measurement -- the
lesson from §11.45/§11.48 (do not ship these kernels' refactors on
reasoning) is holding up well.

### 11.54 A uniformity fix in `stage_radix8_` that is also the first measured win against the §11.53 target (2026-08-01)

Reading the seven stage kernels side by side for redundancy turned up one
that did not match the others. Every kernel precomputes a pointer per leg
and indexes `a1[j * ja]`:

| kernel | leg pointers precomputed |
|---|---|
| radix2 | 2 in, 2 out |
| radix3 | 3 in, 3 out |
| radix4 | 4 in, 4 out |
| radix5 | 5 in, 5 out |
| **radix8** | **1 in, 1 out** |

`stage_radix8_` alone kept the offsets inline -- `a0[(k * q * sa) + j * ja]`
and `b0[(k * ns * sb) + j * jb]` -- for all 16 accesses. It is also the
most expensive kernel in the file (15.2 instr/element against radix-4's
7.1, §11.50) and, per §11.53, addressing is exactly where the instructions
are. Made uniform with the rest.

**Measured** (n=128, 1 MiB working set):

| | before | after |
|---|---:|---:|
| `stage_radix8_` executed Ir | 20,081,600 | **19,756,480 (-1.6%)** |
| whole transform executed Ir | 37.70 M | **37.37 M (-0.9%)** |
| n=128, 1 MiB | 9927 MFLOPS | **10170 (+2.4%)** |
| n=32, 128 KiB | 10515 MFLOPS | 10673 (+1.5%) |
| n=32, 1 MiB | 9901 MFLOPS | 9932 (+0.3%) |
| n=128, 128 KiB | 10403 MFLOPS | 10410 (+0.1%) |

Timings are the mean of two interleaved rounds against the pre-change
binary, using Multi's own throughput rather than the FFTW ratio (FFTW's
own number moved 3-5% between rounds and would have swamped the signal).
Correct under g++ `-Wall -Wextra -Werror` and address+UB sanitizers.

Small, but it is the FIRST thing this session to move in the direction
§11.53 predicted, after three refuted hypotheses aimed at arithmetic
width. Instruction count, not vector width.

**Methodology note worth keeping.** Executed instruction count from
callgrind is DETERMINISTIC -- the same binary reports the same number
every time. On a machine where a 5% timing effect needs interleaved
sampling and seven control shapes to establish, Ir gives an exact answer
in one run. It cannot see stalls or misses, so it must not be the only
evidence (§11.45's butterfly extraction cost 39% of wall clock while
looking fine on a proxy), but for a change that is purely about how many
instructions are issued, it is the right primary instrument and the timing
run is the confirmation.

**Where the remaining instructions are** (n=128, line-level, after this
change): complex arithmetic ~49% (`real()` 18.1%, `operator+=`/`-=`
12.5% each), and the rest is loop control, addressing and stores -- the
inner `for(j...)` line alone is 6.6%. The next candidate in this direction
is the `j * ja` / `j * jb` indexing: with 8 legs in and 8 out, that is 16
strength-reduction candidates in one loop, which is more induction
variables than GCC will maintain. Converting them to pointer increments is
the obvious next experiment. Not attempted here.
