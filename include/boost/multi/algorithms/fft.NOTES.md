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

**Phase B — direction-neutral engines (the §10.1-item-4 target):**

6. Remove `sign_` from `fft_engine`; build `tw_`/`wmat_` with canonical
   (forward) sign; template the Stockham stage kernels on a `Sign` (or
   `bool Backward`) parameter that conjugates twiddles on load; per-pass
   dispatch in `fft_apply_last`/`fft_apply_last_pair` selects the
   instantiation from the plan's `dirs_`. Direct-prime `wmat_` path: same
   conj-on-load treatment.
7. Bluestein: attempt the neutral form (conjugate chirp on load; derive the
   backward convolution from the forward `kernel_ft_` via the
   conjugate/index-reversal identity, or store the one extra table if that's
   cleaner); if it degrades clarity, take the sanctioned fallback of §10.2
   and leave Bluestein engines direction-keyed.
8. Engine reuse key drops back to size alone. Re-run the full test battery;
   the same-size-opposite-direction test from step 5 now also proves the
   sharing (can assert `engines_.size() == 1` for square `{f, b}` via a
   test-only observer or just by the numerics).
9. Update §1's thread-safety text and this file if Phase B changes any of
   the §9.2 immutability conclusions (it shouldn't — it only *removes*
   state from engines).

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
materialize `conj(w)` then multiply. Add a sibling to `fft_ops<T>::mul`:

    conj_mul(a, b) == mul(conj(a), b)
    // std::complex specialization — same 4 mul + 2 add, two signs flipped:
    { (a.real()*b.real()) + (a.imag()*b.imag()),
      (a.real()*b.imag()) - (a.imag()*b.real()) }

(generic fallback: `mul(conj(a), b)` with ADL `conj`). Then a tiny
compile-time selector, e.g. `fft_mul_dir<bool Backward>(w, x)` → `mul` or
`conj_mul`, and the kernel diff is mechanical: every `fft_mul(table_value,
datum)` becomes `fft_mul_dir<Backward>(table_value, datum)`. Convention to
keep: the *table* operand is always the first argument (already true
everywhere today) — conjugation must apply to that operand only.

**Plumbing the sign down:** add `bool Backward` alongside `Batched` on
`run_stages_` and the stage kernels; the runtime `switch(st.kind)` stays
untouched. The engine's public entry gains a runtime direction argument
that selects the `<Batched, Backward>` instantiation once per invocation —
one branch per *pass*, nothing per element.

**Bluestein specifics (Phase B):** the constructor currently builds *two*
convolution sub-engines, `sub_.emplace_back(conv_n_, sign_)` and
`(conv_n_, -sign_)` (~lines 474–475) — with sign-templated kernels these
collapse into ONE sub-engine run with opposite `Backward` values (a real
table-memory win, not just tidiness). The precomputed spectrum obeys
`kernel_ft_backward[k] == conj(kernel_ft_forward[(N-k) mod N])` (FFT of a
conjugated sequence = conjugated, index-reversed FFT) — the index reversal
is why plain conj-on-load doesn't work for `kernel_ft_` and why the
sanctioned fallback (direction-keyed Bluestein engines only) exists.
`chirp_`/`postc_` are plain elementwise conjugates, no reversal.

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
