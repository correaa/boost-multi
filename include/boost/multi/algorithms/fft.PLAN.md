# Execution plan: fft.hpp redesign + partial/mixed-direction FFTs

Audience: an implementing model (Sonnet 5) or developer executing the work
specified in `fft.NOTES.md` §9.2 (T-decoupling, buffer externalization,
allocators) and §10 (per-axis directions). This file says *what to do in
what order, with what gates*; the *why* lives in the NOTES — read those
sections first, they are the contract. Nothing here overrides them.

## Ground rules (apply to every session)

- **Read before writing**: `fft.NOTES.md` §9.1–§9.2 and §10 (all of it,
  §10.5 especially — it has the code tricks and the anchor map). Then
  re-grep every anchor you intend to touch: the file may have moved under
  you since the NOTES were written. The maintainer live-edits `fft.hpp`;
  NEVER revert changes you didn't make — if the file's shape contradicts
  this plan, stop and ask.
- **The gate** — after every task, this must build clean and pass:

      g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
          -Wsign-conversion -Werror -Iinclude \
          test/algorithms_fft.cpp -o /tmp/fft_test.x && /tmp/fft_test.x

  Never proceed past a red gate. Never weaken a flag to get past it.
- **C++17 only.** No `using enum`, no concepts, no spans.
- **Locked decisions — do not relitigate**: wrapper name is `fft_inplace`
  (never `do_fft`); direction spec is runtime `std::array`, dirs-first
  argument order (§10.5 deduction trick); directions live in the plan
  constructor, not `execute()`; engines direction-neutral (target), never
  per-direction tables; unnormalized transforms; no `-ffast-math` anywhere.
- **Session hygiene**: one session per phase below. Commit at each
  checkpoint with a message naming the phase (branch `fftalgo2026`; do not
  push or open MRs unless asked). If a session ends mid-phase, note the
  exact stopping point in the commit message.
- **Stop-and-ask triggers**: the §9 shape already changed under you; a gate
  failure you cannot explain; any test that passes only with loosened
  tolerance; anything requiring a public API break not listed here.

## Session 0 — pre-flight audit (short; may merge into Session 1)

1. Run the gate on the untouched tree; record the baseline.
2. Audit current state vs the NOTES (things may have landed since):
   - Do engines still hold `mutable buf_/out_/xbuf_`? (If yes, §9.2 is
     pending → Session 1 as written. If no, verify against §9.2 and skip
     what's done.)
   - Is the plan still `fft_plan<T, D>` (not `<TW, D>`)?
   - Confirm the §10.5 anchor map (member names, `apply_`,
     `run_stages_<Batched>`, `fft_ops`, Bluestein sub-engine pair).
3. Write the audit result at the top of the session's first commit message.

## Session 1 — §9.2: buffers out, TW decoupling, allocator on execute

**DONE** (all three tasks landed, each as its own checkpoint commit rather
than one combined commit as originally suggested below — commits
`d3c307857` scratch externalization, `efcf0bbb8` T/TW split +
`bd4d211d2` docs reconciliation, `895969431` alloc-zero bugfix,
`045177731` allocator parameter). One API detail changed from what this
plan originally assumed: `fft_plan<TW, D>` became `fft_plan<D, TW>`
(D first) — C++ forbids a defaulted template parameter followed by a
non-defaulted one, and `D` has no sensible default, so `TW`'s default
could only go after `D`. Discussed with the maintainer before landing;
see the T/TW-split commit and NOTES §9.2 for the full rationale. Existing
call sites (`test/`, `benchmark/`) updated accordingly.

Goal state (all from NOTES §9.2 + §10.4(a)(b), which settle every design
question — consult them rather than deciding anything anew):

- Engines/plan hold **no scratch**; they expose required *element counts*
  (e.g. `scratch_elements()`; remember Bluestein/six-step sub-tree needs).
- `fft_plan<TW, D>` where `TW` is the twiddle-table type (default
  `std::complex<double>`); `execute()` deduces the array's element type `T`
  per call. `fft_ops` grows the cross-type `mul(T, TW) -> T`.
- `execute()` takes an allocator; a defaulted overload constructs a fresh
  stateless `std::allocator<T>` per call (never plan-owned state).
- After this session `mutable` must not appear in the file at all; a
  comment on the plan may then state that concurrent `execute()` on one
  plan is safe when each call gets its own scratch.

Suggested task order (gate between each):

1. Externalize scratch only, same `T` everywhere: engines get
   `scratch_elements()`, `run(...)` signatures grow raw scratch-pointer
   parameters, `execute()` allocates via `std::allocator<T>` locally.
   Delete the three `mutable` members. (Biggest mechanical diff — do it
   before the type split so type errors don't compound.)
2. Split `T` → `TW` for tables: retemplate `fft_engine`/`fft_plan` on `TW`,
   `execute()` becomes a template deducing `T`, `fft_ops<T, TW>::mul`.
   The default `TW = std::complex<double>` keeps all existing tests
   compiling unchanged — they use `complex<double>` arrays throughout.
3. Allocator parameter + defaulted overload on `execute()` (and thread the
   allocator to wherever scratch is obtained — allocator abstraction only,
   no raw `new`/`malloc`/owning `std::vector` on the execute path; this is
   the §10.4(c) GPU seam).
4. New tests: `complex<float>` array through a default-`TW` plan (accuracy
   vs `dft_reference` at float-appropriate tolerance ~1e-5); a custom
   counting allocator passed to `execute()` proving it is used and that
   allocation count is stable across repeated executes; re-run benchmark
   compile (`benchmark/algorithms_fft.cpp` per its header instructions) to
   confirm no perf-relevant API broke — a quick 1-D spot-check against the
   `_nowisdom` numbers is enough (idle machine not required for a smoke
   run; do NOT publish numbers from a non-idle run).

Checkpoint commit: "fft: externalize scratch, decouple TW, allocator on
execute (§9.2)". (Landed as three separate checkpoints instead — see the
DONE note above.)

Also landed, benchmarked (not just theorized): with the machine genuinely
idle, the T/TW split showed no measurable regression on the T == TW same-
type path (as expected, since it's compile-time-identical to before); the
scratch-externalization allocation cost from task 1 is real and confirmed
against FFTW (multi::fft_plan lost essentially all its 1-D/2-D wins it had
pre-refactor) — task 3's allocator parameter is the intended fix, but
*using* it (e.g. wiring a stack allocator or pmr pool into the benchmark's
repeated-execute() loop) has not been done — the benchmark still uses the
default `std::allocator<T>` overload. That's the natural next thing to
try before moving to Session 2, if reclaiming the pre-refactor benchmark
standing matters more than the direction feature right now.

## Session 2 — §10.3 Phase A: direction feature, engines stay sign-aware

**DONE** (with one post-session review fix -- see below). All six steps
landed together (steps 1-4 and 6 were small enough not to warrant separate
checkpoints; step 5's seven tests all pass, including under ASan+UBSan).
One real gap found and resolved with the maintainer while implementing
step 4: `std::array<fft_direction, D>` catches too-many directions in a
braced-init-list as a hard compile error, but NOT too-few (aggregate-init
zero-pads, and zero == `fft_direction::none` by construction) -- tried
closing this via independent-N deduction with a static_assert cross-check;
confirmed empirically that a braced-init-list is a non-deduced context for
`std::array<T, N>`'s `N` too, so it can't be closed without a custom
fixed-arity wrapper replacing `std::array` here. Maintainer decision:
accept and document (not worth the added complexity) -- see the
`fft_inplace(dirs, arr)` overload's comment in fft.hpp.

**Post-session review found and fixed one real bug the seven tests
missed**: apply_'s degraded-pair branch used `view.rotated()` where
`view.unrotated()` is required -- correct at D == 2 (they coincide), wrong
axis AND out-of-bounds writes at D >= 3 with a non-cubic shape and dirs
like `{none, forward, none}`. The test battery's only D >= 3 case (the
mixed roundtrip) happened to have its `none` on axis D-2, so the broken
branch was never exercised. Lesson for Session 3's test design: every
branch of apply_'s direction dispatch needs a D >= 3, NON-CUBIC case --
cubic shapes and D == 2 both mask axis mix-ups. Fixed with two such
regression tests (verified to fail against the bugged code); §10.2 now
records the correct rotation.

Execute §10.3 steps 1–5 exactly as written there; supporting detail in
§10.1 (decisions), §10.2 (obstacles), §10.5 (tricks). Order within the
session, gate between each:

1. `fft_direction` enum + helpers (§10.1 item 1; naming-collision note in
   §10.5).
2. `dirs_` member + directions constructor; broadcast delegation from the
   `(extents, sign)` constructor; engine loop skips `none` axes (sentinel
   `which_` entry), reuse keyed `(length, direction)` for now.
3. `apply_()` honors the schedule: skip `none` passes; degrade
   `fft_apply_last_pair` when exactly one of the last two axes is `none`
   (§10.2); `engine_<A>()` gains the not-`none` guard (§10.5). `D == 1`
   with `none` = graceful no-op.
4. `fft_inplace(dirs, arr)` overload (dirs-first; §10.5 deduction trick;
   beware the `fft_is_multi_like` footgun, §10.5).
5. The seven tests of §10.3 step 5, verbatim — note the last paragraph of
   §10.5: `none`-axis untouched checks use exact equality, not `tol`.
6. Update the header's API comment block.

Checkpoint commit: "fft: per-axis directions incl. none (Phase A, NOTES
§10)".

## Session 3 — §10.3 Phase B: direction-neutral engines

**DONE.** Landed in two checkpoints as suggested below (smooth-path
templating, then Bluestein + tests), after fixing a broken hand-off: the
session that started this work had templated the stage kernels
(`stage_radix2_/3_/4_/5_/8_/generic_`) on `bool Backward` but left every
call site passing only `<Batched>` — a non-compiling tree. All of steps
1-7 below are landed as written, with these outcomes on the two
maintainer-left decision points:

- **Bluestein conv sub-engine pair**: collapsed to ONE neutral engine (not
  kept as two identical ones) — run forward (`<false>`) then inverse
  (`<true>`) sequentially. This makes `run_bluestein_`'s pointwise-product
  step's `z` (the second run's default input region) potentially the SAME
  memory as `yf` (the first run's result) when the engine's stage count is
  even — confirmed by direct probe (`fft_engine` constructed standalone)
  that n=101 gives an even (4-stage) conv and n=331/1009 give odd
  (5-stage); both parities pass under ASan+UBSan, covering the aliasing
  case flagged in this file's original text. The write is elementwise
  same-index (`z[i] = f(yf[i])`), safe even fully aliased; documented at
  the `run_bluestein_` definition.
- **`kernel_ft_bwd_`**: computed via the closed form
  `conj(kernel_ft_[(conv_n_ - k) % conv_n_])` at construction (no second
  engine run needed), rather than running the conv engine backward once
  more — cheaper and matches NOTES §10.5's stated invariant directly.
- Task 6's `engine_count()` accessor landed; the pre-existing "same-size
  opposite directions" test (Phase A) was updated in place to assert
  `engine_count() == 1` (was 2 in Phase A) rather than adding a duplicate
  test, since it already built the exact shape task 6 asks for.
- Five new tests landed in `test/algorithms_fft.cpp`: 1-D n=101 BACKWARD
  through Bluestein (the pre-existing large-prime test only ever ran
  forward, so `kernel_ft_bwd_` was previously untested by anything); a
  six-step-length (n=8192) forward-then-backward roundtrip; a non-cubic
  3-D `{backward, forward, none}` vs per-axis reference composition; the
  `engine_count() == 1` sharing proof (above); an all-forward
  bit-identical cross-check between the broadcast int-sign constructor and
  the explicit all-forward directions constructor (both should — and do —
  hit the exact same `Backward = false` instantiations). All pre-existing
  tests (Sessions 0-2's) still pass unchanged, at unchanged tolerances —
  the practical stand-in for "bit-identical to Phase A" now that Phase A's
  sign-baked code no longer exists in the tree to diff against.
- Sweep confirmed: `grep -w sign_`/`grep -w mutable` in fft.hpp → 0 (word-
  boundary grep; naive `grep -c` false-positives on substrings like
  `assign_offsets_`/`immutable`). Gates green: strict `-O2 -Wall -Wextra
  -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Werror` (both g++ and
  clang++), `-O3 -Walloc-zero`, ASan+UBSan. Benchmark
  (`benchmark/algorithms_fft.cpp`) recompiles clean against FFTW and runs;
  no perf-relevant API changed shape (`execute()`'s signature is
  untouched — direction is plan-construction-time state, same as Phase A).
- NOT done, left for Session 4 or a future doc pass: benchmarking whether
  Phase B's engine sharing actually recovers any of the scratch-
  externalization regression noted at the end of Session 1 (that
  regression was about allocator cost, orthogonal to engine count, so no
  change expected here) — not attempted, no claim made either way.

(Original session brief, still accurate for the design rationale — revised
after Sessions 1-2 + the review passes landed; the original thinner
version predated the T/TW split, the allocator threading, and the
recursive-walk dispatch. The enabling facts are in NOTES §10.5 — its
anchor map is current as of the recursive walk — plus the additions
below, derived from re-reading the code as it exists NOW. Re-grep anchors
anyway; the maintainer live-edits.)

Goal: engines lose `sign_` and store canonical-forward tables only;
kernels conjugate table values on load when running backward; engine
reuse keys on length alone, so a square `{forward, backward}` plan shares
ONE engine where Phase A builds two.

Task order, gate between each:

1. **`fft_ops` gains `conj_mul`** — mind the two-type shape it has now:
   `fft_ops<T, TW>::conj_mul(TW const& w, T const& x) -> T`, semantics
   `mul(conj(w), x)`, conjugating ONLY the table operand. Generic default:
   `mul(conj(w), x)` via ADL `conj`. The `fft_ops<complex<R1>,
   complex<R2>>` partial specialization adds the fused branch-free form
   (same 4-mul/2-add as `mul`, two signs flipped):
   `{(wr*xr) + (wi*xi), (wr*xi) - (wi*xr)}` with the same promoted-type
   widening `mul` uses. Then a compile-time selector in `detail`, e.g.
   `template<bool Backward> fft_mul_dir(w, x)` → `mul` or `conj_mul`, so
   the kernel diff is mechanical: every `fft_mul(table_value, datum)`
   becomes `fft_mul_dir<Backward>(table_value, datum)`. (Table operand is
   already first at every call site — verified during the T/TW split,
   which fixed the one violation, in run_sixstep_'s transpose.)

2. **Template the engine's execution path on `bool Backward`** alongside
   the existing `Batched`/`T` parameters: `stage_radix2_/3_/4_/5_/8_/
   generic_/subplan_`, `run_fused_impl_`, `run_stages_`, `run_sixstep_`,
   `run_bluestein_`. Uniform-conjugation rule (§10.5, verified): EVERY
   `tw_`/`wmat_` load conjugates under Backward, INCLUDING `imu` and the
   fixed roots `w1c..w4c` (they are table loads too — do not special-case
   them out), the generic kernel's `wmat` rows, `stage_subplan_`'s input
   twiddles, and run_sixstep_'s transpose twiddle. The public entries
   `run(m, arena)`, `run(m, in, arena)`, `run_fused(...)`,
   `run_contig_inplace(...)` gain a runtime `bool backward` parameter and
   dispatch ONCE per invocation to the `<..., Backward>` instantiation
   (one branch per pass, nothing per element). `stage_subplan_` and
   run_sixstep_'s `e1.run`/`e2.run`/`run_fused_impl_` calls forward the
   SAME Backward (sub-DFTs of a backward transform are backward).

3. **Thread the direction through the orchestration free functions**:
   `fft_exec_fiber`, `fft_exec_slab`, `fft_apply_last` each gain
   `bool backward`; `fft_apply_last_pair` gains TWO independent flags
   (its two axes can have DIFFERENT directions now — `{forward,
   backward}` is exactly the sharing test case). `apply_` passes
   `dirs_[axis] == fft_direction::backward` at its two call sites (the
   pair call and the walk body — the recursive-walk refactor reduced this
   to exactly these two places).

4. **Canonicalize construction**: `tw_` (and hence `wmat_`) built with
   forward sign unconditionally; delete `sign_` and the constructor's
   `sign` parameter; `sub_index_(rr)` and all `sub_.emplace_back` sites
   lose the sign argument; `fft_plan`'s ctor `find_if` drops the
   `e.sign_ == sign` conjunct (length-only reuse again). TRAP: keep the
   six-step `n2 == n1` DISTINCT-engine `emplace_back` (do not "simplify"
   it into `sub_index_`) — the two passes' scratch regions must not alias,
   and they only get distinct arena offsets by being distinct engines.

5. **Bluestein** (`init_bluestein_`/`run_bluestein_`). Facts that make
   this tractable, in preference order:
   - `chirp_`/`postc_` are plain elementwise conjugates across direction
     (`postc_ = fft_mul(inv_m, chirp_)` with REAL `inv_m`, so conj passes
     through) — conj-on-load via `fft_mul_dir<Backward>` just works.
   - `kernel_ft_` does NOT conj-on-load (FFT of a conjugated sequence =
     conjugated, INDEX-REVERSED spectrum). Preferred resolution: also
     precompute the backward-direction spectrum at construction
     (`kernel_ft_bwd_`, one extra conv_n_-sized table — trivial memory
     against what sharing saves) and select the table by flag at
     execution. On-the-fly index-reversal in the pointwise loop is
     possible but mixes forward/backward streams — not recommended.
     Direction-KEYED Bluestein engines (the old fallback wording) is now
     the LAST resort: with keying otherwise gone it would reintroduce
     per-engine direction state and complicate the plan's find_if; prefer
     the second table.
   - The outer `Backward` must NOT change the convolution sub-transform
     directions: the convolution mechanism is fixed (canonical fwd conv
     then inverse conv, i.e. sub runs at `<false>` then `<true>`)
     regardless of the outer transform's direction. Only chirp/postc/
     kernel-table selection depend on the outer direction.
   - The fwd/bwd conv sub-engine PAIR may collapse to ONE neutral engine
     (real table-memory win) — but this changes scratch aliasing:
     today the pointwise product writes into the second engine's region
     while reading the first's; with one engine, `z` and `yf` can be the
     SAME region depending on run_stages_'s ping-pong parity (result lands
     in `out` after an odd stage count, `buf` after even). The write is
     elementwise same-index (`z[i]` from `yf[i]` only), which is safe even
     fully aliased — but this argument must go in a comment and be
     verified under ASan with both parities (pick two conv_n_ values with
     odd and even stage counts). Keeping two now-identical neutral conv
     engines is an acceptable simpler outcome; say which was chosen in the
     commit message.
   - `init_bluestein_`'s construction-time bootstrap runs the conv engine
     forward (`<false>`) — unchanged.

6. **Tests** (extend test/algorithms_fft.cpp; per the Session-2 lesson,
   every new dispatch path needs a D >= 3 NON-CUBIC case):
   - 1-D n = 101 (prime > fft_max_direct_radix → real Bluestein) BACKWARD
     vs `dft_reference` — this is the test that decides kernel_ft
     correctness, per the original plan;
   - 1-D six-step-length (>= 8192) forward-then-backward roundtrip = n·id
     (exercises the transpose-twiddle conjugation without an O(n²)
     reference);
   - non-cubic 3-D `{backward, forward, none}` vs per-axis reference
     composition;
   - sharing proof for square `{forward, backward}`: add a small public
     `engine_count()` accessor (harmless, genuinely useful to callers) and
     assert it returns 1 there and 2 for Phase A semantics... i.e. 1 now;
     also keep the existing numeric check;
   - expectation to CHECK and report (not a hard gate): all-FORWARD
     results should be bit-identical to Phase A's, since forward is the
     canonical sign (tables unchanged) and `<Backward=false>`
     instantiations should be operation-identical — if they are not
     bit-identical, understand why before proceeding.

7. Sweep: `grep -c sign_` → 0 in fft.hpp (or Bluestein-only with the
   last-resort fallback, justified in the commit message); no new
   `mutable`; NOTES §10.5 anchor map updated for the new signatures;
   usual gates (strict O2, CI-like O3 -Walloc-zero) + ASan+UBSan on main
   and stress suites; benchmark rebuilt and smoke-run (all-forward path
   must be perf-neutral — same tables, same kernels at Backward=false).

Checkpoint commits: one for the smooth Stockham path (tasks 1-4 can land
with Bluestein still direction-CONSTRUCTED if split there is cleaner --
tables built with a sign parameter kept temporarily), one for Bluestein +
the keying change + tests. Or all together if the tree stays green
throughout; prefer two.

## Session 4 (optional, lowest priority) — polish

- `D`-bounded inline engine storage replacing `std::vector` (§10.1 item 9 /
  §10.3 optional step). Keep the linear-scan reuse lookup (§9.1: no map).
  **DONE**, on `multi::inplace_array` (array.hpp), not a hand-rolled
  container — revised after the first landing (a hand-written
  `detail::fft_inline_vector<T, Cap>`, aligned-raw-storage + placement-new)
  per maintainer request to dogfood Multi's own facility instead. This
  needed a real design change, not a drop-in swap: `multi::inplace_array`
  (`dynamic_array<T, 1, static_allocator<T, N>>`) is shape-fixed-at-
  construction, no incremental `push_back`/`emplace_back` — a fundamental
  mismatch with the original one-at-a-time dedup loop. Resolved as:
  - A pure `dedup_()` pass over axes first (plain `std::size_t` comparisons,
    no engine construction) produces the distinct lengths and a
    `distinct_count_` (0..D), independent of container choice.
  - `engines_` is then built in ONE SHOT via `index_sequence` pack
    expansion into a `std::array<fft_engine<TW>, D>` (legal for a
    non-default-constructible element given a FULL brace-init list — one
    initializer per slot), then move-constructed into the
    `inplace_array` via its iterator-range constructor
    (`std::make_move_iterator`, avoiding a second deep copy of each
    engine's heap-owned tables).
  - `engines_` is therefore always PHYSICALLY exactly `D` elements — real
    engines occupy `[0, distinct_count_)`, the tail is padded with cheap
    `fft_engine{0}` placeholders (nn<2 returns from the engine constructor
    immediately, no heap tables, matching a prototyped-and-verified
    pattern). This sidesteps a real hazard the naive approach would have
    hit: `inplace_array`'s range constructor evaluates `*first`
    unconditionally even for a nominally empty range (verified in
    isolation), which is UB for a genuinely empty range — never triggered
    here since the physical count is never 0, only the logical
    `distinct_count_` is.
  - `note_reach_`/`assign_offsets_`/`engine_count()` are explicitly bounded
    to `distinct_count_`, never the physical `D`, so the placeholder tail
    costs zero scratch — preserves the "`none` axis costs nothing"
    guarantee (§10.1 decision 3) exactly.
  - `dirs_` had to move earlier in the member declaration order (before
    `engines_`/`which_`) since `engines_`'s initializer now depends on it
    (member init follows declaration order, not initializer-list order).
  Verified: bit-identical against the pre-change baseline (same harness as
  the Tier A task), full gate green (g++/clang++ strict, `-O3
  -Walloc-zero`, ASan+UBSan), the standalone `inplace_array` prototype
  (non-default-constructible element, varying live/padded counts including
  fully-empty, copy/move) clean under ASan+UBSan, and the copy/move stress
  test (copy ctor, copy assignment, move ctor of a two-Bluestein-engine
  plan, including the recursive `sub_` tree) re-run clean against the new
  implementation.
- Batched-1-D benchmark vs `fftw_plan_many_dft` (§10.3 optional follow-up;
  full methodology rules in `benchmark/algorithms_fft.cpp`'s header —
  idle-machine protocol applies to *published* numbers). **DONE.** Added
  `sweep_many()` to `benchmark/algorithms_fft.cpp` ({none, forward} on a
  rank-2 array against FFTW's advanced interface, contiguous-row layout);
  run at howmany = 32 and 256 across the usual 5-smooth fiber-size family.
- Doc pass: NOTES §1 thread-safety text, header comment, and mark §9.2/§10
  items as landed.

## Definition of done

- Gate green; all pre-existing tests untouched and passing; new tests from
  Sessions 1–3 in `test/algorithms_fft.cpp`.
- `grep -c mutable include/boost/multi/algorithms/fft.hpp` → 0.
- `fft_plan<TW, D>` executes `complex<float>` and `complex<double>` arrays
  from one plan; `execute()` has allocator + defaulted overloads.
- `fft_inplace({{forward, none, backward, ...}}, arr)` compiles with double
  braces, wrong-rank spec fails to compile, `none` axes bit-identical.
- Engines shared across same-length axes regardless of direction (modulo a
  taken Bluestein fallback, documented in the commit that took it).
