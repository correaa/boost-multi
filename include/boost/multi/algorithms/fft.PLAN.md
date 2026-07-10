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

**DONE.** All six steps landed together (steps 1-4 and 6 were small enough
not to warrant separate checkpoints; step 5's seven tests all pass,
including under ASan+UBSan). One real gap found and resolved with the
maintainer while implementing step 4: `std::array<fft_direction, D>`
catches too-many directions in a braced-init-list as a hard compile error,
but NOT too-few (aggregate-init zero-pads, and zero == `fft_direction::none`
by construction) -- tried closing this via independent-N deduction with a
static_assert cross-check; confirmed empirically that a braced-init-list
is a non-deduced context for `std::array<T, N>`'s `N` too, so it can't be
closed without a custom fixed-arity wrapper replacing `std::array` here.
Maintainer decision: accept and document (not worth the added complexity)
-- see the `fft_inplace(dirs, arr)` overload's comment in fft.hpp.

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

Execute §10.3 steps 6–9. The enabling facts are all in §10.5 (uniform-
conjugation invariant, fused `conj_mul`, `Backward` plumbing through
`run_stages_`, Bluestein collapse + `kernel_ft_` index-reversal identity).

- Do the smooth Stockham path first (radix kernels + `wmat_`/generic),
  gate, then Bluestein separately.
- **Bluestein fallback rule (pre-authorized — taking it is a valid
  outcome, not a failure)**: attempt the neutral form once; if the
  index-reversal handling of `kernel_ft_` isn't clean and clearly correct,
  keep Bluestein engines direction-keyed (§10.2) and say so in the commit
  message. The prime-length backward test decides correctness, not
  intuition.
- Engine reuse key drops back to length alone (except Bluestein if the
  fallback was taken); the square-`{f, b}` test from Phase A now also
  proves the sharing.
- Sweep: `sign_` gone from engines (or reduced to Bluestein-only), no new
  `mutable`, thread-safety comment still true.

Checkpoint commit: "fft: direction-neutral engines, conj-on-load (Phase B,
NOTES §10)".

## Session 4 (optional, lowest priority) — polish

- `D`-bounded inline engine storage replacing `std::vector` (§10.1 item 9 /
  §10.3 optional step). Keep the linear-scan reuse lookup (§9.1: no map).
- Batched-1-D benchmark vs `fftw_plan_many_dft` (§10.3 optional follow-up;
  full methodology rules in `benchmark/algorithms_fft.cpp`'s header —
  idle-machine protocol applies to *published* numbers).
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
