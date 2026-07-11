# Profiling task: where do the cycles go, multi::fft_plan vs FFTW?

Prompt for an implementing model (Sonnet 5) or developer. Written
2026-07-11, after the §11 optimization experiments (see fft.NOTES.md)
established the FFTW gap is memory passes/locality, not FLOPs — but
nobody has ever actually profiled to confirm where the time goes. This
task is **measurement only**; it exists to arbitrate between the
candidate optimizations listed at the end.

Read `fft.NOTES.md` §11 first (especially §11.13's conclusion) and
`benchmark/algorithms_fft.cpp`'s methodology header. Work on branch
`fftalgo2026`.

## Task

Profile multi::fft_plan against FFTW with Linux perf. Do NOT change
`fft.hpp` or any product code. All harnesses go in the session
scratchpad, never the repo. The deliverable is a new §11.14 in
`fft.NOTES.md`.

## Context

multi::fft_plan runs at ~55-68% of FFTW single-threaded (see the
committed `fft_bench_*_nowisdom.dat`). §11's experiments established the
gap is memory passes/locality, NOT FLOPs (split-radix cut FLOPs and lost
50-80%; unseq/par/fma/-ffast-math were all neutral-to-negative). Worst
regions, in order: 1-D n >= ~390k (25-35% of FFTW, six-step path), the
batched "many" sweeps (55-61%), everything else (60-80%).

## Cases to profile

Each as its own minimal harness: build the plan once, warm up once, then
a fixed-rep timed loop over `plan.execute()` ONLY — do not interleave
with FFTW in the same process; profile FFTW's equivalent loop as a
separate binary so attribution stays clean.

- **P1**: 1-D n=1024 (mid-size smooth, radix-4 dominated)
- **P2**: 1-D n=4096 (deeper stage pipeline)
- **P3**: 1-D n=1048576 (six-step path — the worst measured region)
- **P4**: batched many: `{none, forward}` on [256][1024] row-contiguous
  (the per-fiber contiguous path `fft_exec_slab` takes today)
- **P5**: 1-D n=1009 (Bluestein, prime)
- **F1-F5**: the FFTW equivalents (`fftw_plan_dft_1d` /
  `fftw_plan_many_dft`, FFTW_MEASURE, wisdom fine here — we want its
  best execution)

## Build

The benchmark's exact flags PLUS debug info:

    g++ -std=c++17 -O3 -march=native -mtune=native -funroll-loops \
        -fno-math-errno -DNDEBUG -g -fno-omit-frame-pointer

Do NOT add `-fno-inline` (it distorts exactly what we're measuring);
accept that everything inlines into `execute()` and rely on `-g`
source-line attribution (`perf annotate` / `perf report` with
`--call-graph dwarf`).

## Measure, per case

a) `perf stat -d -d`: cycles, instructions, IPC, L1-dcache miss rate,
   LLC miss rate, branch misses. Same counters for the FFTW twin.

b) `perf record --call-graph dwarf` + `perf report`/`annotate`:
   attribute self-time to source regions of `fft.hpp` — at minimum
   split between:
   - gather/scatter copies (`fft_exec_fiber`/`fft_exec_slab`/
     `stage_subplan_` copy loops),
   - stage kernels' arithmetic (`stage_radix4_`/`8_`/`generic_`),
   - twiddle loads (`tw_[...]` indexing — check for cache-miss hotspots
     on the strided `tw_[r*tstep]` reads specifically; that's candidate
     #2's hypothesis),
   - six-step transpose (P3),
   - Bluestein chirp/pointwise (P5).

   For FFTW, codelet symbol names from libfftw3 are visible — record
   the top few so we know which strategy it picked per size.

## Machine protocol

Same as the benchmark's: AC power, >=95% idle via `mpstat`, package temp
well under the 82°C high (`sensors`), no competing jobs; re-check
between cases. perf may need `/proc/sys/kernel/perf_event_paranoid <= 1`
— check first; if it needs sudo to change, STOP and ask the user to run
the sysctl rather than attempting it yourself.

## Deliverable

New §11.14 in `fft.NOTES.md` (commit that file only, message naming the
section; nothing else, no binaries, no .dat churn):

- a per-case table: our IPC / L1-miss% / LLC-miss% / branch-miss% vs
  FFTW's, plus cycles-per-point;
- per-case self-time breakdown by source region (percentages);
- which FFTW codelets/strategies it used per size (symbol names);
- a short verdict section: for each candidate optimization below, does
  the data SUPPORT, WEAKEN, or say NOTHING about it —
  - **#2 per-stage sequential twiddle tables** (supported iff
    twiddle-load cache misses are a visible fraction),
  - **#3 radix-16 kernel / fewer passes** (supported iff we're
    bandwidth-bound: low IPC + high LLC misses in stage kernels),
  - **#4 six-step rework** (supported iff P3's transpose dominates),
  - **#5 batched-contiguous heuristic** (supported iff P4 shows
    per-fiber overhead or poor IPC vs FFTW's many-codelet);

  and end with a single recommended next task, with the numbers that
  justify it.

## Rules

Standard session gates do not apply (no product code changes), but if
any harness reveals a correctness anomaly, stop and report rather than
investigating unilaterally. If perf is unavailable or the machine can't
be made quiet, report what you got and mark which conclusions are
degraded rather than padding confidence.
