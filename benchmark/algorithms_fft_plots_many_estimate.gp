# Plots multi::fft_plan vs FFTW 3 (ESTIMATE, wisdom disabled) from
# fft_bench_many_h{32,256}_estimate.dat and fft_bench_many3d_h32_estimate.dat.
# usage: run algorithms_fft.cpp.x built with -DDISABLE_WISDOM -DUSE_ESTIMATE
# first (writes the _estimate.dat files to the current directory), then:
# gnuplot algorithms_fft_plots_many_estimate.gp
# data columns: n N_total mine_ms fftw_ms mine_mflops fftw_mflops ratio
# "many": batched 1-D, {none,forward} on [howmany][n] row-contiguous, vs
# fftw_plan_many_dft -- fiber length n on the x-axis, howmany fixed per plot.
# "many3d": batched 2-D, {none,forward,forward} on [depth][n][n] row-contiguous,
# vs fftw_plan_many_dft rank=2 -- layer side n on the x-axis, depth fixed.
# FFTW uses FFTW_ESTIMATE (no runtime codelet search) with wisdom DISABLED;
# plan-build time excluded for both; timing is interleaved per-repetition.

set terminal pngcairo size 900,600 font ",11"
set grid xtics ytics mytics
set logscale x 10
set ylabel "speed  5·N·log_2(N) / time   [MFLOPS]  (higher is better)"
set key top left
set style line 1 lw 2 pt 7  ps 1.1 lc rgb "#c0392b"   # multi
set style line 2 lw 2 pt 5  ps 1.1 lc rgb "#2c3e50"   # fftw

set format x "%.0f"
set xtics rotate by -30

set xlabel "fiber length n, 2^a * 3^b * 5^c"

set output "fft_bench_many_h32_estimate.png"
set title "Batched 1-D {none,forward}, howmany=32 — multi::fft\\_plan vs fftw\\_plan\\_many\\_dft (ESTIMATE, no wisdom)"
plot "fft_bench_many_h32_estimate.dat" using 1:5 with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_many_h32_estimate.dat" using 1:6 with linespoints ls 2 title "FFTW 3 (estimate, no wisdom)"

set output "fft_bench_many_h256_estimate.png"
set title "Batched 1-D {none,forward}, howmany=256 — multi::fft\\_plan vs fftw\\_plan\\_many\\_dft (ESTIMATE, no wisdom)"
plot "fft_bench_many_h256_estimate.dat" using 1:5 with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_many_h256_estimate.dat" using 1:6 with linespoints ls 2 title "FFTW 3 (estimate, no wisdom)"

set xlabel "layer side n (n × n per layer), 2^a * 3^b * 5^c"

set output "fft_bench_many3d_h32_estimate.png"
set title "Batched 2-D {none,forward,forward}, depth=32 — multi::fft\\_plan vs fftw\\_plan\\_many\\_dft rank=2 (ESTIMATE, no wisdom)"
plot "fft_bench_many3d_h32_estimate.dat" using 1:5 with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_many3d_h32_estimate.dat" using 1:6 with linespoints ls 2 title "FFTW 3 (estimate, no wisdom)"
