# Plots multi::fft_plan vs FFTW 3 from fft_bench_{1d,2d,3d}_nowisdom.dat
# usage: run algorithms_fft.cpp.x built with -DDISABLE_WISDOM first (writes
# the _nowisdom.dat files to the current directory), then:
# gnuplot algorithms_fft_plots.gp
# data columns: n_side N_total mine_ms fftw_ms mine_mflops fftw_mflops ratio
# sizes are 2^a * 3^b * 5^c (5-smooth), including every pure single-prime
# power in range, not just powers of two; FFTW uses FFTW_MEASURE with wisdom
# DISABLED (fftw_forget_wisdom() before every plan); plan construction and
# input restoration are excluded; both implementations execute in place.

set terminal pngcairo size 900,600 font ",11"
set grid xtics ytics mytics
set logscale x 10
set yrange [0:*]
set xlabel "transform size n (per axis), 2^a * 3^b * 5^c"
set ylabel "speed  5·N·log_2(N) / time   [MFLOPS]  (higher is better)"
set key top left
set style line 1 lw 2 pt 7  ps 1.1 lc rgb "#c0392b"   # multi
set style line 2 lw 2 pt 5  ps 1.1 lc rgb "#2c3e50"   # fftw

set format x "%.0f"
set xtics rotate by -30

set output "fft_bench_1d.png"
set title "1-D in-place complex FFT — multi::fft\\_plan vs FFTW 3 (cold cache, MEASURE)"
plot "fft_bench_1d_nowisdom.dat" using 1:5 with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_1d_nowisdom.dat" using 1:6 with linespoints ls 2 title "FFTW 3 (measure, no wisdom)"

set output "fft_bench_2d.png"
set title "2-D in-place complex FFT (n × n) — multi::fft\\_plan vs FFTW 3 (cold cache, MEASURE)"
plot "fft_bench_2d_nowisdom.dat" using 1:5 with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_2d_nowisdom.dat" using 1:6 with linespoints ls 2 title "FFTW 3 (measure, no wisdom)"

set output "fft_bench_3d.png"
set title "3-D in-place complex FFT (n × n × n) — multi::fft\\_plan vs FFTW 3 (cold cache, MEASURE)"
plot "fft_bench_3d_nowisdom.dat" using 1:5 with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_3d_nowisdom.dat" using 1:6 with linespoints ls 2 title "FFTW 3 (measure, no wisdom)"
