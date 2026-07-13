# 2-D FFT: multi::fft_plan vs FFTW at three planning levels.
# usage: gnuplot algorithms_fft_2d_planning_levels.gp
# data columns: n N mine_ms est_ms mea_ms exh_ms mine_mflops est_mflops mea_mflops exh_mflops

set terminal pngcairo size 900,600 font ",11"
set output "fft_bench_2d_planning_levels.png"

set title "In-place 2-D FFT n×n — multi::fft\\_plan vs FFTW planning levels (cold cache)"
set xlabel "side length n, 2^a * 3^b * 5^c"
set ylabel "speed  5·N·log_2(N) / time   [MFLOPS]  (higher is better)"

set grid xtics ytics mytics
set logscale x 10
set yrange [0:*]
set key top left
set format x "%.0f"
set xtics rotate by -30

set style line 1 lw 2 pt 7  ps 1.1 lc rgb "#c0392b"   # multi
set style line 2 lw 2 pt 5  ps 1.1 lc rgb "#7f8c8d"   # ESTIMATE
set style line 3 lw 2 pt 9  ps 1.1 lc rgb "#2980b9"   # MEASURE
set style line 4 lw 2 pt 13 ps 1.1 lc rgb "#27ae60"   # EXHAUSTIVE

plot "fft_bench_2d_planning_levels.dat" using 1:7  with linespoints ls 1 title "multi::fft\\_plan", \
     "fft_bench_2d_planning_levels.dat" using 1:8  with linespoints ls 2 title "FFTW ESTIMATE",     \
     "fft_bench_2d_planning_levels.dat" using 1:9  with linespoints ls 3 title "FFTW MEASURE",      \
     "fft_bench_2d_planning_levels.dat" using 1:10 with linespoints ls 4 title "FFTW EXHAUSTIVE"
