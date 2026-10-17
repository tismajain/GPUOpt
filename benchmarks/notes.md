# Benchmark Notes

## Environment
- GPU: NVIDIA Tesla T4
- CUDA Toolkit: 12.8
- Target architecture: sm_75
- FP32 GEMM
- cuBLAS math mode: CUBLAS_DEFAULT_MATH
- Timing: CUDA events
- Warmups: 5
- Measured runs per kernel: 30
- Reported timing: median

## Stage 1 — CPU blocked GEMM
- N=1024
- Block 16: 4.837 GFLOP/s
- Block 32: 5.948 GFLOP/s
- Block 64: 7.855 GFLOP/s
- Best tested block size: 64
- ~1.62x over block size 16

## Stage 2 — Coalescing study
- Poor thread/data mapping: 119.832 GFLOP/s
- Coalesced mapping: 481.312 GFLOP/s
- Speedup: 4.017x
- Max difference: 0

Interpretation:
This was a controlled experiment isolating thread-to-memory mapping.
Do not describe it as a profiler-discovered bottleneck.

## Stage 3 — Shared-memory tiling
Standalone run:
- Coalesced: 344.220 GFLOP/s
- Tiled: 854.531 GFLOP/s
- Speedup: 2.483x

Interpretation:
Shared-memory tiling reduced repeated global-memory loads through block-level reuse.

## Stage 4 — Register blocking
Compiled for sm_75:
- Shared tiled: 39 registers/thread, 2048 B shared memory
- Register blocked: 72 registers/thread, 8192 B shared memory
- Spill stores: 0
- Spill loads: 0

Standalone run:
- Shared tiled: 528.520 GFLOP/s
- Register blocked: 1982.185 GFLOP/s
- Speedup: 3.750x

Interpretation:
Higher register usage enabled more thread-level reuse without spilling.

## Stage 5 — Unified controlled benchmark
At N=1024:
- Coalesced: 355.932 GFLOP/s
- Shared tiled: 808.492 GFLOP/s
- Register blocked: 2937.188 GFLOP/s
- cuBLAS SGEMM: 6041.489 GFLOP/s
- Register-blocked kernel: 48.617% of cuBLAS

Overall improvement:
- 2937.188 / 355.932 = ~8.25x

## Scaling
Register-blocked kernel achieved approximately 53–59% of cuBLAS across N=512–4096 in one scaling sweep.

## Stability
For N=1024 over 5 repeated runs:
- Register-blocked median: ~2.872 TFLOP/s
- Register-blocked range: ~2.830–2.895 TFLOP/s
- cuBLAS median: ~4.992 TFLOP/s
- Relative performance range: ~55.2–61.7%

## Thermal / clock behavior
Observed after shorter N=1024 workload:
- GPU temperature: 56 C
- SM clock: 1590 MHz

Observed after sustained N=4096 workload:
- GPU temperature: 80 C
- SM clock: 1200 MHz

Interpretation:
Absolute throughput drops under sustained large workloads correlate with reduced SM clock.
Therefore final claims should prefer median/range reporting over a single best run.
