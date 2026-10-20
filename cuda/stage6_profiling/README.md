# Stage 6 — Nsight Compute Profiling

## Status

Nsight Compute profiling was attempted on the NVIDIA Tesla T4.

The Kaggle environment returned:

`ERR_NVGPUCTRPERM`

Hardware performance counters were unavailable, so profiler-derived metrics such as occupancy, warp stall reasons, cache behavior, and memory throughput could not be collected reliably.

Rather than estimate these values, this stage documents the environment limitation.

## Next Step

Repeat Nsight Compute profiling on a GPU environment with hardware performance-counter access.
