# Stage 8 — Real-Video Preprocessing Benchmark

The fused preprocessing kernel was benchmarked on a real boxing video.

## Workload

- Resolution: 1920×1080
- Frame rate: 50 FPS
- Frames benchmarked: 300

## Results

| Pipeline | Median | p95 |
|---|---:|---:|
| Baseline | 0.1471 ms | 0.1564 ms |
| Fused CUDA | 0.0460 ms | 0.0513 ms |

Median speedup: **3.18×**

Approximate saving: **0.10 ms/frame**

Maximum output difference: **0.0**
