# Stage 7 — Fused CUDA CV Preprocessing

This stage moves from GEMM optimization to a practical computer-vision preprocessing workload.

The baseline path used separate operations for:
- resize
- normalization
- HWC → CHW conversion
- uint8 → float32 conversion

These operations were fused into a single CUDA kernel.

## Results

| Pipeline | Median |
|---|---:|
| Baseline — 3 kernels | 0.1618 ms |
| Fused — 1 kernel | 0.0576 ms |

Speedup: approximately **2.8×**

Intermediate memory traffic avoided: approximately **18.75 MB/frame**

Maximum output difference: **0.0**
