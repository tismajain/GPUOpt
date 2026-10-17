# GPUOpt

CUDA performance study and real computer-vision kernel optimization.

## Goal

Study GPU performance optimization through progressively optimized FP32 matrix multiplication kernels and apply the same principles to a fused CV preprocessing kernel.

## Environment

- NVIDIA Tesla T4
- CUDA Toolkit 12.8
- CUDA driver support: 13.0
- C++17 / CUDA C++

## Current progress

### Stage 1 — Cache-aware CPU baseline

Blocked FP32 matrix multiplication, N = 1024:

| Block | Time (ms) | GFLOP/s |
|---:|---:|---:|
| 16 | 443.936 | 4.837 |
| 32 | 361.042 | 5.948 |
| 64 | 273.380 | 7.855 |

Best tested configuration: block size 64, achieving ~1.62× the performance of block size 16.
