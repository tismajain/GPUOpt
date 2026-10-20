# Stage 10 — Removing the PCIe Bottleneck

Stage 9 reduced YOLO preprocessing compute from 1.615 ms to 0.063 ms, a 25.65× kernel-level speedup.

However, transferring the full 1920×1080 frame from CPU memory to the GPU cost approximately 1.49 ms, eliminating almost all of the application-level gain.

Stage 10 investigates and removes that bottleneck.

## 10A — Pinned Host Memory

| Path | Median |
|---|---:|
| Pageable HtoD | 1.4709 ms |
| Pinned HtoD DMA | 0.6598 ms |
| CPU staging + pinned HtoD | 1.0113 ms |

Results:

- Raw DMA speedup: **2.23×**
- Real staged-path speedup: **1.45×**

## 10B — Transfer / Inference Overlap

| Pipeline | Time / Frame | Throughput |
|---|---:|---:|
| Sequential pinned | 9.8045 ms | 101.99 FPS |
| Overlapped | 9.0194 ms | 110.87 FPS |

Steady-state throughput improved by approximately **8.7%**.

## 10C — GPU Video Decode with NVDEC

The H.264 boxing video was decoded on the NVIDIA T4 using NVDEC through PyNvVideoCodec.

The decoded frame remained GPU-resident, removing the full-resolution CPU → GPU transfer from the critical path.

## 10D — Direct NV12 Fusion Attempt

A direct NV12 → YOLO fused CUDA path was tested.

Across 100 frames:

- Detection-count agreement: **64%**
- Class-sequence agreement: **40%**

Because it failed the correctness gate, this implementation was not used in the final system.

## 10E — GPU-Resident RGBP Pipeline

The final architecture was:

```text
H.264
  ↓
NVDEC
  ↓
GPU RGBP
  ↓
fused CUDA resize + letterbox + normalize
  ↓
YOLO
