# GPUOpt

A hands-on CUDA optimization project exploring how GPU performance changes across memory access patterns, shared-memory tiling, register blocking, kernel fusion, and real computer-vision workloads.

The project starts from simple matrix multiplication and progressively moves toward application-level optimization in a real YOLO inference pipeline.

The main goal is not just to make individual CUDA kernels faster, but to understand **where the bottleneck moves after each optimization**.

---

## Highlights

- Improved CUDA memory access performance by ~**4.0×** through coalescing.
- Shared-memory tiling delivered a further ~**2.5×** speedup over the coalesced implementation.
- Register-blocked GEMM reached approximately **56% of cuBLAS performance** at 1024×1024 on an NVIDIA Tesla T4.
- Fused real-video preprocessing achieved a **3.18× speedup** on 1920×1080 @ 50 FPS video.
- Fused YOLO preprocessing reduced compute from **1.615 ms to 0.063 ms**, a **25.65× kernel-level speedup**.
- Detection equivalence was validated across 100 frames with **100% matching detection counts**.
- End-to-end YOLO improvement remained ~**1.002×** because host-to-device transfer became the new bottleneck.

---

## Hardware and Environment

Experiments were primarily run on:

- GPU: NVIDIA Tesla T4
- CUDA Toolkit: 12.8
- Driver: 580.178.04
- PyTorch: 2.11.0 + CUDA 12.8
- Ultralytics: 8.4.183
- Nsight Compute: 2025.1.1
- Environment: Kaggle GPU notebook

Some hardware-performance-counter profiling was unavailable because the Kaggle environment returned:

```text
ERR_NVGPUCTRPERM
```

Profiler-dependent metrics were therefore not inferred or fabricated.

---

# Project Progression

| Stage | Focus | Key Result |
|---|---|---|
| 1 | CPU blocked matrix multiplication | Best configuration reached ~7.86 GFLOP/s |
| 2 | CUDA memory coalescing | ~4.0× faster than strided global-memory access |
| 3 | Shared-memory tiling | ~2.5× improvement over coalesced baseline |
| 4 | Register blocking | Higher arithmetic reuse and significantly improved GEMM throughput |
| 5 | cuBLAS comparison | ~56% of cuBLAS performance at 1024×1024 |
| 6 | Nsight Compute profiling | Blocked by restricted GPU performance-counter access |
| 7 | Fused CV preprocessing | Combined resize, normalization, and layout conversion |
| 8 | Real-video preprocessing | 3.18× median speedup on 1920×1080 video |
| 9 | Real YOLO integration | 25.65× preprocessing-kernel speedup; PCIe became bottleneck |
| 10 | GPU-resident pipeline | Planned: pinned memory, async copies, overlap, NVDEC, NV12 |

---

# Stage 1 — CPU Blocked Matrix Multiplication

The project begins with a CPU blocked matrix-multiplication baseline.

Blocking improves cache reuse by processing matrices in smaller tiles instead of repeatedly traversing large regions of memory.

For:

```text
N = 1024
```

the measured results were approximately:

| Block Size | Time | Performance |
|---:|---:|---:|
| 16 | 443.9 ms | 4.84 GFLOP/s |
| 32 | 361.0 ms | 5.95 GFLOP/s |
| 64 | 273.4 ms | 7.86 GFLOP/s |

This establishes the basic idea that data locality matters before moving to GPU-specific optimizations.

---

# Stage 2 — CUDA Memory Coalescing

The first CUDA experiment isolates global-memory access behavior.

Two kernels perform similar work but access memory differently:

- poorly coalesced access
- coalesced access

At 1024×1024:

| Kernel | Time | Performance |
|---|---:|---:|
| Strided / bad access | 17.92 ms | 119.8 GFLOP/s |
| Coalesced | 4.46 ms | 481.3 GFLOP/s |

Result:

```text
~4.0× speedup
```

This demonstrates how strongly GPU performance depends on how adjacent threads access global memory.

---

# Stage 3 — Shared-Memory Tiling

The next step uses shared memory to reuse matrix tiles across multiple operations.

Instead of repeatedly reading the same values from global memory, each thread block loads a tile into shared memory and performs multiple operations using the cached data.

At 1024×1024:

| Kernel | Time | Performance |
|---|---:|---:|
| Coalesced | 6.24 ms | 344.2 GFLOP/s |
| Shared-memory tiled | 2.51 ms | 854.5 GFLOP/s |

Result:

```text
~2.48× speedup over the coalesced implementation
```

---

# Stage 4 — Register Blocking

The tiled kernel was extended with register blocking.

Each thread computes multiple output values and reuses loaded data through registers.

PTXAS resource usage on `sm_75`:

| Kernel | Registers | Shared Memory |
|---|---:|---:|
| Tiled | 39 | ~2 KB |
| Register blocked | 72 | ~8 KB |

This introduced a tradeoff:

- more per-thread reuse
- fewer global-memory operations
- increased register usage
- potentially reduced occupancy

The resulting throughput improvement showed that higher per-thread work could outperform the simpler tiled design despite using more GPU resources.

---

# Stage 5 — Comparison Against cuBLAS

The register-blocked implementation was compared against NVIDIA cuBLAS.

For 1024×1024 matrix multiplication, repeated measurements showed approximately:

```text
Register blocked: ~2872–2889 GFLOP/s
cuBLAS:           ~4902–5140 GFLOP/s
```

The custom CUDA implementation therefore reached approximately:

```text
~56% of cuBLAS performance
```

At 512×512:

```text
Register blocked: ~1209.9 GFLOP/s
~57.3% of cuBLAS
```

This stage provides a realistic reference point against a heavily optimized production library.

---

# Stage 6 — Nsight Compute Profiling

The next planned step was to profile the register-blocked kernel with NVIDIA Nsight Compute and inspect:

- achieved occupancy
- memory throughput
- cache utilization
- warp stall reasons
- arithmetic intensity
- instruction-level behavior

However, Kaggle did not expose NVIDIA GPU hardware performance counters.

Nsight Compute returned:

```text
ERR_NVGPUCTRPERM
```

Rather than infer profiler metrics, the limitation is documented explicitly.

Future profiling should be repeated on a machine with performance-counter access.

---

# Stage 7 — Fused Computer-Vision Preprocessing

The project then moved from synthetic GEMM benchmarks to a practical CV workload.

A YOLO-style preprocessing pipeline was implemented with multiple operations:

```text
resize
→ normalization
→ HWC to CHW conversion
→ float conversion
```

The baseline used multiple kernels and intermediate buffers.

A custom fused CUDA implementation performs the operations in one kernel.

Measured on 1920×1080 input converted to 640×640:

| Pipeline | Median Time |
|---|---:|
| Baseline, 3 kernels | 0.1618 ms |
| Fused, 1 kernel | 0.0576 ms |

Result:

```text
~2.7–3.0× speedup
```

The fused implementation also avoided approximately:

```text
18.75 MB/frame
```

of intermediate memory traffic.

Correctness:

```text
Max difference = 0.0
```

---

# Stage 8 — Real Video Benchmark

The fused preprocessing path was tested on a real boxing video:

```text
Resolution: 1920×1080
Frame rate: 50 FPS
Frames benchmarked: 300
```

Results:

| Pipeline | Median | p95 |
|---|---:|---:|
| Baseline | 0.1471 ms | 0.1564 ms |
| Fused | 0.0460 ms | 0.0513 ms |

Median speedup:

```text
3.18×
```

Approximate saving:

```text
~0.10 ms/frame
```

Correctness remained:

```text
Max difference = 0.0
```

This stage demonstrated that the fused-kernel speedup also held on real video rather than only on synthetic data.

---

# Stage 9 — Real YOLO Inference Integration

The next step was integrating the custom CUDA preprocessing path into an actual YOLOv8n inference pipeline.

Input video:

```text
1920×1080
50 FPS
YOLOv8n
NVIDIA Tesla T4
```

## Actual YOLO Input Geometry

Ultralytics performs rectangular inference rather than forcing this video into 640×640.

The real preprocessing path is:

```text
1920×1080
     ↓
resize to 640×360
     ↓
12 px top padding
12 px bottom padding
     ↓
640×384 input
```

The custom CUDA kernel reproduces:

- bilinear resize
- YOLO letterboxing
- padding value 114
- BGR → RGB
- HWC → CHW
- uint8 → float32
- `/255` normalization

in a single kernel.

---

## Preprocessing Correctness

The custom CUDA preprocessing tensor was compared directly against Ultralytics preprocessing.

Results:

```text
Max absolute difference  : 5.96e-08
Mean absolute difference : 1.25e-08
```

The tensors were effectively numerically identical.

---

## Raw YOLO Output Equivalence

Both tensors were passed through YOLOv8n.

Model output shape:

```text
1 × 84 × 5040
```

Results:

```text
Max raw-output difference  : 8.54e-04
Mean raw-output difference : 7.20e-07
Allclose                   : True
```

---

## Post-NMS Detection Equivalence

Detection equivalence was validated across 100 real video frames.

Results:

```text
Matching detection counts       : 100 / 100
Detection-count match rate      : 100%
Maximum detection-count diff    : 0
Maximum confidence difference   : 1.85e-06
Maximum box-coordinate diff     : 0.000427 px
Mean box-coordinate difference  : 0.000024 px
```

This confirms that the custom preprocessing path produces effectively identical detections.

---

# YOLO Preprocessing Performance

The baseline Ultralytics preprocessing path was compared against the fused CUDA kernel.

| Path | Median | p95 |
|---|---:|---:|
| Ultralytics preprocessing | 1.6152 ms | 1.7981 ms |
| HtoD transfer only | 1.4866 ms | 1.5978 ms |
| Fused CUDA kernel only | 0.0630 ms | 0.0769 ms |
| HtoD + fused kernel | 1.6014 ms | 1.7363 ms |

Kernel-level speedup:

```text
1.6152 / 0.0630 ≈ 25.65×
```

However:

```text
Full preprocessing speedup ≈ 1.01×
```

The fused kernel itself was no longer the bottleneck.

---

# End-to-End YOLO Result

Preprocessing plus YOLO model inference was benchmarked separately.

## Baseline

```text
Preprocessing           : 1.6278 ms
Inference               : 8.0061 ms
Preprocess + inference  : 9.6371 ms
```

## Custom CUDA Path

```text
HtoD + fused preprocess : 1.5842 ms
Inference               : 8.0342 ms
Preprocess + inference  : 9.6181 ms
```

Result:

```text
Overall speedup: 1.002×
```

Theoretical model-pipeline throughput:

```text
Baseline : 103.77 FPS
Custom   : 103.97 FPS
```

---

# Bottleneck Analysis

The fused kernel reduced preprocessing compute from:

```text
1.615 ms
```

to:

```text
0.063 ms
```

Yet the full pipeline barely improved.

The reason is the data path.

## Ultralytics

```text
CPU decoded frame
      ↓
CPU resize + letterbox
      ↓
smaller tensor
      ↓
HtoD
      ↓
YOLO
```

## Custom CUDA

```text
CPU decoded 1920×1080 frame
      ↓
HtoD full-resolution frame
      ↓
fused CUDA preprocessing
      ↓
YOLO
```

Moving the full 1920×1080 BGR frame to the GPU costs approximately:

```text
1.49 ms/frame
```

That transfer dominates the custom preprocessing path.

The optimization therefore shifted the bottleneck from:

```text
preprocessing compute
```

to:

```text
host-to-device data movement
```

This is the main systems-level finding from Stage 9.

---

# Stage 10 — GPU-Resident Pipeline

The next stage investigates whether the PCIe bottleneck can be reduced or removed.

Planned experiments:

### 1. Pinned Host Memory

Measure whether page-locked memory improves HtoD transfer latency.

### 2. Asynchronous HtoD Transfer

Use:

```text
cudaMemcpyAsync
```

and CUDA streams to avoid blocking transfer execution.

### 3. Transfer / Inference Overlap

Pipeline consecutive frames:

```text
Frame N     → inference
Frame N + 1 → HtoD transfer
```

to determine whether transfer latency can be hidden behind GPU computation.

### 4. NVDEC GPU Video Decode

Move video decoding from CPU to the GPU:

```text
compressed video
      ↓
NVDEC
      ↓
GPU-resident frame
      ↓
CUDA preprocessing
      ↓
YOLO
```

This avoids transferring full decoded RGB frames over PCIe.

### 5. NV12-Native Fused Preprocessing

If NVDEC produces NV12 surfaces, implement:

```text
NV12
 ↓
color conversion
 ↓
resize
 ↓
letterbox
 ↓
CHW
 ↓
normalize
 ↓
YOLO
```

as a GPU-resident fused preprocessing pipeline.

---

# Key Takeaways

This project progressively exposed several CUDA optimization principles:

### Memory access can dominate arithmetic

Coalescing alone produced a major improvement without changing the mathematical operation.

### Shared memory improves reuse

Tiling reduced repeated global-memory access and substantially increased throughput.

### Registers introduce tradeoffs

Register blocking increased per-thread reuse but also increased register pressure.

### Kernel fusion reduces memory traffic

Combining CV preprocessing operations removed intermediate buffers and multiple kernel launches.

### Microbenchmark speedup is not application speedup

The YOLO preprocessing kernel became approximately **25.65× faster**, yet end-to-end performance improved by only ~0.2%.

The reason was not the kernel.

It was PCIe transfer.

This is the central lesson from the project:

> Optimizing the current bottleneck often reveals the next one.

---

# Repository Structure

```text
GPUOpt/
├── stage1_cpu_blocked/
├── stage2_coalescing/
├── stage3_tiled/
├── stage4_register_blocking/
├── stage5_cublas_benchmark/
├── stage6_profiling/
├── stage7_fused_preprocessing/
├── stage8_real_video/
├── stage9_yolo_integration/
│   ├── fused_yolo_preprocess.cu
│   ├── yolo_baseline_300.csv
│   ├── yolo_preprocess_benchmark.csv
│   ├── yolo_e2e_benchmark.csv
│   └── README.md
├── results/
└── README.md
```

---

# Current Status

Stages 1–9 are complete.

Stage 10 focuses on removing or hiding the host-to-device transfer bottleneck using asynchronous transfer and GPU-resident video decoding.
