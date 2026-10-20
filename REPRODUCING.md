# Reproducing GPUOpt

This document describes how to rebuild and rerun the project from a fresh checkout. The original measurements were collected primarily on an NVIDIA Tesla T4 in Kaggle with CUDA 12.8.

## Prerequisites

For Stages 1-5:

- C++17 compiler (`g++`)
- NVIDIA CUDA toolkit with `nvcc`
- NVIDIA GPU for CUDA stages
- cuBLAS development library for Stage 5

For Stages 7-10:

- FFmpeg
- Python 3
- CUDA-enabled PyTorch
- packages in `requirements.txt`
- YOLOv8n weights (`yolov8n.pt`, downloaded automatically by Ultralytics when allowed)
- the 1920x1080 @ 50 FPS H.264 boxing video used in the original experiment, or another video with the same geometry for the validated CUDA paths

For Stage 10 NVDEC:

- NVIDIA GPU with NVDEC support
- PyNvVideoCodec

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## One command per stage

From the repository root:

```bash
bash scripts/run_stage1.sh
bash scripts/run_stage2.sh
bash scripts/run_stage3.sh
bash scripts/run_stage4.sh
bash scripts/run_stage5.sh
```

Stage 6 uses Nsight Compute hardware counters:

```bash
bash scripts/run_stage6_ncu.sh
```

Kaggle returns `ERR_NVGPUCTRPERM`, so Stage 6 is expected to fail there. Run it on a machine where NVIDIA performance counters are enabled.

Stages 7-8 use a real 1920x1080 video. The runner extracts frame 6000 to raw BGR, compiles the original CUDA benchmark with a portable frame path, and runs it:

```bash
bash scripts/run_stage7_8.sh /path/to/video.mp4
```

Stage 9 rebuilds the fused YOLO preprocessing library and reruns the core bottleneck experiment:

```bash
bash scripts/run_stage9.sh /path/to/video.mp4 yolov8n.pt
```

It measures Ultralytics preprocessing, HtoD transfer, fused-kernel time, HtoD+fused preprocessing, and preprocess+inference. The original T4 result was approximately 25.65x kernel-level preprocessing speedup but only ~1.002x preprocess+inference speedup because PCIe transfer became the bottleneck.

Stage 10 reruns the final GPU-resident pipeline:

```bash
bash scripts/run_stage10.sh /path/to/video.mp4 yolov8n.pt
```

The Stage 10 directory also contains its own detailed environment checks and reproduction instructions.

## Run the sequence

Stages 1-5 can be run without video input:

```bash
bash scripts/run_all.sh
```

To also run Stages 7-10:

```bash
bash scripts/run_all.sh /path/to/video.mp4 yolov8n.pt
```

Stage 6 is intentionally excluded from `run_all.sh` because profiler permissions are machine-dependent.

## Expected headline results on the original T4

| Stage | Expected result |
|---|---|
| 1 | Best CPU block size 64, ~7.86 GFLOP/s |
| 2 | Coalesced access ~4.0x faster than strided access |
| 3 | Shared-memory tiled ~2.48x faster than coalesced baseline |
| 4-5 | Register-blocked GEMM reaches ~56% of cuBLAS at 1024x1024 |
| 6 | Nsight blocked on Kaggle by `ERR_NVGPUCTRPERM` |
| 7-8 | Fused real-video preprocessing ~3.18x median speedup |
| 9 | 25.65x kernel-level speedup, but ~1.002x preprocess+inference speedup |
| 10 | 11.95 -> 9.03 ms end-to-end, 83.7 -> 110.7 FPS, 24.4% latency reduction |

These are reference measurements, not pass/fail thresholds. GPU clocks, drivers, CUDA/PyTorch versions, codec behavior, and system load can shift absolute timings.

## Benchmark protocol

The video/YOLO experiments use warmup before measurement and synchronize CUDA around timed GPU sections. Stage 10 uses 30 warmup frames and 300 measured frames for the final apples-to-apples comparison. The final validated geometry is 1920x1080 input with Ultralytics rectangular inference producing a 640x384 tensor.

## Data

The project does not commit the source boxing video. Supply the video path explicitly to the reproduction scripts. For the original experiment, the source was H.264, 1920x1080, 50 FPS.
