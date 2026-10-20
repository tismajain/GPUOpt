# Extracted from notebook Stage 10A cells 93-97.

# Stage 10A — Pageable vs pinned-memory HtoD transfer

import time
import numpy as np
import pandas as pd
import torch

NUM_FRAMES = 300
WARMUP = 30

frames10 = bench_frames[:NUM_FRAMES + WARMUP]

H = frames10[0].shape[0]
W = frames10[0].shape[1]
C = frames10[0].shape[2]

print("=" * 80)
print("STAGE 10A — PINNED HOST MEMORY")
print("=" * 80)

print("Frames :", len(frames10))
print("Shape  :", (H, W, C))
print("Bytes/frame:",
      H * W * C,
      f"({H * W * C / (1024**2):.2f} MiB)")

# Reusable pinned CPU buffer
pinned_cpu = torch.empty(
    (H, W, C),
    dtype=torch.uint8,
    pin_memory=True
)

# Reusable GPU buffer
gpu_buffer = torch.empty(
    (H, W, C),
    dtype=torch.uint8,
    device="cuda"
)

copy_stream = torch.cuda.Stream()

print("Pinned:", pinned_cpu.is_pinned())

pageable_times = []
pinned_dma_times = []
pinned_stage_times = []
pinned_full_times = []

# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

for f in frames10[:WARMUP]:

    cpu_t = torch.from_numpy(f)

    # Pageable
    gpu_buffer.copy_(cpu_t)

    # Pageable -> pinned staging
    pinned_cpu.copy_(cpu_t)

    # Async pinned -> device
    with torch.cuda.stream(copy_stream):
        gpu_buffer.copy_(pinned_cpu, non_blocking=True)

torch.cuda.synchronize()


# ------------------------------------------------------------
# Timed benchmark
# ------------------------------------------------------------

for f in frames10[WARMUP:WARMUP + NUM_FRAMES]:

    cpu_t = torch.from_numpy(f)

    # ========================================================
    # A. Current pageable CPU -> GPU
    # ========================================================

    torch.cuda.synchronize()

    t0 = time.perf_counter()

    gpu_buffer.copy_(cpu_t)

    torch.cuda.synchronize()

    t1 = time.perf_counter()

    pageable_times.append(
        (t1 - t0) * 1000.0
    )


    # ========================================================
    # B. CPU pageable -> pinned staging copy
    # ========================================================

    t0 = time.perf_counter()

    pinned_cpu.copy_(cpu_t)

    t1 = time.perf_counter()

    pinned_stage_times.append(
        (t1 - t0) * 1000.0
    )


    # ========================================================
    # C. Already-pinned -> GPU
    #
    # CUDA events measure actual GPU-side DMA time.
    # ========================================================

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(copy_stream):

        start.record(copy_stream)

        gpu_buffer.copy_(
            pinned_cpu,
            non_blocking=True
        )

        end.record(copy_stream)

    end.synchronize()

    pinned_dma_times.append(
        start.elapsed_time(end)
    )


    # ========================================================
    # D. Real staged path:
    #
    # normal OpenCV/numpy frame
    #      ↓
    # pinned host buffer
    #      ↓
    # async HtoD
    #
    # Includes staging cost.
    # ========================================================

    torch.cuda.synchronize()

    t0 = time.perf_counter()

    pinned_cpu.copy_(cpu_t)

    with torch.cuda.stream(copy_stream):
        gpu_buffer.copy_(
            pinned_cpu,
            non_blocking=True
        )

    copy_stream.synchronize()

    t1 = time.perf_counter()

    pinned_full_times.append(
        (t1 - t0) * 1000.0
    )

def report(name, values):
    a = np.asarray(values)

    print(
        f"{name:30s}"
        f"median={np.median(a):8.4f} ms | "
        f"p95={np.percentile(a,95):8.4f} ms | "
        f"mean={np.mean(a):8.4f} ms"
    )


print("\n" + "=" * 90)
print("STAGE 10A RESULTS")
print("=" * 90)

report(
    "Pageable HtoD",
    pageable_times
)

report(
    "Pinned staging CPU copy",
    pinned_stage_times
)

report(
    "Pinned HtoD DMA only",
    pinned_dma_times
)

report(
    "Staging + pinned HtoD",
    pinned_full_times
)


page_med = np.median(pageable_times)
dma_med = np.median(pinned_dma_times)
full_med = np.median(pinned_full_times)

print("\n" + "-" * 90)

print(
    f"Pinned DMA speedup vs pageable : "
    f"{page_med / dma_med:.2f}x"
)

print(
    f"Real staged speedup             : "
    f"{page_med / full_med:.2f}x"
)

df_stage10a = pd.DataFrame({
    "pageable_h2d_ms": pageable_times,
    "pinned_staging_ms": pinned_stage_times,
    "pinned_dma_ms": pinned_dma_times,
    "pinned_full_ms": pinned_full_times,
})

df_stage10a.to_csv(
    "/kaggle/working/stage10a_pinned_memory.csv",
    index=False
)

print(
    "\nSaved: "
    "/kaggle/working/stage10a_pinned_memory.csv"
)