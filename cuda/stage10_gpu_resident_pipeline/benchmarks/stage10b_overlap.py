# Extracted from notebook Stage 10B cells 98-102.

# Stage 10B — Overlap pinned HtoD transfer with YOLO inference

import time
import numpy as np
import pandas as pd
import torch

NUM_FRAMES = 300
WARMUP = 30

frames = bench_frames[:NUM_FRAMES + WARMUP]

H, W, C = frames[0].shape

# ------------------------------------------------------------
# Double buffers
# ------------------------------------------------------------

pinned = [
    torch.empty((H, W, C), dtype=torch.uint8, pin_memory=True),
    torch.empty((H, W, C), dtype=torch.uint8, pin_memory=True)
]

gpu_raw = [
    torch.empty((H, W, C), dtype=torch.uint8, device="cuda"),
    torch.empty((H, W, C), dtype=torch.uint8, device="cuda")
]

gpu_pre = [
    torch.empty((1, 3, 384, 640), dtype=torch.float32, device="cuda"),
    torch.empty((1, 3, 384, 640), dtype=torch.float32, device="cuda")
]

copy_stream = torch.cuda.Stream()
infer_stream = torch.cuda.Stream()

copy_done = [
    torch.cuda.Event(),
    torch.cuda.Event()
]

# ------------------------------------------------------------
# Helper: launch fused kernel on supplied stream
# ------------------------------------------------------------

def launch_fused(src_gpu, out_gpu, stream):
    lib.launch_fused_yolo_preprocess(
        ctypes.c_void_p(src_gpu.data_ptr()),
        ctypes.c_void_p(out_gpu.data_ptr()),
        W,
        H,
        ctypes.c_void_p(stream.cuda_stream)
    )


# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

for i in range(WARMUP):

    b = i % 2
    cpu_t = torch.from_numpy(frames[i])

    pinned[b].copy_(cpu_t)

    with torch.cuda.stream(copy_stream):
        gpu_raw[b].copy_(pinned[b], non_blocking=True)
        copy_done[b].record(copy_stream)

    with torch.cuda.stream(infer_stream):
        infer_stream.wait_event(copy_done[b])

        launch_fused(
            gpu_raw[b],
            gpu_pre[b],
            infer_stream
        )

        with torch.no_grad():
            _ = model.model(gpu_pre[b])

torch.cuda.synchronize()

print("Warmup complete.")

sequential_times = []

for f in frames[WARMUP:WARMUP + NUM_FRAMES]:

    cpu_t = torch.from_numpy(f)

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # CPU -> pinned
    pinned[0].copy_(cpu_t)

    # pinned -> GPU
    with torch.cuda.stream(copy_stream):
        gpu_raw[0].copy_(
            pinned[0],
            non_blocking=True
        )

    copy_stream.synchronize()

    # fused preprocess + inference
    with torch.cuda.stream(infer_stream):

        launch_fused(
            gpu_raw[0],
            gpu_pre[0],
            infer_stream
        )

        with torch.no_grad():
            _ = model.model(gpu_pre[0])

    infer_stream.synchronize()

    t1 = time.perf_counter()

    sequential_times.append(
        (t1 - t0) * 1000.0
    )

# ------------------------------------------------------------
# Overlapped steady-state pipeline
# ------------------------------------------------------------

pipeline_start = torch.cuda.Event(enable_timing=True)
pipeline_end = torch.cuda.Event(enable_timing=True)

# CPU-side total wall time
torch.cuda.synchronize()
wall_start = time.perf_counter()

# ------------------------------------------------------------
# Prime first frame
# ------------------------------------------------------------

first = torch.from_numpy(
    frames[WARMUP]
)

pinned[0].copy_(first)

with torch.cuda.stream(copy_stream):
    gpu_raw[0].copy_(
        pinned[0],
        non_blocking=True
    )
    copy_done[0].record(copy_stream)

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

for i in range(NUM_FRAMES):

    current = i % 2

    # Inference for current frame
    with torch.cuda.stream(infer_stream):

        infer_stream.wait_event(
            copy_done[current]
        )

        launch_fused(
            gpu_raw[current],
            gpu_pre[current],
            infer_stream
        )

        with torch.no_grad():
            _ = model.model(
                gpu_pre[current]
            )

    # While current frame is being processed,
    # stage + copy next frame.
    if i + 1 < NUM_FRAMES:

        nxt = (i + 1) % 2

        cpu_next = torch.from_numpy(
            frames[WARMUP + i + 1]
        )

        # CPU staging copy
        pinned[nxt].copy_(cpu_next)

        with torch.cuda.stream(copy_stream):

            gpu_raw[nxt].copy_(
                pinned[nxt],
                non_blocking=True
            )

            copy_done[nxt].record(
                copy_stream
            )

torch.cuda.synchronize()

wall_end = time.perf_counter()

overlap_total_ms = (
    wall_end - wall_start
) * 1000.0

overlap_ms_per_frame = (
    overlap_total_ms / NUM_FRAMES
)

seq = np.asarray(sequential_times)

seq_median = np.median(seq)

print("\n" + "=" * 90)
print("STAGE 10B — TRANSFER / INFERENCE OVERLAP")
print("=" * 90)

print(
    f"Sequential pinned pipeline median : "
    f"{seq_median:.4f} ms/frame"
)

print(
    f"Overlapped total time             : "
    f"{overlap_total_ms:.4f} ms"
)

print(
    f"Overlapped average                : "
    f"{overlap_ms_per_frame:.4f} ms/frame"
)

print(
    f"Pipeline speedup                  : "
    f"{seq_median / overlap_ms_per_frame:.3f}x"
)

print(
    f"Sequential theoretical FPS        : "
    f"{1000 / seq_median:.2f}"
)

print(
    f"Overlapped throughput             : "
    f"{1000 / overlap_ms_per_frame:.2f} FPS"
)

pd.DataFrame({
    "sequential_ms": sequential_times
}).to_csv(
    "/kaggle/working/stage10b_overlap.csv",
    index=False
)

print(
    "\nSaved: "
    "/kaggle/working/stage10b_overlap.csv"
)