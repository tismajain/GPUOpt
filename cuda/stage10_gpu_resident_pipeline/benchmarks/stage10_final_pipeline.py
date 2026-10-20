# Extracted from notebook final Stage 10 comparison (cells 149-153).
# Expects VIDEO_PATH, model, ultra_preprocess, nvc, rgbplib and ctypes to be initialized.

import cv2
import time
import numpy as np
import torch

NUM_FRAMES = 300
WARMUP = 30

# ============================================================
# A. OpenCV + Ultralytics preprocessing + YOLO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

baseline_decode = []
baseline_pre = []
baseline_infer = []
baseline_total = []

# Warmup
for _ in range(WARMUP):
    ok, f = cap.read()
    if not ok:
        break

    x = ultra_preprocess(f)

    with torch.no_grad():
        _ = model.model(x)

torch.cuda.synchronize()

print("Baseline warmup complete.")

for i in range(NUM_FRAMES):

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    ok, f = cap.read()
    if not ok:
        break

    t1 = time.perf_counter()

    x = ultra_preprocess(f)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    with torch.no_grad():
        _ = model.model(x)

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    baseline_decode.append((t1 - t0) * 1000)
    baseline_pre.append((t2 - t1) * 1000)
    baseline_infer.append((t3 - t2) * 1000)
    baseline_total.append((t3 - t0) * 1000)

cap.release()

# ============================================================
# B. Sequential NVDEC + fused RGBP + YOLO
# ============================================================

nvdec_seq = nvc.SimpleDecoder(
    VIDEO_PATH,
    gpu_id=0,
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.RGBP
)

nvdec_decode = []
nvdec_pre = []
nvdec_infer = []
nvdec_total = []

# Warmup sequentially
for _ in range(WARMUP):

    frames_out = nvdec_seq.get_batch_frames(1)

    if not frames_out:
        break

    frame = frames_out[0]
    views = frame.cuda()

    r_ptr = views[0].__cuda_array_interface__["data"][0]
    g_ptr = views[1].__cuda_array_interface__["data"][0]
    b_ptr = views[2].__cuda_array_interface__["data"][0]

    x = torch.empty(
        (1, 3, 384, 640),
        dtype=torch.float32,
        device="cuda"
    )

    stream = torch.cuda.current_stream()

    rgbplib.launch_rgbp_yolo(
        ctypes.c_void_p(r_ptr),
        ctypes.c_void_p(g_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(x.data_ptr()),
        1920,
        1080,
        ctypes.c_void_p(stream.cuda_stream)
    )

    with torch.no_grad():
        _ = model.model(x)

torch.cuda.synchronize()

print("NVDEC warmup complete.")

for i in range(NUM_FRAMES):

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    frames_out = nvdec_seq.get_batch_frames(1)

    if not frames_out:
        break

    frame = frames_out[0]

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    views = frame.cuda()

    r_ptr = views[0].__cuda_array_interface__["data"][0]
    g_ptr = views[1].__cuda_array_interface__["data"][0]
    b_ptr = views[2].__cuda_array_interface__["data"][0]

    x = torch.empty(
        (1, 3, 384, 640),
        dtype=torch.float32,
        device="cuda"
    )

    stream = torch.cuda.current_stream()

    rgbplib.launch_rgbp_yolo(
        ctypes.c_void_p(r_ptr),
        ctypes.c_void_p(g_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(x.data_ptr()),
        1920,
        1080,
        ctypes.c_void_p(stream.cuda_stream)
    )

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    with torch.no_grad():
        _ = model.model(x)

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    nvdec_decode.append((t1 - t0) * 1000)
    nvdec_pre.append((t2 - t1) * 1000)
    nvdec_infer.append((t3 - t2) * 1000)
    nvdec_total.append((t3 - t0) * 1000)

def summary(name, arr):
    a = np.asarray(arr)

    print(
        f"{name:28s}"
        f"median={np.median(a):8.4f} ms | "
        f"p95={np.percentile(a,95):8.4f} ms | "
        f"mean={np.mean(a):8.4f} ms"
    )


print("\n" + "=" * 100)
print("FINAL STAGE 10 — APPLES-TO-APPLES PIPELINE COMPARISON")
print("=" * 100)

print("\nOPENCV / CPU PATH")
summary("Decode", baseline_decode)
summary("Preprocess", baseline_pre)
summary("Inference", baseline_infer)
summary("TOTAL", baseline_total)

print("\nNVDEC / GPU-RESIDENT PATH")
summary("Decode", nvdec_decode)
summary("Fused preprocess", nvdec_pre)
summary("Inference", nvdec_infer)
summary("TOTAL", nvdec_total)

base_med = np.median(baseline_total)
nvdec_med = np.median(nvdec_total)

print("\n" + "-" * 100)

print(f"Baseline median total : {base_med:.4f} ms")
print(f"NVDEC median total    : {nvdec_med:.4f} ms")

print(
    f"End-to-end speedup    : "
    f"{base_med / nvdec_med:.3f}x"
)

print(
    f"Baseline throughput   : "
    f"{1000 / base_med:.2f} FPS"
)

print(
    f"NVDEC throughput      : "
    f"{1000 / nvdec_med:.2f} FPS"
)

print(
    f"Latency reduction     : "
    f"{100 * (1 - nvdec_med/base_med):.2f}%"
)

def summary(name, arr):
    a = np.asarray(arr)

    print(
        f"{name:28s}"
        f"median={np.median(a):8.4f} ms | "
        f"p95={np.percentile(a,95):8.4f} ms | "
        f"mean={np.mean(a):8.4f} ms"
    )


print("\n" + "=" * 100)
print("FINAL STAGE 10 — APPLES-TO-APPLES PIPELINE COMPARISON")
print("=" * 100)

print("\nOPENCV / CPU PATH")
summary("Decode", baseline_decode)
summary("Preprocess", baseline_pre)
summary("Inference", baseline_infer)
summary("TOTAL", baseline_total)

print("\nNVDEC / GPU-RESIDENT PATH")
summary("Decode", nvdec_decode)
summary("Fused preprocess", nvdec_pre)
summary("Inference", nvdec_infer)
summary("TOTAL", nvdec_total)

base_med = np.median(baseline_total)
nvdec_med = np.median(nvdec_total)

print("\n" + "-" * 100)

print(f"Baseline median total : {base_med:.4f} ms")
print(f"NVDEC median total    : {nvdec_med:.4f} ms")

print(
    f"End-to-end speedup    : "
    f"{base_med / nvdec_med:.3f}x"
)

print(
    f"Baseline throughput   : "
    f"{1000 / base_med:.2f} FPS"
)

print(
    f"NVDEC throughput      : "
    f"{1000 / nvdec_med:.2f} FPS"
)

print(
    f"Latency reduction     : "
    f"{100 * (1 - nvdec_med/base_med):.2f}%"
)

pd.DataFrame({
    "baseline_decode_ms": baseline_decode,
    "baseline_pre_ms": baseline_pre,
    "baseline_infer_ms": baseline_infer,
    "baseline_total_ms": baseline_total,

    "nvdec_decode_ms": nvdec_decode,
    "nvdec_pre_ms": nvdec_pre,
    "nvdec_infer_ms": nvdec_infer,
    "nvdec_total_ms": nvdec_total,
}).to_csv(
    "/kaggle/working/stage10_final_pipeline.csv",
    index=False
)