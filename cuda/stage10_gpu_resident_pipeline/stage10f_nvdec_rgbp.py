# Extracted from notebook Stage 10F (cells 145-148).
# Expects VIDEO_PATH, model, nvc and rgbplib to be initialized as in the notebook.

import time
import numpy as np
import pandas as pd
import torch
import ctypes

NUM_FRAMES = 300
WARMUP = 30

# Fresh decoder for fair sequential access
rgb_bench_decoder = nvc.SimpleDecoder(
    VIDEO_PATH,
    gpu_id=0,
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.RGBP
)

# ------------------------------------------------------------
# Warmup
# ------------------------------------------------------------

for i in range(WARMUP):
    frame = rgb_bench_decoder.get_batch_frames_by_index([i])[0]

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

print("Warmup complete.")

nvdec_total_times = []
nvdec_decode_times = []
nvdec_pre_times = []
nvdec_infer_times = []

for i in range(WARMUP, WARMUP + NUM_FRAMES):

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    t0 = time.perf_counter()

    frame = rgb_bench_decoder.get_batch_frames_by_index([i])[0]

    torch.cuda.synchronize()

    t1 = time.perf_counter()

    # --------------------------------------------------------
    # Fused RGBP preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    with torch.no_grad():
        _ = model.model(x)

    torch.cuda.synchronize()

    t3 = time.perf_counter()

    nvdec_decode_times.append((t1 - t0) * 1000)
    nvdec_pre_times.append((t2 - t1) * 1000)
    nvdec_infer_times.append((t3 - t2) * 1000)
    nvdec_total_times.append((t3 - t0) * 1000)

    if (i - WARMUP + 1) % 50 == 0:
        print(f"Completed {i - WARMUP + 1}/{NUM_FRAMES}")

def report(name, arr):
    arr = np.asarray(arr)

    print(
        f"{name:28s}"
        f"median={np.median(arr):8.4f} ms | "
        f"p95={np.percentile(arr,95):8.4f} ms | "
        f"mean={np.mean(arr):8.4f} ms"
    )


print("\n" + "=" * 95)
print("STAGE 10F — NVDEC RGBP END-TO-END")
print("=" * 95)

report("NVDEC decode", nvdec_decode_times)
report("Fused RGBP preprocess", nvdec_pre_times)
report("YOLO inference", nvdec_infer_times)
report("Total", nvdec_total_times)

total_med = np.median(nvdec_total_times)

print("\n" + "-" * 95)

print(f"Median total         : {total_med:.4f} ms")
print(f"Theoretical FPS      : {1000.0 / total_med:.2f}")

pd.DataFrame({
    "decode_ms": nvdec_decode_times,
    "preprocess_ms": nvdec_pre_times,
    "inference_ms": nvdec_infer_times,
    "total_ms": nvdec_total_times,
}).to_csv(
    "/kaggle/working/stage10f_nvdec_rgbp.csv",
    index=False
)

print("\nSaved: /kaggle/working/stage10f_nvdec_rgbp.csv")