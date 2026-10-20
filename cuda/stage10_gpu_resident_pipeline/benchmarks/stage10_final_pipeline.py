#!/usr/bin/env python3
"""Reproduce the final Stage 10 apples-to-apples benchmark.

Compares:
  A) OpenCV CPU decode + Ultralytics preprocessing + YOLOv8n
  B) NVDEC GPU-resident RGBP decode + fused CUDA preprocessing + YOLOv8n

Example:
  python stage10_final_pipeline.py --video /path/to/video.mp4
"""

from __future__ import annotations

import argparse
import ctypes
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import PyNvVideoCodec as nvc
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to an H.264 video file")
    p.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path/name")
    p.add_argument("--frames", type=int, default=300, help="Measured frames")
    p.add_argument("--warmup", type=int, default=30, help="Warmup frames")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument(
        "--rgbp-lib",
        default=None,
        help="Path to librgbp_yolo.so (defaults to ../build/librgbp_yolo.so)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Optional CSV path for raw per-frame timings",
    )
    return p.parse_args()


def load_rgbp_library(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_cuda.sh before this benchmark."
        )
    lib = ctypes.CDLL(str(path))
    lib.launch_rgbp_yolo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


def stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    return float(np.median(a)), float(np.percentile(a, 95)), float(np.mean(a))


def print_stats(name, arr):
    med, p95, mean = stats(arr)
    print(f"{name:28s} median={med:8.4f} ms | p95={p95:8.4f} ms | mean={mean:8.4f} ms")


def main() -> None:
    args = parse_args()
    video = Path(args.video).expanduser().resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-capable NVIDIA GPU is required.")

    torch.cuda.set_device(args.gpu)
    here = Path(__file__).resolve().parent
    stage_dir = here.parent
    lib_path = (
        Path(args.rgbp_lib).expanduser().resolve()
        if args.rgbp_lib
        else stage_dir / "build" / "librgbp_yolo.so"
    )
    rgbplib = load_rgbp_library(lib_path)

    model = YOLO(args.model)
    model.to(f"cuda:{args.gpu}")

    # Initialize Ultralytics predictor so we can reuse its exact preprocessing path.
    cap0 = cv2.VideoCapture(str(video))
    ok, init_frame = cap0.read()
    cap0.release()
    if not ok:
        raise RuntimeError("Could not decode the first video frame with OpenCV.")
    _ = model.predict(source=init_frame, imgsz=640, device=args.gpu, verbose=False)

    def ultra_preprocess(frame):
        transformed = model.predictor.pre_transform([frame])
        return model.predictor.preprocess(transformed)

    # Inspect source geometry. The validated fused kernel expects 1920x1080 input
    # and produces the same 640x384 rectangular YOLO tensor used in the experiment.
    src_h, src_w = init_frame.shape[:2]
    if (src_w, src_h) != (1920, 1080):
        raise ValueError(
            f"Validated Stage 10 kernel expects 1920x1080 input; got {src_w}x{src_h}. "
            "Use the original boxing workload or adapt the CUDA kernel geometry."
        )

    baseline_decode, baseline_pre, baseline_infer, baseline_total = [], [], [], []

    cap = cv2.VideoCapture(str(video))
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            break
        x = ultra_preprocess(frame)
        with torch.no_grad():
            _ = model.model(x)
    torch.cuda.synchronize()

    for _ in range(args.frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        t1 = time.perf_counter()
        x = ultra_preprocess(frame)
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

    nvdec_decode, nvdec_pre, nvdec_infer, nvdec_total = [], [], [], []

    decoder = nvc.SimpleDecoder(
        str(video),
        gpu_id=args.gpu,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGBP,
    )

    def run_rgbp(frame):
        views = frame.cuda()
        r_ptr = views[0].__cuda_array_interface__["data"][0]
        g_ptr = views[1].__cuda_array_interface__["data"][0]
        b_ptr = views[2].__cuda_array_interface__["data"][0]
        x = torch.empty((1, 3, 384, 640), dtype=torch.float32, device=f"cuda:{args.gpu}")
        stream = torch.cuda.current_stream()
        rgbplib.launch_rgbp_yolo(
            ctypes.c_void_p(r_ptr),
            ctypes.c_void_p(g_ptr),
            ctypes.c_void_p(b_ptr),
            ctypes.c_void_p(x.data_ptr()),
            src_w,
            src_h,
            ctypes.c_void_p(stream.cuda_stream),
        )
        return x

    for _ in range(args.warmup):
        batch = decoder.get_batch_frames(1)
        if not batch:
            break
        x = run_rgbp(batch[0])
        with torch.no_grad():
            _ = model.model(x)
    torch.cuda.synchronize()

    for _ in range(args.frames):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = decoder.get_batch_frames(1)
        if not batch:
            break
        frame = batch[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        x = run_rgbp(frame)
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

    if not baseline_total or not nvdec_total:
        raise RuntimeError("No benchmark frames were collected.")

    print("\n" + "=" * 100)
    print("FINAL STAGE 10 — APPLES-TO-APPLES PIPELINE COMPARISON")
    print("=" * 100)
    print("\nOPENCV / CPU PATH")
    print_stats("Decode", baseline_decode)
    print_stats("Preprocess", baseline_pre)
    print_stats("Inference", baseline_infer)
    print_stats("TOTAL", baseline_total)
    print("\nNVDEC / GPU-RESIDENT PATH")
    print_stats("Decode", nvdec_decode)
    print_stats("Fused preprocess", nvdec_pre)
    print_stats("Inference", nvdec_infer)
    print_stats("TOTAL", nvdec_total)

    base_med = float(np.median(baseline_total))
    nvdec_med = float(np.median(nvdec_total))
    print("\n" + "-" * 100)
    print(f"Baseline median total : {base_med:.4f} ms")
    print(f"NVDEC median total    : {nvdec_med:.4f} ms")
    print(f"End-to-end speedup    : {base_med / nvdec_med:.3f}x")
    print(f"Baseline throughput   : {1000 / base_med:.2f} FPS")
    print(f"NVDEC throughput      : {1000 / nvdec_med:.2f} FPS")
    print(f"Latency reduction     : {100 * (1 - nvdec_med / base_med):.2f}%")

    if args.output:
        n = min(len(baseline_total), len(nvdec_total))
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "baseline_decode_ms": baseline_decode[:n],
            "baseline_pre_ms": baseline_pre[:n],
            "baseline_infer_ms": baseline_infer[:n],
            "baseline_total_ms": baseline_total[:n],
            "nvdec_decode_ms": nvdec_decode[:n],
            "nvdec_pre_ms": nvdec_pre[:n],
            "nvdec_infer_ms": nvdec_infer[:n],
            "nvdec_total_ms": nvdec_total[:n],
        }).to_csv(out, index=False)
        print(f"Saved raw timings to {out}")


if __name__ == "__main__":
    main()
