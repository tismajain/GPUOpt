#!/usr/bin/env python3
"""Reproduce the core Stage 9 bottleneck experiment.

This script compares Ultralytics preprocessing with the custom fused CUDA
preprocessor and then compares preprocess+model forward time for the two paths.
The validated geometry is 1920x1080 input -> 640x384 YOLO tensor.
"""

import argparse
import ctypes
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--library", required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return np.median(a), np.percentile(a, 95), np.mean(a)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA-enabled PyTorch is required.")
    torch.cuda.set_device(args.gpu)

    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read {args.video}")
    h, w = first.shape[:2]
    if (w, h) != (1920, 1080):
        raise SystemExit(f"Validated Stage 9 geometry is 1920x1080; got {w}x{h}.")

    model = YOLO(args.model)
    model.predict(source=first, imgsz=640, device=args.gpu, verbose=False)

    def ultra_preprocess(frame):
        transformed = model.predictor.pre_transform([frame])
        return model.predictor.preprocess(transformed)

    lib = ctypes.CDLL(str(Path(args.library).resolve()))
    lib.launch_fused_yolo_preprocess.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.launch_fused_yolo_preprocess.restype = None

    def fused_from_gpu(src_gpu):
        out = torch.empty((1, 3, 384, 640), dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream()
        lib.launch_fused_yolo_preprocess(
            ctypes.c_void_p(src_gpu.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            1920, 1080,
            ctypes.c_void_p(stream.cuda_stream),
        )
        return out

    # Tensor correctness on the same frame.
    x_ref = ultra_preprocess(first)
    src_gpu = torch.from_numpy(np.ascontiguousarray(first)).to("cuda")
    x_custom = fused_from_gpu(src_gpu)
    torch.cuda.synchronize()
    diff = (x_ref - x_custom).abs()
    print(f"Tensor max abs diff : {diff.max().item():.8e}")
    print(f"Tensor mean abs diff: {diff.mean().item():.8e}")

    cap = cv2.VideoCapture(args.video)
    for _ in range(args.warmup):
        ok, frame = cap.read()
        if not ok:
            break
        _ = ultra_preprocess(frame)
        gpu = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda")
        _ = fused_from_gpu(gpu)
        with torch.no_grad():
            _ = model.model(x_ref)
    torch.cuda.synchronize()

    ultra_ms, h2d_ms, kernel_ms, custom_pre_ms = [], [], [], []
    base_e2e_ms, custom_e2e_ms = [], []

    for _ in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xb = ultra_preprocess(frame)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ultra_ms.append((t1 - t0) * 1000)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        h2d_ms.append((t1 - t0) * 1000)

        # Kernel-only timing: input is already resident on GPU.
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        xc = fused_from_gpu(gpu)
        stop.record()
        stop.synchronize()
        kernel_ms.append(start.elapsed_time(stop))

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu2 = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda")
        xc2 = fused_from_gpu(gpu2)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        custom_pre_ms.append((t1 - t0) * 1000)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xb2 = ultra_preprocess(frame)
        with torch.no_grad():
            _ = model.model(xb2)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        base_e2e_ms.append((t1 - t0) * 1000)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu3 = torch.from_numpy(np.ascontiguousarray(frame)).to("cuda")
        xc3 = fused_from_gpu(gpu3)
        with torch.no_grad():
            _ = model.model(xc3)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        custom_e2e_ms.append((t1 - t0) * 1000)

    cap.release()

    def report(name, xs):
        med, p95, mean = stats(xs)
        print(f"{name:28s} median={med:8.4f} ms  p95={p95:8.4f}  mean={mean:8.4f}")
        return med

    print("\nSTAGE 9 REPRODUCTION")
    u = report("Ultralytics preprocess", ultra_ms)
    report("HtoD only", h2d_ms)
    k = report("Fused kernel only", kernel_ms)
    c = report("HtoD + fused", custom_pre_ms)
    b = report("Baseline pre+infer", base_e2e_ms)
    ce = report("Custom pre+infer", custom_e2e_ms)
    print(f"Kernel-level speedup : {u / k:.2f}x")
    print(f"Full preprocess      : {u / c:.3f}x")
    print(f"End-to-end speedup   : {b / ce:.3f}x")
    print("\nExpected T4 behavior from the original run: ~25.65x kernel-level speedup, ~1.01x full-preprocess speedup, and ~1.002x preprocess+inference speedup.")
    print("Exact timings vary with software versions, clocks, and system load.")


if __name__ == "__main__":
    main()
