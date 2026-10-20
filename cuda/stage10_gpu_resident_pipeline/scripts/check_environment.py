#!/usr/bin/env python3
import importlib
import shutil
import subprocess
import sys

REQUIRED = [
    ("torch", "torch"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("ultralytics", "ultralytics"),
    ("PyNvVideoCodec", "PyNvVideoCodec"),
]

missing = []
for module, package in REQUIRED:
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(package)

if missing:
    print("Missing Python packages:", ", ".join(missing))
    sys.exit(1)

import torch

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime reported by PyTorch:", torch.version.cuda)

for binary in ["nvcc", "ffmpeg", "nvidia-smi"]:
    path = shutil.which(binary)
    print(f"{binary}: {path or 'not found'}")

if shutil.which("ffmpeg"):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print("\nFFmpeg hardware acceleration methods:\n" + proc.stdout.strip())

if not torch.cuda.is_available():
    print("\nERROR: A CUDA-capable NVIDIA GPU is required for Stage 10.")
    sys.exit(1)

print("\nEnvironment check passed.")
