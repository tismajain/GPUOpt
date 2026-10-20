# Extracted from notebook Stage 10C environment cells.

import cv2
import torch
import subprocess
import shutil

print("=" * 80)
print("STAGE 10C — GPU VIDEO DECODE ENVIRONMENT")
print("=" * 80)

print("GPU:", torch.cuda.get_device_name(0))

print("\nFFmpeg:")
print(shutil.which("ffmpeg"))

if shutil.which("ffmpeg"):
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-hwaccels"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)

print("\nNVIDIA utilities:")

for cmd in [
    "nvidia-smi",
    "ffmpeg",
    "nvcc"
]:
    print(f"{cmd:12s}: {shutil.which(cmd)}")

print("\nOpenCV CUDA:")
print("cv2 version:", cv2.__version__)

try:
    print(
        "CUDA devices:",
        cv2.cuda.getCudaEnabledDeviceCount()
    )
except Exception as e:
    print("OpenCV CUDA unavailable:", e)

import importlib.util

print(
    "PyNvVideoCodec installed:",
    importlib.util.find_spec("PyNvVideoCodec") is not None
)

import PyNvVideoCodec as nvc
print("PyNvVideoCodec imported successfully")
print(nvc)

print([
    name for name in dir(nvc)
    if "Decoder" in name or "decode" in name.lower()
])

import inspect
import PyNvVideoCodec as nvc

for obj in [
    nvc.CreateDecoder,
    nvc.CreateSimpleDecoder,
    nvc.SimpleDecoder,
    nvc.DecodedFrame,
]:
    print("\n" + "=" * 80)
    print(obj)
    print("=" * 80)

    try:
        print("Signature:", inspect.signature(obj))
    except Exception as e:
        print("Signature unavailable:", e)

    try:
        print("\nDocstring:")
        print(inspect.getdoc(obj))
    except Exception as e:
        print("Docstring unavailable:", e)

print("DecodedFrame members:")
print([
    x for x in dir(nvc.DecodedFrame)
    if not x.startswith("_")
])

print("SimpleDecoder members:")
print([
    x for x in dir(nvc.SimpleDecoder)
    if not x.startswith("_")
])

import PyNvVideoCodec as nvc

VIDEO_PATH = "/kaggle/input/datasets/tismajain/boxing/GH088416.mp4"

decoder = nvc.SimpleDecoder(
    VIDEO_PATH,
    gpu_id=0,
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.NATIVE
)

print("Decoder created")
print(decoder)

frames = decoder.get_batch_frames(1)

print("Type:", type(frames))
print("Length:", len(frames))

frame0 = frames[0]

print("\nDecodedFrame type:", type(frame0))
print("format:", frame0.format)
print("shape:", frame0.shape)
print("strides:", frame0.strides)
print("dtype:", frame0.dtype)
print("cuda:", frame0.cuda)
print("framesize:", frame0.framesize)

print("\nCUDA array interface:")
print(frame0.__cuda_array_interface__ if hasattr(frame0, "__cuda_array_interface__") else "none")

for i in range(2):
    try:
        ptr = frame0.GetPtrToPlane(i)
        print(f"Plane {i} ptr:", ptr)
    except Exception as e:
        print(f"Plane {i} error:", e)

print("cuda():", frame0.cuda())
print("framesize():", frame0.framesize())