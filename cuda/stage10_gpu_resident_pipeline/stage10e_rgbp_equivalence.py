# Extracted from notebook Stage 10E.
# Expects VIDEO_PATH and model to be initialized as in the notebook.
# Loads the compiled RGBP CUDA library from /kaggle/working/librgbp_yolo.so.

import PyNvVideoCodec as nvc

import PyNvVideoCodec as nvc

rgb_decoder = nvc.SimpleDecoder(
    VIDEO_PATH,
    gpu_id=0,
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.RGBP
)

rgb_frame0 = rgb_decoder.get_batch_frames_by_index([0])[0]

print("format     :", rgb_frame0.format)
print("shape      :", rgb_frame0.shape)
print("strides    :", rgb_frame0.strides)
print("dtype      :", rgb_frame0.dtype)
print("framesize  :", rgb_frame0.framesize())
print("cuda views :", rgb_frame0.cuda())


views = rgb_frame0.cuda()

print("num views:", len(views))

for i, v in enumerate(views):
    print(f"\nView {i}")
    print("type :", type(v))
    print("shape:", getattr(v, "shape", None))
    print("cuda array interface:")
    print(
        getattr(
            v,
            "__cuda_array_interface__",
            None
        )
    )

import cupy as cp

class CudaArrayProxy:
    def __init__(self, obj):
        cai = dict(obj.__cuda_array_interface__)

        # CuPy expects tuples, not lists
        cai["shape"] = tuple(cai["shape"])

        if cai.get("strides") is not None:
            cai["strides"] = tuple(cai["strides"])

        self.__cuda_array_interface__ = cai


rgb_views = rgb_frame0.cuda()

r_cp = cp.asarray(CudaArrayProxy(rgb_views[0]))
g_cp = cp.asarray(CudaArrayProxy(rgb_views[1]))
b_cp = cp.asarray(CudaArrayProxy(rgb_views[2]))

print("CuPy shapes:")
print(r_cp.shape, g_cp.shape, b_cp.shape)

print("dtypes:")
print(r_cp.dtype, g_cp.dtype, b_cp.dtype)

import torch
import torch.nn.functional as F

r_t = torch.utils.dlpack.from_dlpack(r_cp)
g_t = torch.utils.dlpack.from_dlpack(g_cp)
b_t = torch.utils.dlpack.from_dlpack(b_cp)

rgb_full = torch.stack(
    [r_t, g_t, b_t],
    dim=0
).unsqueeze(0)

print("RGB full shape :", rgb_full.shape)
print("dtype          :", rgb_full.dtype)
print("device         :", rgb_full.device)
print("range          :", int(rgb_full.min()), int(rgb_full.max()))

rgb_ref = rgb_full.float() / 255.0

rgb_ref = F.interpolate(
    rgb_ref,
    size=(360, 640),
    mode="bilinear",
    align_corners=False
)

rgb_ref = F.pad(
    rgb_ref,
    (0, 0, 12, 12),
    mode="constant",
    value=114.0 / 255.0
)

print("Reference shape:", rgb_ref.shape)

import ctypes

rgbplib = ctypes.CDLL(
    "/kaggle/working/librgbp_yolo.so"
)

rgbplib.launch_rgbp_yolo.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]

import cupy as cp
import torch
import torch.nn.functional as F
import ctypes

class CudaArrayProxy:
    def __init__(self, obj):
        cai = dict(obj.__cuda_array_interface__)
        cai["shape"] = tuple(cai["shape"])
        if cai.get("strides") is not None:
            cai["strides"] = tuple(cai["strides"])
        self.__cuda_array_interface__ = cai


# ============================================================
# 1. Decode ONE fresh RGBP frame
# ============================================================

fresh_rgb = rgb_decoder.get_batch_frames_by_index([0])[0]

views = fresh_rgb.cuda()

print("format :", fresh_rgb.format)
print("shape  :", fresh_rgb.shape)

# Keep fresh_rgb alive until ALL comparisons are finished.


# ============================================================
# 2. Build reference from EXACT SAME GPU planes
# ============================================================

r_cp = cp.asarray(CudaArrayProxy(views[0]))
g_cp = cp.asarray(CudaArrayProxy(views[1]))
b_cp = cp.asarray(CudaArrayProxy(views[2]))

r_t = torch.utils.dlpack.from_dlpack(r_cp)
g_t = torch.utils.dlpack.from_dlpack(g_cp)
b_t = torch.utils.dlpack.from_dlpack(b_cp)

rgb_full = torch.stack(
    [r_t, g_t, b_t],
    dim=0
).unsqueeze(0)

# This creates independent float GPU storage,
# so decoder-buffer reuse cannot affect the reference.
reference = rgb_full.float() / 255.0

reference = F.interpolate(
    reference,
    size=(360, 640),
    mode="bilinear",
    align_corners=False
)

reference = F.pad(
    reference,
    (0, 0, 12, 12),
    mode="constant",
    value=114.0 / 255.0
)

# Force completion before doing anything else.
torch.cuda.synchronize()


# ============================================================
# 3. Get raw pointers from the SAME fresh frame
# ============================================================

r_ptr = views[0].__cuda_array_interface__["data"][0]
g_ptr = views[1].__cuda_array_interface__["data"][0]
b_ptr = views[2].__cuda_array_interface__["data"][0]


# ============================================================
# 4. Run fused CUDA kernel
# ============================================================

custom = torch.empty(
    (1, 3, 384, 640),
    dtype=torch.float32,
    device="cuda"
)

stream = torch.cuda.current_stream()

rgbplib.launch_rgbp_yolo(
    ctypes.c_void_p(r_ptr),
    ctypes.c_void_p(g_ptr),
    ctypes.c_void_p(b_ptr),
    ctypes.c_void_p(custom.data_ptr()),
    1920,
    1080,
    ctypes.c_void_p(stream.cuda_stream)
)

torch.cuda.synchronize()


# ============================================================
# 5. Compare
# ============================================================

diff = torch.abs(custom - reference)

print("\n" + "=" * 80)
print("FRESH RGBP REFERENCE vs FUSED CUDA")
print("=" * 80)

print("Reference shape :", tuple(reference.shape))
print("Custom shape    :", tuple(custom.shape))

print("Mean abs diff   :", float(diff.mean()))
print("Max abs diff    :", float(diff.max()))

for c, name in enumerate(["R", "G", "B"]):
    d = diff[0, c]

    print(
        f"{name}: "
        f"mean={float(d.mean()):.8f} "
        f"max={float(d.max()):.8f}"
    )


# Compare simple image statistics before resizing

for c, name in enumerate(["R", "G", "B"]):
    print(
        name,
        "min=", int(rgb_full[0, c].min()),
        "max=", int(rgb_full[0, c].max()),
        "mean=", float(rgb_full[0, c].float().mean())
    )

import ctypes
import numpy as np
import cupy as cp
import torch
import torch.nn.functional as F

NUM_CHECK = 100
CONF = 0.25
IOU = 0.7

count_matches = 0
class_matches = 0
total_frames = 0

max_conf_diff = 0.0
max_box_diff_px = 0.0
mean_box_diffs = []

class CudaArrayProxy:
    def __init__(self, obj):
        cai = dict(obj.__cuda_array_interface__)
        cai["shape"] = tuple(cai["shape"])
        if cai.get("strides") is not None:
            cai["strides"] = tuple(cai["strides"])
        self.__cuda_array_interface__ = cai


def rgbp_reference_tensor(rgb_frame):
    views = rgb_frame.cuda()

    r_cp = cp.asarray(CudaArrayProxy(views[0]))
    g_cp = cp.asarray(CudaArrayProxy(views[1]))
    b_cp = cp.asarray(CudaArrayProxy(views[2]))

    r_t = torch.utils.dlpack.from_dlpack(r_cp)
    g_t = torch.utils.dlpack.from_dlpack(g_cp)
    b_t = torch.utils.dlpack.from_dlpack(b_cp)

    x = torch.stack([r_t, g_t, b_t], dim=0).unsqueeze(0)

    x = x.float() / 255.0

    x = F.interpolate(
        x,
        size=(360, 640),
        mode="bilinear",
        align_corners=False
    )

    x = F.pad(
        x,
        (0, 0, 12, 12),
        mode="constant",
        value=114.0 / 255.0
    )

    return x


def rgbp_fused_tensor(rgb_frame):
    views = rgb_frame.cuda()

    r_ptr = views[0].__cuda_array_interface__["data"][0]
    g_ptr = views[1].__cuda_array_interface__["data"][0]
    b_ptr = views[2].__cuda_array_interface__["data"][0]

    out = torch.empty(
        (1, 3, 384, 640),
        dtype=torch.float32,
        device="cuda"
    )

    stream = torch.cuda.current_stream()

    rgbplib.launch_rgbp_yolo(
        ctypes.c_void_p(r_ptr),
        ctypes.c_void_p(g_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(out.data_ptr()),
        1920,
        1080,
        ctypes.c_void_p(stream.cuda_stream)
    )

    return out


for idx in range(NUM_CHECK):

    # One fresh frame, shared by both paths.
    # Important: keep rgb_frame alive until both predictions finish.
    rgb_frame = rgb_decoder.get_batch_frames_by_index([idx])[0]

    ref_x = rgbp_reference_tensor(rgb_frame)
    fused_x = rgbp_fused_tensor(rgb_frame)

    torch.cuda.synchronize()

    res_ref = model.predict(
        source=ref_x,
        conf=CONF,
        iou=IOU,
        device=0,
        verbose=False
    )[0]

    res_fused = model.predict(
        source=fused_x,
        conf=CONF,
        iou=IOU,
        device=0,
        verbose=False
    )[0]

    boxes_r = res_ref.boxes
    boxes_f = res_fused.boxes

    nr = 0 if boxes_r is None else len(boxes_r)
    nf = 0 if boxes_f is None else len(boxes_f)

    total_frames += 1

    if nr == nf:
        count_matches += 1

    if nr > 0 and nr == nf:

        conf_r = boxes_r.conf.detach().cpu()
        conf_f = boxes_f.conf.detach().cpu()

        cls_r = boxes_r.cls.detach().cpu()
        cls_f = boxes_f.cls.detach().cpu()

        xyxy_r = boxes_r.xyxy.detach().cpu()
        xyxy_f = boxes_f.xyxy.detach().cpu()

        # Same detections should have virtually identical confidences,
        # so confidence sort is fine here.
        order_r = torch.argsort(conf_r, descending=True)
        order_f = torch.argsort(conf_f, descending=True)

        conf_r = conf_r[order_r]
        conf_f = conf_f[order_f]

        cls_r = cls_r[order_r]
        cls_f = cls_f[order_f]

        xyxy_r = xyxy_r[order_r]
        xyxy_f = xyxy_f[order_f]

        if torch.equal(cls_r, cls_f):
            class_matches += 1

        conf_diff = torch.max(
            torch.abs(conf_r - conf_f)
        ).item()

        box_abs = torch.abs(
            xyxy_r - xyxy_f
        )

        max_conf_diff = max(
            max_conf_diff,
            conf_diff
        )

        max_box_diff_px = max(
            max_box_diff_px,
            torch.max(box_abs).item()
        )

        mean_box_diffs.append(
            torch.mean(box_abs).item()
        )

    if (idx + 1) % 20 == 0:
        print(f"Checked {idx + 1}/{NUM_CHECK}")


print("\n" + "=" * 90)
print("STAGE 10E — RGBP GPU-RESIDENT YOLO EQUIVALENCE")
print("=" * 90)

print(f"Frames checked                  : {total_frames}")
print(f"Matching detection counts       : {count_matches}/{total_frames}")
print(f"Detection-count match rate      : {100.0 * count_matches / total_frames:.2f}%")
print(f"Matching class sequences        : {class_matches}/{total_frames}")

print(f"\nMax confidence difference        : {max_conf_diff:.8f}")
print(f"Max box-coordinate difference   : {max_box_diff_px:.6f} px")

if mean_box_diffs:
    print(
        f"Mean box-coordinate difference  : "
        f"{np.mean(mean_box_diffs):.6f} px"
    )