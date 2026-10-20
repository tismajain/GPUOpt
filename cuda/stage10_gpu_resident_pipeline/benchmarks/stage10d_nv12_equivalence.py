# Extracted from notebook Stage 10D (cell 137).
# Expects VIDEO_PATH, model, nv12lib to be initialized as in the notebook.
# PyNvVideoCodec (nvc) and CuPy (cp) are imported below for standalone clarity.

import PyNvVideoCodec as nvc
import cupy as cp

import ctypes
import numpy as np
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

rgb_decoder = nvc.SimpleDecoder(
    VIDEO_PATH,
    gpu_id=0,
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.RGBP
)

nv12_decoder = nvc.SimpleDecoder(
    VIDEO_PATH,
    gpu_id=0,
    use_device_memory=True,
    output_color_type=nvc.OutputColorType.NATIVE
)

class CudaArrayProxy:
    def __init__(self, obj):
        cai = dict(obj.__cuda_array_interface__)
        cai["shape"] = tuple(cai["shape"])
        if cai.get("strides") is not None:
            cai["strides"] = tuple(cai["strides"])
        self.__cuda_array_interface__ = cai


def rgbp_to_yolo_tensor(rgb_frame):
    views = rgb_frame.cuda()

    r_cp = cp.asarray(CudaArrayProxy(views[0]))
    g_cp = cp.asarray(CudaArrayProxy(views[1]))
    b_cp = cp.asarray(CudaArrayProxy(views[2]))

    r = torch.utils.dlpack.from_dlpack(r_cp)
    g = torch.utils.dlpack.from_dlpack(g_cp)
    b = torch.utils.dlpack.from_dlpack(b_cp)

    x = torch.stack([r, g, b], dim=0).unsqueeze(0)

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


def nv12_to_yolo_tensor(nv12_frame):
    y_ptr = nv12_frame.GetPtrToPlane(0)
    uv_ptr = nv12_frame.GetPtrToPlane(1)

    out = torch.empty(
        (1, 3, 384, 640),
        dtype=torch.float32,
        device="cuda"
    )

    stream = torch.cuda.current_stream()

    nv12lib.launch_nv12_yolo(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(out.data_ptr()),
        0,  # BT.601
        ctypes.c_void_p(stream.cuda_stream)
    )

    return out


for idx in range(NUM_CHECK):

    # -------------------------------------------------------
    # Decode SAME frame index through both GPU paths
    # -------------------------------------------------------

    rgb_frame = rgb_decoder.get_batch_frames_by_index([idx])[0]
    nv12_frame = nv12_decoder.get_batch_frames_by_index([idx])[0]

    rgb_x = rgbp_to_yolo_tensor(rgb_frame)
    nv12_x = nv12_to_yolo_tensor(nv12_frame)

    torch.cuda.synchronize()

    # -------------------------------------------------------
    # YOLO inference
    # -------------------------------------------------------

    res_rgb = model.predict(
        source=rgb_x,
        conf=CONF,
        iou=IOU,
        device=0,
        verbose=False
    )[0]

    res_nv12 = model.predict(
        source=nv12_x,
        conf=CONF,
        iou=IOU,
        device=0,
        verbose=False
    )[0]

    boxes_r = res_rgb.boxes
    boxes_n = res_nv12.boxes

    nr = 0 if boxes_r is None else len(boxes_r)
    nn = 0 if boxes_n is None else len(boxes_n)

    total_frames += 1

    if nr == nn:
        count_matches += 1

    if nr > 0 and nr == nn:

        conf_r = boxes_r.conf.detach().cpu()
        conf_n = boxes_n.conf.detach().cpu()

        cls_r = boxes_r.cls.detach().cpu()
        cls_n = boxes_n.cls.detach().cpu()

        xyxy_r = boxes_r.xyxy.detach().cpu()
        xyxy_n = boxes_n.xyxy.detach().cpu()

        # Sort by confidence for stable alignment
        order_r = torch.argsort(conf_r, descending=True)
        order_n = torch.argsort(conf_n, descending=True)

        conf_r = conf_r[order_r]
        conf_n = conf_n[order_n]

        cls_r = cls_r[order_r]
        cls_n = cls_n[order_n]

        xyxy_r = xyxy_r[order_r]
        xyxy_n = xyxy_n[order_n]

        if torch.equal(cls_r, cls_n):
            class_matches += 1

        conf_diff = torch.max(
            torch.abs(conf_r - conf_n)
        ).item()

        box_abs = torch.abs(
            xyxy_r - xyxy_n
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
        print(
            f"Checked {idx + 1}/{NUM_CHECK}"
        )


print("\n" + "=" * 90)
print("STAGE 10D — GPU-RESIDENT YOLO EQUIVALENCE")
print("=" * 90)

print(
    f"Frames checked                  : "
    f"{total_frames}"
)

print(
    f"Matching detection counts       : "
    f"{count_matches}/{total_frames}"
)

print(
    f"Detection-count match rate      : "
    f"{100.0 * count_matches / total_frames:.2f}%"
)

print(
    f"Matching class sequences        : "
    f"{class_matches}/{total_frames}"
)

print(
    f"\nMax confidence difference        : "
    f"{max_conf_diff:.8f}"
)

print(
    f"Max box-coordinate difference   : "
    f"{max_box_diff_px:.6f} px"
)

if mean_box_diffs:
    print(
        f"Mean box-coordinate difference  : "
        f"{np.mean(mean_box_diffs):.6f} px"
    )