
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

__global__ void fused_yolo_preprocess(
    const uint8_t* __restrict__ src,   // HWC BGR uint8
    float* __restrict__ dst,           // CHW RGB float32
    int src_w,
    int src_h,
    int dst_w,
    int dst_h,
    int resized_h,
    int pad_top
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h)
        return;

    const int plane = dst_w * dst_h;
    const int out_idx = y * dst_w + x;

    // Padding region: YOLO uses 114
    if (y < pad_top || y >= pad_top + resized_h) {
        const float pad = 114.0f / 255.0f;

        dst[out_idx]             = pad;  // R
        dst[plane + out_idx]     = pad;  // G
        dst[2 * plane + out_idx] = pad;  // B

        return;
    }

    // Coordinates within resized image
    int ry = y - pad_top;

    // OpenCV-style half-pixel mapping
    float scale_x = (float)src_w / (float)dst_w;
    float scale_y = (float)src_h / (float)resized_h;

    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (ry + 0.5f) * scale_y - 0.5f;

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);

    float wx = src_x - x0;
    float wy = src_y - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, src_w - 1));
    x1 = max(0, min(x1, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    y1 = max(0, min(y1, src_h - 1));

    const int idx00 = (y0 * src_w + x0) * 3;
    const int idx01 = (y0 * src_w + x1) * 3;
    const int idx10 = (y1 * src_w + x0) * 3;
    const int idx11 = (y1 * src_w + x1) * 3;

    // Source is BGR.
    for (int c = 0; c < 3; ++c) {
        float v00 = (float)src[idx00 + c];
        float v01 = (float)src[idx01 + c];
        float v10 = (float)src[idx10 + c];
        float v11 = (float)src[idx11 + c];

        float top = v00 + wx * (v01 - v00);
        float bot = v10 + wx * (v11 - v10);

        float value = top + wy * (bot - top);

        // BGR -> RGB
        int out_c = 2 - c;

        dst[out_c * plane + out_idx] = value / 255.0f;
    }
}


extern "C"
void launch_fused_yolo_preprocess(
    const uint8_t* src,
    float* dst,
    int src_w,
    int src_h,
    cudaStream_t stream
) {
    const int dst_w = 640;
    const int dst_h = 384;
    const int resized_h = 360;
    const int pad_top = 12;

    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

    fused_yolo_preprocess<<<grid, block, 0, stream>>>(
        src,
        dst,
        src_w,
        src_h,
        dst_w,
        dst_h,
        resized_h,
        pad_top
    );
}
