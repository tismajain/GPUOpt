#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

__global__ void fused_rgbp_yolo(
    const uint8_t* __restrict__ r_plane,
    const uint8_t* __restrict__ g_plane,
    const uint8_t* __restrict__ b_plane,
    float* __restrict__ output,
    int src_w,
    int src_h
) {
    const int out_w = 640;
    const int out_h = 384;

    const int resized_w = 640;
    const int resized_h = 360;

    const int pad_top = 12;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= out_w || y >= out_h)
        return;

    int out_idx = y * out_w + x;
    int plane = out_w * out_h;

    if (y < pad_top || y >= pad_top + resized_h) {
        float p = 114.0f / 255.0f;

        output[0 * plane + out_idx] = p;
        output[1 * plane + out_idx] = p;
        output[2 * plane + out_idx] = p;

        return;
    }

    int ry = y - pad_top;

    float scale_x = (float)src_w / (float)resized_w;
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

    int idx00 = y0 * src_w + x0;
    int idx01 = y0 * src_w + x1;
    int idx10 = y1 * src_w + x0;
    int idx11 = y1 * src_w + x1;

    const uint8_t* planes[3] = {
        r_plane,
        g_plane,
        b_plane
    };

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        const uint8_t* src = planes[c];

        float v00 = (float)src[idx00];
        float v01 = (float)src[idx01];
        float v10 = (float)src[idx10];
        float v11 = (float)src[idx11];

        float top = v00 + wx * (v01 - v00);
        float bottom = v10 + wx * (v11 - v10);

        float value = top + wy * (bottom - top);

        output[c * plane + out_idx] = value / 255.0f;
    }
}

extern "C"
void launch_rgbp_yolo(
    const uint8_t* r_plane,
    const uint8_t* g_plane,
    const uint8_t* b_plane,
    float* output,
    int src_w,
    int src_h,
    cudaStream_t stream
) {
    dim3 block(16, 16);

    dim3 grid(
        (640 + 15) / 16,
        (384 + 15) / 16
    );

    fused_rgbp_yolo<<<grid, block, 0, stream>>>(
        r_plane,
        g_plane,
        b_plane,
        output,
        src_w,
        src_h
    );
}
