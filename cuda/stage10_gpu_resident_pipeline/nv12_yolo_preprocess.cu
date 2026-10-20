
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// ============================================================
// Limited-range YUV -> RGB
//
// matrix_type:
// 0 = BT.601 limited range
// 1 = BT.709 limited range
// ============================================================

__device__ __forceinline__
void yuv_to_rgb(
    float Y,
    float U,
    float V,
    float &R,
    float &G,
    float &B,
    int matrix_type
) {
    float y = 1.16438356f * (Y - 16.0f);
    float u = U - 128.0f;
    float v = V - 128.0f;

    if (matrix_type == 1) {
        // BT.709 limited range
        R = y + 1.79274107f * v;
        G = y - 0.21324861f * u - 0.53290933f * v;
        B = y + 2.11240179f * u;
    } else {
        // BT.601 limited range
        R = y + 1.59602678f * v;
        G = y - 0.39176229f * u - 0.81296765f * v;
        B = y + 2.01723214f * u;
    }

    R = fminf(255.0f, fmaxf(0.0f, R));
    G = fminf(255.0f, fmaxf(0.0f, G));
    B = fminf(255.0f, fmaxf(0.0f, B));
}


// ============================================================
// Bilinear UV sampling for NV12
//
// Video chroma_location = left
//
// NV12 layout:
// Y plane  : H x W
// UV plane : H/2 x W bytes
//            interleaved U,V,U,V...
// ============================================================

__device__ __forceinline__
void sample_uv_bilinear(
    const uint8_t* uv_plane,
    int width,
    int height,
    float src_x,
    float src_y,
    float &U,
    float &V
) {
    const int uv_width  = width / 2;
    const int uv_height = height / 2;

    // Left chroma siting:
    // horizontal chroma sample k is aligned with x = 2*k
    //
    // Vertically, 4:2:0 chroma is centered between
    // corresponding luma rows.
    float cx = src_x * 0.5f;
    float cy = (src_y - 0.5f) * 0.5f;

    int x0 = (int)floorf(cx);
    int y0 = (int)floorf(cy);

    float wx = cx - x0;
    float wy = cy - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, uv_width - 1));
    x1 = max(0, min(x1, uv_width - 1));

    y0 = max(0, min(y0, uv_height - 1));
    y1 = max(0, min(y1, uv_height - 1));

    // Each chroma sample is two bytes: U,V
    int idx00 = y0 * width + 2 * x0;
    int idx01 = y0 * width + 2 * x1;
    int idx10 = y1 * width + 2 * x0;
    int idx11 = y1 * width + 2 * x1;

    float u00 = (float)uv_plane[idx00 + 0];
    float v00 = (float)uv_plane[idx00 + 1];

    float u01 = (float)uv_plane[idx01 + 0];
    float v01 = (float)uv_plane[idx01 + 1];

    float u10 = (float)uv_plane[idx10 + 0];
    float v10 = (float)uv_plane[idx10 + 1];

    float u11 = (float)uv_plane[idx11 + 0];
    float v11 = (float)uv_plane[idx11 + 1];

    float u_top = u00 + wx * (u01 - u00);
    float u_bot = u10 + wx * (u11 - u10);

    float v_top = v00 + wx * (v01 - v00);
    float v_bot = v10 + wx * (v11 - v10);

    U = u_top + wy * (u_bot - u_top);
    V = v_top + wy * (v_bot - v_top);
}


// ============================================================
// Bilinear luma + bilinear chroma + YUV -> RGB
// ============================================================

__device__ __forceinline__
void sample_nv12_rgb(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    int width,
    int height,
    float src_x,
    float src_y,
    int matrix_type,
    float &R,
    float &G,
    float &B
) {
    // --------------------------------------------------------
    // Bilinear Y sampling
    // --------------------------------------------------------

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);

    float wx = src_x - x0;
    float wy = src_y - y0;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, width - 1));
    x1 = max(0, min(x1, width - 1));

    y0 = max(0, min(y0, height - 1));
    y1 = max(0, min(y1, height - 1));

    float y00 = (float)y_plane[y0 * width + x0];
    float y01 = (float)y_plane[y0 * width + x1];
    float y10 = (float)y_plane[y1 * width + x0];
    float y11 = (float)y_plane[y1 * width + x1];

    float y_top = y00 + wx * (y01 - y00);
    float y_bot = y10 + wx * (y11 - y10);

    float Y = y_top + wy * (y_bot - y_top);

    // --------------------------------------------------------
    // Bilinear UV sampling
    // --------------------------------------------------------

    float U, V;

    sample_uv_bilinear(
        uv_plane,
        width,
        height,
        src_x,
        src_y,
        U,
        V
    );

    // --------------------------------------------------------
    // Convert to RGB
    // --------------------------------------------------------

    yuv_to_rgb(
        Y,
        U,
        V,
        R,
        G,
        B,
        matrix_type
    );
}


// ============================================================
// Fused NV12 -> YOLO tensor
//
// Input:
//   NV12 GPU-resident Y and UV planes
//
// Output:
//   float32 RGB CHW
//   shape = [3, 384, 640]
//
// Operations fused:
//   NV12 -> RGB
//   bilinear resize
//   letterbox
//   CHW layout
//   normalization
// ============================================================

__global__
void fused_nv12_yolo(
    const uint8_t* __restrict__ y_plane,
    const uint8_t* __restrict__ uv_plane,
    float* __restrict__ output,
    int matrix_type
) {
    const int src_w = 1920;
    const int src_h = 1080;

    const int out_w = 640;
    const int out_h = 384;

    const int resized_w = 640;
    const int resized_h = 360;

    const int pad_left = 0;
    const int pad_top = 12;

    int x =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    int y =
        blockIdx.y * blockDim.y +
        threadIdx.y;

    if (x >= out_w || y >= out_h)
        return;

    int pixel =
        y * out_w + x;

    int plane =
        out_w * out_h;

    // --------------------------------------------------------
    // Letterbox padding
    // --------------------------------------------------------

    if (
        x < pad_left ||
        x >= pad_left + resized_w ||
        y < pad_top ||
        y >= pad_top + resized_h
    ) {
        float pad =
            114.0f / 255.0f;

        output[
            0 * plane + pixel
        ] = pad;

        output[
            1 * plane + pixel
        ] = pad;

        output[
            2 * plane + pixel
        ] = pad;

        return;
    }

    // --------------------------------------------------------
    // Position inside resized image
    // --------------------------------------------------------

    int rx = x - pad_left;
    int ry = y - pad_top;

    // --------------------------------------------------------
    // Half-pixel resize mapping
    // --------------------------------------------------------

    float scale_x =
        (float)src_w /
        (float)resized_w;

    float scale_y =
        (float)src_h /
        (float)resized_h;

    float src_x =
        (rx + 0.5f) * scale_x -
        0.5f;

    float src_y =
        (ry + 0.5f) * scale_y -
        0.5f;

    // --------------------------------------------------------
    // NV12 sample -> RGB
    // --------------------------------------------------------

    float R, G, B;

    sample_nv12_rgb(
        y_plane,
        uv_plane,
        src_w,
        src_h,
        src_x,
        src_y,
        matrix_type,
        R,
        G,
        B
    );

    // --------------------------------------------------------
    // RGB -> normalized CHW
    // --------------------------------------------------------

    output[
        0 * plane + pixel
    ] = R / 255.0f;

    output[
        1 * plane + pixel
    ] = G / 255.0f;

    output[
        2 * plane + pixel
    ] = B / 255.0f;
}


// ============================================================
// Launcher
// ============================================================

extern "C"
void launch_nv12_yolo(
    const uint8_t* y_plane,
    const uint8_t* uv_plane,
    float* output,
    int matrix_type,
    cudaStream_t stream
) {
    dim3 block(
        16,
        16
    );

    dim3 grid(
        (640 + block.x - 1) /
            block.x,

        (384 + block.y - 1) /
            block.y
    );

    fused_nv12_yolo<<<
        grid,
        block,
        0,
        stream
    >>>(
        y_plane,
        uv_plane,
        output,
        matrix_type
    );
}
