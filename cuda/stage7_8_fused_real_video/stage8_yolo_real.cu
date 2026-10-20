#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <fstream>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";    \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)


// ============================================================
// Baseline kernel 1
// Letterbox resize + BGR -> RGB
// Output: FP32 HWC, values 0..255
// ============================================================

__global__ void letterbox_bgr_to_rgb(
    const unsigned char* input,
    float* output,
    int inW,
    int inH,
    int outW,
    int outH,
    float scale,
    int resizedW,
    int resizedH,
    int padX,
    int padY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH)
        return;

    int outBase = (y * outW + x) * 3;

    if (x < padX || x >= padX + resizedW ||
        y < padY || y >= padY + resizedH) {

        output[outBase + 0] = 114.0f;
        output[outBase + 1] = 114.0f;
        output[outBase + 2] = 114.0f;
        return;
    }

    float resizedX = x - padX;
    float resizedY = y - padY;

    float srcX = (resizedX + 0.5f) / scale - 0.5f;
    float srcY = (resizedY + 0.5f) / scale - 0.5f;

    srcX = fmaxf(0.0f, fminf(srcX, inW - 1.0f));
    srcY = fmaxf(0.0f, fminf(srcY, inH - 1.0f));

    int x0 = static_cast<int>(floorf(srcX));
    int y0 = static_cast<int>(floorf(srcY));

    int x1 = min(x0 + 1, inW - 1);
    int y1 = min(y0 + 1, inH - 1);

    float dx = srcX - x0;
    float dy = srcY - y0;

    #pragma unroll
    for (int outC = 0; outC < 3; ++outC) {

        int inC = 2 - outC; // BGR -> RGB

        float v00 = input[(y0 * inW + x0) * 3 + inC];
        float v01 = input[(y0 * inW + x1) * 3 + inC];
        float v10 = input[(y1 * inW + x0) * 3 + inC];
        float v11 = input[(y1 * inW + x1) * 3 + inC];

        float top =
            v00 + dx * (v01 - v00);

        float bottom =
            v10 + dx * (v11 - v10);

        output[outBase + outC] =
            top + dy * (bottom - top);
    }
}


// ============================================================
// Baseline kernel 2
// Normalize
// ============================================================

__global__ void normalize(
    const float* input,
    float* output,
    int elements
) {
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < elements)
        output[idx] = input[idx] / 255.0f;
}


// ============================================================
// Baseline kernel 3
// HWC -> CHW
// ============================================================

__global__ void hwc_to_chw(
    const float* input,
    float* output,
    int width,
    int height
) {
    int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    int pixels = width * height;

    if (idx >= pixels)
        return;

    int hwcBase = idx * 3;

    output[0 * pixels + idx] =
        input[hwcBase + 0];

    output[1 * pixels + idx] =
        input[hwcBase + 1];

    output[2 * pixels + idx] =
        input[hwcBase + 2];
}


// ============================================================
// Fused kernel
// Letterbox + BGR->RGB + normalize + HWC->CHW
// ============================================================

__global__ void fused_yolo_preprocess(
    const unsigned char* input,
    float* output,
    int inW,
    int inH,
    int outW,
    int outH,
    float scale,
    int resizedW,
    int resizedH,
    int padX,
    int padY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outW || y >= outH)
        return;

    int pixelIndex = y * outW + x;
    int pixels = outW * outH;

    if (x < padX || x >= padX + resizedW ||
        y < padY || y >= padY + resizedH) {

        float padValue = 114.0f / 255.0f;

        output[0 * pixels + pixelIndex] = padValue;
        output[1 * pixels + pixelIndex] = padValue;
        output[2 * pixels + pixelIndex] = padValue;

        return;
    }

    float resizedX = x - padX;
    float resizedY = y - padY;

    float srcX = (resizedX + 0.5f) / scale - 0.5f;
    float srcY = (resizedY + 0.5f) / scale - 0.5f;

    srcX = fmaxf(0.0f, fminf(srcX, inW - 1.0f));
    srcY = fmaxf(0.0f, fminf(srcY, inH - 1.0f));

    int x0 = static_cast<int>(floorf(srcX));
    int y0 = static_cast<int>(floorf(srcY));

    int x1 = min(x0 + 1, inW - 1);
    int y1 = min(y0 + 1, inH - 1);

    float dx = srcX - x0;
    float dy = srcY - y0;

    #pragma unroll
    for (int outC = 0; outC < 3; ++outC) {

        int inC = 2 - outC; // BGR -> RGB

        float v00 = input[(y0 * inW + x0) * 3 + inC];
        float v01 = input[(y0 * inW + x1) * 3 + inC];
        float v10 = input[(y1 * inW + x0) * 3 + inC];
        float v11 = input[(y1 * inW + x1) * 3 + inC];

        float top =
            v00 + dx * (v01 - v00);

        float bottom =
            v10 + dx * (v11 - v10);

        float value =
            (top + dy * (bottom - top)) / 255.0f;

        output[outC * pixels + pixelIndex] = value;
    }
}


// ============================================================
// Median helper
// ============================================================

float median(std::vector<float>& values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}


// ============================================================
// main
// ============================================================

int main() {

    const int inW = 1920;
    const int inH = 1080;

    const int outW = 640;
    const int outH = 640;

    const int channels = 3;

    const char* framePath =
        "/kaggle/working/GPUOpt/data/boxing_frames/frame_6000.bgr";


    // --------------------------------------------------------
    // YOLO-style letterbox geometry
    // --------------------------------------------------------

    float scale = std::min(
        static_cast<float>(outW) / inW,
        static_cast<float>(outH) / inH
    );

    int resizedW =
        static_cast<int>(std::round(inW * scale));

    int resizedH =
        static_cast<int>(std::round(inH * scale));

    int padX =
        (outW - resizedW) / 2;

    int padY =
        (outH - resizedH) / 2;


    // --------------------------------------------------------
    // Sizes
    // --------------------------------------------------------

    size_t inputElements =
        static_cast<size_t>(inW) * inH * channels;

    size_t outputElements =
        static_cast<size_t>(outW) * outH * channels;

    size_t inputBytes =
        inputElements * sizeof(unsigned char);

    size_t floatBytes =
        outputElements * sizeof(float);


    // --------------------------------------------------------
    // Read real boxing frame
    // --------------------------------------------------------

    std::vector<unsigned char> h_input(inputElements);

    std::ifstream file(framePath, std::ios::binary);

    if (!file) {
        std::cerr
            << "Failed to open real boxing frame: "
            << framePath << "\n";
        return 1;
    }

    file.read(
        reinterpret_cast<char*>(h_input.data()),
        inputBytes
    );

    if (file.gcount() !=
        static_cast<std::streamsize>(inputBytes)) {

        std::cerr
            << "Unexpected frame size. Read "
            << file.gcount()
            << " bytes, expected "
            << inputBytes
            << "\n";

        return 1;
    }

    file.close();


    // --------------------------------------------------------
    // GPU allocation
    // --------------------------------------------------------

    unsigned char* d_input;

    float* d_letterbox;
    float* d_normalized;
    float* d_baseline;
    float* d_fused;

    CUDA_CHECK(
        cudaMalloc(
            &d_input,
            inputBytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_letterbox,
            floatBytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_normalized,
            floatBytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_baseline,
            floatBytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_fused,
            floatBytes
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_input,
            h_input.data(),
            inputBytes,
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // Launch config
    // --------------------------------------------------------

    dim3 block2D(16, 16);

    dim3 grid2D(
        (outW + 15) / 16,
        (outH + 15) / 16
    );

    const int threads = 256;

    int normalizeBlocks =
        (outputElements + threads - 1) / threads;

    int layoutBlocks =
        (outW * outH + threads - 1) / threads;


    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for (int i = 0; i < 10; ++i) {

        letterbox_bgr_to_rgb<<<grid2D, block2D>>>(
            d_input,
            d_letterbox,
            inW,
            inH,
            outW,
            outH,
            scale,
            resizedW,
            resizedH,
            padX,
            padY
        );

        normalize<<<normalizeBlocks, threads>>>(
            d_letterbox,
            d_normalized,
            outputElements
        );

        hwc_to_chw<<<layoutBlocks, threads>>>(
            d_normalized,
            d_baseline,
            outW,
            outH
        );

        fused_yolo_preprocess<<<grid2D, block2D>>>(
            d_input,
            d_fused,
            inW,
            inH,
            outW,
            outH,
            scale,
            resizedW,
            resizedH,
            padX,
            padY
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    const int runs = 100;

    std::vector<float> baselineTimes;
    std::vector<float> fusedTimes;

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // Baseline
    for (int r = 0; r < runs; ++r) {

        CUDA_CHECK(cudaEventRecord(start));

        letterbox_bgr_to_rgb<<<grid2D, block2D>>>(
            d_input,
            d_letterbox,
            inW,
            inH,
            outW,
            outH,
            scale,
            resizedW,
            resizedH,
            padX,
            padY
        );

        normalize<<<normalizeBlocks, threads>>>(
            d_letterbox,
            d_normalized,
            outputElements
        );

        hwc_to_chw<<<layoutBlocks, threads>>>(
            d_normalized,
            d_baseline,
            outW,
            outH
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            )
        );

        baselineTimes.push_back(ms);
    }


    // Fused
    for (int r = 0; r < runs; ++r) {

        CUDA_CHECK(cudaEventRecord(start));

        fused_yolo_preprocess<<<grid2D, block2D>>>(
            d_input,
            d_fused,
            inW,
            inH,
            outW,
            outH,
            scale,
            resizedW,
            resizedH,
            padX,
            padY
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &ms,
                start,
                stop
            )
        );

        fusedTimes.push_back(ms);
    }


    float baselineMedian =
        median(baselineTimes);

    float fusedMedian =
        median(fusedTimes);


    // --------------------------------------------------------
    // Correctness
    // --------------------------------------------------------

    std::vector<float> h_baseline(outputElements);
    std::vector<float> h_fused(outputElements);

    CUDA_CHECK(
        cudaMemcpy(
            h_baseline.data(),
            d_baseline,
            floatBytes,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            h_fused.data(),
            d_fused,
            floatBytes,
            cudaMemcpyDeviceToHost
        )
    );


    double maxDiff = 0.0;

    double checksumBaseline = 0.0;
    double checksumFused = 0.0;

    for (size_t i = 0; i < outputElements; ++i) {

        maxDiff = std::max(
            maxDiff,
            static_cast<double>(
                std::abs(
                    h_baseline[i] -
                    h_fused[i]
                )
            )
        );

        checksumBaseline += h_baseline[i];
        checksumFused += h_fused[i];
    }


    // --------------------------------------------------------
    // Approx. intermediate traffic avoided
    // --------------------------------------------------------

    double tensorMB =
        floatBytes / (1024.0 * 1024.0);

    double avoidedTrafficMB =
        4.0 * tensorMB;


    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout
        << "Frame source          : REAL boxing video frame 6000\n";

    std::cout
        << "Input                 : "
        << inW << "x" << inH
        << " BGR uint8\n";

    std::cout
        << "YOLO input            : "
        << outW << "x" << outH
        << " RGB FP32 CHW\n\n";

    std::cout
        << "Resize scale          : "
        << scale << "\n";

    std::cout
        << "Resized region        : "
        << resizedW << "x"
        << resizedH << "\n";

    std::cout
        << "Padding X/Y           : "
        << padX << " / "
        << padY << "\n\n";

    std::cout
        << "Baseline median       : "
        << baselineMedian
        << " ms\n";

    std::cout
        << "Fused median          : "
        << fusedMedian
        << " ms\n";

    std::cout
        << "Speedup               : "
        << baselineMedian / fusedMedian
        << "x\n\n";

    std::cout
        << "Baseline launches     : 3\n";

    std::cout
        << "Fused launches        : 1\n";

    std::cout
        << "Baseline intermediates: 2\n";

    std::cout
        << "Fused intermediates   : 0\n";

    std::cout
        << "Traffic avoided       : "
        << avoidedTrafficMB
        << " MB/frame\n\n";

    std::cout << std::scientific
              << std::setprecision(6);

    std::cout
        << "Max abs difference    : "
        << maxDiff
        << "\n";

    std::cout << std::fixed
              << std::setprecision(4);

    std::cout
        << "Baseline checksum     : "
        << checksumBaseline
        << "\n";

    std::cout
        << "Fused checksum        : "
        << checksumFused
        << "\n";


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_letterbox));
    CUDA_CHECK(cudaFree(d_normalized));
    CUDA_CHECK(cudaFree(d_baseline));
    CUDA_CHECK(cudaFree(d_fused));

    return 0;
}
