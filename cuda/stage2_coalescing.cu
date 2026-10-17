
// cuda/stage2_coalescing.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>

__global__ void matmul_bad(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

__global__ void matmul_coalesced(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

double benchmark(
    void (*kernel)(const float*, const float*, float*, int),
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N,
    dim3 grid,
    dim3 block
) {
    const int warmups = 5;
    const int runs = 20;

    for (int i = 0; i < warmups; ++i) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }

    cudaDeviceSynchronize();

    std::vector<float> times;

    for (int i = 0; i < runs; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        kernel<<<grid, block>>>(d_A, d_B, d_C, N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}

int main(int argc, char** argv) {

    int N = 1024;
    if (argc > 1) N = std::stoi(argv[1]);

    size_t bytes =
        static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> A(N * N);
    std::vector<float> B(N * N);
    std::vector<float> C_bad(N * N);
    std::vector<float> C_good(N * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    double bad_ms = benchmark(
        matmul_bad,
        d_A, d_B, d_C,
        N, grid, block
    );

    cudaMemcpy(
        C_bad.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    double good_ms = benchmark(
        matmul_coalesced,
        d_A, d_B, d_C,
        N, grid, block
    );

    cudaMemcpy(
        C_good.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    double ops =
        2.0 * static_cast<double>(N) * N * N;

    double bad_gflops =
        ops / (bad_ms / 1000.0) / 1e9;

    double good_gflops =
        ops / (good_ms / 1000.0) / 1e9;

    double max_diff = 0.0;

    for (size_t i = 0; i < C_bad.size(); ++i) {
        max_diff = std::max(
            max_diff,
            static_cast<double>(
                std::abs(C_bad[i] - C_good[i])
            )
        );
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "N                   : " << N << "\n\n";

    std::cout << "Bad mapping time    : "
              << bad_ms << " ms\n";

    std::cout << "Bad mapping GFLOP/s : "
              << bad_gflops << "\n\n";

    std::cout << "Coalesced time      : "
              << good_ms << " ms\n";

    std::cout << "Coalesced GFLOP/s   : "
              << good_gflops << "\n\n";

    std::cout << "Speedup             : "
              << bad_ms / good_ms << "x\n";

    std::cout << "Max difference      : "
              << max_diff << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
