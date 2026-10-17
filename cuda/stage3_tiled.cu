
// cuda/stage3_tiled.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>

#define TILE 16

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

__global__ void matmul_tiled(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {

        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        if (row < N && aCol < N)
            As[ty][tx] = A[row * N + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < N && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
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

    size_t elements = static_cast<size_t>(N) * N;
    size_t bytes = elements * sizeof(float);

    std::vector<float> A(elements);
    std::vector<float> B(elements);

    std::vector<float> C_ref(elements);
    std::vector<float> C_tiled(elements);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);

    dim3 grid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    double ref_ms = benchmark(
        matmul_coalesced,
        d_A, d_B, d_C,
        N, grid, block
    );

    cudaMemcpy(
        C_ref.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    double tiled_ms = benchmark(
        matmul_tiled,
        d_A, d_B, d_C,
        N, grid, block
    );

    cudaMemcpy(
        C_tiled.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );

    double ops =
        2.0 * static_cast<double>(N) * N * N;

    double ref_gflops =
        ops / (ref_ms / 1000.0) / 1e9;

    double tiled_gflops =
        ops / (tiled_ms / 1000.0) / 1e9;

    double max_diff = 0.0;

    for (size_t i = 0; i < elements; ++i) {
        max_diff = std::max(
            max_diff,
            static_cast<double>(
                std::abs(C_ref[i] - C_tiled[i])
            )
        );
    }

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "N                  : " << N << "\n\n";

    std::cout << "Coalesced time     : "
              << ref_ms << " ms\n";

    std::cout << "Coalesced GFLOP/s  : "
              << ref_gflops << "\n\n";

    std::cout << "Tiled time         : "
              << tiled_ms << " ms\n";

    std::cout << "Tiled GFLOP/s      : "
              << tiled_gflops << "\n\n";

    std::cout << "Speedup            : "
              << ref_ms / tiled_ms << "x\n";

    std::cout << "Max difference     : "
              << max_diff << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
