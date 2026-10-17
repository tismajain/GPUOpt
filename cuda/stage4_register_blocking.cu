
// cuda/stage4_register_blocking.cu

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>

#define TILE 16

// ---------------- Stage 3 reference ----------------

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

        As[ty][tx] =
            (row < N && aCol < N)
            ? A[row * N + aCol]
            : 0.0f;

        Bs[ty][tx] =
            (bRow < N && col < N)
            ? B[bRow * N + col]
            : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}


// ---------------- Stage 4 ----------------

#define TX 16
#define TY 16

#define TM 4
#define TN 4

#define BM (TY * TM)   // 64 output rows/block
#define BN (TX * TN)   // 64 output cols/block
#define BK 16

__global__ void matmul_register_blocked(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int tid = ty * blockDim.x + tx;
    int numThreads = blockDim.x * blockDim.y;

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    float acc[TM][TN] = {0.0f};

    int numKTiles = (N + BK - 1) / BK;

    for (int t = 0; t < numKTiles; ++t) {

        // Cooperatively load A tile
        for (int idx = tid; idx < BM * BK; idx += numThreads) {

            int localRow = idx / BK;
            int localCol = idx % BK;

            int globalRow = blockRow + localRow;
            int globalCol = t * BK + localCol;

            As[localRow][localCol] =
                (globalRow < N && globalCol < N)
                ? A[globalRow * N + globalCol]
                : 0.0f;
        }

        // Cooperatively load B tile
        for (int idx = tid; idx < BK * BN; idx += numThreads) {

            int localRow = idx / BN;
            int localCol = idx % BN;

            int globalRow = t * BK + localRow;
            int globalCol = blockCol + localCol;

            Bs[localRow][localCol] =
                (globalRow < N && globalCol < N)
                ? B[globalRow * N + globalCol]
                : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BK; ++k) {

            float aFrag[TM];
            float bFrag[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i)
                aFrag[i] = As[ty * TM + i][k];

            #pragma unroll
            for (int j = 0; j < TN; ++j)
                bFrag[j] = Bs[k][tx * TN + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {

                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += aFrag[i] * bFrag[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {

        int row = blockRow + ty * TM + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {

            int col = blockCol + tx * TN + j;

            if (row < N && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}


// ---------------- timing ----------------

template <typename Kernel>
double benchmark(
    Kernel kernel,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N,
    dim3 grid,
    dim3 block
) {
    const int warmups = 5;
    const int runs = 20;

    for (int i = 0; i < warmups; ++i)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);

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

    if (argc > 1)
        N = std::stoi(argv[1]);

    size_t elements = static_cast<size_t>(N) * N;
    size_t bytes = elements * sizeof(float);

    std::vector<float> A(elements);
    std::vector<float> B(elements);

    std::vector<float> C_tiled(elements);
    std::vector<float> C_reg(elements);

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


    // Stage 3 configuration

    dim3 tiledBlock(TILE, TILE);

    dim3 tiledGrid(
        (N + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    double tiled_ms = benchmark(
        matmul_tiled,
        d_A,
        d_B,
        d_C,
        N,
        tiledGrid,
        tiledBlock
    );

    cudaMemcpy(
        C_tiled.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    // Stage 4 configuration

    dim3 regBlock(TX, TY);

    dim3 regGrid(
        (N + BN - 1) / BN,
        (N + BM - 1) / BM
    );

    double reg_ms = benchmark(
        matmul_register_blocked,
        d_A,
        d_B,
        d_C,
        N,
        regGrid,
        regBlock
    );

    cudaMemcpy(
        C_reg.data(),
        d_C,
        bytes,
        cudaMemcpyDeviceToHost
    );


    double operations =
        2.0 * static_cast<double>(N) * N * N;

    double tiled_gflops =
        operations / (tiled_ms / 1000.0) / 1e9;

    double reg_gflops =
        operations / (reg_ms / 1000.0) / 1e9;


    double max_diff = 0.0;

    for (size_t i = 0; i < elements; ++i) {

        max_diff = std::max(
            max_diff,
            static_cast<double>(
                std::abs(C_tiled[i] - C_reg[i])
            )
        );
    }


    std::cout << std::fixed << std::setprecision(3);

    std::cout << "N                        : "
              << N << "\n\n";

    std::cout << "Shared tiled time        : "
              << tiled_ms << " ms\n";

    std::cout << "Shared tiled GFLOP/s     : "
              << tiled_gflops << "\n\n";

    std::cout << "Register blocked time    : "
              << reg_ms << " ms\n";

    std::cout << "Register blocked GFLOP/s : "
              << reg_gflops << "\n\n";

    std::cout << "Speedup                   : "
              << tiled_ms / reg_ms << "x\n";

    std::cout << "Max difference            : "
              << max_diff << "\n";


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
