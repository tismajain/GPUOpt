
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";    \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

#define CUBLAS_CHECK(call)                                                 \
do {                                                                       \
    cublasStatus_t status = call;                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                 \
        std::cerr << "cuBLAS error: " << status << "\n";                   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)


// ============================================================
// Stage 2 — coalesced global-memory kernel
// ============================================================

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


// ============================================================
// Stage 3 — shared-memory tiled kernel
// ============================================================

#define TILE 16

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

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}


// ============================================================
// Stage 4 — 4x4 register-blocked kernel
// ============================================================

#define TX 16
#define TY 16

#define TM 4
#define TN 4

#define BM (TY * TM)
#define BN (TX * TN)
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

        #pragma unroll
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


// ============================================================
// CUDA kernel benchmark
// ============================================================

template <typename Kernel>
double benchmark_kernel(
    Kernel kernel,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N,
    dim3 grid,
    dim3 block
) {
    const int warmups = 5;
    const int runs = 30;

    for (int i = 0; i < warmups; ++i)
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < runs; ++i) {

        CUDA_CHECK(cudaEventRecord(start));

        kernel<<<grid, block>>>(d_A, d_B, d_C, N);

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}


// ============================================================
// cuBLAS benchmark
// ============================================================

double benchmark_cublas(
    cublasHandle_t handle,
    const float* d_A,
    const float* d_B,
    float* d_C,
    int N
) {
    const int warmups = 5;
    const int runs = 30;

    float alpha = 1.0f;
    float beta = 0.0f;

    /*
       Our matrices are row-major.

       cuBLAS expects column-major.

       To compute:

           C = A * B

       in row-major memory, we exploit:

           C^T = B^T * A^T

       so we swap A and B in the cuBLAS call.
    */

    for (int i = 0; i < warmups; ++i) {

        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N
            )
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < runs; ++i) {

        CUDA_CHECK(cudaEventRecord(start));

        CUBLAS_CHECK(
            cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_B, N,
                d_A, N,
                &beta,
                d_C, N
            )
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());

    return times[times.size() / 2];
}


// ============================================================
// correctness helper
// ============================================================

double max_abs_difference(
    const std::vector<float>& A,
    const std::vector<float>& B
) {
    double maxDiff = 0.0;

    for (size_t i = 0; i < A.size(); ++i) {
        maxDiff = std::max(
            maxDiff,
            static_cast<double>(
                std::abs(A[i] - B[i])
            )
        );
    }

    return maxDiff;
}


// ============================================================
// main
// ============================================================

int main(int argc, char** argv) {

    int N = 1024;

    if (argc > 1)
        N = std::stoi(argv[1]);

    size_t elements =
        static_cast<size_t>(N) * N;

    size_t bytes =
        elements * sizeof(float);


    // Host matrices

    std::vector<float> A(elements);
    std::vector<float> B(elements);

    std::vector<float> C_coalesced(elements);
    std::vector<float> C_tiled(elements);
    std::vector<float> C_reg(elements);
    std::vector<float> C_cublas(elements);


    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }


    // Device memory

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            B.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );


    // cuBLAS

    cublasHandle_t handle;

    CUBLAS_CHECK(cublasCreate(&handle));

    // Explicit normal FP32 math mode.
    CUBLAS_CHECK(
        cublasSetMathMode(
            handle,
            CUBLAS_DEFAULT_MATH
        )
    );


    // --------------------------------------------------------
    // Stage 2
    // --------------------------------------------------------

    dim3 basicBlock(16, 16);

    dim3 basicGrid(
        (N + 15) / 16,
        (N + 15) / 16
    );

    double coalesced_ms =
        benchmark_kernel(
            matmul_coalesced,
            d_A, d_B, d_C,
            N,
            basicGrid,
            basicBlock
        );

    CUDA_CHECK(
        cudaMemcpy(
            C_coalesced.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Stage 3
    // --------------------------------------------------------

    double tiled_ms =
        benchmark_kernel(
            matmul_tiled,
            d_A, d_B, d_C,
            N,
            basicGrid,
            basicBlock
        );

    CUDA_CHECK(
        cudaMemcpy(
            C_tiled.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Stage 4
    // --------------------------------------------------------

    dim3 regBlock(TX, TY);

    dim3 regGrid(
        (N + BN - 1) / BN,
        (N + BM - 1) / BM
    );

    double reg_ms =
        benchmark_kernel(
            matmul_register_blocked,
            d_A, d_B, d_C,
            N,
            regGrid,
            regBlock
        );

    CUDA_CHECK(
        cudaMemcpy(
            C_reg.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // cuBLAS
    // --------------------------------------------------------

    double cublas_ms =
        benchmark_cublas(
            handle,
            d_A,
            d_B,
            d_C,
            N
        );

    CUDA_CHECK(
        cudaMemcpy(
            C_cublas.data(),
            d_C,
            bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Metrics
    // --------------------------------------------------------

    double ops =
        2.0 * static_cast<double>(N) * N * N;

    auto gflops = [&](double ms) {
        return ops / (ms / 1000.0) / 1e9;
    };

    double coalesced_gflops = gflops(coalesced_ms);
    double tiled_gflops     = gflops(tiled_ms);
    double reg_gflops       = gflops(reg_ms);
    double cublas_gflops    = gflops(cublas_ms);


    double coalesced_diff =
        max_abs_difference(
            C_coalesced,
            C_cublas
        );

    double tiled_diff =
        max_abs_difference(
            C_tiled,
            C_cublas
        );

    double reg_diff =
        max_abs_difference(
            C_reg,
            C_cublas
        );


    // --------------------------------------------------------
    // Results
    // --------------------------------------------------------

    std::cout << std::fixed << std::setprecision(3);

    std::cout << "\nN = " << N << "\n\n";

    std::cout
        << "Kernel              Time (ms)    GFLOP/s      % cuBLAS\n";

    std::cout
        << "------------------------------------------------------\n";

    std::cout
        << "Coalesced           "
        << std::setw(9) << coalesced_ms << "    "
        << std::setw(9) << coalesced_gflops << "    "
        << std::setw(7)
        << (coalesced_gflops / cublas_gflops * 100.0)
        << "%\n";

    std::cout
        << "Shared tiled        "
        << std::setw(9) << tiled_ms << "    "
        << std::setw(9) << tiled_gflops << "    "
        << std::setw(7)
        << (tiled_gflops / cublas_gflops * 100.0)
        << "%\n";

    std::cout
        << "Register blocked    "
        << std::setw(9) << reg_ms << "    "
        << std::setw(9) << reg_gflops << "    "
        << std::setw(7)
        << (reg_gflops / cublas_gflops * 100.0)
        << "%\n";

    std::cout
        << "cuBLAS SGEMM        "
        << std::setw(9) << cublas_ms << "    "
        << std::setw(9) << cublas_gflops << "    "
        << "100.000%\n";


    std::cout << "\nCorrectness vs cuBLAS\n";
    std::cout << "---------------------\n";

    std::cout
        << "Coalesced max abs diff : "
        << coalesced_diff << "\n";

    std::cout
        << "Tiled max abs diff     : "
        << tiled_diff << "\n";

    std::cout
        << "Register max abs diff  : "
        << reg_diff << "\n";


    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
