
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

using namespace std;

void matmul_blocked(
    const float* A,
    const float* B,
    float* C,
    int N,
    int BLOCK
) {
    for (int ii = 0; ii < N; ii += BLOCK) {
        for (int kk = 0; kk < N; kk += BLOCK) {
            for (int jj = 0; jj < N; jj += BLOCK) {

                int i_end = min(ii + BLOCK, N);
                int k_end = min(kk + BLOCK, N);
                int j_end = min(jj + BLOCK, N);

                for (int i = ii; i < i_end; ++i) {
                    for (int k = kk; k < k_end; ++k) {

                        float a = A[i * N + k];

                        for (int j = jj; j < j_end; ++j) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {

    int N = 1024;
    int BLOCK = 32;
    int runs = 5;

    if (argc > 1) N = stoi(argv[1]);
    if (argc > 2) BLOCK = stoi(argv[2]);

    size_t elements = static_cast<size_t>(N) * N;

    vector<float> A(elements);
    vector<float> B(elements);
    vector<float> C(elements, 0.0f);

    mt19937 rng(42);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i) {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    vector<double> times_ms;

    for (int r = 0; r < runs; ++r) {

        fill(C.begin(), C.end(), 0.0f);

        auto start = chrono::high_resolution_clock::now();

        matmul_blocked(
            A.data(),
            B.data(),
            C.data(),
            N,
            BLOCK
        );

        auto end = chrono::high_resolution_clock::now();

        double ms =
            chrono::duration<double, milli>(end - start).count();

        times_ms.push_back(ms);
    }

    sort(times_ms.begin(), times_ms.end());
    double median_ms = times_ms[times_ms.size() / 2];

    double seconds = median_ms / 1000.0;

    double operations =
        2.0 * static_cast<double>(N) * N * N;

    double gflops =
        operations / seconds / 1e9;

    double checksum = 0.0;
    for (float x : C) {
        checksum += x;
    }

    cout << fixed << setprecision(3);

    cout << "N             : " << N << "\n";
    cout << "Block size    : " << BLOCK << "\n";
    cout << "Median time   : " << median_ms << " ms\n";
    cout << "Performance   : " << gflops << " GFLOP/s\n";
    cout << "Checksum      : " << checksum << "\n";

    return 0;
}
