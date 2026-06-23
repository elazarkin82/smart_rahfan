#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// Shape definitions matching the NPU model
const int CHANNELS = 64;
const int SEARCH_H = 16;
const int SEARCH_W = 16;
const int KERNEL_H = 8;
const int KERNEL_W = 8;

// 1. Baseline NCHW (with standard branching and single thread)
void depthwise_correlation_nchw(const float* search, const float* kernel, float* output) {
    const int pad_top = 3;
    const int pad_left = 3;

    for (int c = 0; c < CHANNELS; ++c) {
        const float* search_chan = search + c * SEARCH_H * SEARCH_W;
        const float* kernel_chan = kernel + c * KERNEL_H * KERNEL_W;
        float* output_chan = output + c * SEARCH_H * SEARCH_W;

        for (int y = 0; y < SEARCH_H; ++y) {
            for (int x = 0; x < SEARCH_W; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    int sy = y + ky - pad_top;
                    if (sy >= 0 && sy < SEARCH_H) {
                        for (int kx = 0; kx < KERNEL_W; ++kx) {
                            int sx = x + kx - pad_left;
                            if (sx >= 0 && sx < SEARCH_W) {
                                sum += search_chan[sy * SEARCH_W + sx] * kernel_chan[ky * KERNEL_W + kx];
                            }
                        }
                    }
                }
                output_chan[y * SEARCH_W + x] = sum;
            }
        }
    }
}

// 2. OpenMP NCHW (adds multi-threading over channels)
void depthwise_correlation_nchw_omp(const float* search, const float* kernel, float* output) {
    const int pad_top = 3;
    const int pad_left = 3;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < CHANNELS; ++c) {
        const float* search_chan = search + c * SEARCH_H * SEARCH_W;
        const float* kernel_chan = kernel + c * KERNEL_H * KERNEL_W;
        float* output_chan = output + c * SEARCH_H * SEARCH_W;

        for (int y = 0; y < SEARCH_H; ++y) {
            for (int x = 0; x < SEARCH_W; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    int sy = y + ky - pad_top;
                    if (sy >= 0 && sy < SEARCH_H) {
                        for (int kx = 0; kx < KERNEL_W; ++kx) {
                            int sx = x + kx - pad_left;
                            if (sx >= 0 && sx < SEARCH_W) {
                                sum += search_chan[sy * SEARCH_W + sx] * kernel_chan[ky * KERNEL_W + kx];
                            }
                        }
                    }
                }
                output_chan[y * SEARCH_W + x] = sum;
            }
        }
    }
}

// 3. Explicitly Padded + Unrolled (Single Thread)
// Removes all inside-loop branching by padding the input channel beforehand.
// Padded size must accommodate 3 top padding and 4 bottom padding = 23 height.
// We align padded width to 24 (multiple of 8) for vector memory alignment.
void depthwise_correlation_nchw_padded(const float* search, const float* kernel, float* output) {
    const int PADDED_H = 23;
    const int PADDED_W = 24;

    for (int c = 0; c < CHANNELS; ++c) {
        const float* search_chan = search + c * SEARCH_H * SEARCH_W;
        const float* kernel_chan = kernel + c * KERNEL_H * KERNEL_W;
        float* output_chan = output + c * SEARCH_H * SEARCH_W;

        float padded_search[PADDED_H * PADDED_W] = {0.0f};

        // Copy search channel to center of padded buffer (top offset: 3, left offset: 3)
        for (int y = 0; y < SEARCH_H; ++y) {
            std::copy(search_chan + y * SEARCH_W, search_chan + (y + 1) * SEARCH_W, padded_search + (y + 3) * PADDED_W + 3);
        }

        // Perform convolution with unrolled inner loops (no boundary checks needed)
        for (int y = 0; y < SEARCH_H; ++y) {
            for (int x = 0; x < SEARCH_W; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    const float* s_row = padded_search + (y + ky) * PADDED_W + (x + 3 - 3); // starts at x
                    const float* k_row = kernel_chan + ky * KERNEL_W;

                    sum += s_row[0] * k_row[0];
                    sum += s_row[1] * k_row[1];
                    sum += s_row[2] * k_row[2];
                    sum += s_row[3] * k_row[3];
                    sum += s_row[4] * k_row[4];
                    sum += s_row[5] * k_row[5];
                    sum += s_row[6] * k_row[6];
                    sum += s_row[7] * k_row[7];
                }
                output_chan[y * SEARCH_W + x] = sum;
            }
        }
    }
}

// 4. Explicitly Padded + Unrolled + OpenMP
void depthwise_correlation_nchw_padded_omp(const float* search, const float* kernel, float* output) {
    const int PADDED_H = 23;
    const int PADDED_W = 24;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < CHANNELS; ++c) {
        const float* search_chan = search + c * SEARCH_H * SEARCH_W;
        const float* kernel_chan = kernel + c * KERNEL_H * KERNEL_W;
        float* output_chan = output + c * SEARCH_H * SEARCH_W;

        float padded_search[PADDED_H * PADDED_W] = {0.0f};

        for (int y = 0; y < SEARCH_H; ++y) {
            std::copy(search_chan + y * SEARCH_W, search_chan + (y + 1) * SEARCH_W, padded_search + (y + 3) * PADDED_W + 3);
        }

        for (int y = 0; y < SEARCH_H; ++y) {
            for (int x = 0; x < SEARCH_W; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    const float* s_row = padded_search + (y + ky) * PADDED_W + x;
                    const float* k_row = kernel_chan + ky * KERNEL_W;

                    sum += s_row[0] * k_row[0];
                    sum += s_row[1] * k_row[1];
                    sum += s_row[2] * k_row[2];
                    sum += s_row[3] * k_row[3];
                    sum += s_row[4] * k_row[4];
                    sum += s_row[5] * k_row[5];
                    sum += s_row[6] * k_row[6];
                    sum += s_row[7] * k_row[7];
                }
                output_chan[y * SEARCH_W + x] = sum;
            }
        }
    }
}

// 5. Explicitly Padded + ARM NEON Vectorization + OpenMP
void depthwise_correlation_nchw_neon_omp(const float* search, const float* kernel, float* output) {
    const int PADDED_H = 23;
    const int PADDED_W = 24;

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < CHANNELS; ++c) {
        const float* search_chan = search + c * SEARCH_H * SEARCH_W;
        const float* kernel_chan = kernel + c * KERNEL_H * KERNEL_W;
        float* output_chan = output + c * SEARCH_H * SEARCH_W;

        float padded_search[PADDED_H * PADDED_W] = {0.0f};

        for (int y = 0; y < SEARCH_H; ++y) {
            std::copy(search_chan + y * SEARCH_W, search_chan + (y + 1) * SEARCH_W, padded_search + (y + 3) * PADDED_W + 3);
        }

        for (int y = 0; y < SEARCH_H; ++y) {
            for (int x = 0; x < SEARCH_W; ++x) {
                float32x4_t sum_vec = vdupq_n_f32(0.0f);

                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    const float* s_row = padded_search + (y + ky) * PADDED_W + x;
                    const float* k_row = kernel_chan + ky * KERNEL_W;

                    // Load 8 elements using two NEON registers (4 floats each)
                    float32x4_t s_vec0 = vld1q_f32(s_row);
                    float32x4_t s_vec1 = vld1q_f32(s_row + 4);
                    float32x4_t k_vec0 = vld1q_f32(k_row);
                    float32x4_t k_vec1 = vld1q_f32(k_row + 4);

                    // Multiply and accumulate
                    sum_vec = vmlaq_f32(sum_vec, s_vec0, k_vec0);
                    sum_vec = vmlaq_f32(sum_vec, s_vec1, k_vec1);
                }
                
                // Horizontal add the vector to a single float
#if defined(__aarch64__)
                output_chan[y * SEARCH_W + x] = vaddvq_f32(sum_vec);
#else
                float temp[4];
                vst1q_f32(temp, sum_vec);
                output_chan[y * SEARCH_W + x] = temp[0] + temp[1] + temp[2] + temp[3];
#endif
            }
        }
    }
}

int main() {
    std::cout << "====================================================\n";
    std::cout << "CPU Depthwise Correlation Optimization Benchmark\n";
    std::cout << "====================================================\n";
    std::cout << "Channels: " << CHANNELS << "\n";
    std::cout << "Search Features: " << SEARCH_H << "x" << SEARCH_W << "\n";
    std::cout << "Template Features: " << KERNEL_H << "x" << KERNEL_W << "\n";
    std::cout << "Total operations: " << CHANNELS * SEARCH_H * SEARCH_W * KERNEL_H * KERNEL_W << " MACs (approx. 2.1M FLOPs)\n\n";

    // Allocate buffers
    std::vector<float> search(SEARCH_H * SEARCH_W * CHANNELS);
    std::vector<float> kernel(KERNEL_H * KERNEL_W * CHANNELS);
    std::vector<float> output1(SEARCH_H * SEARCH_W * CHANNELS, 0.0f);
    std::vector<float> output2(SEARCH_H * SEARCH_W * CHANNELS, 0.0f);
    std::vector<float> output3(SEARCH_H * SEARCH_W * CHANNELS, 0.0f);
    std::vector<float> output4(SEARCH_H * SEARCH_W * CHANNELS, 0.0f);
    std::vector<float> output5(SEARCH_H * SEARCH_W * CHANNELS, 0.0f);

    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& val : search) val = dist(rng);
    for (auto& val : kernel) val = dist(rng);

    const int iterations = 1000;

    // Run baseline to verify correctness of optimized versions
    depthwise_correlation_nchw(search.data(), kernel.data(), output1.data());
    depthwise_correlation_nchw_omp(search.data(), kernel.data(), output2.data());
    depthwise_correlation_nchw_padded(search.data(), kernel.data(), output3.data());
    depthwise_correlation_nchw_padded_omp(search.data(), kernel.data(), output4.data());
    depthwise_correlation_nchw_neon_omp(search.data(), kernel.data(), output5.data());

    // Correctness checks
    auto verify = [](const std::vector<float>& ref, const std::vector<float>& opt, const std::string& name) {
        float max_diff = 0.0f;
        for (size_t i = 0; i < ref.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(ref[i] - opt[i]));
        }
        if (max_diff > 1e-4) {
            std::cout << "[ERROR] Correctness check FAILED for " << name << "! Max diff: " << max_diff << "\n";
        } else {
            std::cout << "[+] Correctness check PASSED for " << name << "\n";
        }
    };

    verify(output1, output2, "OpenMP Baseline");
    verify(output1, output3, "Padded Unrolled ST");
    verify(output1, output4, "Padded Unrolled OpenMP");
    verify(output1, output5, "NEON OpenMP");
    std::cout << "\n";

    // 1. Benchmark Baseline
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        depthwise_correlation_nchw(search.data(), kernel.data(), output1.data());
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_baseline = std::chrono::duration<double, std::milli>(t2 - t1).count() / iterations;
    std::cout << "1. Baseline (Single-Thread, NCHW): " << time_baseline << " ms\n";

    // 2. Benchmark OpenMP
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        depthwise_correlation_nchw_omp(search.data(), kernel.data(), output2.data());
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double time_omp = std::chrono::duration<double, std::milli>(t4 - t3).count() / iterations;
    std::cout << "2. OpenMP (Multi-Thread, NCHW):     " << time_omp << " ms\n";

    // 3. Benchmark Padded + Unrolled ST
    auto t5 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        depthwise_correlation_nchw_padded(search.data(), kernel.data(), output3.data());
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    double time_padded = std::chrono::duration<double, std::milli>(t6 - t5).count() / iterations;
    std::cout << "3. Padded + Unrolled (Single-Thread): " << time_padded << " ms\n";

    // 4. Benchmark Padded + Unrolled OpenMP
    auto t7 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        depthwise_correlation_nchw_padded_omp(search.data(), kernel.data(), output4.data());
    }
    auto t8 = std::chrono::high_resolution_clock::now();
    double time_padded_omp = std::chrono::duration<double, std::milli>(t8 - t7).count() / iterations;
    std::cout << "4. Padded + Unrolled + OpenMP:        " << time_padded_omp << " ms\n";

    // 5. Benchmark NEON + OpenMP
    auto t9 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        depthwise_correlation_nchw_neon_omp(search.data(), kernel.data(), output5.data());
    }
    auto t10 = std::chrono::high_resolution_clock::now();
    double time_neon_omp = std::chrono::duration<double, std::milli>(t10 - t9).count() / iterations;
    std::cout << "5. NEON SIMD + OpenMP:                " << time_neon_omp << " ms\n";

    std::cout << "====================================================\n";
    return 0;
}
