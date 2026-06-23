#ifndef NN_OPERATIONS_CPU_HPP
#define NN_OPERATIONS_CPU_HPP

#include <arm_neon.h>
#include <omp.h>
#include <algorithm>
#include <vector>

namespace NNOperationsCpu {
    template <typename T>
    inline void transpose_nchw_to_nhwc(const T* src, T* dst, int N, int C, int H, int W) {
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int src_idx = n * C * H * W + c * H * W + h * W + w;
                        int dst_idx = n * H * W * C + h * W * C + w * C + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }

    template <typename T>
    inline void transpose_nhwc_to_nchw(const T* src, T* dst, int N, int C, int H, int W) {
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        int src_idx = n * H * W * C + h * W * C + w * C + c;
                        int dst_idx = n * C * H * W + c * H * W + h * W + w;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }

    inline void depthwise_correlation_nchw_neon_omp(
        const float* search,     // Shape: [channels, search_h, search_w]
        const float* kernel,     // Shape: [channels, kernel_h, kernel_w]
        float* output,           // Shape: [channels, search_h, search_w]
        int channels,
        int search_h,
        int search_w,
        int kernel_h,
        int kernel_w
    ) {
        int padded_h = search_h + kernel_h - 1;
        int padded_w = (search_w + kernel_w - 1 + 3) & ~3;
        int pad_top = (kernel_h - 1) / 2;
        int pad_left = (kernel_w - 1) / 2;

        #pragma omp parallel
        {
            // Thread-local scratch buffer allocated once per thread to avoid allocation overhead
            std::vector<float> padded_search(padded_h * padded_w, 0.0f);

            #pragma omp for
            for (int c = 0; c < channels; ++c) {
                const float* s_chan = search + c * search_h * search_w;
                const float* k_chan = kernel + c * kernel_h * kernel_w;
                float* o_chan = output + c * search_h * search_w;

                // Zero-initialize the padded buffer
                std::fill(padded_search.begin(), padded_search.end(), 0.0f);

                // Copy search feature map to the middle of the padded buffer
                for (int y = 0; y < search_h; ++y) {
                    std::copy_n(s_chan + y * search_w, search_w, padded_search.data() + (y + pad_top) * padded_w + pad_left);
                }

                // Compute convolution
                for (int y = 0; y < search_h; ++y) {
                    for (int x = 0; x < search_w; ++x) {
                        float32x4_t sum_vec = vdupq_n_f32(0.0f);
                        float remainder_sum = 0.0f;

                        for (int ky = 0; ky < kernel_h; ++ky) {
                            const float* s_row = padded_search.data() + (y + ky) * padded_w + x;
                            const float* k_row = k_chan + ky * kernel_w;

                            int kx = 0;
                            for (; kx <= kernel_w - 4; kx += 4) {
                                float32x4_t s_val = vld1q_f32(s_row + kx);
                                float32x4_t k_val = vld1q_f32(k_row + kx);
                                sum_vec = vmlaq_f32(sum_vec, s_val, k_val);
                            }
                            for (; kx < kernel_w; ++kx) {
                                remainder_sum += s_row[kx] * k_row[kx];
                            }
                        }

                        float sum = vgetq_lane_f32(sum_vec, 0) + vgetq_lane_f32(sum_vec, 1) +
                                    vgetq_lane_f32(sum_vec, 2) + vgetq_lane_f32(sum_vec, 3);
                        o_chan[y * search_w + x] = sum + remainder_sum;
                    }
                }
            }
        }
    }
}

#endif // NN_OPERATIONS_CPU_HPP
