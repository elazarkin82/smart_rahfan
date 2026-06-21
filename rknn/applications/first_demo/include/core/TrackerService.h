#ifndef TRACKER_SERVICE_H
#define TRACKER_SERVICE_H

#include <mutex>
#include "rknn_api.h"
#if defined(USE_RGA)
#include <RgaApi.h>
#include <im2d.h>
#endif

typedef unsigned char uchar;

class TrackerService
{
public:
    class TrackerCallback
    {
    public:
        virtual ~TrackerCallback()
        {
        }
        virtual void onTargetDetected(int x, int y) = 0;
        virtual void onHeatmapCreated(const float* heatmap, int w, int h) = 0;
        virtual void onStackCreated(const uchar* stack, int w, int h, int c) = 0;
    };


private:
    rknn_context m_ctx_template;
    rknn_context m_ctx_search_backbone;
    rknn_context m_ctx_decoder;
    bool m_is_model_loaded;
    bool m_is_target_defined;
    TrackerCallback* m_callback;
    std::mutex m_mutex;

    // Model tensor attributes
    int m_in_width_ref;
    int m_in_height_ref;
    int m_in_channels_ref;

    int m_in_width_search;
    int m_in_height_search;
    int m_in_channels_search;

    int m_out_width_hm;
    int m_out_height_hm;

    // Crop boundaries in absolute pixels
    float m_min_crop;
    float m_max_crop;
    bool m_quality_enabled;

    // Resolved tensor indices
    int m_idx_skip3;
    int m_idx_search_features;
    int m_idx_dec_corr;
    int m_idx_dec_skip3;
    int m_backbone_out_num;

    // Resolved tensor formats
    rknn_tensor_format m_fmt_skip3;
    rknn_tensor_format m_fmt_search_features;
    rknn_tensor_format m_fmt_template;

    // Pre-allocated buffers to prevent runtime heap allocation (MISRA-compliant fixed-size arrays)
    uchar m_ref_stack_buf[MAX_STACK_TARGET_SIZE * MAX_STACK_TARGET_SIZE * MAX_STACK_LAYERS];
    uchar m_search_buf[MAX_STACK_TARGET_SIZE * MAX_STACK_TARGET_SIZE];
    float m_heatmap_buf[MAX_HEATMAP_PXL_SIZE * MAX_HEATMAP_PXL_SIZE];
    float m_template_features[64 * 8 * 8];
    float m_template_features_nchw[64 * 8 * 8];
    float m_search_features_nchw[64 * 16 * 16];
    float m_corr_out_buf[64 * 16 * 16];
    float m_corr_out_nhwc[64 * 16 * 16];
    float m_skip3_nhwc[32 * 32 * 32];

    // Resizing helpers
    void resize_bilinear_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);
    void resize_center_square_bilinear_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);
    void resize_nearest_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);
#if defined(USE_RGA)
    void resize_rga(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);
    bool init_rga_buffers(int src_w, int src_h);
    void release_rga_buffers();
    // DMA heap buffer state for RGA hardware acceleration
    bool                m_rga_initialized;
    int                 m_rga_src_w;
    int                 m_rga_src_h;
    int                 m_rga_src_fd;
    int                 m_rga_dst_fd;
    uchar*              m_rga_src_va;
    uchar*              m_rga_dst_va;
    rga_buffer_handle_t m_rga_src_handle;
    rga_buffer_handle_t m_rga_dst_handle;
    rga_buffer_t        m_rga_src_buf;
    rga_buffer_t        m_rga_dst_buf;
#elif defined(USE_OMP)
    void resize_bilinear_omp(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);
#endif

    // Heatmap post-processing
    void decode_heatmap(const float* raw_heatmap, int* out_x, int* out_y);

public:
    TrackerService(const char* template_path, const char* search_backbone_path, const char* decoder_path, float min_crop, float max_crop, bool quality_enabled);
    ~TrackerService();

    bool is_model_loaded() const;
    bool is_target_defined() const;
    void refresh_target(const uchar* frame, int w, int h, int target_x, int target_y);
    void clear_target();
    void set_tracker_callback(TrackerCallback* cb);
    void update_frame(uchar* frame, int w, int h);
};

#endif
