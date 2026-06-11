#ifndef TRACKER_SERVICE_H
#define TRACKER_SERVICE_H

#include <mutex>
#include "rknn_api.h"

typedef unsigned char uchar;

class TrackerService
{
public:
    class TrackerCallback
    {
    public:
        virtual ~TrackerCallback() {}
        virtual void onTargetDetected(int x, int y) = 0;
    };

private:
    rknn_context m_ctx;
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

    // Pre-allocated buffers to prevent runtime heap allocation
    uchar* m_ref_stack_buf;
    uchar* m_search_buf;
    float* m_heatmap_buf;
    bool* m_bfs_visited;
    int* m_bfs_queue_x;
    int* m_bfs_queue_y;

    // Resizing helpers
    void resize_bilinear_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);
    void resize_nearest_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h);

    // Heatmap post-processing
    void decode_heatmap(const float* raw_heatmap, int* out_x, int* out_y);

public:
    TrackerService(const char* model_path);
    ~TrackerService();

    bool is_model_loaded() const;
    bool is_target_defined() const;
    void refresh_target(uchar* frame, int w, int h);
    void set_tracker_callback(TrackerCallback* cb);
    void update_frame(uchar* frame, int w, int h);
};

#endif
