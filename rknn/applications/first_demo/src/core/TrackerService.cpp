#include "core/TrackerService.h"
#include "utils/StatusObject.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <math.h>

TrackerService::TrackerService(const char* model_path)
{
    FILE* fp;
    long model_size;
    void* model_data;
    int ret;
    rknn_input_output_num io_num;
    rknn_tensor_attr in_attrs[2];
    rknn_tensor_attr out_attrs[2];

    m_ctx = 0;
    m_is_model_loaded = false;
    m_is_target_defined = false;
    m_callback = NULL;

    m_in_width_ref = 32;
    m_in_height_ref = 32;
    m_in_channels_ref = 16;

    m_in_width_search = 256;
    m_in_height_search = 256;
    m_in_channels_search = 1;

    m_out_width_hm = 256;
    m_out_height_hm = 256;

    // Load model file into memory
    fp = fopen(model_path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "[TrackerService] Failed to open model file: %s\n", model_path);
        StatusObject::instance()->update("tracker_model_status", "Error: File Not Found");
        return;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    model_data = malloc(model_size);
    if (model_data == NULL)
    {
        fclose(fp);
        fprintf(stderr, "[TrackerService] Failed to allocate memory for model buffer.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: OOM on Load");
        return;
    }

    if (fread(model_data, 1, model_size, fp) != (size_t)model_size)
    {
        free(model_data);
        fclose(fp);
        fprintf(stderr, "[TrackerService] Failed to read model file contents.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: Read Failure");
        return;
    }
    fclose(fp);

    // Initialize RKNN NPU context
    ret = rknn_init(&m_ctx, model_data, model_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_init failed with error code: %d\n", ret);
        StatusObject::instance()->update("tracker_model_status", "Error: NPU Init Failed");
        return;
    }

    // Query inputs & outputs attributes
    ret = rknn_query(m_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] Failed to query model IO numbers: %d\n", ret);
        StatusObject::instance()->update("tracker_model_status", "Error: Query Failed");
        return;
    }

    if (io_num.n_input < 2 || io_num.n_output < 1)
    {
        fprintf(stderr, "[TrackerService] Invalid model architecture: needs 2 inputs, 1+ outputs.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: Invalid RKNN Architecture");
        return;
    }

    // Query dynamic tensor dimensions
    memset(in_attrs, 0, sizeof(in_attrs));
    in_attrs[0].index = 0;
    rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[0], sizeof(rknn_tensor_attr));
    in_attrs[1].index = 1;
    rknn_query(m_ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[1], sizeof(rknn_tensor_attr));

    memset(out_attrs, 0, sizeof(out_attrs));
    out_attrs[0].index = 0;
    rknn_query(m_ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[0], sizeof(rknn_tensor_attr));

    // Directly parse input/output dimensions assuming HWC layout
    m_in_height_ref = in_attrs[0].dims[1];
    m_in_width_ref = in_attrs[0].dims[2];
    m_in_channels_ref = in_attrs[0].dims[3];

    m_in_height_search = in_attrs[1].dims[1];
    m_in_width_search = in_attrs[1].dims[2];
    m_in_channels_search = in_attrs[1].dims[3];

    m_out_height_hm = out_attrs[0].dims[1];
    m_out_width_hm = out_attrs[0].dims[2];

    fprintf(stdout, "[DEBUG] Input 0 name: %s, n_dims: %d, dims: [%d, %d, %d, %d], size: %d, fmt: %d, type: %d\n",
            in_attrs[0].name, in_attrs[0].n_dims, in_attrs[0].dims[0], in_attrs[0].dims[1], in_attrs[0].dims[2], in_attrs[0].dims[3],
            in_attrs[0].size, in_attrs[0].fmt, in_attrs[0].type);
    fprintf(stdout, "[DEBUG] Input 1 name: %s, n_dims: %d, dims: [%d, %d, %d, %d], size: %d, fmt: %d, type: %d\n",
            in_attrs[1].name, in_attrs[1].n_dims, in_attrs[1].dims[0], in_attrs[1].dims[1], in_attrs[1].dims[2], in_attrs[1].dims[3],
            in_attrs[1].size, in_attrs[1].fmt, in_attrs[1].type);
    fprintf(stdout, "[DEBUG] Output 0 name: %s, n_dims: %d, dims: [%d, %d, %d, %d], size: %d, fmt: %d, type: %d\n",
            out_attrs[0].name, out_attrs[0].n_dims, out_attrs[0].dims[0], out_attrs[0].dims[1], out_attrs[0].dims[2], out_attrs[0].dims[3],
            out_attrs[0].size, out_attrs[0].fmt, out_attrs[0].type);

    fprintf(stdout, "[TrackerService] Loaded RKNN model successfully.\n");
    fprintf(stdout, " - Input 0 (Reference): %dx%dx%d\n", m_in_width_ref, m_in_height_ref, m_in_channels_ref);
    fprintf(stdout, " - Input 1 (Search): %dx%dx%d\n", m_in_width_search, m_in_height_search, m_in_channels_search);
    fprintf(stdout, " - Output 0 (Heatmap): %dx%d\n", m_out_width_hm, m_out_height_hm);

    m_is_model_loaded = true;
    StatusObject::instance()->update("tracker_model_status", "Loaded & Ready");

    // Allocate runtime internal buffers
    m_ref_stack_buf = (uchar*)malloc(m_in_width_ref * m_in_height_ref * m_in_channels_ref);
    memset(m_ref_stack_buf, 0, m_in_width_ref * m_in_height_ref * m_in_channels_ref);

    m_search_buf = (uchar*)malloc(m_in_width_search * m_in_height_search * m_in_channels_search);
    m_heatmap_buf = (float*)malloc(m_out_width_hm * m_out_height_hm * sizeof(float));


}

TrackerService::~TrackerService()
{
    if (m_is_model_loaded)
    {
        rknn_destroy(m_ctx);
        free(m_ref_stack_buf);
        free(m_search_buf);
        free(m_heatmap_buf);

    }
}

bool TrackerService::is_model_loaded() const
{
    return m_is_model_loaded;
}

bool TrackerService::is_target_defined() const
{
    return m_is_target_defined;
}

void TrackerService::clear_target()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_is_target_defined = false;
}

void TrackerService::set_tracker_callback(TrackerCallback* cb)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callback = cb;
}

void TrackerService::refresh_target(const uchar* frame, int w, int h, int target_x, int target_y)
{
    int c;
    float max_sz;
    float min_sz;
    int y, x;
    uchar* temp_64x64;

    if (!m_is_model_loaded)
    {
        return;
    }

    max_sz = (float)(w < h ? w : h);
    min_sz = 16.0f;
    temp_64x64 = (uchar*)malloc(m_in_width_ref * m_in_height_ref);

    for (c = 0; c < m_in_channels_ref; ++c)
    {
        float sz = max_sz - (c * (max_sz - min_sz) / (m_in_channels_ref - 1));
        float half = sz / 2.0f;
        int x1 = (int)roundf((float)target_x - half);
        int y1 = (int)roundf((float)target_y - half);
        int sz_int = (int)sz;
        uchar* crop_buf = (uchar*)calloc(sz_int * sz_int, 1);
        int cy, cx;

        for (cy = 0; cy < sz_int; ++cy)
        {
            int sy = y1 + cy;
            if (sy >= 0 && sy < h)
            {
                for (cx = 0; cx < sz_int; ++cx)
                {
                    int sx = x1 + cx;
                    if (sx >= 0 && sx < w)
                    {
                        crop_buf[cy * sz_int + cx] = frame[sy * w + sx];
                    }
                }
            }
        }

        resize_bilinear_gray(crop_buf, sz_int, sz_int, temp_64x64, m_in_width_ref, m_in_height_ref);
        
        for (y = 0; y < m_in_height_ref; ++y)
        {
            for (x = 0; x < m_in_width_ref; ++x)
            {
                m_ref_stack_buf[(y * m_in_width_ref + x) * m_in_channels_ref + c] = temp_64x64[y * m_in_width_ref + x];
            }
        }

        free(crop_buf);
    }

    free(temp_64x64);

    std::lock_guard<std::mutex> lock(m_mutex);
    m_is_target_defined = true;
    if (m_callback != NULL)
    {
        m_callback->onStackCreated(m_ref_stack_buf, m_in_width_ref, m_in_height_ref, m_in_channels_ref);
    }
    fprintf(stdout, "[TrackerService] Target refreshed & initialized with multi-scale reference stack.\n");
}

void TrackerService::update_frame(uchar* frame, int w, int h)
{
    rknn_input inputs[2];
    rknn_output outputs[1];
    int ret;
    int out_x;
    int out_y;
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_resize_start;
    std::chrono::steady_clock::time_point t_resize_end;
    std::chrono::steady_clock::time_point t_npu_start;
    std::chrono::steady_clock::time_point t_npu_end;
    std::chrono::steady_clock::time_point t_decode_start;
    std::chrono::steady_clock::time_point t_decode_end;
    std::chrono::steady_clock::time_point t_end;
    float resize_ms;
    float npu_ms;
    float decode_ms;
    float total_ms;
    char time_buf[64];

    if (!m_is_model_loaded || !m_is_target_defined)
    {
        return;
    }

    t_start = std::chrono::steady_clock::now();

    // 1. Resize incoming frame to search window size (using Bilinear Interpolation)
    t_resize_start = std::chrono::steady_clock::now();
    resize_bilinear_gray(frame, w, h, m_search_buf, m_in_width_search, m_in_height_search);
    t_resize_end = std::chrono::steady_clock::now();

    // 2. Setup inputs
    memset(inputs, 0, sizeof(inputs));
    // Input 0: Reference Stack
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_in_width_ref * m_in_height_ref * m_in_channels_ref;
    inputs[0].buf = m_ref_stack_buf;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    // Input 1: Search Frame
    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_UINT8;
    inputs[1].size = m_in_width_search * m_in_height_search * m_in_channels_search;
    inputs[1].buf = m_search_buf;
    inputs[1].fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(m_ctx, 2, inputs);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_inputs_set failed: %d\n", ret);
        return;
    }

    // 3. Run inference on NPU
    t_npu_start = std::chrono::steady_clock::now();
    ret = rknn_run(m_ctx, NULL);
    t_npu_end = std::chrono::steady_clock::now();
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_run failed: %d\n", ret);
        return;
    }

    // 4. Retrieve outputs
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // get floating point values for heatmap post-processing

    ret = rknn_outputs_get(m_ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_outputs_get failed: %d\n", ret);
        return;
    }

    // Copy raw output heatmap buffer
    memcpy(m_heatmap_buf, outputs[0].buf, m_out_width_hm * m_out_height_hm * sizeof(float));

    // Release outputs immediately
    rknn_outputs_release(m_ctx, 1, outputs);

    // 5. Decode heatmap using local 5x5 sub-pixel centroid
    t_decode_start = std::chrono::steady_clock::now();
    out_x = -1;
    out_y = -1;
    decode_heatmap(m_heatmap_buf, &out_x, &out_y);
    t_decode_end = std::chrono::steady_clock::now();

    t_end = std::chrono::steady_clock::now();

    // Calculate times in milliseconds
    resize_ms = std::chrono::duration<float, std::milli>(t_resize_end - t_resize_start).count();
    npu_ms = std::chrono::duration<float, std::milli>(t_npu_end - t_npu_start).count();
    decode_ms = std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();
    total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Update StatusObject
    snprintf(time_buf, sizeof(time_buf), "%.2f ms", resize_ms);
    StatusObject::instance()->update("tracker_time_resize", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms", npu_ms);
    StatusObject::instance()->update("tracker_time_npu", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms", decode_ms);
    StatusObject::instance()->update("tracker_time_decode", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms", total_ms);
    StatusObject::instance()->update("tracker_time_total", time_buf);

    // 6. Trigger TrackerCallback
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_callback != NULL)
        {
            m_callback->onTargetDetected(out_x, out_y);
            m_callback->onHeatmapCreated(m_heatmap_buf, m_out_width_hm, m_out_height_hm);
        }
    }
}

void TrackerService::resize_bilinear_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h)
{
    int x, y;
    float x_ratio;
    float y_ratio;
    int x_l, y_l, x_h, y_h;
    float x_weight, y_weight;
    uchar a, b, c, d;

    x_ratio = ((float)(src_w - 1)) / dst_w;
    y_ratio = ((float)(src_h - 1)) / dst_h;

    for (y = 0; y < dst_h; ++y)
    {
        for (x = 0; x < dst_w; ++x)
        {
            x_l = (int)(x_ratio * x);
            y_l = (int)(y_ratio * y);
            x_h = x_l + 1;
            y_h = y_l + 1;

            if (x_h >= src_w)
            {
                x_h = src_w - 1;
            }
            if (y_h >= src_h)
            {
                y_h = src_h - 1;
            }

            x_weight = (x_ratio * x) - x_l;
            y_weight = (y_ratio * y) - y_l;

            a = src[y_l * src_w + x_l];
            b = src[y_l * src_w + x_h];
            c = src[y_h * src_w + x_l];
            d = src[y_h * src_w + x_h];

            dst[y * dst_w + x] = (uchar)(
                a * (1.0f - x_weight) * (1.0f - y_weight) +
                b * x_weight * (1.0f - y_weight) +
                c * (1.0f - x_weight) * y_weight +
                d * x_weight * y_weight
            );
        }
    }
}

void TrackerService::resize_nearest_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h)
{
    int x, y;
    int src_x, src_y;

    for (y = 0; y < dst_h; ++y)
    {
        src_y = (y * src_h) / dst_h;
        for (x = 0; x < dst_w; ++x)
        {
            src_x = (x * src_w) / dst_w;
            dst[y * dst_w + x] = src[src_y * src_w + src_x];
        }
    }
}

void TrackerService::decode_heatmap(const float* raw_heatmap, int* out_x, int* out_y)
{
    int max_flat_idx = 0;
    float max_val = raw_heatmap[0];
    for (int i = 1; i < m_out_width_hm * m_out_height_hm; ++i)
    {
        if (raw_heatmap[i] > max_val)
        {
            max_val = raw_heatmap[i];
            max_flat_idx = i;
        }
    }

    *out_y = max_flat_idx / m_out_width_hm;
    *out_x = max_flat_idx % m_out_width_hm;
}
