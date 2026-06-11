#include "core/TrackerService.h"
#include "utils/StatusObject.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    // Dynamic mapping of dimensions based on raw queried NPU attributes (supporting 4D/5D)
    if (in_attrs[0].n_dims == 5)
    {
        m_in_width_ref = in_attrs[0].dims[3];
        m_in_height_ref = in_attrs[0].dims[2];
        m_in_channels_ref = in_attrs[0].dims[4];
    }
    else
    {
        m_in_width_ref = in_attrs[0].dims[3];
        m_in_height_ref = in_attrs[0].dims[2];
        m_in_channels_ref = in_attrs[0].dims[0] * in_attrs[0].dims[1];
    }

    if (in_attrs[1].n_dims == 4 && in_attrs[1].dims[1] == 256 && in_attrs[1].dims[3] == 256)
    {
        m_in_width_search = in_attrs[1].dims[3];
        m_in_height_search = in_attrs[1].dims[1];
        m_in_channels_search = in_attrs[1].dims[2];
    }
    else
    {
        m_in_width_search = in_attrs[1].dims[3];
        m_in_height_search = in_attrs[1].dims[2];
        m_in_channels_search = in_attrs[1].dims[1];
    }

    if (out_attrs[0].n_dims == 4 && out_attrs[0].dims[1] == 256 && out_attrs[0].dims[2] == 256)
    {
        m_out_width_hm = out_attrs[0].dims[2];
        m_out_height_hm = out_attrs[0].dims[1];
    }
    else
    {
        m_out_width_hm = out_attrs[0].dims[3];
        m_out_height_hm = out_attrs[0].dims[2];
    }

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

    m_bfs_visited = (bool*)malloc(m_out_width_hm * m_out_height_hm);
    m_bfs_queue_x = (int*)malloc(m_out_width_hm * m_out_height_hm * sizeof(int));
    m_bfs_queue_y = (int*)malloc(m_out_width_hm * m_out_height_hm * sizeof(int));
}

TrackerService::~TrackerService()
{
    if (m_is_model_loaded)
    {
        rknn_destroy(m_ctx);
        free(m_ref_stack_buf);
        free(m_search_buf);
        free(m_heatmap_buf);
        free(m_bfs_visited);
        free(m_bfs_queue_x);
        free(m_bfs_queue_y);
    }
}

bool TrackerService::is_model_loaded() const
{
    return m_is_model_loaded;
}

void TrackerService::set_tracker_callback(TrackerCallback* cb)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callback = cb;
}

void TrackerService::refresh_target(uchar* frame, int w, int h)
{
    int c;
    uchar* temp_crop;

    if (!m_is_model_loaded)
    {
        return;
    }

    // Allocate a temporary crop buffer
    temp_crop = (uchar*)malloc(m_in_width_ref * m_in_height_ref);

    // Rescale input frame into the target reference resolution (using Nearest-Neighbor as configured)
    resize_nearest_gray(frame, w, h, temp_crop, m_in_width_ref, m_in_height_ref);

    std::lock_guard<std::mutex> lock(m_mutex);
    // Fill the historical reference stack with the newly initialized target template
    for (c = 0; c < m_in_channels_ref; ++c)
    {
        memcpy(m_ref_stack_buf + (c * m_in_width_ref * m_in_height_ref), temp_crop, m_in_width_ref * m_in_height_ref);
    }

    free(temp_crop);
    fprintf(stdout, "[TrackerService] Target refreshed & initialized.\n");
}

void TrackerService::update_frame(uchar* frame, int w, int h)
{
    rknn_input inputs[2];
    rknn_output outputs[1];
    int ret;
    int out_x;
    int out_y;

    if (!m_is_model_loaded)
    {
        return;
    }

    // 1. Resize incoming frame to search window size (using Bilinear Interpolation)
    resize_bilinear_gray(frame, w, h, m_search_buf, m_in_width_search, m_in_height_search);

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
    ret = rknn_run(m_ctx, NULL);
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
    out_x = -1;
    out_y = -1;
    decode_heatmap(m_heatmap_buf, &out_x, &out_y);

    // 6. Trigger TrackerCallback
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_callback != NULL)
        {
            m_callback->onTargetDetected(out_x, out_y);
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
    int y, x, i;
    int dy, dx, ny, nx;
    int q_head, q_tail;
    int q_x, q_y;
    int cy, cx;
    int blob_size;
    int max_flat_idx;
    float max_val;
    int y_max, x_max;
    int half_w;
    int y_start, y_end, x_start, x_end;
    float total_mass;
    float x_sum, y_sum;

    // 1. Copy and apply Threshold gate (0.5 threshold noise gate)
    for (i = 0; i < m_out_width_hm * m_out_height_hm; ++i)
    {
        if (raw_heatmap[i] >= 0.5f)
        {
            m_heatmap_buf[i] = raw_heatmap[i];
        }
        else
        {
            m_heatmap_buf[i] = 0.0f;
        }
    }

    // 2. Connected Component (Blob Size) Filter using BFS
    memset(m_bfs_visited, 0, m_out_width_hm * m_out_height_hm);
    for (y = 0; y < m_out_height_hm; ++y)
    {
        for (x = 0; x < m_out_width_hm; ++x)
        {
            if (m_heatmap_buf[y * m_out_width_hm + x] > 0.0f && !m_bfs_visited[y * m_out_width_hm + x])
            {
                // Queue start
                q_head = 0;
                q_tail = 0;

                m_bfs_queue_x[q_tail] = x;
                m_bfs_queue_y[q_tail] = y;
                m_bfs_visited[y * m_out_width_hm + x] = true;
                q_tail++;

                while (q_head < q_tail)
                {
                    q_x = m_bfs_queue_x[q_head];
                    q_y = m_bfs_queue_y[q_head];
                    q_head++;

                    // Check 8 neighbors
                    for (dy = -1; dy <= 1; ++dy)
                    {
                        for (dx = -1; dx <= 1; ++dx)
                        {
                            if (dy == 0 && dx == 0)
                            {
                                continue;
                            }
                            ny = q_y + dy;
                            nx = q_x + dx;

                            if (ny >= 0 && ny < m_out_height_hm && nx >= 0 && nx < m_out_width_hm)
                            {
                                if (m_heatmap_buf[ny * m_out_width_hm + nx] > 0.0f && !m_bfs_visited[ny * m_out_width_hm + nx])
                                {
                                    m_bfs_visited[ny * m_out_width_hm + nx] = true;
                                    m_bfs_queue_x[q_tail] = nx;
                                    m_bfs_queue_y[q_tail] = ny;
                                    q_tail++;
                                }
                            }
                        }
                    }
                }

                // If component blob contains fewer than 30 pixels, zero them out
                blob_size = q_tail;
                if (blob_size < 30)
                {
                    for (i = 0; i < blob_size; ++i)
                    {
                        cx = m_bfs_queue_x[i];
                        cy = m_bfs_queue_y[i];
                        m_heatmap_buf[cy * m_out_width_hm + cx] = 0.0f;
                    }
                }
            }
        }
    }

    // 3. Find global maximum peak in filtered heatmap
    max_flat_idx = 0;
    max_val = m_heatmap_buf[0];
    for (i = 1; i < m_out_width_hm * m_out_height_hm; ++i)
    {
        if (m_heatmap_buf[i] > max_val)
        {
            max_val = m_heatmap_buf[i];
            max_flat_idx = i;
        }
    }

    // If no signal survives, return tracker lost (-1, -1)
    if (max_val <= 0.0001f)
    {
        *out_x = -1;
        *out_y = -1;
        return;
    }

    y_max = max_flat_idx / m_out_width_hm;
    x_max = max_flat_idx % m_out_width_hm;

    // 4. Local 5x5 sub-pixel centroid calculation
    half_w = 2;
    y_start = y_max - half_w;
    if (y_start < 0)
    {
        y_start = 0;
    }
    y_end = y_max + half_w + 1;
    if (y_end > m_out_height_hm)
    {
        y_end = m_out_height_hm;
    }

    x_start = x_max - half_w;
    if (x_start < 0)
    {
        x_start = 0;
    }
    x_end = x_max + half_w + 1;
    if (x_end > m_out_width_hm)
    {
        x_end = m_out_width_hm;
    }

    total_mass = 0.0f;
    x_sum = 0.0f;
    y_sum = 0.0f;

    for (cy = y_start; cy < y_end; ++cy)
    {
        for (cx = x_start; cx < x_end; ++cx)
        {
            float w = m_heatmap_buf[cy * m_out_width_hm + cx];
            total_mass += w;
            x_sum += cx * w;
            y_sum += cy * w;
        }
    }

    if (total_mass > 0.000001f)
    {
        *out_x = (int)(x_sum / total_mass);
        *out_y = (int)(y_sum / total_mass);
    }
    else
    {
        *out_x = x_max;
        *out_y = y_max;
    }
}
