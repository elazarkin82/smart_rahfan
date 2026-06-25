#include "core/TrackerService.h"
#include "utils/StatusObject.hpp"
#include "utils/NNOperationsCpu.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <math.h>
#include <utility>

#if defined(USE_RGA)
#include <RgaApi.h>
#include <im2d.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <linux/dma-heap.h>
#endif

static int tracker_min_int(int a, int b)
{
    return (a < b) ? a : b;
}

static int tracker_max_int(int a, int b)
{
    return (a > b) ? a : b;
}

static int tracker_clamp_int(int value, int min_value, int max_value)
{
    int result;

    result = value;
    if (result < min_value)
    {
        result = min_value;
    }
    else if (result > max_value)
    {
        result = max_value;
    }

    return result;
}

static double tracker_max_double(double a, double b)
{
    return (a > b) ? a : b;
}

TrackerService::TrackerService(const char* model_path, float min_crop, float max_crop, bool quality_enabled, bool use_argmax_only, int iterations_num)
{
    FILE* fp;
    long model_size;
    void* model_data;
    int ret;
    int ref_idx;
    int search_idx;
    int hm_idx;
    int q_idx;
    uint32_t i;
    rknn_input_output_num io_num;
    rknn_tensor_attr attr;
    rknn_tensor_attr ref_attr;
    rknn_tensor_attr search_attr;
    rknn_tensor_attr hm_attr;

    m_ctx_model = 0;
    m_is_model_loaded = false;
    m_is_target_defined = false;
    m_callback = NULL;
    m_drone_cb = NULL;

    m_min_crop = min_crop;
    m_max_crop = max_crop;
    m_quality_enabled = quality_enabled;
    m_use_argmax_only = use_argmax_only;
    m_iterations_num = iterations_num;
    if (m_iterations_num < 1)
    {
        m_iterations_num = 1;
    }
    else if (m_iterations_num > MAX_TRACKER_ITERATIONS)
    {
        m_iterations_num = MAX_TRACKER_ITERATIONS;
    }
#if defined(USE_RGA)
    m_rga_initialized = false;
    m_rga_src_w      = 0;
    m_rga_src_h      = 0;
    m_rga_src_fd     = -1;
    m_rga_dst_fd     = -1;
    m_rga_src_va     = NULL;
    m_rga_dst_va     = NULL;
    m_rga_src_handle = 0;
    m_rga_dst_handle = 0;
#endif

    m_in_width_ref = 128;
    m_in_height_ref = 128;
    m_in_channels_ref = 2;

    m_in_width_search = 256;
    m_in_height_search = 256;
    m_in_channels_search = 1;

    m_out_width_hm = 32;
    m_out_height_hm = 32;

    // Load model file into memory
    fp = fopen(model_path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "[TrackerService] Failed to open model file: %s\n", model_path);
        StatusObject::instance()->update("tracker_model_status", "Error: Model File Not Found");
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

    // Initialize RKNN context for unified model
    ret = rknn_init(&m_ctx_model, model_data, model_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_init failed: %d\n", ret);
        StatusObject::instance()->update("tracker_model_status", "Error: NPU Init Failed");
        return;
    }

    // Query dynamic tensor info
    ret = rknn_query(m_ctx_model, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_query RKNN_QUERY_IN_OUT_NUM failed: %d\n", ret);
        rknn_destroy(m_ctx_model);
        StatusObject::instance()->update("tracker_model_status", "Error: Query Failed");
        return;
    }

    if (io_num.n_input != 2 || io_num.n_output != 2)
    {
        fprintf(stderr, "[TrackerService] Expected 2 inputs and 2 outputs. Found inputs=%d outputs=%d\n", io_num.n_input, io_num.n_output);
        rknn_destroy(m_ctx_model);
        StatusObject::instance()->update("tracker_model_status", "Error: IO Count Mismatch");
        return;
    }

    // Map input tensors dynamically by name or shape
    ref_idx = -1;
    search_idx = -1;
    for (i = 0; i < io_num.n_input; ++i)
    {
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(m_ctx_model, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        if (strstr(attr.name, "reference_stack") != NULL)
        {
            ref_idx = (int)i;
        }
        else if (strstr(attr.name, "search_frame") != NULL)
        {
            search_idx = (int)i;
        }
    }
    if (ref_idx == -1 || search_idx == -1)
    {
        fprintf(stdout, "[WARNING] Dynamic input name matching failed. Falling back to default order.\n");
        ref_idx = 0;
        search_idx = 1;
    }
    m_idx_ref_stack = ref_idx;
    m_idx_search_frame = search_idx;

    // Map output tensors dynamically by element count
    hm_idx = -1;
    q_idx = -1;
    for (i = 0; i < io_num.n_output; ++i)
    {
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(m_ctx_model, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (attr.n_elems == 1)
        {
            q_idx = (int)i;
        }
        else
        {
            hm_idx = (int)i;
        }
    }
    if (hm_idx == -1 || q_idx == -1)
    {
        fprintf(stdout, "[WARNING] Dynamic output mapping failed. Falling back to default order.\n");
        hm_idx = 0;
        q_idx = 1;
    }
    m_idx_heatmap = hm_idx;
    m_idx_quality = q_idx;

    // Query attributes to extract shapes
    memset(&ref_attr, 0, sizeof(ref_attr));
    ref_attr.index = m_idx_ref_stack;
    rknn_query(m_ctx_model, RKNN_QUERY_INPUT_ATTR, &ref_attr, sizeof(ref_attr));

    memset(&search_attr, 0, sizeof(search_attr));
    search_attr.index = m_idx_search_frame;
    rknn_query(m_ctx_model, RKNN_QUERY_INPUT_ATTR, &search_attr, sizeof(search_attr));

    memset(&hm_attr, 0, sizeof(hm_attr));
    hm_attr.index = m_idx_heatmap;
    rknn_query(m_ctx_model, RKNN_QUERY_OUTPUT_ATTR, &hm_attr, sizeof(hm_attr));

    // Reference stack shape extraction (dynamic)
    if (ref_attr.dims[1] == 1 || ref_attr.dims[1] == 2 || ref_attr.dims[1] == 16)
    {
        m_in_height_ref = ref_attr.dims[2];
        m_in_width_ref = ref_attr.dims[3];
        m_in_channels_ref = ref_attr.dims[1];
    }
    else
    {
        m_in_height_ref = ref_attr.dims[1];
        m_in_width_ref = ref_attr.dims[2];
        m_in_channels_ref = ref_attr.dims[3];
    }

    // Search frame shape extraction (dynamic)
    if (search_attr.dims[1] == 1)
    {
        m_in_height_search = search_attr.dims[2];
        m_in_width_search = search_attr.dims[3];
        m_in_channels_search = search_attr.dims[1];
    }
    else
    {
        m_in_height_search = search_attr.dims[1];
        m_in_width_search = search_attr.dims[2];
        m_in_channels_search = search_attr.dims[3];
    }

    // Heatmap shape extraction (dynamic)
    if (hm_attr.dims[1] == 1)
    {
        m_out_height_hm = hm_attr.dims[2];
        m_out_width_hm = hm_attr.dims[3];
    }
    else
    {
        m_out_height_hm = hm_attr.dims[1];
        m_out_width_hm = hm_attr.dims[2];
    }

    m_is_model_loaded = true;
    StatusObject::instance()->update("tracker_model_status", "Model Loaded Successfully");

    // Initialize pre-allocated buffers (static sizing)
    memset(m_ref_stack_buf, 0, sizeof(m_ref_stack_buf));
    memset(m_search_buf, 0, sizeof(m_search_buf));
    memset(m_heatmap_buf, 0, sizeof(m_heatmap_buf));
}

TrackerService::~TrackerService()
{
    if (m_is_model_loaded)
    {
        rknn_destroy(m_ctx_model);
#if defined(USE_RGA)
        release_rga_buffers();
#endif
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

void TrackerService::set_drone_callback(IControlerCallback* cb)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_drone_cb = cb;
}




void TrackerService::refresh_target(const uchar* frame, int w, int h, int target_x, int target_y)
{
    int c;
    float max_sz;
    float min_sz;
    float sz;
    float half;
    int x1;
    int y1;
    int sz_int;
    int y;
    int x;
    int cy;
    int cx;
    int sy;
    int sx;
    uchar* temp_ref;
    uchar* crop_buf;

    if (!m_is_model_loaded)
    {
        return;
    }

    max_sz = m_max_crop;
    min_sz = m_min_crop;
    temp_ref = (uchar*)malloc(m_in_width_ref * m_in_height_ref);

    for (c = 0; c < m_in_channels_ref; ++c)
    {
        if (m_in_channels_ref <= 1)
        {
            sz = max_sz;
        }
        else
        {
            sz = max_sz - (c * (max_sz - min_sz) / (m_in_channels_ref - 1));
        }
        half = sz / 2.0f;
        x1 = (int)roundf((float)target_x - half);
        y1 = (int)roundf((float)target_y - half);
        sz_int = (int)sz;
        if (sz_int < 1)
        {
            sz_int = 1;
        }
        crop_buf = (uchar*)calloc(sz_int * sz_int, 1);

        for (cy = 0; cy < sz_int; ++cy)
        {
            sy = y1 + cy;
            if (sy >= 0 && sy < h)
            {
                for (cx = 0; cx < sz_int; ++cx)
                {
                    sx = x1 + cx;
                    if (sx >= 0 && sx < w)
                    {
                        crop_buf[cy * sz_int + cx] = frame[sy * w + sx];
                    }
                }
            }
        }

        resize_bilinear_gray(crop_buf, sz_int, sz_int, temp_ref, m_in_width_ref, m_in_height_ref);

        for (y = 0; y < m_in_height_ref; ++y)
        {
            for (x = 0; x < m_in_width_ref; ++x)
            {
                m_ref_stack_buf[(y * m_in_width_ref + x) * m_in_channels_ref + c] = temp_ref[y * m_in_width_ref + x];
            }
        }

        free(crop_buf);
    }

    free(temp_ref);

    std::lock_guard<std::mutex> lock(m_mutex);
    m_is_target_defined = true;
    if (m_callback != NULL)
    {
        m_callback->onStackCreated(m_ref_stack_buf, m_in_width_ref, m_in_height_ref, m_in_channels_ref);
    }
    fprintf(stdout, "[TrackerService] Target refreshed & initialized with multi-scale reference stack.\n");
}

void TrackerService::crop_and_resize_gray(const uchar* src, int src_w, int src_h, float cx, float cy, float crop_size, uchar* dst, int dst_w, int dst_h)
{
    int sz;
    uchar* crop_buf;
    float half;
    int x1;
    int y1;
    int y;
    int x;
    int sy;
    int sx;
    int clamped_sy;
    int clamped_sx;

    sz = (int)roundf(crop_size);
    if (sz < 1)
    {
        sz = 1;
    }
    crop_buf = (uchar*)calloc(sz * sz, 1);

    half = crop_size / 2.0f;
    x1 = (int)roundf(cx - half);
    y1 = (int)roundf(cy - half);

    for (y = 0; y < sz; ++y)
    {
        sy = y1 + y;
        clamped_sy = tracker_clamp_int(sy, 0, src_h - 1);
        for (x = 0; x < sz; ++x)
        {
            sx = x1 + x;
            clamped_sx = tracker_clamp_int(sx, 0, src_w - 1);
            crop_buf[y * sz + x] = src[clamped_sy * src_w + clamped_sx];
        }
    }

    resize_bilinear_gray(crop_buf, sz, sz, dst, dst_w, dst_h);
    free(crop_buf);
}

void TrackerService::crop_and_resize_ref_stack(float crop_size, float target_size)
{
    int tgt_sz;
    uchar* temp_crop;
    uchar* new_ref_stack;
    uchar* chan_buf;
    float cx;
    float cy;
    int c;
    int y;
    int x;

    tgt_sz = (int)target_size;
    temp_crop = (uchar*)malloc(tgt_sz * tgt_sz);
    new_ref_stack = (uchar*)malloc(tgt_sz * tgt_sz * m_in_channels_ref);
    chan_buf = (uchar*)malloc(m_in_width_ref * m_in_height_ref);

    cx = m_in_width_ref / 2.0f;
    cy = m_in_height_ref / 2.0f;

    for (c = 0; c < m_in_channels_ref; ++c)
    {
        for (y = 0; y < m_in_height_ref; ++y)
        {
            for (x = 0; x < m_in_width_ref; ++x)
            {
                chan_buf[y * m_in_width_ref + x] = m_ref_stack_buf[(y * m_in_width_ref + x) * m_in_channels_ref + c];
            }
        }

        crop_and_resize_gray(chan_buf, m_in_width_ref, m_in_height_ref, cx, cy, crop_size, temp_crop, tgt_sz, tgt_sz);

        for (y = 0; y < tgt_sz; ++y)
        {
            for (x = 0; x < tgt_sz; ++x)
            {
                new_ref_stack[(y * tgt_sz + x) * m_in_channels_ref + c] = temp_crop[y * tgt_sz + x];
            }
        }
    }

    memcpy(m_ref_stack_buf, new_ref_stack, tgt_sz * tgt_sz * m_in_channels_ref);
    free(chan_buf);
    free(temp_crop);
    free(new_ref_stack);
}

void TrackerService::update_frame(uchar* frame, int w, int h)
{
    rknn_input inputs[2];
    rknn_output outputs[2];
    int ret;
    int out_x;
    int out_y;
    int iter;
    int i;
    int peak_x_hm;
    int peak_y_hm;
    int crop_history_count;
    int iter_pred_count;
    int map_idx;
    int dx;
    int dy;
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
    float pcx;
    float pcy;
    float q_val;
    float crop_size;
    float mx;
    float my;
    float cx;
    float cy;
    float tl_x;
    float tl_y;
    char time_buf[64];
    uchar* ref_stack_backup;
    uchar* curr_search;
    uchar* next_search;
    const float* raw_hm;
    CropHistory crop_history[MAX_TRACKER_ITERATIONS];
    float iter_pred_x[MAX_TRACKER_ITERATIONS];
    float iter_pred_y[MAX_TRACKER_ITERATIONS];
    bool run_success;

    if (!m_is_model_loaded || !m_is_target_defined)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_drone_cb != NULL)
        {
            m_drone_cb->send_command(1000, 1000, 1000, 1000);
        }
        return;
    }

    resize_ms = 0.0f;
    npu_ms = 0.0f;
    decode_ms = 0.0f;
    total_ms = 0.0f;
    pcx = 0.0f;
    pcy = 0.0f;
    q_val = 0.0f;
    crop_history_count = 0;
    iter_pred_count = 0;
    run_success = true;
    out_x = -1;
    out_y = -1;

    t_start = std::chrono::steady_clock::now();

    // Backup reference stack to allow modifying it in iterations
    ref_stack_backup = (uchar*)malloc(m_in_width_ref * m_in_height_ref * m_in_channels_ref);
    memcpy(ref_stack_backup, m_ref_stack_buf, m_in_width_ref * m_in_height_ref * m_in_channels_ref);

    // Initialize temporary search frame
    curr_search = (uchar*)malloc(m_in_width_search * m_in_height_search);

    // Crop the centered square search region for the first iteration
    t_resize_start = std::chrono::steady_clock::now();
    resize_center_square_bilinear_gray(frame, w, h, curr_search, m_in_width_search, m_in_height_search);
    t_resize_end = std::chrono::steady_clock::now();
    resize_ms += std::chrono::duration<float, std::milli>(t_resize_end - t_resize_start).count();

    for (iter = 0; iter < m_iterations_num; ++iter)
    {
        // Setup current search buffer
        memcpy(m_search_buf, curr_search, m_in_width_search * m_in_height_search);

        // Setup inputs for NPU
        memset(inputs, 0, sizeof(inputs));

        inputs[m_idx_ref_stack].index = m_idx_ref_stack;
        inputs[m_idx_ref_stack].type = RKNN_TENSOR_UINT8;
        inputs[m_idx_ref_stack].size = m_in_width_ref * m_in_height_ref * m_in_channels_ref;
        inputs[m_idx_ref_stack].buf = m_ref_stack_buf;
        inputs[m_idx_ref_stack].fmt = RKNN_TENSOR_NHWC;

        inputs[m_idx_search_frame].index = m_idx_search_frame;
        inputs[m_idx_search_frame].type = RKNN_TENSOR_UINT8;
        inputs[m_idx_search_frame].size = m_in_width_search * m_in_height_search * m_in_channels_search;
        inputs[m_idx_search_frame].buf = m_search_buf;
        inputs[m_idx_search_frame].fmt = RKNN_TENSOR_NHWC;

        t_npu_start = std::chrono::steady_clock::now();
        ret = rknn_inputs_set(m_ctx_model, 2, inputs);
        if (ret < 0)
        {
            fprintf(stderr, "[TrackerService] rknn_inputs_set failed at iter %d: %d\n", iter, ret);
            run_success = false;
            break;
        }

        ret = rknn_run(m_ctx_model, NULL);
        if (ret < 0)
        {
            fprintf(stderr, "[TrackerService] rknn_run failed at iter %d: %d\n", iter, ret);
            run_success = false;
            break;
        }

        memset(outputs, 0, sizeof(outputs));
        outputs[0].index = 0;
        outputs[0].want_float = 1;
        outputs[1].index = 1;
        outputs[1].want_float = 1;

        ret = rknn_outputs_get(m_ctx_model, 2, outputs, NULL);
        if (ret < 0)
        {
            fprintf(stderr, "[TrackerService] rknn_outputs_get failed at iter %d: %d\n", iter, ret);
            run_success = false;
            break;
        }
        t_npu_end = std::chrono::steady_clock::now();
        npu_ms += std::chrono::duration<float, std::milli>(t_npu_end - t_npu_start).count();

        // Decode heatmap and extract quality
        t_decode_start = std::chrono::steady_clock::now();

        // Scale raw heatmap outputs from pre_threshold range [-1.0, 1.0] back to standard [0.0, 1.0]
        raw_hm = (const float*)outputs[m_idx_heatmap].buf;
        for (i = 0; i < m_out_width_hm * m_out_height_hm; ++i)
        {
            m_heatmap_buf[i] = (raw_hm[i] + 1.0f) / 2.0f;
        }

        peak_x_hm = -1;
        peak_y_hm = -1;
        decode_heatmap(m_heatmap_buf, &peak_x_hm, &peak_y_hm);

        if (peak_x_hm >= 0 && peak_y_hm >= 0)
        {
            pcx = (peak_x_hm * (float)m_in_width_search) / m_out_width_hm;
            pcy = (peak_y_hm * (float)m_in_height_search) / m_out_height_hm;
        }
        else
        {
            pcx = m_in_width_search / 2.0f;
            pcy = m_in_height_search / 2.0f;
        }

        iter_pred_x[iter_pred_count] = pcx;
        iter_pred_y[iter_pred_count] = pcy;
        iter_pred_count++;

        t_decode_end = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();

        if (m_quality_enabled)
        {
            q_val = ((float*)outputs[m_idx_quality].buf)[0];
        }
        
        if (q_val < 0.2f)
        {
			run_success = false;
			break;
		}

        // Release NPU outputs for this iteration
        rknn_outputs_release(m_ctx_model, 2, outputs);

        // Update StatusObject with quality telemetry from the last iteration
        if (iter == m_iterations_num - 1)
        {
            if (m_quality_enabled)
            {
                snprintf(time_buf, sizeof(time_buf), "%.4f", q_val);
                StatusObject::instance()->update("tracker_quality", time_buf);
            }
            else
            {
                StatusObject::instance()->update("tracker_quality", "Disabled");
            }
        }

        // Prepare next iteration search crop and reference crop
        if (iter < m_iterations_num - 1)
        {
            t_resize_start = std::chrono::steady_clock::now();

            crop_size = m_in_width_search / 2.0f;
            crop_history[crop_history_count].cx = pcx;
            crop_history[crop_history_count].cy = pcy;
            crop_history[crop_history_count].crop_size = crop_size;
            crop_history_count++;

            // Crop search frame
            next_search = (uchar*)malloc(m_in_width_search * m_in_height_search);
            crop_and_resize_gray(curr_search, m_in_width_search, m_in_height_search, pcx, pcy, crop_size, next_search, m_in_width_search, m_in_height_search);
            memcpy(curr_search, next_search, m_in_width_search * m_in_height_search);
            free(next_search);

            // Crop reference stack
            crop_and_resize_ref_stack(m_in_width_ref / 2.0f, m_in_width_ref);

            t_resize_end = std::chrono::steady_clock::now();
            resize_ms += std::chrono::duration<float, std::milli>(t_resize_end - t_resize_start).count();
        }
    }

    if (run_success)
    {
        // 3. Map final coordinates back to the original starting search space
        t_decode_start = std::chrono::steady_clock::now();
        mx = pcx;
        my = pcy;
        if (iter_pred_count > 0)
        {
            mx = iter_pred_x[iter_pred_count - 1];
            my = iter_pred_y[iter_pred_count - 1];
            for (map_idx = crop_history_count - 1; map_idx >= 0; --map_idx)
            {
                cx = crop_history[map_idx].cx;
                cy = crop_history[map_idx].cy;
                crop_size = crop_history[map_idx].crop_size;
                tl_x = cx - crop_size / 2.0f;
                tl_y = cy - crop_size / 2.0f;
                mx = tl_x + mx * (crop_size / m_in_width_search);
                my = tl_y + my * (crop_size / m_in_height_search);
            }
        }

        // Scale mapped coordinates to standard 256x256 display space
        out_x = (int)roundf(mx * (256.0f / m_in_width_search));
        out_y = (int)roundf(my * (256.0f / m_in_height_search));
        t_decode_end = std::chrono::steady_clock::now();
        decode_ms += std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();
    }
    else
    {
        out_x = -1;
        out_y = -1;
    }

    // Restore reference stack and clean up temporary memory
    memcpy(m_ref_stack_buf, ref_stack_backup, m_in_width_ref * m_in_height_ref * m_in_channels_ref);
    free(ref_stack_backup);
    free(curr_search);

    t_end = std::chrono::steady_clock::now();
    total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Update StatusObject
    snprintf(time_buf, sizeof(time_buf), "%.2f ms", resize_ms);
    StatusObject::instance()->update("tracker_time_resize", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms", npu_ms);
    StatusObject::instance()->update("tracker_time_npu", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms (CPU decode)", decode_ms);
    StatusObject::instance()->update("tracker_time_decode", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms", total_ms);
    StatusObject::instance()->update("tracker_time_total", time_buf);

    // 4. Trigger TrackerCallback
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_callback != NULL)
        {
            m_callback->onTargetDetected(out_x, out_y);
            m_callback->onHeatmapCreated(m_heatmap_buf, m_out_width_hm, m_out_height_hm);
        }
        if (m_drone_cb != NULL && out_x >= 0 && out_y >= 0)
        {
            int16_t yaw = 1000;
            int16_t roll = 1000; // <- ->
            int16_t pitch = 1500;
            int16_t throttle = 1000;
            dx = out_x - 128;
            dy = out_y - 128;
            DroneControlerHal::calculate_tracking_commands(dx, dy, roll, throttle);
            m_drone_cb->send_command(roll, pitch, yaw, throttle);
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

void TrackerService::resize_center_square_bilinear_gray(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h)
{
#if defined(USE_RGA)
    int crop_size_rga = tracker_min_int(src_w, src_h);
    if (crop_size_rga <= 0)
    {
        memset(dst, 0, (size_t)(dst_w * dst_h));
        return;
    }
    int x0_rga = (src_w - crop_size_rga) / 2;
    int y0_rga = (src_h - crop_size_rga) / 2;

    if (init_rga_buffers(src_w, src_h))
    {
        memcpy(m_rga_src_va, src, (size_t)(src_w * src_h));

        im_rect crop_rect;
        crop_rect.x = x0_rga;
        crop_rect.y = y0_rga;
        crop_rect.width = crop_size_rga;
        crop_rect.height = crop_size_rga;

        IM_STATUS status = imcrop(m_rga_src_buf, m_rga_dst_buf, crop_rect);
        if (status == IM_STATUS_SUCCESS)
        {
            memcpy(dst, m_rga_dst_va, (size_t)(dst_w * dst_h));
            return;
        }
        else
        {
            fprintf(stderr, "[TrackerService] RGA hardware crop failed: %d. Falling back to CPU.\n", status);
        }
    }
#endif

    int crop_size;
    int x0, y0;
    int x, y;
    float x_ratio;
    float y_ratio;
    int x_l, y_l, x_h, y_h;
    int sx_l, sx_h, sy_l, sy_h;
    float x_weight, y_weight;
    uchar a, b, c, d;

    crop_size = tracker_min_int(src_w, src_h);
    if (crop_size <= 0)
    {
        memset(dst, 0, (size_t)(dst_w * dst_h));
        return;
    }

    x0 = (src_w - crop_size) / 2;
    y0 = (src_h - crop_size) / 2;

    x_ratio = ((float)(crop_size - 1)) / dst_w;
    y_ratio = ((float)(crop_size - 1)) / dst_h;

#if defined(USE_OMP)
    #pragma omp parallel for schedule(dynamic)
#endif
    for (y = 0; y < dst_h; ++y)
    {
        for (x = 0; x < dst_w; ++x)
        {
            x_l = (int)(x_ratio * x);
            y_l = (int)(y_ratio * y);
            x_h = x_l + 1;
            y_h = y_l + 1;

            if (x_h >= crop_size)
            {
                x_h = crop_size - 1;
            }
            if (y_h >= crop_size)
            {
                y_h = crop_size - 1;
            }

            x_weight = (x_ratio * x) - x_l;
            y_weight = (y_ratio * y) - y_l;

            sx_l = x0 + x_l;
            sx_h = x0 + x_h;
            sy_l = y0 + y_l;
            sy_h = y0 + y_h;

            a = src[sy_l * src_w + sx_l];
            b = src[sy_l * src_w + sx_h];
            c = src[sy_h * src_w + sx_l];
            d = src[sy_h * src_w + sx_h];

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

#if defined(USE_RGA)
bool TrackerService::init_rga_buffers(int src_w, int src_h)
{
    if (m_rga_initialized && m_rga_src_w == src_w && m_rga_src_h == src_h)
    {
        return true;
    }

    release_rga_buffers();

    // Store source dimensions early so release_rga_buffers() can compute
    // munmap sizes if init fails partway through.
    m_rga_src_w = src_w;
    m_rga_src_h = src_h;

    int heap_fd = open("/dev/dma_heap/system", O_RDONLY | O_CLOEXEC);
    if (heap_fd < 0)
    {
        fprintf(stderr, "[TrackerService] Cannot open /dev/dma_heap/system: %s\n", strerror(errno));
        return false;
    }

    struct dma_heap_allocation_data alloc;
    bool ok = true;

    // Allocate src DMA buffer (grayscale camera frame: src_w * src_h bytes)
    memset(&alloc, 0, sizeof(alloc));
    alloc.len      = (uint64_t)(src_w * src_h);
    alloc.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0)
    {
        fprintf(stderr, "[TrackerService] DMA heap alloc (src) failed: %s\n", strerror(errno));
        ok = false;
    }
    else
    {
        m_rga_src_fd = (int)alloc.fd;
    }

    // Allocate dst DMA buffer (resized search window)
    if (ok)
    {
        size_t dst_size = (size_t)(m_in_width_search * m_in_height_search * m_in_channels_search);
        memset(&alloc, 0, sizeof(alloc));
        alloc.len      = (uint64_t)dst_size;
        alloc.fd_flags = O_RDWR | O_CLOEXEC;
        if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0)
        {
            fprintf(stderr, "[TrackerService] DMA heap alloc (dst) failed: %s\n", strerror(errno));
            ok = false;
        }
        else
        {
            m_rga_dst_fd = (int)alloc.fd;
        }
    }

    close(heap_fd);

    if (!ok)
    {
        release_rga_buffers();
        return false;
    }

    // Map src buffer into process address space
    m_rga_src_va = (uchar*)mmap(NULL, (size_t)(src_w * src_h),
                                 PROT_READ | PROT_WRITE, MAP_SHARED, m_rga_src_fd, 0);
    if (m_rga_src_va == (uchar*)MAP_FAILED)
    {
        fprintf(stderr, "[TrackerService] mmap RGA src failed: %s\n", strerror(errno));
        m_rga_src_va = NULL;
        release_rga_buffers();
        return false;
    }

    // Map dst buffer into process address space
    size_t dst_size = (size_t)(m_in_width_search * m_in_height_search * m_in_channels_search);
    m_rga_dst_va = (uchar*)mmap(NULL, dst_size,
                                 PROT_READ | PROT_WRITE, MAP_SHARED, m_rga_dst_fd, 0);
    if (m_rga_dst_va == (uchar*)MAP_FAILED)
    {
        fprintf(stderr, "[TrackerService] mmap RGA dst failed: %s\n", strerror(errno));
        m_rga_dst_va = NULL;
        release_rga_buffers();
        return false;
    }

    // Register buffers with the RGA driver (one-time import per session)
    m_rga_src_handle = importbuffer_fd(m_rga_src_fd, src_w * src_h);
    if (m_rga_src_handle == 0)
    {
        fprintf(stderr, "[TrackerService] RGA importbuffer_fd (src) failed\n");
        release_rga_buffers();
        return false;
    }

    m_rga_dst_handle = importbuffer_fd(m_rga_dst_fd, (int)dst_size);
    if (m_rga_dst_handle == 0)
    {
        fprintf(stderr, "[TrackerService] RGA importbuffer_fd (dst) failed\n");
        release_rga_buffers();
        return false;
    }

    // Build rga_buffer_t descriptors — reused on every call to imresize
    m_rga_src_buf = wrapbuffer_handle(m_rga_src_handle, src_w, src_h, RK_FORMAT_YCbCr_400);
    m_rga_dst_buf = wrapbuffer_handle(m_rga_dst_handle,
                                       m_in_width_search, m_in_height_search, RK_FORMAT_YCbCr_400);

    m_rga_initialized = true;
    fprintf(stdout, "[TrackerService] RGA DMA buffers ready: src=%dx%d dst=%dx%d\n",
            src_w, src_h, m_in_width_search, m_in_height_search);
    return true;
}

void TrackerService::release_rga_buffers()
{
    if (m_rga_src_handle != 0)
    {
        releasebuffer_handle(m_rga_src_handle);
        m_rga_src_handle = 0;
    }
    if (m_rga_dst_handle != 0)
    {
        releasebuffer_handle(m_rga_dst_handle);
        m_rga_dst_handle = 0;
    }
    if (m_rga_src_va != NULL)
    {
        munmap(m_rga_src_va, (size_t)(m_rga_src_w * m_rga_src_h));
        m_rga_src_va = NULL;
    }
    if (m_rga_dst_va != NULL)
    {
        munmap(m_rga_dst_va, (size_t)(m_in_width_search * m_in_height_search * m_in_channels_search));
        m_rga_dst_va = NULL;
    }
    if (m_rga_src_fd >= 0)
    {
        close(m_rga_src_fd);
        m_rga_src_fd = -1;
    }
    if (m_rga_dst_fd >= 0)
    {
        close(m_rga_dst_fd);
        m_rga_dst_fd = -1;
    }
    m_rga_initialized = false;
}

void TrackerService::resize_rga(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h)
{
    if (!init_rga_buffers(src_w, src_h))
    {
        fprintf(stderr, "[TrackerService] RGA init failed, falling back to CPU resize.\n");
        resize_bilinear_gray(src, src_w, src_h, dst, dst_w, dst_h);
        return;
    }

    // Copy the incoming (non-DMA) camera frame into the DMA-accessible src buffer
    memcpy(m_rga_src_va, src, (size_t)(src_w * src_h));

    IM_STATUS status = imresize(m_rga_src_buf, m_rga_dst_buf);
    if (status != IM_STATUS_SUCCESS)
    {
        fprintf(stderr, "[TrackerService] RGA hardware resize failed: %d. Falling back to CPU.\n", status);
        resize_bilinear_gray(src, src_w, src_h, dst, dst_w, dst_h);
        return;
    }

    // Copy RGA output from DMA buffer into the caller's dst (m_search_buf)
    memcpy(dst, m_rga_dst_va, (size_t)(dst_w * dst_h));
}
#elif defined(USE_OMP)
void TrackerService::resize_bilinear_omp(const uchar* src, int src_w, int src_h, uchar* dst, int dst_w, int dst_h)
{
    float x_ratio = ((float)(src_w - 1)) / dst_w;
    float y_ratio = ((float)(src_h - 1)) / dst_h;

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < dst_h; ++y)
    {
        for (int x = 0; x < dst_w; ++x)
        {
            int x_l = (int)(x_ratio * x);
            int y_l = (int)(y_ratio * y);
            int x_h = x_l + 1;
            int y_h = y_l + 1;

            if (x_h >= src_w)
            {
                x_h = src_w - 1;
            }
            if (y_h >= src_h)
            {
                y_h = src_h - 1;
            }

            float x_weight = (x_ratio * x) - x_l;
            float y_weight = (y_ratio * y) - y_l;

            uchar a = src[y_l * src_w + x_l];
            uchar b = src[y_l * src_w + x_h];
            uchar c = src[y_h * src_w + x_l];
            uchar d = src[y_h * src_w + x_h];

            dst[y * dst_w + x] = (uchar)(
                a * (1.0f - x_weight) * (1.0f - y_weight) +
                b * x_weight * (1.0f - y_weight) +
                c * (1.0f - x_weight) * y_weight +
                d * x_weight * y_weight
            );
        }
    }
}
#endif

void TrackerService::decode_heatmap(const float* raw_heatmap, int* out_x, int* out_y)
{
    // 1. Find discrete peak (argmax)
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

    int max_y = max_flat_idx / m_out_width_hm;
    int max_x = max_flat_idx % m_out_width_hm;

    if (m_use_argmax_only)
    {
        *out_x = max_x;
        *out_y = max_y;
        return;
    }

    // 2. Define 15x15 local window centered at peak
    int y_start = tracker_max_int(0, max_y - 7);
    int y_end = tracker_min_int(m_out_height_hm, max_y + 8);
    int x_start = tracker_max_int(0, max_x - 7);
    int x_end = tracker_min_int(m_out_width_hm, max_x + 8);

    // 3. Compute Mean & StdDev in this window
    double sum = 0.0;
    double sq_sum = 0.0;
    int count = 0;
    for (int y = y_start; y < y_end; ++y)
    {
        for (int x = x_start; x < x_end; ++x)
        {
            float val = raw_heatmap[y * m_out_width_hm + x];
            sum += val;
            sq_sum += val * val;
            count++;
        }
    }

    double mean = (count > 0) ? (sum / count) : 0.0;
    double variance = (count > 0) ? (sq_sum / count - mean * mean) : 0.0;
    double std_dev = sqrt(tracker_max_double(0.0, variance));
    float threshold = (float)(mean + 1.5 * std_dev);

    // 4. Non-recursive local BFS to extract connected blob above threshold
    // Using fixed stack arrays of size 225 (15x15) to guarantee no runtime allocation
    bool visited[15][15];
    memset(visited, 0, sizeof(visited));

    std::pair<int, int> queue[225];
    int head = 0;
    int tail = 0;

    int local_peak_y = max_y - y_start;
    int local_peak_x = max_x - x_start;

    queue[tail++] = {local_peak_y, local_peak_x};
    visited[local_peak_y][local_peak_x] = true;

    while (head < tail)
    {
        auto curr = queue[head++];
        int cy = curr.first;
        int cx = curr.second;

        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                if (dy == 0 && dx == 0) continue;
                int ny = cy + dy;
                int nx = cx + dx;

                int win_h = y_end - y_start;
                int win_w = x_end - x_start;

                if (ny >= 0 && ny < win_h && nx >= 0 && nx < win_w)
                {
                    if (!visited[ny][nx])
                    {
                        float val = raw_heatmap[(y_start + ny) * m_out_width_hm + (x_start + nx)];
                        if (val > threshold)
                        {
                            visited[ny][nx] = true;
                            queue[tail++] = {ny, nx};
                        }
                    }
                }
            }
        }
    }

    // 5. Compute Weighted Centroid of the extracted blob
    double sum_x = 0.0;
    double sum_y = 0.0;
    double total_mass = 0.0;
    for (int i = 0; i < tail; ++i)
    {
        int cy = queue[i].first;
        int cx = queue[i].second;
        int global_y = y_start + cy;
        int global_x = x_start + cx;
        float val = raw_heatmap[global_y * m_out_width_hm + global_x];
        sum_x += global_x * val;
        sum_y += global_y * val;
        total_mass += val;
    }

    if (total_mass > 1e-6)
    {
        *out_x = (int)round(sum_x / total_mass);
        *out_y = (int)round(sum_y / total_mass);
    }
    else
    {
        *out_x = max_x;
        *out_y = max_y;
    }
}
