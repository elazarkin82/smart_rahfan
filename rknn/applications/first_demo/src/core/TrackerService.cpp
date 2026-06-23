#include "core/TrackerService.h"
#include "utils/StatusObject.hpp"
#include "utils/NNOperationsCpu.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <vector>
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

TrackerService::TrackerService(const char* template_path, const char* search_backbone_path, const char* decoder_path, float min_crop, float max_crop, bool quality_enabled, bool use_argmax_only)
{
    FILE* fp;
    long model_size;
    void* model_data;
    int ret;
    rknn_input_output_num io_num_temp;
    rknn_input_output_num io_num_backbone;
    rknn_input_output_num io_num_decoder;
    rknn_tensor_attr template_in_attrs[1];
    rknn_tensor_attr template_out_attrs[1];
    rknn_tensor_attr backbone_in_attrs[1];
    rknn_tensor_attr backbone_out_attrs[4];
    rknn_tensor_attr decoder_in_attrs[2];
    rknn_tensor_attr decoder_out_attrs[1];
 
    m_ctx_template = 0;
    m_ctx_search_backbone = 0;
    m_ctx_decoder = 0;
    m_is_model_loaded = false;
    m_is_target_defined = false;
    m_callback = NULL;
    m_drone_cb = NULL;
 
    m_min_crop = min_crop;
    m_max_crop = max_crop;
    m_quality_enabled = quality_enabled;
    m_use_argmax_only = use_argmax_only;
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

    m_in_width_ref = 64;
    m_in_height_ref = 64;
    m_in_channels_ref = 16;

    m_in_width_search = 256;
    m_in_height_search = 256;
    m_in_channels_search = 1;

    m_out_width_hm = 256;
    m_out_height_hm = 256;

    // 1. Load template model file into memory
    fp = fopen(template_path, "rb");
    if (fp == NULL)
    {
        fprintf(stderr, "[TrackerService] Failed to open template model file: %s\n", template_path);
        StatusObject::instance()->update("tracker_model_status", "Error: Template File Not Found");
        return;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    model_data = malloc(model_size);
    if (model_data == NULL)
    {
        fclose(fp);
        fprintf(stderr, "[TrackerService] Failed to allocate memory for template model buffer.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: OOM on Load");
        return;
    }

    if (fread(model_data, 1, model_size, fp) != (size_t)model_size)
    {
        free(model_data);
        fclose(fp);
        fprintf(stderr, "[TrackerService] Failed to read template model file contents.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: Read Failure");
        return;
    }
    fclose(fp);

    // Initialize RKNN context for template encoder
    ret = rknn_init(&m_ctx_template, model_data, model_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_init for template failed: %d\n", ret);
        StatusObject::instance()->update("tracker_model_status", "Error: Template NPU Init Failed");
        return;
    }

    // 2. Load search backbone model file into memory
    fp = fopen(search_backbone_path, "rb");
    if (fp == NULL)
    {
        rknn_destroy(m_ctx_template);
        fprintf(stderr, "[TrackerService] Failed to open search backbone model file: %s\n", search_backbone_path);
        StatusObject::instance()->update("tracker_model_status", "Error: Backbone File Not Found");
        return;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    model_data = malloc(model_size);
    if (model_data == NULL)
    {
        fclose(fp);
        rknn_destroy(m_ctx_template);
        fprintf(stderr, "[TrackerService] Failed to allocate memory for search backbone model buffer.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: OOM on Load");
        return;
    }

    if (fread(model_data, 1, model_size, fp) != (size_t)model_size)
    {
        free(model_data);
        fclose(fp);
        rknn_destroy(m_ctx_template);
        fprintf(stderr, "[TrackerService] Failed to read search backbone model file contents.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: Read Failure");
        return;
    }
    fclose(fp);

    // Initialize RKNN context for search backbone
    ret = rknn_init(&m_ctx_search_backbone, model_data, model_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        rknn_destroy(m_ctx_template);
        fprintf(stderr, "[TrackerService] rknn_init for search backbone failed: %d\n", ret);
        StatusObject::instance()->update("tracker_model_status", "Error: Backbone NPU Init Failed");
        return;
    }

    // 3. Load decoder model file into memory
    fp = fopen(decoder_path, "rb");
    if (fp == NULL)
    {
        rknn_destroy(m_ctx_template);
        rknn_destroy(m_ctx_search_backbone);
        fprintf(stderr, "[TrackerService] Failed to open decoder model file: %s\n", decoder_path);
        StatusObject::instance()->update("tracker_model_status", "Error: Decoder File Not Found");
        return;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    model_data = malloc(model_size);
    if (model_data == NULL)
    {
        fclose(fp);
        rknn_destroy(m_ctx_template);
        rknn_destroy(m_ctx_search_backbone);
        fprintf(stderr, "[TrackerService] Failed to allocate memory for decoder model buffer.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: OOM on Load");
        return;
    }

    if (fread(model_data, 1, model_size, fp) != (size_t)model_size)
    {
        free(model_data);
        fclose(fp);
        rknn_destroy(m_ctx_template);
        rknn_destroy(m_ctx_search_backbone);
        fprintf(stderr, "[TrackerService] Failed to read decoder model file contents.\n");
        StatusObject::instance()->update("tracker_model_status", "Error: Read Failure");
        return;
    }
    fclose(fp);

    // Initialize RKNN context for decoder
    ret = rknn_init(&m_ctx_decoder, model_data, model_size, 0, NULL);
    free(model_data);

    if (ret < 0)
    {
        rknn_destroy(m_ctx_template);
        rknn_destroy(m_ctx_search_backbone);
        fprintf(stderr, "[TrackerService] rknn_init for decoder failed: %d\n", ret);
        StatusObject::instance()->update("tracker_model_status", "Error: Decoder NPU Init Failed");
        return;
    }

    // 4. Query dynamic tensor dimensions & attributes
    rknn_query(m_ctx_template, RKNN_QUERY_IN_OUT_NUM, &io_num_temp, sizeof(io_num_temp));
    rknn_query(m_ctx_search_backbone, RKNN_QUERY_IN_OUT_NUM, &io_num_backbone, sizeof(io_num_backbone));
    rknn_query(m_ctx_decoder, RKNN_QUERY_IN_OUT_NUM, &io_num_decoder, sizeof(io_num_decoder));
    m_backbone_out_num = (int)io_num_backbone.n_output;

    // Query attributes for template model
    memset(template_in_attrs, 0, sizeof(template_in_attrs));
    template_in_attrs[0].index = 0;
    rknn_query(m_ctx_template, RKNN_QUERY_INPUT_ATTR, &template_in_attrs[0], sizeof(rknn_tensor_attr));

    memset(template_out_attrs, 0, sizeof(template_out_attrs));
    template_out_attrs[0].index = 0;
    rknn_query(m_ctx_template, RKNN_QUERY_OUTPUT_ATTR, &template_out_attrs[0], sizeof(rknn_tensor_attr));
    m_fmt_template = template_out_attrs[0].fmt;

    // Query attributes for backbone model
    memset(backbone_in_attrs, 0, sizeof(backbone_in_attrs));
    backbone_in_attrs[0].index = 0;
    rknn_query(m_ctx_search_backbone, RKNN_QUERY_INPUT_ATTR, &backbone_in_attrs[0], sizeof(rknn_tensor_attr));

    // Query attributes for decoder output model
    memset(decoder_out_attrs, 0, sizeof(decoder_out_attrs));
    decoder_out_attrs[0].index = 0;
    rknn_query(m_ctx_decoder, RKNN_QUERY_OUTPUT_ATTR, &decoder_out_attrs[0], sizeof(rknn_tensor_attr));

    // Keep shapes (handle NCHW vs NHWC layouts dynamically based on channel position)
    // Reference stack input has 2 channels
    if (template_in_attrs[0].dims[1] == 2)
    {
        m_in_height_ref = template_in_attrs[0].dims[2];
        m_in_width_ref = template_in_attrs[0].dims[3];
        m_in_channels_ref = template_in_attrs[0].dims[1];
    }
    else
    {
        m_in_height_ref = template_in_attrs[0].dims[1];
        m_in_width_ref = template_in_attrs[0].dims[2];
        m_in_channels_ref = template_in_attrs[0].dims[3];
    }

    // Search frame input has 1 channel
    if (backbone_in_attrs[0].dims[1] == 1)
    {
        m_in_height_search = backbone_in_attrs[0].dims[2];
        m_in_width_search = backbone_in_attrs[0].dims[3];
        m_in_channels_search = backbone_in_attrs[0].dims[1];
    }
    else
    {
        m_in_height_search = backbone_in_attrs[0].dims[1];
        m_in_width_search = backbone_in_attrs[0].dims[2];
        m_in_channels_search = backbone_in_attrs[0].dims[3];
    }

    // Heatmap output has 1 channel
    if (decoder_out_attrs[0].dims[1] == 1)
    {
        m_out_height_hm = decoder_out_attrs[0].dims[2];
        m_out_width_hm = decoder_out_attrs[0].dims[3];
    }
    else
    {
        m_out_height_hm = decoder_out_attrs[0].dims[1];
        m_out_width_hm = decoder_out_attrs[0].dims[2];
    }

    // Dynamic output index resolution for Search Backbone based on shape (handles NCHW/NHWC layout query)
    m_idx_skip3 = -1;
    m_idx_search_features = -1;
    m_fmt_skip3 = RKNN_TENSOR_NCHW;
    m_fmt_search_features = RKNN_TENSOR_NCHW;

    for (uint32_t i = 0; i < io_num_backbone.n_output; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(m_ctx_search_backbone, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        
        uint32_t channels = 0, height = 0, width = 0;
        if (attr.dims[2] == attr.dims[3]) // NCHW layout: [batch, channels, height, width]
        {
            channels = attr.dims[1];
            height = attr.dims[2];
            width = attr.dims[3];
        }
        else // NHWC layout: [batch, height, width, channels]
        {
            height = attr.dims[1];
            width = attr.dims[2];
            channels = attr.dims[3];
        }
        
        if (height == 32 && width == 32 && channels == 32)
        {
            m_idx_skip3 = (int)i;
            m_fmt_skip3 = RKNN_TENSOR_NHWC;
        }
        else if (height == 16 && width == 16 && channels == 64)
        {
            m_idx_search_features = (int)i;
            m_fmt_search_features = RKNN_TENSOR_NHWC;
        }
    }

    // Dynamic input index resolution for Decoder based on shape (NHWC layout query)
    m_idx_dec_corr = -1;
    m_idx_dec_skip3 = -1;

    for (uint32_t i = 0; i < io_num_decoder.n_input; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(m_ctx_decoder, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        
        uint32_t height = attr.dims[1];
        uint32_t width = attr.dims[2];
        uint32_t channels = attr.dims[3];
        
        if (height == 32 && width == 32 && channels == 32) m_idx_dec_skip3 = (int)i;
        else if (height == 16 && width == 16 && channels == 64) m_idx_dec_corr = (int)i;
    }

    if (m_idx_skip3 == -1 || m_idx_search_features == -1 || m_idx_dec_corr == -1 || m_idx_dec_skip3 == -1)
    {
        fprintf(stderr, "[TrackerService] Error: Failed to resolve all tensor indices.\n");
        m_is_model_loaded = false;
        return;
    }

    fprintf(stdout, "[TrackerService] Loaded split RKNN model subgraphs successfully.\n");
    fprintf(stdout, " - Template Input (Reference): %dx%dx%d\n", m_in_width_ref, m_in_height_ref, m_in_channels_ref);
    fprintf(stdout, " - Backbone Input (Search): %dx%dx%d\n", m_in_width_search, m_in_height_search, m_in_channels_search);
    fprintf(stdout, " - Decoder Output (Heatmap): %dx%d\n", m_out_width_hm, m_out_height_hm);
    fprintf(stdout, " - Backbone Outputs: skip3_idx=%d, search_features_idx=%d\n", m_idx_skip3, m_idx_search_features);
    fprintf(stdout, " - Decoder Inputs: corr_features_idx=%d, skip3_idx=%d\n", m_idx_dec_corr, m_idx_dec_skip3);

    m_is_model_loaded = true;
    StatusObject::instance()->update("tracker_model_status", "Loaded & Ready");

    // Initialize pre-allocated buffers (static sizing)
    memset(m_ref_stack_buf, 0, sizeof(m_ref_stack_buf));
    memset(m_search_buf, 0, sizeof(m_search_buf));
    memset(m_heatmap_buf, 0, sizeof(m_heatmap_buf));
    memset(m_template_features, 0, sizeof(m_template_features));
    memset(m_corr_out_buf, 0, sizeof(m_corr_out_buf));
}

TrackerService::~TrackerService()
{
    if (m_is_model_loaded)
    {
        rknn_destroy(m_ctx_template);
        rknn_destroy(m_ctx_search_backbone);
        rknn_destroy(m_ctx_decoder);
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
    int y, x;
    uchar* temp_64x64;
    int ret;
    rknn_input inputs[1];
    rknn_output outputs[1];

    if (!m_is_model_loaded)
    {
        return;
    }

    max_sz = m_max_crop;
    min_sz = m_min_crop;
    temp_64x64 = (uchar*)malloc(m_in_width_ref * m_in_height_ref);

    for (c = 0; c < m_in_channels_ref; ++c)
    {
        float sz = max_sz - (c * (max_sz - min_sz) / (m_in_channels_ref - 1));
        float half = sz / 2.0f;
        int x1 = (int)roundf((float)target_x - half);
        int y1 = (int)roundf((float)target_y - half);
        int sz_int = (int)sz;
        if (sz_int < 1) sz_int = 1;
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

    // Run NPU inference on template context to compute template features
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_in_width_ref * m_in_height_ref * m_in_channels_ref;
    inputs[0].buf = m_ref_stack_buf;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(m_ctx_template, 1, inputs);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] Failed to set inputs for template model: %d\n", ret);
        return;
    }

    ret = rknn_run(m_ctx_template, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] Failed to run template model: %d\n", ret);
        return;
    }

    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1; // Get features in FP32 format for CPU depthwise correlation
    ret = rknn_outputs_get(m_ctx_template, 1, outputs, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] Failed to get outputs for template model: %d\n", ret);
        return;
    }

    // Copy to template features cache
    memcpy(m_template_features, outputs[0].buf, 64 * 8 * 8 * sizeof(float));
    if (m_fmt_template == RKNN_TENSOR_NHWC)
    {
        NNOperationsCpu::transpose_nhwc_to_nchw(m_template_features, m_template_features_nchw, 1, 64, 8, 8);
    }
    else
    {
        memcpy(m_template_features_nchw, m_template_features, 64 * 8 * 8 * sizeof(float));
    }
    rknn_outputs_release(m_ctx_template, 1, outputs);

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
    rknn_input inputs_sb[1];
    rknn_output outputs_sb[4];
    rknn_input inputs_d[2];
    rknn_output outputs_d[1];
    int ret;
    int out_x;
    int out_y;
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_resize_start;
    std::chrono::steady_clock::time_point t_resize_end;
    std::chrono::steady_clock::time_point t_sb_start;
    std::chrono::steady_clock::time_point t_sb_end;
    std::chrono::steady_clock::time_point t_corr_start;
    std::chrono::steady_clock::time_point t_corr_end;
    std::chrono::steady_clock::time_point t_dec_start;
    std::chrono::steady_clock::time_point t_dec_end;
    std::chrono::steady_clock::time_point t_decode_start;
    std::chrono::steady_clock::time_point t_decode_end;
    std::chrono::steady_clock::time_point t_end;
    float resize_ms;
    float sb_ms;
    float corr_ms;
    float dec_ms;
    float decode_ms;
    float total_ms;
    char time_buf[64];

    if (!m_is_model_loaded || !m_is_target_defined)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_drone_cb != NULL)
        {
            m_drone_cb->send_command(1000, 1000, 1000, 1000);
        }
        return;
    }

    t_start = std::chrono::steady_clock::now();

    // 1. Crop the centered square search region, then resize to model input size.
    t_resize_start = std::chrono::steady_clock::now();
    resize_center_square_bilinear_gray(frame, w, h, m_search_buf, m_in_width_search, m_in_height_search);
    t_resize_end = std::chrono::steady_clock::now();

    // 2. Setup Search Backbone inputs
    memset(inputs_sb, 0, sizeof(inputs_sb));
    inputs_sb[0].index = 0;
    inputs_sb[0].type = RKNN_TENSOR_UINT8;
    inputs_sb[0].size = m_in_width_search * m_in_height_search * m_in_channels_search;
    inputs_sb[0].buf = m_search_buf;
    inputs_sb[0].fmt = RKNN_TENSOR_NHWC;

    t_sb_start = std::chrono::steady_clock::now();
    ret = rknn_inputs_set(m_ctx_search_backbone, 1, inputs_sb);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_inputs_set search backbone failed: %d\n", ret);
        return;
    }

    ret = rknn_run(m_ctx_search_backbone, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_run search backbone failed: %d\n", ret);
        return;
    }

    memset(outputs_sb, 0, sizeof(outputs_sb));
    for (int i = 0; i < m_backbone_out_num; ++i)
    {
        outputs_sb[i].index = i;
        outputs_sb[i].want_float = 1;
    }

    ret = rknn_outputs_get(m_ctx_search_backbone, m_backbone_out_num, outputs_sb, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_outputs_get search backbone failed: %d\n", ret);
        return;
    }
    t_sb_end = std::chrono::steady_clock::now();

    // 3. Run CPU Depthwise Correlation (expects NCHW, produces NCHW)
    t_corr_start = std::chrono::steady_clock::now();
    if (m_fmt_search_features == RKNN_TENSOR_NHWC)
    {
        NNOperationsCpu::transpose_nhwc_to_nchw(
            (const float*)outputs_sb[m_idx_search_features].buf,
            m_search_features_nchw,
            1, 64, 16, 16
        );
    }
    else
    {
        memcpy(m_search_features_nchw, outputs_sb[m_idx_search_features].buf, 64 * 16 * 16 * sizeof(float));
    }

    NNOperationsCpu::depthwise_correlation_nchw_neon_omp(
        m_search_features_nchw,      // search features
        m_template_features_nchw,     // kernel features
        m_corr_out_buf,               // output
        64, 16, 16, 8, 8
    );

    // Transpose correlation output and skip3 output to NHWC layout (needed for Decoder inputs)
    NNOperationsCpu::transpose_nchw_to_nhwc(m_corr_out_buf, m_corr_out_nhwc, 1, 64, 16, 16);
    
    if (m_fmt_skip3 == RKNN_TENSOR_NHWC)
    {
        memcpy(m_skip3_nhwc, outputs_sb[m_idx_skip3].buf, 32 * 32 * 32 * sizeof(float));
    }
    else
    {
        NNOperationsCpu::transpose_nchw_to_nhwc(
            (const float*)outputs_sb[m_idx_skip3].buf,
            m_skip3_nhwc,
            1, 32, 32, 32
        );
    }
    t_corr_end = std::chrono::steady_clock::now();

    // 4. Setup Decoder inputs
    memset(inputs_d, 0, sizeof(inputs_d));

    inputs_d[m_idx_dec_corr].index = m_idx_dec_corr;
    inputs_d[m_idx_dec_corr].type = RKNN_TENSOR_FLOAT32;
    inputs_d[m_idx_dec_corr].size = 16 * 16 * 64 * sizeof(float);
    inputs_d[m_idx_dec_corr].buf = m_corr_out_nhwc;
    inputs_d[m_idx_dec_corr].fmt = RKNN_TENSOR_NHWC;

    inputs_d[m_idx_dec_skip3].index = m_idx_dec_skip3;
    inputs_d[m_idx_dec_skip3].type = RKNN_TENSOR_FLOAT32;
    inputs_d[m_idx_dec_skip3].size = 32 * 32 * 32 * sizeof(float);
    inputs_d[m_idx_dec_skip3].buf = m_skip3_nhwc;
    inputs_d[m_idx_dec_skip3].fmt = RKNN_TENSOR_NHWC;

    t_dec_start = std::chrono::steady_clock::now();
    ret = rknn_inputs_set(m_ctx_decoder, 2, inputs_d);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_inputs_set decoder failed: %d\n", ret);
        rknn_outputs_release(m_ctx_search_backbone, m_backbone_out_num, outputs_sb);
        return;
    }

    ret = rknn_run(m_ctx_decoder, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_run decoder failed: %d\n", ret);
        rknn_outputs_release(m_ctx_search_backbone, m_backbone_out_num, outputs_sb);
        return;
    }

    memset(outputs_d, 0, sizeof(outputs_d));
    outputs_d[0].index = 0;
    outputs_d[0].want_float = 1;

    ret = rknn_outputs_get(m_ctx_decoder, 1, outputs_d, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[TrackerService] rknn_outputs_get decoder failed: %d\n", ret);
        rknn_outputs_release(m_ctx_search_backbone, m_backbone_out_num, outputs_sb);
        return;
    }
    t_dec_end = std::chrono::steady_clock::now();

    // 5. Decode heatmap using standard argmax
    t_decode_start = std::chrono::steady_clock::now();
    
    // Scale raw heatmap outputs from pre_threshold range [-1.0, 1.0] back to standard [0.0, 1.0]
    const float* raw_hm = (const float*)outputs_d[0].buf;
    for (int i = 0; i < m_out_width_hm * m_out_height_hm; ++i)
    {
        m_heatmap_buf[i] = (raw_hm[i] + 1.0f) / 2.0f;
    }

    out_x = -1;
    out_y = -1;
    decode_heatmap(m_heatmap_buf, &out_x, &out_y);

    // Scale coordinates dynamically back to the standard 256x256 grid expected by the UI/WebServer
    if (out_x >= 0 && out_y >= 0)
    {
        out_x = (out_x * 256) / m_out_width_hm;
        out_y = (out_y * 256) / m_out_height_hm;
    }
    t_decode_end = std::chrono::steady_clock::now();

    // Release all outputs
    rknn_outputs_release(m_ctx_decoder, 1, outputs_d);
    rknn_outputs_release(m_ctx_search_backbone, m_backbone_out_num, outputs_sb);

    t_end = std::chrono::steady_clock::now();

    // Calculate times in milliseconds
    resize_ms = std::chrono::duration<float, std::milli>(t_resize_end - t_resize_start).count();
    sb_ms = std::chrono::duration<float, std::milli>(t_sb_end - t_sb_start).count();
    corr_ms = std::chrono::duration<float, std::milli>(t_corr_end - t_corr_start).count();
    dec_ms = std::chrono::duration<float, std::milli>(t_dec_end - t_dec_start).count();
    decode_ms = std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();
    total_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    // Update StatusObject
    snprintf(time_buf, sizeof(time_buf), "%.2f ms", resize_ms);
    StatusObject::instance()->update("tracker_time_resize", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms (SB:%.1f, Dec:%.1f)", sb_ms + dec_ms, sb_ms, dec_ms);
    StatusObject::instance()->update("tracker_time_npu", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms (CPU Corr:%.1f)", decode_ms + corr_ms, corr_ms);
    StatusObject::instance()->update("tracker_time_decode", time_buf);

    snprintf(time_buf, sizeof(time_buf), "%.2f ms", total_ms);
    StatusObject::instance()->update("tracker_time_total", time_buf);

    // Update StatusObject with quality telemetry
    StatusObject::instance()->update("tracker_quality", "Disabled");

    // 6. Trigger TrackerCallback
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_callback != NULL)
        {
            m_callback->onTargetDetected(out_x, out_y);
            m_callback->onHeatmapCreated(m_heatmap_buf, m_out_width_hm, m_out_height_hm);
        }
        if (m_drone_cb != NULL)
        {
            int16_t yaw = 1000;
            int16_t pitch = 1000;
            int dx = out_x - 128;
            int dy = out_y - 128;
            DroneControlerHal::calculate_tracking_commands(dx, dy, yaw, pitch);
            m_drone_cb->send_command(1000, pitch, yaw, 1000);
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
    int crop_size_rga = std::min(src_w, src_h);
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

    crop_size = std::min(src_w, src_h);
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
    int y_start = std::max(0, max_y - 7);
    int y_end = std::min(m_out_height_hm, max_y + 8);
    int x_start = std::max(0, max_x - 7);
    int x_end = std::min(m_out_width_hm, max_x + 8);

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
    double std_dev = std::sqrt(std::max(0.0, variance));
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
        *out_x = (int)std::round(sum_x / total_mass);
        *out_y = (int)std::round(sum_y / total_mass);
    }
    else
    {
        *out_x = max_x;
        *out_y = max_y;
    }
}
