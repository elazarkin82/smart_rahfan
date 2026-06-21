#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <sstream>
#include "rknn_api.h"
#include "utils/NNOperationsCpu.hpp"

struct Metrics
{
    double mae;
    double mse;
    double max_diff;
    int max_diff_idx;
    double cosine_sim;
    double sqnr;
};

static Metrics compute_metrics(const float* a, const float* b, int count)
{
    Metrics m;
    m.mae = 0.0;
    m.mse = 0.0;
    m.max_diff = 0.0;
    m.max_diff_idx = 0;
    m.sqnr = 0.0;
    
    double sum_abs_diff = 0.0;
    double sum_sq_diff = 0.0;
    double sum_a2 = 0.0;
    double sum_b2 = 0.0;
    double sum_ab = 0.0;
    
    for (int i = 0; i < count; ++i)
    {
        double diff = std::abs((double)a[i] - (double)b[i]);
        if (diff > m.max_diff)
        {
            m.max_diff = diff;
            m.max_diff_idx = i;
        }
        sum_abs_diff += diff;
        sum_sq_diff += diff * diff;
        sum_ab += (double)a[i] * (double)b[i];
        sum_a2 += (double)a[i] * (double)a[i];
        sum_b2 += (double)b[i] * (double)b[i];
    }
    
    m.mae = sum_abs_diff / count;
    m.mse = sum_sq_diff / count;
    m.cosine_sim = (sum_a2 > 0.0 && sum_b2 > 0.0) ? (sum_ab / (std::sqrt(sum_a2) * std::sqrt(sum_b2))) : 0.0;
    m.sqnr = (sum_sq_diff > 0.0) ? (10.0 * std::log10(sum_b2 / sum_sq_diff)) : 999.0;
    return m;
}

static void* load_file(const char* filepath, int* size)
{
    FILE* fp = fopen(filepath, "rb");
    if (fp == NULL)
    {
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    void* data = malloc(*size);
    if (data)
    {
        int bytes_read = fread(data, 1, *size, fp);
        (void)bytes_read;
    }
    fclose(fp);
    return data;
}

static void print_tensor_attr(std::ostream& out, const rknn_tensor_attr& attr)
{
    out << "  - name: " << attr.name << "\n"
        << "    index: " << attr.index << "\n"
        << "    n_dims: " << attr.n_dims << "\n"
        << "    dims: [";
    for (uint32_t i = 0; i < attr.n_dims; ++i)
    {
        out << attr.dims[i] << (i == attr.n_dims - 1 ? "" : ", ");
    }
    out << "]\n"
        << "    size: " << attr.size << "\n"
        << "    type: " << attr.type << "\n"
        << "    fmt: " << attr.fmt << "\n";
}

int main(int argc, char** argv)
{
    const char* template_path = "tracker_template.rknn";
    const char* backbone_path = "tracker_search_backbone.rknn";
    const char* decoder_path = "tracker_decoder.rknn";
    const char* test_data_path = "test_sample.bin";
    const char* log_path = "tester_results.log";

    if (argc > 1) template_path = argv[1];
    if (argc > 2) backbone_path = argv[2];
    if (argc > 3) decoder_path = argv[3];
    if (argc > 4) test_data_path = argv[4];

    std::ofstream log_file(log_path);
    if (!log_file.is_open())
    {
        std::cerr << "Failed to open log file for writing: " << log_path << std::endl;
        return -1;
    }

    auto log_write = [&](const std::string& msg) {
        std::cout << msg << std::flush;
        log_file << msg << std::flush;
    };

    log_write("====================================================\n");
    log_write("RKNN TARGET TRACKER SPLIT PIPELINE NPU-CPU-NPU TESTER\n");
    log_write("====================================================\n\n");

    // 1. Read test_sample.bin (Version 3)
    log_write("[*] Loading test data: " + std::string(test_data_path) + " ...\n");
    std::ifstream file(test_data_path, std::ios::binary);
    if (!file.is_open())
    {
        log_write("[ERROR] Failed to open test data file.\n");
        return -1;
    }

    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "TSTD", 4) != 0)
    {
        log_write("[ERROR] Invalid magic header in test data!\n");
        return -1;
    }

    uint32_t version;
    file.read((char*)&version, 4);
    if (version != 3)
    {
        log_write("[ERROR] Tester requires Version 3 test data. Found version: " + std::to_string(version) + "\n");
        return -1;
    }

    std::vector<uint8_t> ref_stack(128 * 128 * 2);
    std::vector<uint8_t> search_frame(256 * 256 * 1);
    std::vector<float> expected_features(8 * 8 * 64);
    std::vector<float> expected_skip1(128 * 128 * 8);
    std::vector<float> expected_skip2(64 * 64 * 16);
    std::vector<float> expected_skip3(32 * 32 * 32);
    std::vector<float> expected_search_features(16 * 16 * 64);
    std::vector<float> expected_corr_out(16 * 16 * 64);
    std::vector<float> expected_heatmap(32 * 32 * 1);
    float expected_quality = 0.0f;

    file.read((char*)ref_stack.data(), ref_stack.size());
    file.read((char*)search_frame.data(), search_frame.size());
    file.read((char*)expected_features.data(), expected_features.size() * sizeof(float));
    file.read((char*)expected_skip1.data(), expected_skip1.size() * sizeof(float));
    file.read((char*)expected_skip2.data(), expected_skip2.size() * sizeof(float));
    file.read((char*)expected_skip3.data(), expected_skip3.size() * sizeof(float));
    file.read((char*)expected_search_features.data(), expected_search_features.size() * sizeof(float));
    file.read((char*)expected_corr_out.data(), expected_corr_out.size() * sizeof(float));
    file.read((char*)expected_heatmap.data(), expected_heatmap.size() * sizeof(float));
    file.read((char*)&expected_quality, sizeof(float));

    log_write("[SUCCESS] Test sample loaded (Version " + std::to_string(version) + ").\n");

    auto print_stats_uint8 = [&](const std::string& name, const std::vector<uint8_t>& vec) {
        double sum = 0;
        int min_val = 255;
        int max_val = 0;
        for (size_t i = 0; i < vec.size(); ++i) {
            sum += vec[i];
            if (vec[i] < min_val) min_val = vec[i];
            if (vec[i] > max_val) max_val = vec[i];
        }
        log_write("  - " + name + ": min=" + std::to_string(min_val) + ", max=" + std::to_string(max_val) + ", mean=" + std::to_string(sum / vec.size()) + "\n");
    };

    auto print_stats_float = [&](const std::string& name, const std::vector<float>& vec) {
        double sum = 0;
        float min_val = vec[0];
        float max_val = vec[0];
        for (size_t i = 0; i < vec.size(); ++i) {
            sum += vec[i];
            if (vec[i] < min_val) min_val = vec[i];
            if (vec[i] > max_val) max_val = vec[i];
        }
        log_write("  - " + name + ": min=" + std::to_string(min_val) + ", max=" + std::to_string(max_val) + ", mean=" + std::to_string(sum / vec.size()) + "\n");
    };

    log_write("Loaded Tensor Statistics:\n");
    print_stats_uint8("ref_stack", ref_stack);
    print_stats_uint8("search_frame", search_frame);
    print_stats_float("expected_features", expected_features);
    print_stats_float("expected_skip1", expected_skip1);
    print_stats_float("expected_skip2", expected_skip2);
    print_stats_float("expected_skip3", expected_skip3);
    print_stats_float("expected_search_features", expected_search_features);
    print_stats_float("expected_corr_out", expected_corr_out);
    print_stats_float("expected_heatmap", expected_heatmap);
    log_write("\n");

    // 2. Load models
    log_write("[*] Loading RKNN models...\n");
    int size_t = 0;
    void* data_t = load_file(template_path, &size_t);
    if (!data_t)
    {
        log_write("[ERROR] Failed to load template model: " + std::string(template_path) + "\n");
        return -1;
    }

    int size_b = 0;
    void* data_b = load_file(backbone_path, &size_b);
    if (!data_b)
    {
        log_write("[ERROR] Failed to load search backbone model: " + std::string(backbone_path) + "\n");
        free(data_t);
        return -1;
    }

    int size_d = 0;
    void* data_d = load_file(decoder_path, &size_d);
    if (!data_d)
    {
        log_write("[ERROR] Failed to load decoder model: " + std::string(decoder_path) + "\n");
        free(data_t);
        free(data_b);
        return -1;
    }
    log_write("[SUCCESS] RKNN models loaded into memory.\n\n");

    // 3. Initialize RKNN contexts
    log_write("[*] Initializing RKNN contexts on NPU...\n");
    rknn_context ctx_t;
    rknn_context ctx_sb;
    rknn_context ctx_d;

    int ret = rknn_init(&ctx_t, data_t, size_t, 0, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_init for template failed: " + std::to_string(ret) + "\n");
        free(data_t);
        free(data_b);
        free(data_d);
        return -1;
    }

    ret = rknn_init(&ctx_sb, data_b, size_b, 0, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_init for search backbone failed: " + std::to_string(ret) + "\n");
        rknn_destroy(ctx_t);
        free(data_t);
        free(data_b);
        free(data_d);
        return -1;
    }

    ret = rknn_init(&ctx_d, data_d, size_d, 0, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_init for decoder failed: " + std::to_string(ret) + "\n");
        rknn_destroy(ctx_t);
        rknn_destroy(ctx_sb);
        free(data_t);
        free(data_b);
        free(data_d);
        return -1;
    }
    log_write("[SUCCESS] RKNN NPU contexts initialized.\n\n");

    rknn_output outputs_t[1];
    std::memset(outputs_t, 0, sizeof(outputs_t));
    bool outputs_t_acquired = false;

    rknn_output outputs_sb[4];
    std::memset(outputs_sb, 0, sizeof(outputs_sb));
    bool outputs_sb_acquired = false;

    rknn_output outputs_d[1];
    std::memset(outputs_d, 0, sizeof(outputs_d));
    bool outputs_d_acquired = false;

    auto cleanup_and_exit = [&](int exit_code) -> int {
        log_write("\n[*] Cleaning up contexts...\n");
        if (outputs_d_acquired)
        {
            rknn_outputs_release(ctx_d, 1, outputs_d);
        }
        if (outputs_sb_acquired)
        {
            rknn_outputs_release(ctx_sb, 4, outputs_sb);
        }
        if (outputs_t_acquired)
        {
            rknn_outputs_release(ctx_t, 1, outputs_t);
        }
        rknn_destroy(ctx_t);
        rknn_destroy(ctx_sb);
        rknn_destroy(ctx_d);
        free(data_t);
        free(data_b);
        free(data_d);
        log_write("====================================================\n");
        log_write("TESTING COMPLETED\n");
        log_write("====================================================\n");
        return exit_code;
    };

    // 4. Query & Print Model Attributes
    log_write("====================================================\n");
    log_write("TENSOR ATTRIBUTES:\n");
    log_write("====================================================\n");
    
    rknn_input_output_num io_num_t;
    rknn_query(ctx_t, RKNN_QUERY_IN_OUT_NUM, &io_num_t, sizeof(io_num_t));
    
    rknn_input_output_num io_num_sb;
    rknn_query(ctx_sb, RKNN_QUERY_IN_OUT_NUM, &io_num_sb, sizeof(io_num_sb));

    rknn_input_output_num io_num_d;
    rknn_query(ctx_d, RKNN_QUERY_IN_OUT_NUM, &io_num_d, sizeof(io_num_d));

    log_write("Template Model Inputs: " + std::to_string(io_num_t.n_input) + ", Outputs: " + std::to_string(io_num_t.n_output) + "\n");
    for (uint32_t i = 0; i < io_num_t.n_input; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_t, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        std::stringstream ss;
        print_tensor_attr(ss, attr);
        log_write(ss.str());
    }
    for (uint32_t i = 0; i < io_num_t.n_output; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_t, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        std::stringstream ss;
        print_tensor_attr(ss, attr);
        log_write(ss.str());
    }

    log_write("\nSearch Backbone Model Inputs: " + std::to_string(io_num_sb.n_input) + ", Outputs: " + std::to_string(io_num_sb.n_output) + "\n");
    for (uint32_t i = 0; i < io_num_sb.n_input; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_sb, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        std::stringstream ss;
        print_tensor_attr(ss, attr);
        log_write(ss.str());
    }
    for (uint32_t i = 0; i < io_num_sb.n_output; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_sb, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        std::stringstream ss;
        print_tensor_attr(ss, attr);
        log_write(ss.str());
    }

    log_write("\nDecoder Model Inputs: " + std::to_string(io_num_d.n_input) + ", Outputs: " + std::to_string(io_num_d.n_output) + "\n");
    for (uint32_t i = 0; i < io_num_d.n_input; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_d, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        std::stringstream ss;
        print_tensor_attr(ss, attr);
        log_write(ss.str());
    }
    for (uint32_t i = 0; i < io_num_d.n_output; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_d, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        std::stringstream ss;
        print_tensor_attr(ss, attr);
        log_write(ss.str());
    }
    log_write("\n");

    // Dynamic output index resolution for Search Backbone based on shape (handles NCHW/NHWC layout query)
    int idx_skip1 = -1;
    int idx_skip2 = -1;
    int idx_skip3 = -1;
    int idx_search_features = -1;
    rknn_tensor_format fmt_skip1 = RKNN_TENSOR_NCHW;
    rknn_tensor_format fmt_skip2 = RKNN_TENSOR_NCHW;
    rknn_tensor_format fmt_skip3 = RKNN_TENSOR_NCHW;
    rknn_tensor_format fmt_search_features = RKNN_TENSOR_NCHW;

    for (uint32_t i = 0; i < io_num_sb.n_output; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_sb, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        
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
        
        if (height == 128 && width == 128 && channels == 8)
        {
            idx_skip1 = i;
            fmt_skip1 = RKNN_TENSOR_NHWC;
        }
        else if (height == 64 && width == 64 && channels == 16)
        {
            idx_skip2 = i;
            fmt_skip2 = RKNN_TENSOR_NHWC;
        }
        else if (height == 32 && width == 32 && channels == 32)
        {
            idx_skip3 = i;
            fmt_skip3 = RKNN_TENSOR_NHWC;
        }
        else if (height == 16 && width == 16 && channels == 64)
        {
            idx_search_features = i;
            fmt_search_features = RKNN_TENSOR_NHWC;
        }
    }

    log_write("Resolved Search Backbone output indexes:\n");
    log_write("  - skip1: " + std::to_string(idx_skip1) + "\n");
    log_write("  - skip2: " + std::to_string(idx_skip2) + "\n");
    log_write("  - skip3: " + std::to_string(idx_skip3) + "\n");
    log_write("  - search_features: " + std::to_string(idx_search_features) + "\n\n");

    if (idx_skip3 == -1 || idx_search_features == -1)
    {
        log_write("[ERROR] Could not resolve all required search backbone outputs (skip3 and search_features) by shape.\n");
        return cleanup_and_exit(-1);
    }

    // Dynamic input index resolution for Decoder based on shape (handles NCHW/NHWC layout query)
    int idx_dec_corr = -1;
    int idx_dec_skip3 = -1;

    for (uint32_t i = 0; i < io_num_d.n_input; ++i)
    {
        rknn_tensor_attr attr;
        attr.index = i;
        rknn_query(ctx_d, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        
        uint32_t channels = 0, height = 0, width = 0;
        if (attr.dims[2] == attr.dims[3]) // NCHW layout
        {
            channels = attr.dims[1];
            height = attr.dims[2];
            width = attr.dims[3];
        }
        else // NHWC layout
        {
            height = attr.dims[1];
            width = attr.dims[2];
            channels = attr.dims[3];
        }
        
        if (height == 32 && width == 32 && channels == 32) idx_dec_skip3 = i;
        else if (height == 16 && width == 16 && channels == 64) idx_dec_corr = i;
    }

    log_write("Resolved Decoder input indexes:\n");
    log_write("  - corr_features: " + std::to_string(idx_dec_corr) + "\n");
    log_write("  - skip3: " + std::to_string(idx_dec_skip3) + "\n\n");

    if (idx_dec_corr == -1 || idx_dec_skip3 == -1)
    {
        log_write("[ERROR] Could not resolve all decoder inputs by shape.\n");
        return cleanup_and_exit(-1);
    }

    // 5. Test Template Subgraph
    log_write("====================================================\n");
    log_write("TESTING TEMPLATE SUBGRAPH:\n");
    log_write("====================================================\n");

    rknn_input inputs_t[1];
    std::memset(inputs_t, 0, sizeof(inputs_t));
    inputs_t[0].index = 0;
    inputs_t[0].type = RKNN_TENSOR_UINT8;  // NPU handles division by 255
    inputs_t[0].size = ref_stack.size();
    inputs_t[0].buf = ref_stack.data();
    inputs_t[0].fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx_t, 1, inputs_t);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_inputs_set template failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }

    auto t_t_start = std::chrono::steady_clock::now();
    ret = rknn_run(ctx_t, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_run template failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }

    std::memset(outputs_t, 0, sizeof(outputs_t));
    outputs_t[0].index = 0;
    outputs_t[0].want_float = 1;

    ret = rknn_outputs_get(ctx_t, 1, outputs_t, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_outputs_get template failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }
    outputs_t_acquired = true;
    auto t_t_end = std::chrono::steady_clock::now();
    double template_ms = std::chrono::duration<double, std::milli>(t_t_end - t_t_start).count();

    rknn_tensor_format fmt_template = RKNN_TENSOR_NCHW;
    {
        rknn_tensor_attr attr;
        attr.index = 0;
        rknn_query(ctx_t, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        fmt_template = attr.fmt;
    }

    // Compare computed template features to expected features
    {
        const float* computed_features = (const float*)outputs_t[0].buf;
        int count = expected_features.size();
        std::vector<float> computed_features_nhwc(8 * 8 * 64, 0.0f);
        
        // Transpose NCHW [1, 64, 8, 8] to NHWC [1, 8, 8, 64]
        NNOperationsCpu::transpose_nchw_to_nhwc(computed_features, computed_features_nhwc.data(), 1, 64, 8, 8);

        Metrics m = compute_metrics(computed_features_nhwc.data(), expected_features.data(), count);
        
        log_write("Template Subgraph Execution Latency: " + std::to_string(template_ms) + " ms\n");
        log_write("Template Features Match Metrics (TFLite vs RKNN FP16):\n");
        log_write("  - Mean Absolute Error (MAE): " + std::to_string(m.mae) + "\n");
        log_write("  - Cosine Similarity: " + std::to_string(m.cosine_sim) + "\n");
        log_write("  - SQNR: " + std::to_string(m.sqnr) + " dB\n\n");
    }

    // 6. Test Split pipeline execution (Backbone NPU -> CPU Correlation -> Decoder NPU)
    log_write("====================================================\n");
    log_write("TESTING SPLIT NPU-CPU-NPU PIPELINE:\n");
    log_write("====================================================\n");

    // 6a. Run Backbone NPU
    rknn_input inputs_sb[1];
    std::memset(inputs_sb, 0, sizeof(inputs_sb));
    inputs_sb[0].index = 0;
    inputs_sb[0].type = RKNN_TENSOR_UINT8;  // NPU handles division by 255
    inputs_sb[0].size = search_frame.size();
    inputs_sb[0].buf = search_frame.data();
    inputs_sb[0].fmt = RKNN_TENSOR_NHWC;

    log_write("[*] 6a. Setting search backbone inputs...\n");
    ret = rknn_inputs_set(ctx_sb, 1, inputs_sb);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_inputs_set search backbone failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }

    log_write("[*] 6a. Running search backbone on NPU...\n");
    auto t_start = std::chrono::steady_clock::now();
    ret = rknn_run(ctx_sb, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_run search backbone failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }

    log_write("[*] 6a. Getting search backbone outputs...\n");
    std::memset(outputs_sb, 0, sizeof(outputs_sb));
    for (uint32_t i = 0; i < io_num_sb.n_output; ++i)
    {
        outputs_sb[i].index = i;
        outputs_sb[i].want_float = 1;
    }

    ret = rknn_outputs_get(ctx_sb, io_num_sb.n_output, outputs_sb, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_outputs_get search backbone failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }
    outputs_sb_acquired = true;
    auto t_backbone = std::chrono::steady_clock::now();

    log_write("[*] 6b. Running CPU depthwise correlation...\n");
    auto t_corr_start = std::chrono::steady_clock::now();

    // 6b. Run CPU Depthwise Correlation (expects NCHW, produces NCHW)
    std::vector<float> sf_nchw(64 * 16 * 16, 0.0f);
    if (fmt_search_features == RKNN_TENSOR_NHWC)
    {
        NNOperationsCpu::transpose_nhwc_to_nchw((const float*)outputs_sb[idx_search_features].buf, sf_nchw.data(), 1, 64, 16, 16);
    }
    else
    {
        std::memcpy(sf_nchw.data(), outputs_sb[idx_search_features].buf, 64 * 16 * 16 * sizeof(float));
    }

    std::vector<float> template_nchw(64 * 8 * 8, 0.0f);
    if (fmt_template == RKNN_TENSOR_NHWC)
    {
        NNOperationsCpu::transpose_nhwc_to_nchw((const float*)outputs_t[0].buf, template_nchw.data(), 1, 64, 8, 8);
    }
    else
    {
        std::memcpy(template_nchw.data(), outputs_t[0].buf, 64 * 8 * 8 * sizeof(float));
    }

    std::vector<float> corr_out(64 * 16 * 16, 0.0f);
    NNOperationsCpu::depthwise_correlation_nchw_neon_omp(
        sf_nchw.data(),
        template_nchw.data(),
        corr_out.data(),
        64, 16, 16, 8, 8
    );

    // Transpose correlation output and skip3 output to NHWC layout (needed for Decoder inputs)
    std::vector<float> corr_out_nhwc(64 * 16 * 16, 0.0f);
    NNOperationsCpu::transpose_nchw_to_nhwc(corr_out.data(), corr_out_nhwc.data(), 1, 64, 16, 16);

    std::vector<float> skip3_nhwc(32 * 32 * 32, 0.0f);
    if (fmt_skip3 == RKNN_TENSOR_NHWC)
    {
        std::memcpy(skip3_nhwc.data(), outputs_sb[idx_skip3].buf, 32 * 32 * 32 * sizeof(float));
    }
    else
    {
        NNOperationsCpu::transpose_nchw_to_nhwc((const float*)outputs_sb[idx_skip3].buf, skip3_nhwc.data(), 1, 32, 32, 32);
    }
    auto t_corr_end = std::chrono::steady_clock::now();

    // --- Metric Calculations (Run outside the timed CPU correlation path) ---
    {
        // 1. skip1 (if index resolved)
        if (idx_skip1 != -1)
        {
            std::vector<float> skip1_nhwc(128 * 128 * 8, 0.0f);
            if (fmt_skip1 == RKNN_TENSOR_NHWC)
            {
                std::memcpy(skip1_nhwc.data(), outputs_sb[idx_skip1].buf, 128 * 128 * 8 * sizeof(float));
            }
            else
            {
                NNOperationsCpu::transpose_nchw_to_nhwc((const float*)outputs_sb[idx_skip1].buf, skip1_nhwc.data(), 1, 8, 128, 128);
            }
            Metrics m_skip1 = compute_metrics(skip1_nhwc.data(), expected_skip1.data(), 128 * 128 * 8);
            log_write("Backbone Skip1 Match Metrics:\n");
            log_write("  - Cosine Similarity: " + std::to_string(m_skip1.cosine_sim) + " (MAE: " + std::to_string(m_skip1.mae) + ")\n");
            log_write("  - SQNR: " + std::to_string(m_skip1.sqnr) + " dB\n");
        }

        // 2. skip2 (if index resolved)
        if (idx_skip2 != -1)
        {
            std::vector<float> skip2_nhwc(64 * 64 * 16, 0.0f);
            if (fmt_skip2 == RKNN_TENSOR_NHWC)
            {
                std::memcpy(skip2_nhwc.data(), outputs_sb[idx_skip2].buf, 64 * 64 * 16 * sizeof(float));
            }
            else
            {
                NNOperationsCpu::transpose_nchw_to_nhwc((const float*)outputs_sb[idx_skip2].buf, skip2_nhwc.data(), 1, 16, 64, 64);
            }
            Metrics m_skip2 = compute_metrics(skip2_nhwc.data(), expected_skip2.data(), 64 * 64 * 16);
            log_write("Backbone Skip2 Match Metrics:\n");
            log_write("  - Cosine Similarity: " + std::to_string(m_skip2.cosine_sim) + " (MAE: " + std::to_string(m_skip2.mae) + ")\n");
            log_write("  - SQNR: " + std::to_string(m_skip2.sqnr) + " dB\n");
            log_write("  - First 10 expected_skip2 values: ");
            for (int i = 0; i < 10; ++i) log_write(std::to_string(expected_skip2[i]) + " ");
            log_write("\n  - First 10 computed skip2_nhwc values: ");
            for (int i = 0; i < 10; ++i) log_write(std::to_string(skip2_nhwc[i]) + " ");
            log_write("\n  - First 10 raw computed values: ");
            const float* raw_skip2 = (const float*)outputs_sb[idx_skip2].buf;
            for (int i = 0; i < 10; ++i) log_write(std::to_string(raw_skip2[i]) + " ");
            log_write("\n");
        }

        // 3. skip3
        {
            Metrics m_skip3 = compute_metrics(skip3_nhwc.data(), expected_skip3.data(), 32 * 32 * 32);
            log_write("Backbone Skip3 Match Metrics:\n");
            log_write("  - Cosine Similarity: " + std::to_string(m_skip3.cosine_sim) + " (MAE: " + std::to_string(m_skip3.mae) + ")\n");
            log_write("  - SQNR: " + std::to_string(m_skip3.sqnr) + " dB\n");
        }

        // 4. search features
        {
            std::vector<float> search_features_nhwc(16 * 16 * 64, 0.0f);
            if (fmt_search_features == RKNN_TENSOR_NHWC)
            {
                std::memcpy(search_features_nhwc.data(), outputs_sb[idx_search_features].buf, 16 * 16 * 64 * sizeof(float));
            }
            else
            {
                NNOperationsCpu::transpose_nchw_to_nhwc((const float*)outputs_sb[idx_search_features].buf, search_features_nhwc.data(), 1, 64, 16, 16);
            }
            Metrics m_sf = compute_metrics(search_features_nhwc.data(), expected_search_features.data(), 16 * 16 * 64);
            log_write("Backbone Search Features Match Metrics:\n");
            log_write("  - Cosine Similarity: " + std::to_string(m_sf.cosine_sim) + " (MAE: " + std::to_string(m_sf.mae) + ")\n");
            log_write("  - SQNR: " + std::to_string(m_sf.sqnr) + " dB\n\n");
        }

        // 5. Depthwise correlation output
        {
            Metrics m_corr = compute_metrics(corr_out_nhwc.data(), expected_corr_out.data(), 16 * 16 * 64);
            log_write("Depthwise Correlation Output Match Metrics (TFLite vs CPU NEON+OMP):\n");
            log_write("  - MAE: " + std::to_string(m_corr.mae) + "\n");
            log_write("  - Cosine Similarity: " + std::to_string(m_corr.cosine_sim) + "\n");
            log_write("  - SQNR: " + std::to_string(m_corr.sqnr) + " dB\n\n");
        }
    }

    log_write("[*] 6c. Setting decoder inputs...\n");
    // 6c. Run Decoder NPU
    rknn_input inputs_d[2];
    std::memset(inputs_d, 0, sizeof(inputs_d));

    // Assign dynamically mapped indices
    inputs_d[idx_dec_corr].index = idx_dec_corr;
    inputs_d[idx_dec_corr].type = RKNN_TENSOR_FLOAT32;
    inputs_d[idx_dec_corr].size = corr_out_nhwc.size() * sizeof(float);
    inputs_d[idx_dec_corr].buf = corr_out_nhwc.data();
    inputs_d[idx_dec_corr].fmt = RKNN_TENSOR_NHWC;

    inputs_d[idx_dec_skip3].index = idx_dec_skip3;
    inputs_d[idx_dec_skip3].type = RKNN_TENSOR_FLOAT32;
    inputs_d[idx_dec_skip3].size = skip3_nhwc.size() * sizeof(float);
    inputs_d[idx_dec_skip3].buf = skip3_nhwc.data();
    inputs_d[idx_dec_skip3].fmt = RKNN_TENSOR_NHWC;

    auto t_dec_start = std::chrono::steady_clock::now();
    ret = rknn_inputs_set(ctx_d, 2, inputs_d);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_inputs_set decoder failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }

    ret = rknn_run(ctx_d, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_run decoder failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }

    std::memset(outputs_d, 0, sizeof(outputs_d));
    outputs_d[0].index = 0;
    outputs_d[0].want_float = 1;

    ret = rknn_outputs_get(ctx_d, 1, outputs_d, NULL);
    if (ret < 0)
    {
        log_write("[ERROR] rknn_outputs_get decoder failed: " + std::to_string(ret) + "\n");
        return cleanup_and_exit(ret);
    }
    outputs_d_acquired = true;
    auto t_decoder = std::chrono::steady_clock::now();

    // Latency Telemetry
    double backbone_ms = std::chrono::duration<double, std::milli>(t_backbone - t_start).count();
    double corr_ms = std::chrono::duration<double, std::milli>(t_corr_end - t_corr_start).count();
    double decoder_ms = std::chrono::duration<double, std::milli>(t_decoder - t_dec_start).count();
    double total_ms = backbone_ms + corr_ms + decoder_ms;

    log_write("Pipeline Execution Latency:\n");
    log_write("  - Search Backbone (NPU): " + std::to_string(backbone_ms) + " ms\n");
    log_write("  - Depthwise Corr (CPU):  " + std::to_string(corr_ms) + " ms\n");
    log_write("  - Heatmap Decoder (NPU): " + std::to_string(decoder_ms) + " ms\n");
    log_write("  - Total Frame Pipeline:  " + std::to_string(total_ms) + " ms\n\n");

    // Compare Heatmap outputs
    {
        const float* computed_heatmap = (const float*)outputs_d[0].buf;
        int count = expected_heatmap.size();
        Metrics m = compute_metrics(computed_heatmap, expected_heatmap.data(), count);
        
        log_write("Heatmap Match Metrics (TFLite vs Split Pipeline):\n");
        log_write("  - Mean Absolute Error (MAE): " + std::to_string(m.mae) + "\n");
        log_write("  - Mean Squared Error (MSE): " + std::to_string(m.mse) + "\n");
        log_write("  - Max Absolute Difference: " + std::to_string(m.max_diff) + " (at flat index: " + std::to_string(m.max_diff_idx) + ")\n");
        log_write("  - Cosine Similarity: " + std::to_string(m.cosine_sim) + "\n");
        log_write("  - SQNR: " + std::to_string(m.sqnr) + " dB\n");

        // Find argmax locations (peak location)
        int exp_argmax = 0;
        float exp_max_val = expected_heatmap[0];
        int rknn_argmax = 0;
        float rknn_max_val = computed_heatmap[0];
        
        for (int i = 1; i < count; ++i)
        {
            if (expected_heatmap[i] > exp_max_val)
            {
                exp_max_val = expected_heatmap[i];
                exp_argmax = i;
            }
            if (computed_heatmap[i] > rknn_max_val)
            {
                rknn_max_val = computed_heatmap[i];
                rknn_argmax = i;
            }
        }

        int exp_x = exp_argmax % 32;
        int exp_y = exp_argmax / 32;
        
        int rknn_x = rknn_argmax % 32;
        int rknn_y = rknn_argmax / 32;

        log_write("\nArgmax Peak Mismatch:\n");
        log_write("  - Expected (TFLite) Argmax: index " + std::to_string(exp_argmax) + " -> Coordinate [" + std::to_string(exp_x) + ", " + std::to_string(exp_y) + "] (value: " + std::to_string(exp_max_val) + ")\n");
        log_write("  - Actual (Split Pipeline) Argmax: index " + std::to_string(rknn_argmax) + " -> Coordinate [" + std::to_string(rknn_x) + ", " + std::to_string(rknn_y) + "] (value: " + std::to_string(rknn_max_val) + ")\n");
        
        double peak_dist = std::sqrt((exp_x - rknn_x) * (exp_x - rknn_x) + (exp_y - rknn_y) * (exp_y - rknn_y));
        log_write("  - Peak Euclidean Mismatch Distance: " + std::to_string(peak_dist) + " pixels\n");

        log_write("\nFirst 10 Heatmap Values Comparison:\n");
        for (int i = 0; i < 10 && i < count; ++i)
        {
            log_write("  [" + std::to_string(i) + "] TFLite: " + std::to_string(expected_heatmap[i]) + " | RKNN: " + std::to_string(computed_heatmap[i]) + "\n");
        }
        log_write("\n");
    }

    return cleanup_and_exit(0);
}
