#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include "rknn_api.h"

// Helper to load binary files
static void* load_file(const char* filename, int* size)
{
    FILE* fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    void* data = malloc(*size);
    if (data == NULL)
    {
        fclose(fp);
        return NULL;
    }
    
    if (fread(data, 1, *size, fp) != (size_t)*size)
    {
        free(data);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    return data;
}

int main(int argc, char** argv)
{
    const char* model_path = "matmul_corr.rknn";
    int model_size = 0;
    void* model_data = NULL;
    rknn_context ctx = 0;
    int ret = 0;
    
    rknn_input_output_num io_num;
    rknn_tensor_attr in_attrs[2];
    rknn_tensor_attr out_attrs[2];
    
    float* data_ref_stack = NULL;
    float* data_search_frame = NULL;
    float* keras_hm = NULL;
    float* keras_q = NULL;
    
    rknn_input inputs[2];
    rknn_output outputs[2];
    
    if (argc > 1)
    {
        model_path = argv[1];
    }
    
    fprintf(stdout, "=== Loading RKNN model: %s ===\n", model_path);
    model_data = load_file(model_path, &model_size);
    if (model_data == NULL)
    {
        fprintf(stderr, "[ERROR] Failed to load model file %s\n", model_path);
        return -1;
    }
    
    fprintf(stdout, "=== Initializing RKNN context ===\n");
    ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0)
    {
        fprintf(stderr, "[ERROR] rknn_init failed with code %d\n", ret);
        return -1;
    }
    
    // Query dynamic tensor info
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        fprintf(stderr, "[ERROR] rknn_query RKNN_QUERY_IN_OUT_NUM failed: %d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    fprintf(stdout, "Model has %d inputs and %d outputs.\n", io_num.n_input, io_num.n_output);
    if (io_num.n_input != 2 || io_num.n_output != 2)
    {
        fprintf(stderr, "[ERROR] Expected 2 inputs and 2 outputs for this model. Found inputs=%d outputs=%d\n", io_num.n_input, io_num.n_output);
        rknn_destroy(ctx);
        return -1;
    }
    
    // Query attributes
    memset(in_attrs, 0, sizeof(in_attrs));
    for (uint32_t i = 0; i < io_num.n_input; ++i)
    {
        in_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn_tensor_attr));
        fprintf(stdout, "Input [%d] name: '%s', shape: [%u, %u, %u, %u]\n",
                i, in_attrs[i].name, in_attrs[i].dims[0], in_attrs[i].dims[1], in_attrs[i].dims[2], in_attrs[i].dims[3]);
    }
    
    memset(out_attrs, 0, sizeof(out_attrs));
    for (uint32_t i = 0; i < io_num.n_output; ++i)
    {
        out_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn_tensor_attr));
        fprintf(stdout, "Output [%d] name: '%s', shape: [%u, %u, %u, %u]\n",
                i, out_attrs[i].name, out_attrs[i].dims[0], out_attrs[i].dims[1], out_attrs[i].dims[2], out_attrs[i].dims[3]);
    }
    
    // Map input tensors dynamically by name
    int ref_idx = -1;
    int search_idx = -1;
    for (int i = 0; i < 2; ++i)
    {
        if (strstr(in_attrs[i].name, "reference_stack") != NULL) ref_idx = i;
        else if (strstr(in_attrs[i].name, "search_frame") != NULL) search_idx = i;
    }
    if (ref_idx == -1 || search_idx == -1)
    {
        fprintf(stdout, "[WARNING] Dynamic input name matching failed. Falling back to default order.\n");
        ref_idx = 0;
        search_idx = 1;
    }
    
    // Map output tensors dynamically by element count
    int hm_idx = -1;
    int q_idx = -1;
    for (int i = 0; i < 2; ++i)
    {
        if (out_attrs[i].n_elems == 1) q_idx = i;
        else hm_idx = i;
    }
    if (hm_idx == -1 || q_idx == -1)
    {
        fprintf(stdout, "[WARNING] Dynamic output mapping failed. Falling back to default order.\n");
        hm_idx = 0;
        q_idx = 1;
    }
    
    // Load inputs
    int sz;
    data_ref_stack = (float*)load_file("ref_stack.bin", &sz);
    if (data_ref_stack == NULL || (uint32_t)sz != in_attrs[ref_idx].n_elems * sizeof(float))
    {
        fprintf(stderr, "[ERROR] Failed to load ref_stack.bin or size mismatched (got %d bytes, expected %u)\n", sz, (uint32_t)(in_attrs[ref_idx].n_elems * sizeof(float)));
        return -1;
    }
    
    data_search_frame = (float*)load_file("search_frame.bin", &sz);
    if (data_search_frame == NULL || (uint32_t)sz != in_attrs[search_idx].n_elems * sizeof(float))
    {
        fprintf(stderr, "[ERROR] Failed to load search_frame.bin or size mismatched (got %d bytes, expected %u)\n", sz, (uint32_t)(in_attrs[search_idx].n_elems * sizeof(float)));
        return -1;
    }
    
    // Load Keras reference outputs
    keras_hm = (float*)load_file("keras_heatmap.bin", &sz);
    if (keras_hm == NULL || (uint32_t)sz != out_attrs[hm_idx].n_elems * sizeof(float))
    {
        fprintf(stderr, "[ERROR] Failed to load keras_heatmap.bin or size mismatched (got %d bytes, expected %u)\n", sz, (uint32_t)(out_attrs[hm_idx].n_elems * sizeof(float)));
        return -1;
    }
    
    keras_q = (float*)load_file("keras_quality.bin", &sz);
    if (keras_q == NULL || (uint32_t)sz != out_attrs[q_idx].n_elems * sizeof(float))
    {
        fprintf(stderr, "[ERROR] Failed to load keras_quality.bin or size mismatched\n");
        return -1;
    }
    
    // Set inputs
    memset(inputs, 0, sizeof(inputs));
    inputs[ref_idx].index = ref_idx;
    inputs[ref_idx].type = RKNN_TENSOR_FLOAT32;
    inputs[ref_idx].size = in_attrs[ref_idx].n_elems * sizeof(float);
    inputs[ref_idx].buf = data_ref_stack;
    inputs[ref_idx].fmt = RKNN_TENSOR_NHWC;
    
    inputs[search_idx].index = search_idx;
    inputs[search_idx].type = RKNN_TENSOR_FLOAT32;
    inputs[search_idx].size = in_attrs[search_idx].n_elems * sizeof(float);
    inputs[search_idx].buf = data_search_frame;
    inputs[search_idx].fmt = RKNN_TENSOR_NHWC;
    
    ret = rknn_inputs_set(ctx, 2, inputs);
    if (ret < 0)
    {
        fprintf(stderr, "[ERROR] rknn_inputs_set failed: %d\n", ret);
        return -1;
    }
    
    // 1. Run and benchmark NPU performance (average over 100 runs)
    fprintf(stdout, "=== Running NPU benchmark (100 iterations) ===\n");
    int loops = 100;
    auto t_start = std::chrono::steady_clock::now();
    for (int l = 0; l < loops; ++l)
    {
        ret = rknn_run(ctx, NULL);
        if (ret < 0)
        {
            fprintf(stderr, "[ERROR] rknn_run failed at iteration %d with code %d\n", l, ret);
            return -1;
        }
    }
    auto t_end = std::chrono::steady_clock::now();
    double avg_npu_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count() / loops;
    fprintf(stdout, "NPU average inference execution time: %.3f ms\n", avg_npu_ms);
    
    // 2. Retrieve outputs
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 1;
    outputs[1].index = 1;
    outputs[1].want_float = 1;
    
    ret = rknn_outputs_get(ctx, 2, outputs, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "[ERROR] rknn_outputs_get failed: %d\n", ret);
        return -1;
    }
    
    float* npu_hm = (float*)outputs[hm_idx].buf;
    float* npu_q = (float*)outputs[q_idx].buf;
    
    // 3. Compute SQNR and metrics comparing NPU heatmap vs Keras heatmap
    double sum_sig = 0.0;
    double sum_noise = 0.0;
    double max_diff = 0.0;
    double sum_abs_diff = 0.0;
    int hm_len = out_attrs[hm_idx].n_elems;
    
    for (int i = 0; i < hm_len; ++i)
    {
        double diff = std::abs((double)npu_hm[i] - (double)keras_hm[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
        sum_abs_diff += diff;
        sum_noise += diff * diff;
        sum_sig += (double)keras_hm[i] * (double)keras_hm[i];
    }
    
    double mae = sum_abs_diff / hm_len;
    double sqnr = (sum_noise > 0.0) ? (10.0 * std::log10(sum_sig / sum_noise)) : 999.0;
    
    fprintf(stdout, "\n==================================================\n");
    fprintf(stdout, "=== HEATMAP MATCHING METRICS (NPU FP16 vs KERAS) ===\n");
    fprintf(stdout, "==================================================\n");
    fprintf(stdout, "  - Max Absolute Difference: %f\n", max_diff);
    fprintf(stdout, "  - Mean Absolute Error (MAE): %f\n", mae);
    fprintf(stdout, "  - Signal Power (Keras):      %f\n", sum_sig);
    fprintf(stdout, "  - Noise Power (Diff MSE):    %f\n", sum_noise);
    fprintf(stdout, "  - Heatmap SQNR:              %.2f dB\n", sqnr);
    fprintf(stdout, "--------------------------------------------------\n");
    fprintf(stdout, "=== QUALITY HEAD MATCHING ===\n");
    fprintf(stdout, "  - Keras Quality Output:      %f\n", keras_q[0]);
    fprintf(stdout, "  - NPU Quality Output:        %f\n", npu_q[0]);
    fprintf(stdout, "  - Quality Output Delta:      %f\n", std::abs(keras_q[0] - npu_q[0]));
    fprintf(stdout, "==================================================\n\n");
    
    // Cleanup
    rknn_outputs_release(ctx, 2, outputs);
    free(data_ref_stack);
    free(data_search_frame);
    free(keras_hm);
    free(keras_q);
    rknn_destroy(ctx);
    
    return 0;
}
