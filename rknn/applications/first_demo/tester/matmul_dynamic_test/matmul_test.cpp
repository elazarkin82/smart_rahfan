#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <random>
#include "rknn_api.h"

// Helper function to load RKNN model from file
static void* load_model(const char* filename, int* size)
{
    FILE* fp;
    void* data;
    
    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    *size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    data = malloc(*size);
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
    const char* model_path = "matmul.rknn";
    int model_size = 0;
    void* model_data = NULL;
    rknn_context ctx = 0;
    int ret = 0;
    
    rknn_input_output_num io_num;
    rknn_tensor_attr in_attrs[2];
    rknn_tensor_attr out_attrs[1];
    
    float* data_A = NULL;
    float* data_B = NULL;
    float* ref_C = NULL;
    
    rknn_input inputs[2];
    rknn_output outputs[1];
    
    float* npu_C = NULL;
    
    double sum_sig = 0.0;
    double sum_noise = 0.0;
    double max_diff = 0.0;
    double sum_abs_diff = 0.0;
    double sqnr = 0.0;
    double mae = 0.0;
    
    int M = 128;
    int K = 64;
    int N = 128;
    int i, j, k;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    if (argc > 1)
    {
        model_path = argv[1];
    }
    
    fprintf(stdout, "=== Loading model: %s ===\n", model_path);
    model_data = load_model(model_path, &model_size);
    if (model_data == NULL)
    {
        fprintf(stderr, "Error: Failed to load model file %s\n", model_path);
        return -1;
    }
    
    fprintf(stdout, "=== Initializing RKNN context ===\n");
    ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_init failed with code %d\n", ret);
        return -1;
    }
    
    // Query dynamic tensor info
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_query RKNN_QUERY_IN_OUT_NUM failed: %d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    fprintf(stdout, "Model has %d inputs and %d outputs.\n", io_num.n_input, io_num.n_output);
    if (io_num.n_input != 2 || io_num.n_output != 1)
    {
        fprintf(stderr, "Error: Expected 2 inputs and 1 output for MatMul model.\n");
        rknn_destroy(ctx);
        return -1;
    }
    
    memset(in_attrs, 0, sizeof(in_attrs));
    in_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[0], sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_query input[0] attr failed: %d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    in_attrs[1].index = 1;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attrs[1], sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_query input[1] attr failed: %d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    memset(out_attrs, 0, sizeof(out_attrs));
    out_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[0], sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_query output[0] attr failed: %d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }
    
    fprintf(stdout, "Input 0 shape: [%u, %u, %u, %u]\n", in_attrs[0].dims[0], in_attrs[0].dims[1], in_attrs[0].dims[2], in_attrs[0].dims[3]);
    fprintf(stdout, "Input 1 shape: [%u, %u, %u, %u]\n", in_attrs[1].dims[0], in_attrs[1].dims[1], in_attrs[1].dims[2], in_attrs[1].dims[3]);
    fprintf(stdout, "Output shape:  [%u, %u, %u, %u]\n", out_attrs[0].dims[0], out_attrs[0].dims[1], out_attrs[0].dims[2], out_attrs[0].dims[3]);
    
    // Allocate buffer for inputs and CPU reference output
    data_A = (float*)malloc(M * K * sizeof(float));
    data_B = (float*)malloc(K * N * sizeof(float));
    ref_C  = (float*)malloc(M * N * sizeof(float));
    
    if (data_A == NULL || data_B == NULL || ref_C == NULL)
    {
        fprintf(stderr, "Error: Out of memory during buffer allocation.\n");
        free(data_A); free(data_B); free(ref_C);
        rknn_destroy(ctx);
        return -1;
    }
    
    // Fill inputs with random values
    for (i = 0; i < M * K; ++i)
    {
        data_A[i] = dis(gen);
    }
    for (i = 0; i < K * N; ++i)
    {
        data_B[i] = dis(gen);
    }
    
    // Setup inputs for NPU
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].size = M * K * sizeof(float);
    inputs[0].buf = data_A;
    inputs[0].fmt = RKNN_TENSOR_NHWC; // using default layout
    
    inputs[1].index = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT32;
    inputs[1].size = K * N * sizeof(float);
    inputs[1].buf = data_B;
    inputs[1].fmt = RKNN_TENSOR_NHWC;
    
    fprintf(stdout, "=== Running NPU Inference ===\n");
    ret = rknn_inputs_set(ctx, 2, inputs);
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_inputs_set failed: %d\n", ret);
        free(data_A); free(data_B); free(ref_C);
        rknn_destroy(ctx);
        return -1;
    }
    
    auto t0 = std::chrono::steady_clock::now();
    ret = rknn_run(ctx, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_run failed: %d\n", ret);
        free(data_A); free(data_B); free(ref_C);
        rknn_destroy(ctx);
        return -1;
    }
    
    memset(outputs, 0, sizeof(outputs));
    outputs[0].index = 0;
    outputs[0].want_float = 1;
    
    ret = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0)
    {
        fprintf(stderr, "Error: rknn_outputs_get failed: %d\n", ret);
        free(data_A); free(data_B); free(ref_C);
        rknn_destroy(ctx);
        return -1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double npu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    npu_C = (float*)outputs[0].buf;
    
    fprintf(stdout, "NPU Inference completed in %.3f ms\n", npu_ms);
    
    fprintf(stdout, "=== Simulating CPU MatMul ===\n");
    auto t2 = std::chrono::steady_clock::now();
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            float val = 0.0f;
            for (k = 0; k < K; ++k)
            {
                val += data_A[i * K + k] * data_B[k * N + j];
            }
            ref_C[i * N + j] = val;
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fprintf(stdout, "CPU MatMul completed in %.3f ms\n", cpu_ms);
    
    fprintf(stdout, "=== Comparing Results ===\n");
    for (i = 0; i < M * N; ++i)
    {
        double diff = std::abs((double)npu_C[i] - (double)ref_C[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
        sum_abs_diff += diff;
        sum_noise += diff * diff;
        sum_sig += (double)ref_C[i] * (double)ref_C[i];
    }
    
    mae = sum_abs_diff / (M * N);
    sqnr = (sum_noise > 0.0) ? (10.0 * std::log10(sum_sig / sum_noise)) : 999.0;
    
    fprintf(stdout, "\n=== METRICS ===\n");
    fprintf(stdout, "  - Max Absolute Difference: %f\n", max_diff);
    fprintf(stdout, "  - Mean Absolute Error (MAE): %f\n", mae);
    fprintf(stdout, "  - Signal Power:              %f\n", sum_sig);
    fprintf(stdout, "  - Noise Power (MSE * count): %f\n", sum_noise);
    fprintf(stdout, "  - SQNR:                      %.2f dB\n\n", sqnr);
    
    // Clean up
    rknn_outputs_release(ctx, 1, outputs);
    free(data_A);
    free(data_B);
    free(ref_C);
    rknn_destroy(ctx);
    
    fprintf(stdout, "Test finished.\n");
    return 0;
}
