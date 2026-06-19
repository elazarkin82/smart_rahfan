#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#define LOG_TAG "Tracker3Lite_NDK"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jstring JNICALL
Java_elazarkin_ksg_tracker4_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++ Tracker3-Lite NDK (16KB aligned)";
    return env->NewStringUTF(hello.c_str());
}

JNIEXPORT void JNICALL
Java_elazarkin_ksg_tracker4_MainActivity_rotateYPlane(
        JNIEnv* env,
        jclass clazz,
        jbyteArray srcArray,
        jbyteArray destArray,
        jint srcW,
        jint srcH,
        jint rowStride,
        jint rotationDegrees) {
    jbyte* src = env->GetByteArrayElements(srcArray, nullptr);
    jbyte* dest = env->GetByteArrayElements(destArray, nullptr);
    
    if (src && dest) {
        uint8_t* s = (uint8_t*)src;
        uint8_t* d = (uint8_t*)dest;
        
        if (rotationDegrees == 90) {
            for (int y = 0; y < srcH; ++y) {
                for (int x = 0; x < srcW; ++x) {
                    d[x * srcH + (srcH - 1 - y)] = s[y * rowStride + x];
                }
            }
        } else if (rotationDegrees == 180) {
            for (int y = 0; y < srcH; ++y) {
                for (int x = 0; x < srcW; ++x) {
                    d[(srcH - 1 - y) * srcW + (srcW - 1 - x)] = s[y * rowStride + x];
                }
            }
        } else if (rotationDegrees == 270) {
            for (int y = 0; y < srcH; ++y) {
                for (int x = 0; x < srcW; ++x) {
                    d[(srcW - 1 - x) * srcH + y] = s[y * rowStride + x];
                }
            }
        } else {
            for (int y = 0; y < srcH; ++y) {
                std::memcpy(d + y * srcW, s + y * rowStride, srcW);
            }
        }
    }
    
    if (src) env->ReleaseByteArrayElements(srcArray, src, JNI_ABORT);
    if (dest) env->ReleaseByteArrayElements(destArray, dest, 0);
}

JNIEXPORT void JNICALL
Java_elazarkin_ksg_tracker4_MainActivity_downsampleSearchCrop(
        JNIEnv* env,
        jclass clazz,
        jbyteArray yPlane,
        jint srcW,
        jint srcH,
        jint rowStride,
        jfloat cx,
        jfloat cy,
        jfloat cropSize,
        jint outW,
        jint outH,
        jfloatArray outBuffer) {
        
    jbyte* yData = env->GetByteArrayElements(yPlane, nullptr);
    if (!yData) return;

    jfloat* out = env->GetFloatArrayElements(outBuffer, nullptr);
    if (!out) {
        env->ReleaseByteArrayElements(yPlane, yData, JNI_ABORT);
        return;
    }
    
    float halfSize = cropSize / 2.0f;
    float srcX_start = cx - halfSize;
    float srcY_start = cy - halfSize;
    
    // Bilinear Interpolated Resizing to target (Grayscale 1-Channel)
    for (int outY = 0; outY < outH; ++outY) {
        for (int outX = 0; outX < outW; ++outX) {
            int baseIdx = outY * outW + outX;
            
            // Map output index to local crop coordinates
            float cropX = ((float)outX / (float)(outW - 1)) * cropSize;
            float cropY = ((float)outY / (float)(outH - 1)) * cropSize;
            
            // Map local crop coordinates to absolute source frame coordinates
            float sx = srcX_start + cropX;
            float sy = srcY_start + cropY;
            
            int x0 = (int)std::floor(sx);
            int y0 = (int)std::floor(sy);
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            
            float dx = sx - (float)x0;
            float dy = sy - (float)y0;
            
            // Apply boundary replication padding
            int x0_c = std::max(0, std::min((int)srcW - 1, x0));
            int x1_c = std::max(0, std::min((int)srcW - 1, x1));
            int y0_c = std::max(0, std::min((int)srcH - 1, y0));
            int y1_c = std::max(0, std::min((int)srcH - 1, y1));
            
            // Fetch four neighbor pixels
            uint8_t p00 = (uint8_t)yData[y0_c * rowStride + x0_c];
            uint8_t p10 = (uint8_t)yData[y0_c * rowStride + x1_c];
            uint8_t p01 = (uint8_t)yData[y1_c * rowStride + x0_c];
            uint8_t p11 = (uint8_t)yData[y1_c * rowStride + x1_c];
            
            // Perform Bilinear Interpolation
            float val = (1.0f - dx) * (1.0f - dy) * p00 +
                        dx * (1.0f - dy) * p10 +
                        (1.0f - dx) * dy * p01 +
                        dx * dy * p11;
                        
            out[baseIdx] = val / 255.0f;
        }
    }
    
    env->ReleaseByteArrayElements(yPlane, yData, JNI_ABORT);
    env->ReleaseFloatArrayElements(outBuffer, out, 0);
}

JNIEXPORT jfloatArray JNICALL
Java_elazarkin_ksg_tracker4_MainActivity_calculateLocalRefinedArgmaxCentroid(
        JNIEnv* env,
        jclass clazz,
        jfloatArray heatmap,
        jint hmW,
        jint hmH) {
        
    jfloat* hm = env->GetFloatArrayElements(heatmap, nullptr);
    if (!hm) return nullptr;
    
    // 1. Find discrete peak (argmax)
    float max_val = -1.0f;
    int max_x = hmW / 2;
    int max_y = hmH / 2;
    
    for (int y = 0; y < hmH; ++y) {
        for (int x = 0; x < hmW; ++x) {
            float val = hm[y * hmW + x];
            if (val > max_val) {
                max_val = val;
                max_x = x;
                max_y = y;
            }
        }
    }
    
    // 2. Define 15x15 local window centered at peak
    int y_start = std::max(0, max_y - 7);
    int y_end = std::min((int)hmH, max_y + 8);
    int x_start = std::max(0, max_x - 7);
    int x_end = std::min((int)hmW, max_x + 8);
    
    // 3. Compute Mean & StdDev in this window
    double sum = 0.0;
    double sq_sum = 0.0;
    int count = 0;
    for (int y = y_start; y < y_end; ++y) {
        for (int x = x_start; x < x_end; ++x) {
            float val = hm[y * hmW + x];
            sum += val;
            sq_sum += val * val;
            count++;
        }
    }
    
    double mean = (count > 0) ? (sum / count) : 0.0;
    double variance = (count > 0) ? (sq_sum / count - mean * mean) : 0.0;
    double std_dev = std::sqrt(std::max(0.0, variance));
    float threshold = (float)(mean + 1.5 * std_dev);
    
    // 4. Non-recursive BFS to extract connected blob above threshold
    std::vector<bool> visited(hmW * hmH, false);
    std::vector<std::pair<int, int>> queue;
    
    queue.push_back({max_y, max_x});
    visited[max_y * hmW + max_x] = true;
    
    size_t head = 0;
    while (head < queue.size()) {
        auto curr = queue[head++];
        int cy = curr.first;
        int cx = curr.second;
        
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                int ny = cy + dy;
                int nx = cx + dx;
                
                // Keep BFS search strictly within the local 15x15 window
                if (ny >= y_start && ny < y_end && nx >= x_start && nx < x_end) {
                    int idx = ny * hmW + nx;
                    if (!visited[idx] && hm[idx] > threshold) {
                        visited[idx] = true;
                        queue.push_back({ny, nx});
                    }
                }
            }
        }
    }
    
    // 5. Compute Weighted Centroid of all pixels in the blob
    double sum_x = 0.0;
    double sum_y = 0.0;
    double total_mass = 0.0;
    for (const auto& p : queue) {
        int cy = p.first;
        int cx = p.second;
        float val = hm[cy * hmW + cx];
        sum_x += cx * val;
        sum_y += cy * val;
        total_mass += val;
    }
    
    LOGD("JNI Argmax search: max_val=%f, max_x=%d, max_y=%d", max_val, max_x, max_y);
    LOGD("JNI Centroid window: sum_x=%f, sum_y=%f, total_mass=%f, count=%zu, threshold=%f", sum_x, sum_y, total_mass, queue.size(), threshold);
    
    env->ReleaseFloatArrayElements(heatmap, hm, JNI_ABORT);
    
    // 6. Return coordinates normalized to [0.0, 1.0]
    jfloatArray result = env->NewFloatArray(2);
    if (!result) return nullptr;
    
    float res[2];
    if (total_mass > 1e-6) {
        res[0] = (float)(sum_x / total_mass) / (float)hmW;
        res[1] = (float)(sum_y / total_mass) / (float)hmH;
    } else {
        res[0] = (float)max_x / (float)hmW;
        res[1] = (float)max_y / (float)hmH;
    }
    
    LOGD("JNI Return: res[0]=%f, res[1]=%f", res[0], res[1]);
    
    env->SetFloatArrayRegion(result, 0, 2, res);
    return result;
}

}