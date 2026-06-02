#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>

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
    
    // Bilinear Interpolated Resizing to 256x256 (Grayscale 1-Channel)
    for (int outY = 0; outY < 256; ++outY) {
        for (int outX = 0; outX < 256; ++outX) {
            int baseIdx = outY * 256 + outX;
            
            // Map output index to local crop coordinates
            float cropX = ((float)outX / 255.0f) * cropSize;
            float cropY = ((float)outY / 255.0f) * cropSize;
            
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
        jfloatArray heatmap) {
        
    jfloat* hm = env->GetFloatArrayElements(heatmap, nullptr);
    if (!hm) return nullptr;
    
    // 1. Find absolute global peak (argmax)
    float max_val = -1.0f;
    int max_x = 128;
    int max_y = 128;
    
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 256; ++x) {
            float val = hm[y * 256 + x];
            if (val > max_val) {
                max_val = val;
                max_x = x;
                max_y = y;
            }
        }
    }
    
    // 2. Compute Center of Mass strictly within a local 5x5 window
    double sum_x = 0.0;
    double sum_y = 0.0;
    double total_mass = 0.0;
    
    for (int dy = -2; dy <= 2; ++dy) {
        int sy = max_y + dy;
        if (sy < 0 || sy >= 256) continue;
        
        for (int dx = -2; dx <= 2; ++dx) {
            int sx = max_x + dx;
            if (sx < 0 || sx >= 256) continue;
            
            float val = hm[sy * 256 + sx];
            sum_x += sx * val;
            sum_y += sy * val;
            total_mass += val;
        }
    }
    
    env->ReleaseFloatArrayElements(heatmap, hm, JNI_ABORT); // read-only
    
    // 3. Construct and return result array
    jfloatArray result = env->NewFloatArray(2);
    if (!result) return nullptr;
    
    float res[2];
    if (total_mass > 1e-6) {
        res[0] = (float)(sum_x / total_mass) / 256.0f;
        res[1] = (float)(sum_y / total_mass) / 256.0f;
    } else {
        res[0] = (float)max_x / 256.0f;
        res[1] = (float)max_y / 256.0f;
    }
    
    env->SetFloatArrayRegion(result, 0, 2, res);
    return result;
}

}