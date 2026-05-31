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
Java_elazarkin_ksg_tracker4_MainActivity_downsampleSearchFrame(
        JNIEnv* env,
        jclass clazz,
        jbyteArray yPlane,
        jint srcW,
        jint srcH,
        jint rowStride,
        jfloatArray outBuffer) {
        
    jbyte* yData = env->GetByteArrayElements(yPlane, nullptr);
    if (!yData) return;

    jfloat* out = env->GetFloatArrayElements(outBuffer, nullptr);
    if (!out) {
        env->ReleaseByteArrayElements(yPlane, yData, JNI_ABORT);
        return;
    }
    
    // Area-Averaging Downsampling to 256x256 (Grayscale 1-Channel)
    for (int outY = 0; outY < 256; ++outY) {
        for (int outX = 0; outX < 256; ++outX) {
            int baseIdx = outY * 256 + outX;
            
            float srcX_start = (float)outX * (float)srcW / 256.0f;
            float srcX_end = (float)(outX + 1) * (float)srcW / 256.0f;
            float srcY_start = (float)outY * (float)srcH / 256.0f;
            float srcY_end = (float)(outY + 1) * (float)srcH / 256.0f;
            
            int x_start = std::max(0, (int)srcX_start);
            int x_end = std::min(srcW, (int)std::ceil(srcX_end));
            int y_start = std::max(0, (int)srcY_start);
            int y_end = std::min(srcH, (int)std::ceil(srcY_end));
            
            if (x_end <= x_start) x_end = x_start + 1;
            if (y_end <= y_start) y_end = y_start + 1;
            if (x_end > srcW) x_end = srcW;
            if (y_end > srcH) y_end = srcH;
            
            double sumY = 0.0;
            double totalWeight = 0.0;
            
            for (int sy = y_start; sy < y_end; ++sy) {
                float y_overlap = 1.0f;
                if (sy == y_start) {
                    y_overlap = (float)(sy + 1) - srcY_start;
                } else if (sy == y_end - 1) {
                    y_overlap = srcY_end - (float)sy;
                }
                
                for (int sx = x_start; sx < x_end; ++sx) {
                    float x_overlap = 1.0f;
                    if (sx == x_start) {
                        x_overlap = (float)(sx + 1) - srcX_start;
                    } else if (sx == x_end - 1) {
                        x_overlap = srcX_end - (float)sx;
                    }
                    
                    float weight = x_overlap * y_overlap;
                    uint8_t pixel = (uint8_t) yData[sy * rowStride + sx];
                    sumY += (pixel / 255.0f) * weight;
                    totalWeight += weight;
                }
            }
            
            if (totalWeight > 0.0) {
                out[baseIdx] = (float)(sumY / totalWeight);
            } else {
                out[baseIdx] = 0.0f;
            }
        }
    }
    
    env->ReleaseByteArrayElements(yPlane, yData, JNI_ABORT);
    env->ReleaseFloatArrayElements(outBuffer, out, 0);
}

JNIEXPORT jfloatArray JNICALL
Java_elazarkin_ksg_tracker4_MainActivity_calculateCenterOfMass(
        JNIEnv* env,
        jclass clazz,
        jfloatArray heatmap,
        jfloat threshold) {
        
    jfloat* hm = env->GetFloatArrayElements(heatmap, nullptr);
    if (!hm) return nullptr;
    
    double sum_x = 0.0;
    double sum_y = 0.0;
    double total_mass = 0.0;
    
    // Spatial integration over 256x256 grid
    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 256; ++x) {
            float val = hm[y * 256 + x];
            if (val > threshold) {
                sum_x += x * val;
                sum_y += y * val;
                total_mass += val;
            }
        }
    }
    
    env->ReleaseFloatArrayElements(heatmap, hm, JNI_ABORT); // read-only
    
    // Prepare output array
    jfloatArray result = env->NewFloatArray(2);
    if (!result) return nullptr;
    
    float res[2];
    if (total_mass > 1e-6) {
        res[0] = (float)(sum_x / total_mass) / 256.0f;
        res[1] = (float)(sum_y / total_mass) / 256.0f;
    } else {
        res[0] = 0.5f;
        res[1] = 0.5f;
    }
    
    env->SetFloatArrayRegion(result, 0, 2, res);
    return result;
}

}