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
Java_elazarkin_ksg_tracker3_1lite_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++ Tracker3-Lite NDK (16KB aligned)";
    return env->NewStringUTF(hello.c_str());
}

JNIEXPORT void JNICALL
Java_elazarkin_ksg_tracker3_1lite_MainActivity_downsampleAndMaskFrameV3(
        JNIEnv* env,
        jclass clazz,
        jobject srcBitmap,
        jfloat targetX,
        jfloat targetY,
        jfloat maskRadius,
        jboolean useExponentialMask,
        jfloat maskSigma,
        jboolean isSearchFrame,
        jint numChannels,
        jfloatArray outBuffer) {
        
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    
    // 1. Validate & Lock Bitmap
    if (AndroidBitmap_getInfo(env, srcBitmap, &info) < 0) {
        LOGE("AndroidBitmap_getInfo failed");
        return;
    }
    
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Invalid bitmap format, expected RGBA_8888");
        return;
    }
    
    if (AndroidBitmap_lockPixels(env, srcBitmap, &pixels) < 0) {
        LOGE("AndroidBitmap_lockPixels failed");
        return;
    }
    
    // 2. Get output float buffer pointer
    jfloat* out = env->GetFloatArrayElements(outBuffer, nullptr);
    if (!out) {
        AndroidBitmap_unlockPixels(env, srcBitmap);
        return;
    }
    
    int srcW = info.width;
    int srcH = info.height;
    int stride = info.stride;
    
    // Target position mapped directly into 256x256 output coordinate space
    float targetX_256 = targetX * 256.0f;
    float targetY_256 = targetY * 256.0f;
    
    auto getGrayscale = [&](int px, int py) -> float {
        uint8_t* row = (uint8_t*)pixels + py * stride;
        uint32_t* pixelPtr = (uint32_t*)row + px;
        uint8_t* rgba = (uint8_t*)pixelPtr;
        // Standard Luminance: 0.299R + 0.587G + 0.114B
        return 0.299f * rgba[0] + 0.587f * rgba[1] + 0.114f * rgba[2];
    };
    
    // 3. Area-Averaging Downsampling to 256x256
    for (int outY = 0; outY < 256; ++outY) {
        for (int outX = 0; outX < 256; ++outX) {
            int baseIdx = (outY * 256 + outX) * numChannels;
            
            // Map output pixel boundaries back to fractional source image space
            float srcX_start = (float)outX * (float)srcW / 256.0f;
            float srcX_end = (float)(outX + 1) * (float)srcW / 256.0f;
            float srcY_start = (float)outY * (float)srcH / 256.0f;
            float srcY_end = (float)(outY + 1) * (float)srcH / 256.0f;
            
            int x_start = std::max(0, (int)srcX_start);
            int x_end = std::min(srcW, (int)std::ceil(srcX_end));
            int y_start = std::max(0, (int)srcY_start);
            int y_end = std::min(srcH, (int)std::ceil(srcY_end));
            
            // Boundary safety
            if (x_end <= x_start) x_end = x_start + 1;
            if (y_end <= y_start) y_end = y_start + 1;
            if (x_end > srcW) x_end = srcW;
            if (y_end > srcH) y_end = srcH;
            
            double sum = 0.0;
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
                    sum += getGrayscale(sx, sy) * weight;
                    totalWeight += weight;
                }
            }
            
            float averaged = (totalWeight > 0.0) ? (float)(sum / totalWeight) : 0.0f;
            
            // Channel 0: Grayscale pixel value [0.0, 1.0]
            out[baseIdx + 0] = averaged / 255.0f;
            
            // Channel 1: Exponential Cone Attention Mask (for 2-channel inputs like hist)
            if (numChannels >= 2) {
                if (isSearchFrame) {
                    out[baseIdx + 1] = 0.0f;
                } else {
                    float dx = (float)outX - targetX_256;
                    float dy = (float)outY - targetY_256;
                    float distSq = dx * dx + dy * dy;
                    
                    if (useExponentialMask) {
                        float d = std::sqrt(distSq);
                        out[baseIdx + 1] = std::exp(-d / maskSigma);
                    } else {
                        out[baseIdx + 1] = (distSq <= (maskRadius * maskRadius)) ? 1.0f : 0.0f;
                    }
                }
            }
        }
    }
    
    // 4. Cleanup and release
    AndroidBitmap_unlockPixels(env, srcBitmap);
    env->ReleaseFloatArrayElements(outBuffer, out, 0);
}

JNIEXPORT jfloatArray JNICALL
Java_elazarkin_ksg_tracker3_1lite_MainActivity_calculateCenterOfMass(
        JNIEnv* env,
        jclass clazz,
        jfloatArray heatmap,
        jfloat threshold) {
        
    jfloat* hm = env->GetFloatArrayElements(heatmap, nullptr);
    if (!hm) return nullptr;
    
    double sum_x = 0.0;
    double sum_y = 0.0;
    double total_mass = 0.0;
    
    // Spatial integration over 64x64 grid
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            float val = hm[y * 64 + x];
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
        res[0] = (float)(sum_x / total_mass) / 64.0f;
        res[1] = (float)(sum_y / total_mass) / 64.0f;
    } else {
        res[0] = 0.5f;
        res[1] = 0.5f;
    }
    
    env->SetFloatArrayRegion(result, 0, 2, res);
    return result;
}

}