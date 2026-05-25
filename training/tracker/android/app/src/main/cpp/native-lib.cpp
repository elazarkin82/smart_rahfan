#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>

#define LOG_TAG "TrackerTester_NDK"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

JNIEXPORT jstring JNICALL
Java_elazarkin_ksg_external_trackertester_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++ NDK (16KB aligned)";
    return env->NewStringUTF(hello.c_str());
}

JNIEXPORT void JNICALL
Java_elazarkin_ksg_external_trackertester_MainActivity_cropAndMaskFrame(
        JNIEnv* env,
        jclass clazz,
        jobject srcBitmap,
        jfloat targetX,
        jfloat targetY,
        jfloat cropScale,
        jfloat maskRadius,
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
    int stride = info.stride; // Bytes per row
    
    // 3. Calculate Crop Window dimensions & center
    float cropW = cropScale * srcW;
    float cropH = cropScale * srcH;
    
    float cx = targetX * srcW;
    float cy = targetY * srcH;
    
    // Crop window bounds [x1, y1] with clamping to image boundaries
    float x1 = cx - cropW / 2.0f;
    float y1 = cy - cropH / 2.0f;
    
    x1 = std::max(0.0f, std::min(x1, (float)srcW - cropW));
    y1 = std::max(0.0f, std::min(y1, (float)srcH - cropH));
    
    // 4. Calculate relative target coordinates inside the clamped crop window (mapped to 256x256)
    float targetX_crop = ((targetX * srcW) - x1) / cropW * 256.0f;
    float targetY_crop = ((targetY * srcH) - y1) / cropH * 256.0f;
    
    // 5. Bilinear Resize & Grayscale Conversion
    for (int outY = 0; outY < 256; ++outY) {
        for (int outX = 0; outX < 256; ++outX) {
            
            // Check circular mask first to save resizing calculations
            if (maskRadius > 0.0f) {
                float dx = (float)outX - targetX_crop;
                float dy = (float)outY - targetY_crop;
                if ((dx * dx + dy * dy) > (maskRadius * maskRadius)) {
                    out[outY * 256 + outX] = 0.0f; // Black out
                    continue;
                }
            }
            
            // Map out coordinates to source image crop space
            float srcX = x1 + ((float)outX / 255.0f) * cropW;
            float srcY = y1 + ((float)outY / 255.0f) * cropH;
            
            // Bilinear interpolation neighbors
            int x_low = std::max(0, std::min((int)srcX, srcW - 2));
            int y_low = std::max(0, std::min((int)srcY, srcH - 2));
            int x_high = x_low + 1;
            int y_high = y_low + 1;
            
            float weightX = srcX - (float)x_low;
            float weightY = srcY - (float)y_low;
            
            auto getGrayscale = [&](int px, int py) -> float {
                uint8_t* row = (uint8_t*)pixels + py * stride;
                uint32_t* pixelPtr = (uint32_t*)row + px;
                uint8_t* rgba = (uint8_t*)pixelPtr;
                // Luminance formula: 0.299R + 0.587G + 0.114B
                return 0.299f * rgba[0] + 0.587f * rgba[1] + 0.114f * rgba[2];
            };
            
            float p00 = getGrayscale(x_low, y_low);
            float p10 = getGrayscale(x_high, y_low);
            float p01 = getGrayscale(x_low, y_high);
            float p11 = getGrayscale(x_high, y_high);
            
            float interpolatedGray = (1.0f - weightX) * (1.0f - weightY) * p00 +
                                     weightX * (1.0f - weightY) * p10 +
                                     (1.0f - weightX) * weightY * p01 +
                                     weightX * weightY * p11;
                                     
            // Normalize to [0.0, 1.0]
            out[outY * 256 + outX] = interpolatedGray / 255.0f;
        }
    }
    
    // 6. Cleanup and release
    AndroidBitmap_unlockPixels(env, srcBitmap);
    env->ReleaseFloatArrayElements(outBuffer, out, 0);
}

JNIEXPORT jfloatArray JNICALL
Java_elazarkin_ksg_external_trackertester_MainActivity_calculateCenterOfMass(
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
        res[0] = (float)(sum_x / total_mass) / 64.0f; // Normalized [0, 1]
        res[1] = (float)(sum_y / total_mass) / 64.0f;
    } else {
        res[0] = 0.5f; // Fallback to center if no activation peak found
        res[1] = 0.5f;
    }
    
    env->SetFloatArrayRegion(result, 0, 2, res);
    return result;
}

}