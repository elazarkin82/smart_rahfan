#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_elazarkin_ksg_external_trackertester_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}