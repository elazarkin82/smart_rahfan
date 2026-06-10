#ifndef CAMERA_CAPTURE_H
#define CAMERA_CAPTURE_H

#include <thread>
#include <mutex>
#include <condition_variable>

typedef unsigned char uchar;

class CameraCapture
{
public:
    class CaptureCallback
    {
    public:
        virtual ~CaptureCallback() {}
        virtual void onFrame(uchar* frame, int w, int h, int frame_rate = 30) = 0;
    };

private:
    struct V4L2Buffer
    {
        void* start;
        size_t length;
    };

    char m_dev_path[256];
    int m_width;
    int m_height;
    int m_fd;
    V4L2Buffer* m_buffers;
    unsigned int m_buffer_count;

    CaptureCallback* m_callback;
    std::thread m_thread;
    std::mutex m_mutex;
    bool m_is_running;

    // Buffer allocated once to store extracted grayscale (Y-channel) frames
    uchar* m_gray_buffer;

    // Private lifecycle methods
    bool open_device(const char* dev);
    bool config_device();
    void close_device();
    void capture_thread_loop();

public:
    CameraCapture(const char* dev, int w, int h);
    ~CameraCapture();

    void reconfig(const char* dev, int w, int h);
    void set_capture_callback(CaptureCallback* cb);
    void start_capture_thread();
    void stop_capture_thread();
};

#endif
