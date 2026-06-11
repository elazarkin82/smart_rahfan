#include "core/CameraCapture.h"
#include "utils/StatusObject.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#include <errno.h>

CameraCapture::CameraCapture(const char* dev, int w, int h)
{
    m_width = w;
    m_height = h;
    m_fd = -1;
    m_buffers = NULL;
    m_buffer_count = 0;
    m_callback = NULL;
    m_is_running = false;

    snprintf(m_dev_path, sizeof(m_dev_path), "%s", dev);

    // Pre-allocate the grayscale frame buffer (max expected size: 1920x1280)
    m_gray_buffer = (uchar*)malloc(1920 * 1280);
}

CameraCapture::~CameraCapture()
{
    stop_capture_thread();
    free(m_gray_buffer);
}

void CameraCapture::reconfig(const char* dev, int w, int h)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    snprintf(m_dev_path, sizeof(m_dev_path), "%s", dev);
    m_width = w;
    m_height = h;
    fprintf(stdout, "[CameraCapture] Reconfiguring to device: %s (%dx%d)\n", m_dev_path, m_width, m_height);
}

void CameraCapture::set_capture_callback(CaptureCallback* cb)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_callback = cb;
}

void CameraCapture::start_capture_thread()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_is_running)
    {
        m_is_running = true;
        m_thread = std::thread(&CameraCapture::capture_thread_loop, this);
    }
}

void CameraCapture::stop_capture_thread()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_is_running)
        {
            return;
        }
        m_is_running = false;
    }

    if (m_thread.joinable())
    {
        m_thread.join();
    }
}

bool CameraCapture::open_device(const char* dev)
{
    m_fd = open(dev, O_RDWR | O_NONBLOCK, 0);
    if (m_fd < 0)
    {
        return false;
    }
    return true;
}

bool CameraCapture::config_device()
{
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;
    unsigned int i;

    // 1. Set video format (YUYV is standard for UVC uncompressed streams)
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = m_width;
    fmt.fmt.pix.height = m_height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(m_fd, VIDIOC_S_FMT, &fmt) < 0)
    {
        fprintf(stderr, "[CameraCapture] Error setting format: %s\n", strerror(errno));
        return false;
    }

    // 2. Request memory-mapped buffers
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(m_fd, VIDIOC_REQBUFS, &req) < 0)
    {
        fprintf(stderr, "[CameraCapture] Error requesting buffers: %s\n", strerror(errno));
        return false;
    }

    m_buffer_count = req.count;
    m_buffers = (V4L2Buffer*)calloc(m_buffer_count, sizeof(V4L2Buffer));

    // 3. Map buffers to user space
    for (i = 0; i < m_buffer_count; ++i)
    {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(m_fd, VIDIOC_QUERYBUF, &buf) < 0)
        {
            fprintf(stderr, "[CameraCapture] Error querying buffer %d: %s\n", i, strerror(errno));
            return false;
        }

        m_buffers[i].length = buf.length;
        m_buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, buf.m.offset);

        if (m_buffers[i].start == MAP_FAILED)
        {
            fprintf(stderr, "[CameraCapture] Error mmapping buffer %d: %s\n", i, strerror(errno));
            return false;
        }
    }

    // 4. Queue buffers
    for (i = 0; i < m_buffer_count; ++i)
    {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(m_fd, VIDIOC_QBUF, &buf) < 0)
        {
            fprintf(stderr, "[CameraCapture] Error queueing buffer %d: %s\n", i, strerror(errno));
            return false;
        }
    }

    // 5. Start streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_STREAMON, &type) < 0)
    {
        fprintf(stderr, "[CameraCapture] Error starting stream: %s\n", strerror(errno));
        return false;
    }

    return true;
}

void CameraCapture::close_device()
{
    enum v4l2_buf_type type;
    unsigned int i;

    if (m_fd >= 0)
    {
        type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(m_fd, VIDIOC_STREAMOFF, &type);

        if (m_buffers != NULL)
        {
            for (i = 0; i < m_buffer_count; ++i)
            {
                munmap(m_buffers[i].start, m_buffers[i].length);
            }
            free(m_buffers);
            m_buffers = NULL;
        }

        close(m_fd);
        m_fd = -1;
    }
}

void CameraCapture::capture_thread_loop()
{
    char current_dev[256];
    int current_w;
    int current_h;
    bool is_connected;
    int select_ret;
    fd_set fds;
    struct timeval tv;
    struct v4l2_buffer buf;
    unsigned char* src;
    int i;

    fprintf(stdout, "[CameraCapture] Thread loop started.\n");
    StatusObject::instance()->update("camera_status", "Disconnected");

    while (true)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_is_running)
            {
                break;
            }
            snprintf(current_dev, sizeof(current_dev), "%s", m_dev_path);
            current_w = m_width;
            current_h = m_height;
        }

        // 1. Try to open the device node
        is_connected = open_device(current_dev);
        if (!is_connected)
        {
            StatusObject::instance()->update("camera_status", "Disconnected (retrying...)");
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        // 2. Configure V4L2 mappings and formats
        if (!config_device())
        {
            StatusObject::instance()->update("camera_status", "Configuration Error");
            close_device();
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        StatusObject::instance()->update("camera_status", "Connected & Streaming");
        fprintf(stdout, "[CameraCapture] Successfully started capture on %s\n", current_dev);

        // 3. Capture frames loop as long as the device path has not changed
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (!m_is_running || strcmp(current_dev, m_dev_path) != 0 || current_w != m_width || current_h != m_height)
                {
                    break;
                }
            }

            // Wait for V4L2 frame availability with timeout (select)
            FD_ZERO(&fds);
            FD_SET(m_fd, &fds);
            tv.tv_sec = 1;
            tv.tv_usec = 0;

            select_ret = select(m_fd + 1, &fds, NULL, NULL, &tv);
            if (select_ret < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }
                fprintf(stderr, "[CameraCapture] Select error: %s\n", strerror(errno));
                break; // reconnection trigger
            }
            else if (select_ret == 0)
            {
                // Timeout
                fprintf(stderr, "[CameraCapture] Frame timeout, camera might be disconnected.\n");
                break; // reconnection trigger
            }

            // Dequeue buffer
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            if (ioctl(m_fd, VIDIOC_DQBUF, &buf) < 0)
            {
                fprintf(stderr, "[CameraCapture] Error dequeueing frame buffer: %s\n", strerror(errno));
                break;
            }

            // Extract only the Y-channel (Grayscale) from YUYV buffer
            // YUYV structure: Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
            // Y0 is byte 0, Y1 is byte 2, Y2 is byte 4...
            src = (unsigned char*)m_buffers[buf.index].start;
            for (i = 0; i < current_w * current_h; ++i)
            {
                m_gray_buffer[i] = src[i * 2];
            }

            // Fire callback to MainService
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_callback != NULL)
                {
                    m_callback->onFrame(m_gray_buffer, current_w, current_h);
                }
            }

            // Queue buffer back
            if (ioctl(m_fd, VIDIOC_QBUF, &buf) < 0)
            {
                fprintf(stderr, "[CameraCapture] Error queueing back frame buffer: %s\n", strerror(errno));
                break;
            }
        }

        // Clean up current V4L2 capture and retry/update loop
        close_device();
        fprintf(stdout, "[CameraCapture] Capture stream stopped on %s\n", current_dev);
        StatusObject::instance()->update("camera_status", "Disconnected");
    }

    fprintf(stdout, "[CameraCapture] Thread loop terminated.\n");
}
