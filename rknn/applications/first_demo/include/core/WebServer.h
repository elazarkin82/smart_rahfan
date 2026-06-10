#ifndef WEB_SERVER_H
#define WEB_SERVER_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include "CivetServer.h"

typedef unsigned char uchar;

class WebServer
{
public:
    enum Command
    {
        CMD_UPDATE_CAMERA_PARAMS = 1,
        CMD_SAVE_PARAMS = 2,
        CMD_CHOOSE_TARGET = 3
    };

    class CommandCallback
    {
    public:
        virtual ~CommandCallback() {}
        virtual void onCommand(Command key, const char* values, int len) = 0;
    };

private:
    CivetServer* m_server;
    CommandCallback* m_cmd_callback;
    std::mutex m_mutex;
    std::condition_variable m_condvar;

    // Stream state & buffers
    bool m_is_streaming;
    uchar* m_frame_buf;
    int m_frame_w;
    int m_frame_h;
    int m_target_x;
    int m_target_y;
    bool m_has_new_frame;

    // Pre-allocated JPEG output buffers
    uchar* m_jpeg_buf;
    unsigned long m_jpeg_size;

    // Helper compression function
    void compress_gray_to_jpeg(const uchar* gray_buf, int w, int h, uchar* dest_buf, unsigned long* dest_size);

public:
    WebServer(int port);
    ~WebServer();

    void set_command_callback(CommandCallback* cb);
    void update(uchar* frame, int w, int h, int target_x, int target_y);
    void trigger_command(Command key, const char* values);

    // Stream synchronization API called by CivetWeb handlers
    bool is_streaming() const;
    void set_streaming(bool state);
    void wait_for_frame(uchar* jpeg_dest, unsigned long* jpeg_len);
};

#endif
