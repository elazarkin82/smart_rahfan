#ifndef MAIN_SERVICE_H
#define MAIN_SERVICE_H

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include "core/CameraCapture.h"
#include "core/TrackerService.h"
#include "core/WebServer.h"
#include "core/DroneControler.h"

class MainService : public CameraCapture::CaptureCallback,
                    public TrackerService::TrackerCallback,
                    public WebServer::CommandCallback
{
public:
    struct Params
    {
        char cam_dev[256];
        int width;
        int height;
        char rknn_model_path[256];
        char quality_mode[64];
        float quality_lost_threshold;
        float quality_display_threshold;
        float min_crop;
        float max_crop;
        int decode_argmax_only;
        int iterations_num;
        char drone_serial_port[256];
        int drone_controller_id;
    };

private:
    enum
    {
        FPS_HISTORY_MAX = 1024
    };

    Params m_params;
    char m_params_path[256];

    CameraCapture* m_camera;
    TrackerService* m_tracker;
    WebServer* m_web_server;
    DroneControler* m_drone;

    // Last captured frame buffer
    uchar* m_lastFrame;
    int m_lastFrame_w;
    int m_lastFrame_h;
    std::mutex m_last_frame_copy_mutex;
    bool m_has_last_frame;

    // Track coordinates
    int m_target_x;
    int m_target_y;
    bool m_target_low_quality;

    // Service lifecycle
    bool m_is_running;
    std::thread m_main_loop_thread;

    // Web Command Queuing / Signaling
    std::condition_variable m_cmd_condvar;
    std::mutex m_cmd_mutex;
    bool m_has_pending_command;
    WebServer::Command m_pending_command_key;
    char m_pending_command_val[512];

    // Real FPS calculations (rolling 5-second window)
    std::chrono::steady_clock::time_point m_camera_frame_times[FPS_HISTORY_MAX];
    std::chrono::steady_clock::time_point m_tracker_frame_times[FPS_HISTORY_MAX];
    int m_camera_frame_count;
    int m_camera_frame_next;
    int m_tracker_frame_count;
    int m_tracker_frame_next;
    std::mutex m_fps_mutex;

    // Helper functions
    bool parse_params_file(const char* params_path, Params& out);
    void save_params_file(const char* params_path, const Params& in);
    void process_command_internal(WebServer::Command key, const char* values);
    void main_loop();

public:
    MainService(const char* params_path);
    ~MainService();

    void start();
    void stop();

    // Callbacks implementation
    void onFrame(uchar* frame, int w, int h) override;
    void onTargetDetected(int x, int y, bool low_quality) override;
    void onHeatmapCreated(const float* heatmap, int w, int h) override;
    void onStackCreated(const uchar* stack, int w, int h, int c) override;
    void onCommand(WebServer::Command key, const char* values, int len) override;
};

#endif
