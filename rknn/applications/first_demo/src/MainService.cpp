#include "MainService.h"
#include "utils/StatusObject.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MainService::MainService(const char* params_path)
{
    m_camera = NULL;
    m_tracker = NULL;
    m_web_server = NULL;
    m_is_running = false;
    m_has_last_frame = false;
    m_has_pending_command = false;

    m_lastFrame_w = 0;
    m_lastFrame_h = 0;
    m_target_x = -1;
    m_target_y = -1;

    snprintf(m_params_path, sizeof(m_params_path), "%s", params_path);

    // Pre-allocate the shared frame buffer (max 1920x1280)
    m_lastFrame = (uchar*)malloc(1920 * 1280);

    // Load initial parameters from file
    if (!parse_params_file(m_params_path, m_params))
    {
        fprintf(stdout, "[MainService] Failed to load params file. Setting default parameters.\n");
        snprintf(m_params.cam_dev, sizeof(m_params.cam_dev), "/dev/video0");
        m_params.width = 640;
        m_params.height = 480;
        snprintf(m_params.rknn_model_path, sizeof(m_params.rknn_model_path), "/usr/bin/tracker_model.rknn");
        save_params_file(m_params_path, m_params);
    }

    // Initialize StatusObject telemetry defaults
    char res_buf[64];
    StatusObject::instance()->update("camera_fps", "0.0 FPS");
    StatusObject::instance()->update("tracker_fps", "0.0 FPS");
    StatusObject::instance()->update("tracker_time_resize", "N/A");
    StatusObject::instance()->update("tracker_time_npu", "N/A");
    StatusObject::instance()->update("tracker_time_decode", "N/A");
    StatusObject::instance()->update("tracker_time_total", "N/A");
    StatusObject::instance()->update("web_time_jpeg", "N/A");
    StatusObject::instance()->update("tracking_status", "Target Not Selected");
    StatusObject::instance()->update("target_position", "N/A");
    snprintf(res_buf, sizeof(res_buf), "%dx%d", m_params.width, m_params.height);
    StatusObject::instance()->update("camera_resolution", res_buf);
}

MainService::~MainService()
{
    stop();
    free(m_lastFrame);
}

void MainService::start()
{
    if (m_is_running)
    {
        return;
    }

    m_is_running = true;

    // 1. Create sub-services
    m_camera = new CameraCapture(m_params.cam_dev, m_params.width, m_params.height);
    m_tracker = new TrackerService(m_params.rknn_model_path);
    m_web_server = new WebServer(8080);

    // 2. Setup callbacks integration
    m_camera->set_capture_callback(this);
    m_tracker->set_tracker_callback(this);
    m_web_server->set_command_callback(this);

    // 3. Start background threads
    m_camera->start_capture_thread();
    m_main_loop_thread = std::thread(&MainService::main_loop, this);

    fprintf(stdout, "[MainService] All sub-services initialized and started.\n");
}

void MainService::stop()
{
    {
        std::lock_guard<std::mutex> lock(m_cmd_mutex);
        if (!m_is_running)
        {
            return;
        }
        m_is_running = false;
        m_cmd_condvar.notify_all();
    }

    if (m_main_loop_thread.joinable())
    {
        m_main_loop_thread.join();
    }

    if (m_camera != NULL)
    {
        m_camera->stop_capture_thread();
        delete m_camera;
        m_camera = NULL;
    }

    if (m_tracker != NULL)
    {
        delete m_tracker;
        m_tracker = NULL;
    }

    if (m_web_server != NULL)
    {
        delete m_web_server;
        m_web_server = NULL;
    }

    StatusObject::instance()->update("web_server_status", "Offline");
    StatusObject::instance()->update("camera_status", "Disconnected");
    StatusObject::instance()->update("tracker_model_status", "Unloaded");
    fprintf(stdout, "[MainService] All sub-services terminated successfully.\n");
}

void MainService::onFrame(uchar* frame, int w, int h)
{
    std::lock_guard<std::mutex> lock(m_last_frame_copy_mutex);
    std::chrono::steady_clock::time_point now;
    long long elapsed;
    float real_fps;
    char fps_buf[32];

    memcpy(m_lastFrame, frame, w * h);
    m_lastFrame_w = w;
    m_lastFrame_h = h;
    m_has_last_frame = true;

    // Report Camera FPS to Status using 5-second rolling window
    now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock_fps(m_fps_mutex);
        m_camera_frame_times.push(now);
        while (!m_camera_frame_times.empty())
        {
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_camera_frame_times.front()).count();
            if (elapsed >= 5)
            {
                m_camera_frame_times.pop();
            }
            else
            {
                break;
            }
        }
        real_fps = m_camera_frame_times.size() / 5.0f;
        snprintf(fps_buf, sizeof(fps_buf), "%.1f FPS", real_fps);
        StatusObject::instance()->update("camera_fps", fps_buf);
    }
}

void MainService::onTargetDetected(int x, int y)
{
    // Capture tracking coordinates (called by Tracker thread context)
    m_target_x = x;
    m_target_y = y;

    // Report tracker status
    char pos_buf[64];
    if (x >= 0 && y >= 0)
    {
        snprintf(pos_buf, sizeof(pos_buf), "[%d, %d]", x, y);
        StatusObject::instance()->update("tracking_status", "Target Acquired");
        StatusObject::instance()->update("target_position", pos_buf);
    }
    else
    {
        StatusObject::instance()->update("tracking_status", "Target Lost");
        StatusObject::instance()->update("target_position", "N/A");
    }
}

void MainService::onHeatmapCreated(const float* heatmap, int w, int h)
{
    if (m_web_server != NULL)
    {
        m_web_server->update_heatmap(heatmap, w, h);
    }
}

void MainService::onStackCreated(const uchar* stack, int w, int h, int c)
{
    if (m_web_server != NULL)
    {
        m_web_server->update_stack(stack, w, h, c);
    }
}

void MainService::onCommand(WebServer::Command key, const char* values, int len)
{
    std::lock_guard<std::mutex> lock(m_cmd_mutex);
    m_pending_command_key = key;
    snprintf(m_pending_command_val, sizeof(m_pending_command_val), "%s", values);
    m_has_pending_command = true;
    m_cmd_condvar.notify_all();
}

bool MainService::parse_params_file(const char* params_path, Params& out)
{
    FILE* fp;
    char line[256];
    char* eq;
    char* key;
    char* val;
    char* nl;

    fp = fopen(params_path, "r");
    if (fp == NULL)
    {
        return false;
    }

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        eq = strchr(line, '=');
        if (eq != NULL)
        {
            *eq = '\0';
            key = line;
            val = eq + 1;

            nl = strchr(val, '\n');
            if (nl != NULL)
            {
                *nl = '\0';
            }

            if (strcmp(key, "cam_dev") == 0)
            {
                snprintf(out.cam_dev, sizeof(out.cam_dev), "%s", val);
            }
            else if (strcmp(key, "capture_width") == 0)
            {
                out.width = atoi(val);
            }
            else if (strcmp(key, "capture_height") == 0)
            {
                out.height = atoi(val);
            }
            else if (strcmp(key, "rknn_model_path") == 0)
            {
                snprintf(out.rknn_model_path, sizeof(out.rknn_model_path), "%s", val);
            }
        }
    }

    fclose(fp);
    return true;
}

void MainService::save_params_file(const char* params_path, const Params& in)
{
    FILE* fp;

    fp = fopen(params_path, "w");
    if (fp != NULL)
    {
        fprintf(fp, "cam_dev=%s\n", in.cam_dev);
        fprintf(fp, "capture_width=%d\n", in.width);
        fprintf(fp, "capture_height=%d\n", in.height);
        fprintf(fp, "rknn_model_path=%s\n", in.rknn_model_path);
        fclose(fp);
        fprintf(stdout, "[MainService] Permanent configuration saved to %s\n", params_path);
    }
}

void MainService::process_command_internal(WebServer::Command key, const char* values)
{
    char val_copy[512];
    char* first_hash;
    char* second_hash;
    char dev[256];
    int w;
    int h;
    float x_n;
    float y_n;
    int target_px;
    int target_py;
    char res_buf[64];

    w = 640;
    h = 480;
    x_n = 0.0f;
    y_n = 0.0f;

    switch (key)
    {
        case WebServer::CMD_UPDATE_CAMERA_PARAMS:
            snprintf(val_copy, sizeof(val_copy), "%s", values);
            first_hash = strchr(val_copy, '#');
            if (first_hash != NULL)
            {
                *first_hash = '\0';
                strncpy(dev, val_copy, sizeof(dev));
                dev[sizeof(dev) - 1] = '\0';

                second_hash = strchr(first_hash + 1, '#');
                if (second_hash != NULL)
                {
                    *second_hash = '\0';
                    w = atoi(first_hash + 1);
                    h = atoi(second_hash + 1);
                }

                // Apply changes to Camera component
                m_camera->reconfig(dev, w, h);

                // Update local params state
                strncpy(m_params.cam_dev, dev, sizeof(m_params.cam_dev));
                m_params.cam_dev[sizeof(m_params.cam_dev) - 1] = '\0';
                m_params.width = w;
                m_params.height = h;

                snprintf(res_buf, sizeof(res_buf), "%dx%d", w, h);
                StatusObject::instance()->update("camera_resolution", res_buf);
            }
            break;

        case WebServer::CMD_SAVE_PARAMS:
            save_params_file(m_params_path, m_params);
            break;

        case WebServer::CMD_RESET_TARGET:
            m_tracker->clear_target();
            if (m_web_server != NULL)
            {
                m_web_server->update_heatmap(NULL, 256, 256);
                m_web_server->update_stack(NULL, 64, 64, 16);
            }
            m_target_x = -1;
            m_target_y = -1;
            StatusObject::instance()->update("tracking_status", "Target Not Selected");
            StatusObject::instance()->update("target_position", "N/A");
            break;

        case WebServer::CMD_CHOOSE_TARGET:
            sscanf(values, "%f,%f", &x_n, &y_n);
            
            // Only crop if we have valid camera frames active
            if (m_lastFrame_w > 0 && m_lastFrame_h > 0)
            {
                target_px = (int)(x_n * m_lastFrame_w);
                target_py = (int)(y_n * m_lastFrame_h);

                // Initialize tracker reference templates with full frame and target pixel coordinates
                m_tracker->refresh_target(m_lastFrame, m_lastFrame_w, m_lastFrame_h, target_px, target_py);

                // Reset tracker outputs coordinates to start fresh
                m_target_x = (int)(x_n * 256.0f);
                m_target_y = (int)(y_n * 256.0f);
            }
            break;
    }
}

void MainService::main_loop()
{
    uchar* work_frame;
    int work_w;
    int work_h;
    bool has_frame;
    std::chrono::steady_clock::time_point now;
    long long elapsed;
    float real_fps;
    char fps_buf[32];

    work_frame = (uchar*)malloc(1920 * 1280);
    work_w = 0;
    work_h = 0;
    has_frame = false;

    while (m_is_running)
    {
        // 1. Safe Processing of Web Commands inside main execution thread
        {
            std::unique_lock<std::mutex> lock(m_cmd_mutex);
            if (m_has_pending_command)
            {
                process_command_internal(m_pending_command_key, m_pending_command_val);
                m_has_pending_command = false;
            }
        }

        // 2. Fetch the latest camera frame
        {
            std::lock_guard<std::mutex> lock(m_last_frame_copy_mutex);
            if (m_has_last_frame)
            {
                memcpy(work_frame, m_lastFrame, m_lastFrame_w * m_lastFrame_h);
                work_w = m_lastFrame_w;
                work_h = m_lastFrame_h;
                has_frame = true;
                m_has_last_frame = false;
            }
        }

        // 3. Execute sequential tracking (RKNN Inference)
        if (has_frame)
        {
            m_tracker->update_frame(work_frame, work_w, work_h);
            
            // Calculate Tracker FPS using 5-second rolling window if target is defined
            if (m_tracker->is_target_defined())
            {
                now = std::chrono::steady_clock::now();
                {
                    std::lock_guard<std::mutex> lock_fps(m_fps_mutex);
                    m_tracker_frame_times.push(now);
                    while (!m_tracker_frame_times.empty())
                    {
                        elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_tracker_frame_times.front()).count();
                        if (elapsed >= 5)
                        {
                            m_tracker_frame_times.pop();
                        }
                        else
                        {
                            break;
                        }
                    }
                    real_fps = m_tracker_frame_times.size() / 5.0f;
                    snprintf(fps_buf, sizeof(fps_buf), "%.1f FPS", real_fps);
                    StatusObject::instance()->update("tracker_fps", fps_buf);
                }
            }
            else
            {
                // Clear rolling queue when idle
                {
                    std::lock_guard<std::mutex> lock_fps(m_fps_mutex);
                    while (!m_tracker_frame_times.empty())
                    {
                        m_tracker_frame_times.pop();
                    }
                }
                StatusObject::instance()->update("tracker_fps", "0.0 FPS");
                // Clear times in status
                StatusObject::instance()->update("tracker_time_resize", "N/A");
                StatusObject::instance()->update("tracker_time_npu", "N/A");
                StatusObject::instance()->update("tracker_time_decode", "N/A");
                StatusObject::instance()->update("tracker_time_total", "N/A");
            }
            
            // Send tracking overlays to web server stream
            m_web_server->update(work_frame, work_w, work_h, m_target_x, m_target_y);
            
            has_frame = false;
        }

        // Sleep/wait on CV to avoid thread spin
        {
            std::unique_lock<std::mutex> lock(m_cmd_mutex);
            m_cmd_condvar.wait_for(lock, std::chrono::milliseconds(10), [this]() {
                return m_has_pending_command || !m_is_running;
            });
        }
    }

    free(work_frame);
}
