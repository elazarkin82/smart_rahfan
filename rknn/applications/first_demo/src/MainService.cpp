#include "MainService.h"
#include "utils/StatusObject.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

MainService::MainService(const char* params_path)
{
    char res_buf[64];

    m_camera = NULL;
    m_tracker = NULL;
    m_web_server = NULL;
    m_drone = NULL;
    m_is_running = false;
    m_has_last_frame = false;
    m_has_pending_command = false;

    m_lastFrame_w = 0;
    m_lastFrame_h = 0;
    m_target_x = -1;
    m_target_y = -1;
    m_target_low_quality = false;
    m_camera_frame_count = 0;
    m_camera_frame_next = 0;
    m_tracker_frame_count = 0;
    m_tracker_frame_next = 0;

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
        snprintf(m_params.rknn_model_path, sizeof(m_params.rknn_model_path), "matmul_corr.rknn");
        snprintf(m_params.quality_mode, sizeof(m_params.quality_mode), "disabled");
        m_params.quality_lost_threshold = 0.20f;
        m_params.quality_display_threshold = 0.20f;
        m_params.min_crop = 64.0f;
        m_params.max_crop = 256.0f;
        m_params.decode_argmax_only = 0;
        m_params.iterations_num = 1;
        snprintf(m_params.drone_serial_port, sizeof(m_params.drone_serial_port), "/dev/ttyUSB0");
        m_params.drone_controller_id = -1;
        save_params_file(m_params_path, m_params);
    }

    // Initialize StatusObject telemetry defaults
    StatusObject::instance()->update("camera_fps", "0.0 FPS");
    StatusObject::instance()->update("tracker_fps", "0.0 FPS");
    StatusObject::instance()->update("tracker_time_resize", "N/A");
    StatusObject::instance()->update("tracker_time_npu", "N/A");
    StatusObject::instance()->update("tracker_time_decode", "N/A");
    StatusObject::instance()->update("tracker_time_total", "N/A");
    StatusObject::instance()->update("web_time_jpeg", "N/A");
    StatusObject::instance()->update("tracking_status", "Target Not Selected");
    StatusObject::instance()->update("target_position", "N/A");
    StatusObject::instance()->update("flight_mode", "Manual");
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
    bool quality_enabled;

    if (m_is_running)
    {
        return;
    }

    m_is_running = true;

    // 1. Create sub-services
    m_camera = new CameraCapture(m_params.cam_dev, m_params.width, m_params.height);
    quality_enabled = (strcmp(m_params.quality_mode, "disabled") != 0);
    m_tracker = new TrackerService(
        m_params.rknn_model_path,
        m_params.min_crop,
        m_params.max_crop,
        quality_enabled,
        m_params.decode_argmax_only != 0,
        m_params.iterations_num,
        m_params.quality_lost_threshold,
        m_params.quality_display_threshold
    );
    m_web_server = new WebServer(8080);
    m_drone = new DroneControler(m_params.drone_serial_port, m_params.drone_controller_id);

    // 2. Setup callbacks integration
    m_camera->set_capture_callback(this);
    m_tracker->set_tracker_callback(this);
    m_web_server->set_command_callback(this);
    m_web_server->set_drone_callback(m_drone);
    m_tracker->set_drone_callback(NULL);

    // 3. Start background threads
    m_camera->start_capture_thread();
    m_drone->start();
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

    if (m_drone != NULL)
    {
        m_drone->stop();
        delete m_drone;
        m_drone = NULL;
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
    int i;
    int valid_count;

    if (w * h > 1920 * 1280)
    {
        fprintf(stderr, "[MainService] Frame size %dx%d exceeds pre-allocated buffer size!\n", w, h);
        return;
    }

    memcpy(m_lastFrame, frame, w * h);
    m_lastFrame_w = w;
    m_lastFrame_h = h;
    m_has_last_frame = true;

    // Report Camera FPS to Status using 5-second rolling window
    now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock_fps(m_fps_mutex);
        m_camera_frame_times[m_camera_frame_next] = now;
        m_camera_frame_next = (m_camera_frame_next + 1) % FPS_HISTORY_MAX;
        if (m_camera_frame_count < FPS_HISTORY_MAX)
        {
            m_camera_frame_count++;
        }

        valid_count = 0;
        for (i = 0; i < m_camera_frame_count; ++i)
        {
            elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_camera_frame_times[i]).count();
            if (elapsed < 5)
            {
                valid_count++;
            }
        }

        real_fps = valid_count / 5.0f;
        snprintf(fps_buf, sizeof(fps_buf), "%.1f FPS", real_fps);
        StatusObject::instance()->update("camera_fps", fps_buf);
    }
}

void MainService::onTargetDetected(int x, int y, bool low_quality)
{
    char pos_buf[64];

    m_target_x = x;
    m_target_y = y;
    m_target_low_quality = low_quality;

    // Report tracker status
    if (x >= 0 && y >= 0)
    {
        snprintf(pos_buf, sizeof(pos_buf), "[%d, %d]", x, y);
        if (low_quality)
        {
            StatusObject::instance()->update("tracking_status", "Target Low Quality");
        }
        else
        {
            StatusObject::instance()->update("tracking_status", "Target Acquired");
        }
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

    out.decode_argmax_only = 0; // Default fallback
    out.iterations_num = 1;
    out.quality_lost_threshold = 0.20f;
    out.quality_display_threshold = 0.20f;
    snprintf(out.drone_serial_port, sizeof(out.drone_serial_port), "/dev/ttyUSB0");
    out.drone_controller_id = -1;

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
            else if (strcmp(key, "quality_mode") == 0)
            {
                snprintf(out.quality_mode, sizeof(out.quality_mode), "%s", val);
            }
            else if (strcmp(key, "quality_lost_threshold") == 0)
            {
                out.quality_lost_threshold = (float)atof(val);
            }
            else if (strcmp(key, "quality_display_threshold") == 0)
            {
                out.quality_display_threshold = (float)atof(val);
            }
            else if (strcmp(key, "min_crop") == 0)
            {
                out.min_crop = (float)atof(val);
            }
            else if (strcmp(key, "max_crop") == 0)
            {
                out.max_crop = (float)atof(val);
            }
            else if (strcmp(key, "decode_argmax_only") == 0)
            {
                out.decode_argmax_only = atoi(val);
            }
            else if (strcmp(key, "iterations_num") == 0)
            {
                out.iterations_num = atoi(val);
            }
            else if (strcmp(key, "drone_serial_port") == 0)
            {
                snprintf(out.drone_serial_port, sizeof(out.drone_serial_port), "%s", val);
            }
            else if (strcmp(key, "drone_controller_id") == 0)
            {
                out.drone_controller_id = atoi(val);
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
        fprintf(fp, "quality_mode=%s\n", in.quality_mode);
        fprintf(fp, "quality_lost_threshold=%.2f\n", in.quality_lost_threshold);
        fprintf(fp, "quality_display_threshold=%.2f\n", in.quality_display_threshold);
        fprintf(fp, "min_crop=%.1f\n", in.min_crop);
        fprintf(fp, "max_crop=%.1f\n", in.max_crop);
        fprintf(fp, "decode_argmax_only=%d\n", in.decode_argmax_only);
        fprintf(fp, "iterations_num=%d\n", in.iterations_num);
        fprintf(fp, "drone_serial_port=%s\n", in.drone_serial_port);
        fprintf(fp, "drone_controller_id=%d\n", in.drone_controller_id);
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
    int crop_size;
    int x0;
    int y0;
    bool auto_mode;
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
                m_web_server->set_drone_callback(m_drone);
            }
            m_tracker->set_drone_callback(NULL);
            m_target_x = -1;
            m_target_y = -1;
            m_target_low_quality = false;
            StatusObject::instance()->update("tracking_status", "Target Not Selected");
            StatusObject::instance()->update("target_position", "N/A");
            StatusObject::instance()->update("flight_mode", "Manual");
            if (m_drone != NULL)
            {
                m_drone->send_command(1000, 1000, 1000, 1000);
            }
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

                // Reset tracker outputs coordinates relative to the 256x256 search crop frame space
                if (m_lastFrame_w < m_lastFrame_h)
                {
                    crop_size = m_lastFrame_w;
                }
                else
                {
                    crop_size = m_lastFrame_h;
                }
                x0 = (m_lastFrame_w - crop_size) / 2;
                y0 = (m_lastFrame_h - crop_size) / 2;
                m_target_x = ((target_px - x0) * 256) / crop_size;
                m_target_y = ((target_py - y0) * 256) / crop_size;
                m_target_low_quality = false;
            }
            break;

        case WebServer::CMD_SET_AUTONOMOUS:
            {
                auto_mode = (atoi(values) != 0);
                if (auto_mode)
                {
                    m_web_server->set_drone_callback(NULL);
                    m_tracker->set_drone_callback(m_drone);
                    StatusObject::instance()->update("flight_mode", "Autonomous");
                }
                else
                {
                    m_web_server->set_drone_callback(m_drone);
                    m_tracker->set_drone_callback(NULL);
                    StatusObject::instance()->update("flight_mode", "Manual");
                }
                // Send neutral commands to the drone upon switching modes for safety
                if (m_drone != NULL)
                {
                    m_drone->send_command(1000, 1000, 1000, 1000);
                }
                fprintf(stdout, "[MainService] Autonomous mode set to: %d\n", auto_mode);
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
    int i;
    int valid_count;

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
                    m_tracker_frame_times[m_tracker_frame_next] = now;
                    m_tracker_frame_next = (m_tracker_frame_next + 1) % FPS_HISTORY_MAX;
                    if (m_tracker_frame_count < FPS_HISTORY_MAX)
                    {
                        m_tracker_frame_count++;
                    }

                    valid_count = 0;
                    for (i = 0; i < m_tracker_frame_count; ++i)
                    {
                        elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - m_tracker_frame_times[i]).count();
                        if (elapsed < 5)
                        {
                            valid_count++;
                        }
                    }

                    real_fps = valid_count / 5.0f;
                    snprintf(fps_buf, sizeof(fps_buf), "%.1f FPS", real_fps);
                    StatusObject::instance()->update("tracker_fps", fps_buf);
                }
            }
            else
            {
                // Clear rolling queue when idle
                {
                    std::lock_guard<std::mutex> lock_fps(m_fps_mutex);
                    m_tracker_frame_count = 0;
                    m_tracker_frame_next = 0;
                }
                StatusObject::instance()->update("tracker_fps", "0.0 FPS");
                // Clear times in status
                StatusObject::instance()->update("tracker_time_resize", "N/A");
                StatusObject::instance()->update("tracker_time_npu", "N/A");
                StatusObject::instance()->update("tracker_time_decode", "N/A");
                StatusObject::instance()->update("tracker_time_total", "N/A");
            }
            
            // Send tracking overlays to web server stream
            m_web_server->update(work_frame, work_w, work_h, m_target_x, m_target_y, m_target_low_quality);
            
            has_frame = false;
        }

        // Sleep/wait on CV to avoid thread spin
        {
            std::unique_lock<std::mutex> lock(m_cmd_mutex);
            m_cmd_condvar.wait_for(lock, std::chrono::milliseconds(10), [this]()
            {
                return m_has_pending_command || !m_is_running;
            });
        }
    }

    free(work_frame);
}
