#include "core/WebServer.h"
#include "utils/StatusObject.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>
#include "civetweb.h"
#include <chrono>

// Web browser UI HTML
const char* INDEX_HTML = 
"<!DOCTYPE html>\n"
"<html>\n"
"<head>\n"
"<meta charset=\"utf-8\">\n"
"<title>Radxa Zero 3E Tracker Dashboard</title>\n"
"<link href=\"https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Outfit:wght@600;800&family=JetBrains+Mono&display=swap\" rel=\"stylesheet\">\n"
"<style>\n"
"body {\n"
"    background-color: #0b0d10;\n"
"    color: #e2e8f0;\n"
"    font-family: 'Inter', sans-serif;\n"
"    margin: 0;\n"
"    padding: 0;\n"
"    display: flex;\n"
"    flex-direction: column;\n"
"    align-items: center;\n"
"}\n"
"header {\n"
"    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);\n"
"    width: 100%;\n"
"    padding: 20px 0;\n"
"    text-align: center;\n"
"    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);\n"
"}\n"
"h1 {\n"
"    font-family: 'Outfit', sans-serif;\n"
"    font-size: 24px;\n"
"    margin: 0;\n"
"    letter-spacing: 1px;\n"
"    color: #38bdf8;\n"
"}\n"
".container {\n"
"    max-width: 1400px;\n"
"    display: flex;\n"
"    flex-direction: row;\n"
"    gap: 30px;\n"
"    margin: 40px auto;\n"
"    padding: 0 20px;\n"
"}\n"
".card {\n"
"    background: #111827;\n"
"    border: 1px solid #1f2937;\n"
"    border-radius: 12px;\n"
"    padding: 25px;\n"
"    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4);\n"
"}\n"
".stream-container {\n"
"    position: relative;\n"
"    cursor: crosshair;\n"
"}\n"
".stream-img {\n"
"    border-radius: 8px;\n"
"    border: 2px solid #38bdf8;\n"
"    max-width: 640px;\n"
"    display: block;\n"
"}\n"
".status-card {\n"
"    width: 380px;\n"
"    display: flex;\n"
"    flex-direction: column;\n"
"    gap: 20px;\n"
"}\n"
"h2 {\n"
"    font-family: 'Outfit', sans-serif;\n"
"    font-size: 18px;\n"
"    margin-top: 0;\n"
"    border-bottom: 2px solid #1f2937;\n"
"    padding-bottom: 8px;\n"
"    color: #38bdf8;\n"
"}\n"
"pre {\n"
"    font-family: 'JetBrains Mono', monospace;\n"
"    font-size: 13px;\n"
"    background: #030712;\n"
"    padding: 15px;\n"
"    border-radius: 8px;\n"
"    border: 1px solid #1f2937;\n"
"    margin: 0;\n"
"    overflow-x: auto;\n"
"    color: #10b981;\n"
"}\n"
".form-group {\n"
"    display: flex;\n"
"    flex-direction: column;\n"
"    gap: 8px;\n"
"    margin-bottom: 15px;\n"
"}\n"
"label {\n"
"    font-size: 12px;\n"
"    font-weight: 600;\n"
"    color: #9ca3af;\n"
"}\n"
"input, select {\n"
"    background: #030712;\n"
"    border: 1px solid #374151;\n"
"    color: #e2e8f0;\n"
"    padding: 10px;\n"
"    border-radius: 6px;\n"
"    font-family: inherit;\n"
"    font-size: 14px;\n"
"}\n"
"input:focus, select:focus {\n"
"    border-color: #38bdf8;\n"
"    outline: none;\n"
"}\n"
".btn {\n"
"    background: #0284c7;\n"
"    color: #ffffff;\n"
"    font-weight: 600;\n"
"    border: none;\n"
"    padding: 12px;\n"
"    border-radius: 6px;\n"
"    cursor: pointer;\n"
"    transition: background 0.2s;\n"
"    font-size: 14px;\n"
"    display: block;\n"
"    width: 100%;\n"
"}\n"
".btn:hover {\n"
"    background: #0369a1;\n"
"}\n"
".btn-save {\n"
"    background: #10b981;\n"
"}\n"
".btn-save:hover {\n"
"    background: #059669;\n"
"}\n"
".btn-reset {\n"
"    background: #ef4444;\n"
"}\n"
".btn-reset:hover {\n"
"    background: #dc2626;\n"
"}\n"
".coords-info {\n"
"    font-size: 12px;\n"
"    color: #9ca3af;\n"
"    text-align: center;\n"
"}\n"
"</style>\n"
"</head>\n"
"<body>\n"
"<header>\n"
"    <h1>TARGET TRACKER CONTROL DASHBOARD</h1>\n"
"</header>\n"
"<div class=\"container\">\n"
"    <div class=\"card\">\n"
"        <h2>Live Stream (Click on target to track)</h2>\n"
"        <div class=\"stream-container\" id=\"streamContainer\">\n"
"            <img class=\"stream-img\" id=\"streamImg\" src=\"/stream\" alt=\"Live Camera Stream\">\n"
"        </div>\n"
"        <div class=\"coords-info\" style=\"margin-top: 10px;\" id=\"coordsInfo\">Click to initialize tracker</div>\n"
"    </div>\n"
"    <div class=\"card\" style=\"width: 256px;\">\n"
"        <h2>Heatmap Debug</h2>\n"
"        <div class=\"stream-container\">\n"
"            <img class=\"stream-img\" src=\"/heatmap_stream\" alt=\"Heatmap Stream\" style=\"width: 256px; height: 256px; border: 2px solid #ef4444;\">\n"
"        </div>\n"
"        <div class=\"coords-info\" style=\"margin-top: 10px;\">256x256 Filtered Output</div>\n"
"    </div>\n"
"    <div class=\"card\" style=\"width: 256px;\">\n"
"        <h2>Reference Stack</h2>\n"
"        <div class=\"stream-container\" style=\"width: 256px; height: 256px; display: flex; justify-content: center; align-items: center; background: #000; border: 2px solid #3b82f6;\">\n"
"            <img id=\"stackImg\" src=\"/stack_layer?idx=0\" alt=\"Stack Layer\" style=\"width: 128px; height: 128px; image-rendering: pixelated;\">\n"
"        </div>\n"
"        <div class=\"coords-info\" style=\"margin-top: 10px;\" id=\"stackInfo\">Layer 0 (Crop: N/A)</div>\n"
"        <div class=\"form-group\" style=\"margin-top: 10px; display: flex; align-items: center; gap: 5px;\">\n"
"            <button class=\"btn\" style=\"width: 40px; padding: 5px;\" onclick=\"stepStack(-1)\">&lt;</button>\n"
"            <input type=\"range\" id=\"stackSlider\" min=\"0\" max=\"15\" value=\"0\" style=\"flex-grow: 1;\" oninput=\"onStackSlider(this.value)\">\n"
"            <button class=\"btn\" style=\"width: 40px; padding: 5px;\" onclick=\"stepStack(1)\">&gt;</button>\n"
"        </div>\n"
"    </div>\n"
"    <div class=\"status-card\">\n"
"        <div class=\"card\">\n"
"            <h2>Telemetry Status</h2>\n"
"            <pre id=\"statusPre\">Loading telemetry...</pre>\n"
"        </div>\n"
"        <div class=\"card\">\n"
"            <h2>Configuration</h2>\n"
"            <div class=\"form-group\">\n"
"                <label>Camera Device</label>\n"
"                <input type=\"text\" id=\"camDev\" value=\"/dev/video0\">\n"
"            </div>\n"
"            <div class=\"form-group\">\n"
"                <label>Resolution</label>\n"
"                <select id=\"resSelect\">\n"
"                    <option value=\"640#480\">640 x 480 (Recommended)</option>\n"
"                    <option value=\"1280#720\">1280 x 720</option>\n"
"                    <option value=\"1920#1080\">1920 x 1080</option>\n"
"                </select>\n"
"            </div>\n"
"            <button class=\"btn\" onclick=\"updateConfig()\">Apply Configuration</button>\n"
"            <button class=\"btn btn-save\" onclick=\"saveConfig()\" style=\"margin-top: 10px;\">Save Permanent</button>\n"
"            <button class=\"btn btn-reset\" onclick=\"resetTarget()\" style=\"margin-top: 10px;\">Reset Target</button>\n"
"        </div>\n"
"    </div>\n"
"</div>\n"
"<script>\n"
"const img = document.getElementById('streamImg');\n"
"const coordsInfo = document.getElementById('coordsInfo');\n"
"\n"
"img.addEventListener('click', function(e) {\n"
"    const rect = e.target.getBoundingClientRect();\n"
"    const x = (e.clientX - rect.left) / rect.width;\n"
"    const y = (e.clientY - rect.top) / rect.height;\n"
"    \n"
"    coordsInfo.innerText = `Selected target coordinates: X=${x.toFixed(3)}, Y=${y.toFixed(3)}`;\n"
"    \n"
"    fetch(`/command?cmd=CHOOSE_TARGET&val=${x.toFixed(3)},${y.toFixed(3)}`)\n"
"        .then(res => {\n"
"            console.log('Command sent:', res.status);\n"
"            currentStackIdx = 0;\n"
"            const slider = document.getElementById('stackSlider');\n"
"            if (slider) {\n"
"                slider.value = 0;\n"
"            }\n"
"            setTimeout(updateStackView, 200);\n"
"        });\n"
"});\n"
"\n"
"function updateConfig() {\n"
"    const dev = document.getElementById('camDev').value;\n"
"    const res = document.getElementById('resSelect').value;\n"
"    const cmdVal = `${dev}#${res}`;\n"
"    \n"
"    fetch(`/command?cmd=UPDATE_CAMERA_PARAMS&val=${encodeURIComponent(cmdVal)}`)\n"
"        .then(res => {\n"
"            alert('Reconfiguration command sent!');\n"
"            setTimeout(() => location.reload(), 1000);\n"
"        });\n"
"}\n"
"\n"
"function saveConfig() {\n"
"    fetch('/command?cmd=SAVE_PARAMS')\n"
"        .then(res => alert('Configuration saved to params.conf'));\n"
"}\n"
"\n"
"let cameraW = 640;\n"
"let cameraH = 480;\n"
"let currentStackIdx = 0;\n"
"let stackLayersCount = 16;\n"
"\n"
"function resetTarget() {\n"
"    fetch('/command?cmd=RESET_TARGET')\n"
"        .then(res => {\n"
"            coordsInfo.innerText = 'Click to initialize tracker';\n"
"            currentStackIdx = 0;\n"
"            const slider = document.getElementById('stackSlider');\n"
"            if (slider) {\n"
"                slider.value = 0;\n"
"            }\n"
"            updateStackView();\n"
"            alert('Target cleared!');\n"
"        });\n"
"}\n"
"\n"
"function onStackSlider(val) {\n"
"    currentStackIdx = parseInt(val);\n"
"    updateStackView();\n"
"}\n"
"\n"
"function stepStack(dir) {\n"
"    currentStackIdx += dir;\n"
"    if (currentStackIdx < 0) {\n"
"        currentStackIdx = stackLayersCount - 1;\n"
"    }\n"
"    if (currentStackIdx >= stackLayersCount) {\n"
"        currentStackIdx = 0;\n"
"    }\n"
"    const slider = document.getElementById('stackSlider');\n"
"    if (slider) {\n"
"        slider.value = currentStackIdx;\n"
"    }\n"
"    updateStackView();\n"
"}\n"
"\n"
"function updateStackView() {\n"
"    const stackImg = document.getElementById('stackImg');\n"
"    stackImg.src = `/stack_layer?idx=${currentStackIdx}&t=${Date.now()}`;\n"
"    updateStackLabel();\n"
"}\n"
"\n"
"function updateStackLabel() {\n"
"    const max_sz = Math.min(cameraW, cameraH);\n"
"    const min_sz = 16;\n"
"    const denom = Math.max(1, stackLayersCount - 1);\n"
"    const sz = Math.round(max_sz - (currentStackIdx * (max_sz - min_sz) / denom));\n"
"    document.getElementById('stackInfo').innerText = `Layer ${currentStackIdx} (Crop: ${sz}x${sz})`;\n"
"}\n"
"\n"
"function parseResolutionFromStatus(statusText) {\n"
"    const lines = statusText.split('\\n');\n"
"    for (let i = 0; i < lines.length; i++) {\n"
"        if (lines[i].startsWith('camera_resolution=')) {\n"
"            const val = lines[i].split('=')[1];\n"
"            const parts = val.split('x');\n"
"            if (parts.length === 2) {\n"
"                cameraW = parseInt(parts[0]);\n"
"                cameraH = parseInt(parts[1]);\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"function parseStackLayersFromStatus(statusText) {\n"
"    const lines = statusText.split('\\n');\n"
"    for (let i = 0; i < lines.length; i++) {\n"
"        if (lines[i].startsWith('stack_layers=')) {\n"
"            const val = parseInt(lines[i].split('=')[1]);\n"
"            if (!isNaN(val) && val > 0) {\n"
"                stackLayersCount = val;\n"
"                const slider = document.getElementById('stackSlider');\n"
"                if (slider) {\n"
"                    slider.max = stackLayersCount - 1;\n"
"                }\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"function fetchStatus() {\n"
"    fetch('/status')\n"
"        .then(res => res.text())\n"
"        .then(text => {\n"
"            document.getElementById('statusPre').innerText = text;\n"
"            parseResolutionFromStatus(text);\n"
"            parseStackLayersFromStatus(text);\n"
"            updateStackLabel();\n"
"        });\n"
"}\n"
"\n"
"setInterval(fetchStatus, 1000);\n"
"fetchStatus();\n"
"</script>\n"
"</body>\n"
"</html>\n";

// CivetWeb Request Handler Implementations

class WebPageHandler : public CivetHandler
{
public:
    bool handleGet(CivetServer* server, struct mg_connection* conn) override
    {
        mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/html\r\n"
                        "Content-Length: %d\r\n"
                        "Connection: close\r\n\r\n", (int)strlen(INDEX_HTML));
        mg_write(conn, INDEX_HTML, strlen(INDEX_HTML));
        return true;
    }
};

class StatusHandler : public CivetHandler
{
public:
    bool handleGet(CivetServer* server, struct mg_connection* conn) override
    {
        char report[2048];
        report[0] = '\0';
        StatusObject::instance()->get_status_report(report, sizeof(report));

        mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/plain\r\n"
                        "Content-Length: %d\r\n"
                        "Connection: close\r\n\r\n", (int)strlen(report));
        mg_write(conn, report, strlen(report));
        return true;
    }
};

class CommandHandler : public CivetHandler
{
private:
    WebServer* m_web_server;

public:
    CommandHandler(WebServer* ws) : m_web_server(ws) {}

    bool handleGet(CivetServer* server, struct mg_connection* conn) override
    {
        char cmd_buf[64];
        char val_buf[512];
        const struct mg_request_info* req_info;
        WebServer::Command cmd_key;

        cmd_buf[0] = '\0';
        val_buf[0] = '\0';

        req_info = mg_get_request_info(conn);
        if (req_info->query_string != NULL)
        {
            mg_get_var(req_info->query_string, strlen(req_info->query_string), "cmd", cmd_buf, sizeof(cmd_buf));
            mg_get_var(req_info->query_string, strlen(req_info->query_string), "val", val_buf, sizeof(val_buf));
        }

        cmd_key = (WebServer::Command)0;
        if (strcmp(cmd_buf, "UPDATE_CAMERA_PARAMS") == 0)
        {
            cmd_key = WebServer::CMD_UPDATE_CAMERA_PARAMS;
        }
        else if (strcmp(cmd_buf, "SAVE_PARAMS") == 0)
        {
            cmd_key = WebServer::CMD_SAVE_PARAMS;
        }
        else if (strcmp(cmd_buf, "CHOOSE_TARGET") == 0)
        {
            cmd_key = WebServer::CMD_CHOOSE_TARGET;
        }
        else if (strcmp(cmd_buf, "RESET_TARGET") == 0)
        {
            cmd_key = WebServer::CMD_RESET_TARGET;
        }

        if (cmd_key != 0)
        {
            m_web_server->trigger_command(cmd_key, val_buf);
            mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 2\r\n"
                            "Connection: close\r\n\r\nOK");
        }
        else
        {
            mg_printf(conn, "HTTP/1.1 400 Bad Request\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 15\r\n"
                            "Connection: close\r\n\r\nInvalid Command");
        }
        return true;
    }
};

class StreamHandler : public CivetHandler
{
private:
    WebServer* m_web_server;

public:
    StreamHandler(WebServer* ws) : m_web_server(ws) {}

    bool handleGet(CivetServer* server, struct mg_connection* conn) override
    {
        uchar* jpeg_frame;
        unsigned long jpeg_len;

        // Enforce Single Client Limitation
        if (m_web_server->is_streaming())
        {
            mg_printf(conn, "HTTP/1.1 503 Service Unavailable\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 28\r\n"
                            "Connection: close\r\n\r\n"
                            "Another active stream exists");
            return true;
        }

        m_web_server->set_streaming(true);
        StatusObject::instance()->update("web_stream_status", "Active");

        mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                        "Content-Type: multipart/x-mixed-replace; boundary=--frameboundary\r\n"
                        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                        "Pragma: no-cache\r\n"
                        "Expires: 0\r\n"
                        "Connection: close\r\n\r\n");

        jpeg_frame = (uchar*)malloc(1920 * 1280);

        while (true)
        {
            jpeg_len = 0;
            m_web_server->wait_for_frame(jpeg_frame, &jpeg_len);

            if (jpeg_len > 0)
            {
                // Send MJPEG boundary and frame
                if (mg_printf(conn, "--frameboundary\r\n"
                                    "Content-Type: image/jpeg\r\n"
                                    "Content-Length: %lu\r\n\r\n", jpeg_len) <= 0)
                {
                    break; // write failed (client disconnected)
                }

                if (mg_write(conn, jpeg_frame, jpeg_len) <= 0)
                {
                    break; // write failed
                }

                if (mg_printf(conn, "\r\n") <= 0)
                {
                    break;
                }
            }
            else
            {
                // Safety sleep
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        free(jpeg_frame);
        m_web_server->set_streaming(false);
        StatusObject::instance()->update("web_stream_status", "Inactive");
        return true;
    }
};

class HeatmapStreamHandler : public CivetHandler
{
private:
    WebServer* m_web_server;

public:
    HeatmapStreamHandler(WebServer* ws) : m_web_server(ws)
    {
    }

    bool handleGet(CivetServer* server, struct mg_connection* conn) override
    {
        uchar* jpeg_frame;
        unsigned long jpeg_len;

        if (m_web_server->is_heatmap_streaming())
        {
            mg_printf(conn, "HTTP/1.1 503 Service Unavailable\r\n"
                            "Content-Type: text/plain\r\n"
                            "Content-Length: 28\r\n"
                            "Connection: close\r\n\r\n"
                            "Another active stream exists");
            return true;
        }

        m_web_server->set_heatmap_streaming(true);

        mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                        "Content-Type: multipart/x-mixed-replace; boundary=--frameboundary\r\n"
                        "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                        "Pragma: no-cache\r\n"
                        "Expires: 0\r\n"
                        "Connection: close\r\n\r\n");

        jpeg_frame = (uchar*)malloc(256 * 256 * 3);

        while (true)
        {
            jpeg_len = 0;
            m_web_server->wait_for_heatmap(jpeg_frame, &jpeg_len);

            if (jpeg_len > 0)
            {
                if (mg_printf(conn, "--frameboundary\r\n"
                                    "Content-Type: image/jpeg\r\n"
                                    "Content-Length: %lu\r\n\r\n", jpeg_len) <= 0)
                {
                    break;
                }

                if (mg_write(conn, jpeg_frame, jpeg_len) <= 0)
                {
                    break;
                }

                if (mg_printf(conn, "\r\n") <= 0)
                {
                    break;
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }

        free(jpeg_frame);
        m_web_server->set_heatmap_streaming(false);
        return true;
    }
};

class StackLayerHandler : public CivetHandler
{
private:
    WebServer* m_web_server;

public:
    StackLayerHandler(WebServer* ws) : m_web_server(ws)
    {
    }

    bool handleGet(CivetServer* server, struct mg_connection* conn) override
    {
        char idx_buf[32];
        int idx;
        const uchar* jpeg_data;
        unsigned long jpeg_len;
        int ret;
        const struct mg_request_info* req_info;

        idx = 0;
        req_info = mg_get_request_info(conn);
        if (req_info != NULL && req_info->query_string != NULL)
        {
            ret = mg_get_var(req_info->query_string, strlen(req_info->query_string), "idx", idx_buf, sizeof(idx_buf));
            if (ret >= 0)
            {
                idx = atoi(idx_buf);
            }
        }

        if (idx < 0 || idx >= MAX_STACK_LAYERS)
        {
            idx = 0;
        }

        jpeg_data = NULL;
        jpeg_len = 0;
        m_web_server->get_stack_layer(idx, &jpeg_data, &jpeg_len);

        if (jpeg_data != NULL && jpeg_len > 0)
        {
            mg_printf(conn, "HTTP/1.1 200 OK\r\n"
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: %lu\r\n"
                            "Connection: close\r\n\r\n", jpeg_len);
            mg_write(conn, jpeg_data, jpeg_len);
        }
        else
        {
            mg_printf(conn, "HTTP/1.1 444 No Response\r\n"
                            "Connection: close\r\n\r\n");
        }
        return true;
    }
};


// WebServer Wrapper Class Implementation

WebServer::WebServer(int port)
{
    char port_str[16];
    const char* options[5];
    int i;

    m_cmd_callback = NULL;
    m_is_streaming = false;
    m_has_new_frame = false;

    m_frame_w = 0;
    m_frame_h = 0;
    m_target_x = -1;
    m_target_y = -1;

    m_has_new_heatmap = false;
    m_is_heatmap_streaming = false;

    m_frame_buf = (uchar*)malloc(1920 * 1280);
    m_jpeg_buf = (uchar*)malloc(1920 * 1280);
    m_jpeg_size = 0;

    m_heatmap_rgb_buf = (uchar*)malloc(256 * 256 * 3);
    m_heatmap_jpeg_buf = (uchar*)malloc(256 * 256 * 3);
    m_heatmap_jpeg_size = 0;
    memset(m_heatmap_rgb_buf, 0, 256 * 256 * 3);

    for (i = 0; i < MAX_STACK_LAYERS; ++i)
    {
        m_stack_jpeg_bufs[i] = (uchar*)malloc(16384);
        m_stack_jpeg_sizes[i] = 0;
    }

    update_stack(NULL, 64, 64, MAX_STACK_LAYERS);

    snprintf(port_str, sizeof(port_str), "%d", port);

    options[0] = "listening_ports";
    options[1] = port_str;
    options[2] = NULL;

    m_server = new CivetServer(options);

    // Register HTTP Handlers
    m_server->addHandler("/", new WebPageHandler());
    m_server->addHandler("/status", new StatusHandler());
    m_server->addHandler("/command", new CommandHandler(this));
    m_server->addHandler("/stream", new StreamHandler(this));
    m_server->addHandler("/heatmap_stream", new HeatmapStreamHandler(this));
    m_server->addHandler("/stack_layer", new StackLayerHandler(this));

    StatusObject::instance()->update("web_server_status", "Online");
    StatusObject::instance()->update("web_stream_status", "Inactive");
    fprintf(stdout, "[WebServer] Running on port %s\n", port_str);
}

WebServer::~WebServer()
{
    int i;

    // Clean handlers
    delete m_server;
    free(m_frame_buf);
    free(m_jpeg_buf);
    free(m_heatmap_rgb_buf);
    free(m_heatmap_jpeg_buf);

    for (i = 0; i < MAX_STACK_LAYERS; ++i)
    {
        free(m_stack_jpeg_bufs[i]);
    }
}

void WebServer::set_command_callback(CommandCallback* cb)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_cmd_callback = cb;
}

void WebServer::update(uchar* frame, int w, int h, int target_x, int target_y)
{
    int x_cam, y_cam;
    int box_size, half;
    int x_start, x_end, y_start, y_end;
    int cx, cy;
    std::chrono::steady_clock::time_point t_comp_start;
    std::chrono::steady_clock::time_point t_comp_end;
    float comp_ms;
    char comp_buf[64];

    std::lock_guard<std::mutex> lock(m_mutex);

    // Copy raw grayscale frame internally
    memcpy(m_frame_buf, frame, w * h);
    m_frame_w = w;
    m_frame_h = h;
    m_target_x = target_x;
    m_target_y = target_y;

    // Draw the tracking bounding box borders if target coordinates are valid
    if (m_target_x >= 0 && m_target_y >= 0)
    {
        // Scale coordinate system from 256x256 to camera frame space
        x_cam = (m_target_x * w) / 256;
        y_cam = (m_target_y * h) / 256;

        box_size = 30;
        half = box_size / 2;

        x_start = x_cam - half;
        x_end = x_cam + half;
        y_start = y_cam - half;
        y_end = y_cam + half;

        // Draw horizontal boundaries (white = 255)
        for (cx = x_start; cx <= x_end; ++cx)
        {
            if (cx >= 0 && cx < w)
            {
                if (y_start >= 0 && y_start < h)
                {
                    m_frame_buf[y_start * w + cx] = 255;
                }
                if (y_end >= 0 && y_end < h)
                {
                    m_frame_buf[y_end * w + cx] = 255;
                }
            }
        }

        // Draw vertical boundaries
        for (cy = y_start; cy <= y_end; ++cy)
        {
            if (cy >= 0 && cy < h)
            {
                if (x_start >= 0 && x_start < w)
                {
                    m_frame_buf[cy * w + x_start] = 255;
                }
                if (x_end >= 0 && x_end < w)
                {
                    m_frame_buf[cy * w + x_end] = 255;
                }
            }
        }
    }

    // Perform the grayscale JPEG compression inside Web context thread trigger
    t_comp_start = std::chrono::steady_clock::now();
    compress_gray_to_jpeg(m_frame_buf, m_frame_w, m_frame_h, m_jpeg_buf, &m_jpeg_size);
    t_comp_end = std::chrono::steady_clock::now();

    comp_ms = std::chrono::duration<float, std::milli>(t_comp_end - t_comp_start).count();
    snprintf(comp_buf, sizeof(comp_buf), "%.2f ms", comp_ms);
    StatusObject::instance()->update("web_time_jpeg", comp_buf);

    m_has_new_frame = true;
    m_condvar.notify_all();
}

void WebServer::trigger_command(Command key, const char* values)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_cmd_callback != NULL)
    {
        m_cmd_callback->onCommand(key, values, (int)strlen(values));
    }
}

bool WebServer::is_streaming() const
{
    return m_is_streaming;
}

void WebServer::set_streaming(bool state)
{
    m_is_streaming = state;
}

void WebServer::wait_for_frame(uchar* jpeg_dest, unsigned long* jpeg_len)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condvar.wait_for(lock, std::chrono::milliseconds(100), [this]() { return m_has_new_frame; });

    if (m_has_new_frame)
    {
        memcpy(jpeg_dest, m_jpeg_buf, m_jpeg_size);
        *jpeg_len = m_jpeg_size;
        m_has_new_frame = false;
    }
}

void WebServer::compress_gray_to_jpeg(const uchar* gray_buf, int w, int h, uchar* dest_buf, unsigned long* dest_size)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    uchar* outbuffer;
    unsigned long outsize;
    JSAMPROW row_pointer[1];

    outbuffer = NULL;
    outsize = 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = (JSAMPROW)(gray_buf + cinfo.next_scanline * w);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);

    if (outsize < 1920 * 1280)
    {
        memcpy(dest_buf, outbuffer, outsize);
        *dest_size = outsize;
    }

    jpeg_destroy_compress(&cinfo);
    free(outbuffer);
}

static void jet_colormap(float v, uchar& r, uchar& g, uchar& b)
{
    float r_f;
    float g_f;
    float b_f;

    if (v < 0.0f)
    {
        v = 0.0f;
    }
    if (v > 1.0f)
    {
        v = 1.0f;
    }
    
    r_f = std::max(0.0f, std::min(1.0f, 1.5f - std::abs(4.0f * v - 3.0f)));
    g_f = std::max(0.0f, std::min(1.0f, 1.5f - std::abs(4.0f * v - 2.0f)));
    b_f = std::max(0.0f, std::min(1.0f, 1.5f - std::abs(4.0f * v - 1.0f)));
    
    r = (uchar)(r_f * 255.0f);
    g = (uchar)(g_f * 255.0f);
    b = (uchar)(b_f * 255.0f);
}

void WebServer::compress_rgb_to_jpeg(const uchar* rgb_buf, int w, int h, uchar* dest_buf, unsigned long* dest_size)
{
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    uchar* outbuffer;
    unsigned long outsize;
    JSAMPROW row_pointer[1];

    outbuffer = NULL;
    outsize = 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    jpeg_mem_dest(&cinfo, &outbuffer, &outsize);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 80, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    while (cinfo.next_scanline < cinfo.image_height)
    {
        row_pointer[0] = (JSAMPROW)(rgb_buf + cinfo.next_scanline * w * 3);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);

    if (outsize < 256 * 256 * 3)
    {
        memcpy(dest_buf, outbuffer, outsize);
        *dest_size = outsize;
    }

    jpeg_destroy_compress(&cinfo);
    free(outbuffer);
}

void WebServer::update_heatmap(const float* heatmap, int w, int h)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (heatmap != NULL)
    {
        for (int i = 0; i < w * h; ++i)
        {
            uchar r, g, b;
            jet_colormap(heatmap[i], r, g, b);
            m_heatmap_rgb_buf[i * 3 + 0] = r;
            m_heatmap_rgb_buf[i * 3 + 1] = g;
            m_heatmap_rgb_buf[i * 3 + 2] = b;
        }
    }
    else
    {
        memset(m_heatmap_rgb_buf, 0, w * h * 3);
    }

    compress_rgb_to_jpeg(m_heatmap_rgb_buf, w, h, m_heatmap_jpeg_buf, &m_heatmap_jpeg_size);
    m_has_new_heatmap = true;
    m_condvar.notify_all();
}

bool WebServer::is_heatmap_streaming() const
{
    return m_is_heatmap_streaming;
}

void WebServer::set_heatmap_streaming(bool state)
{
    m_is_heatmap_streaming = state;
}

void WebServer::wait_for_heatmap(uchar* jpeg_dest, unsigned long* jpeg_len)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_condvar.wait_for(lock, std::chrono::milliseconds(100), [this]() { return m_has_new_heatmap; });

    if (m_has_new_heatmap)
    {
        memcpy(jpeg_dest, m_heatmap_jpeg_buf, m_heatmap_jpeg_size);
        *jpeg_len = m_heatmap_jpeg_size;
        m_has_new_heatmap = false;
    }
}

void WebServer::update_stack(const uchar* stack, int w, int h, int c)
{
    int i;
    uchar* temp_black;
    uchar* temp_planar;
    int y, x;

    std::lock_guard<std::mutex> lock(m_mutex);

    int layers = std::min(c, (int)MAX_STACK_LAYERS);

    if (stack != NULL)
    {
        temp_planar = (uchar*)malloc(w * h);
        for (i = 0; i < layers; ++i)
        {
            for (y = 0; y < h; ++y)
            {
                for (x = 0; x < w; ++x)
                {
                    temp_planar[y * w + x] = stack[(y * w + x) * c + i];
                }
            }
            compress_gray_to_jpeg(temp_planar, w, h, m_stack_jpeg_bufs[i], &m_stack_jpeg_sizes[i]);
        }
        free(temp_planar);

        for (i = layers; i < MAX_STACK_LAYERS; ++i)
        {
            m_stack_jpeg_sizes[i] = 0;
        }
    }
    else
    {
        temp_black = (uchar*)calloc(w * h, 1);
        for (i = 0; i < MAX_STACK_LAYERS; ++i)
        {
            compress_gray_to_jpeg(temp_black, w, h, m_stack_jpeg_bufs[i], &m_stack_jpeg_sizes[i]);
        }
        free(temp_black);
    }

    char layers_str[16];
    snprintf(layers_str, sizeof(layers_str), "%d", layers);
    StatusObject::instance()->update("stack_layers", layers_str);
}

void WebServer::get_stack_layer(int idx, const uchar** jpeg_dest, unsigned long* jpeg_len)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (idx >= 0 && idx < MAX_STACK_LAYERS)
    {
        *jpeg_dest = m_stack_jpeg_bufs[idx];
        *jpeg_len = m_stack_jpeg_sizes[idx];
    }
    else
    {
        *jpeg_dest = NULL;
        *jpeg_len = 0;
    }
}
