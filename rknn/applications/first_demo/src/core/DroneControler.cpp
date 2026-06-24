#include "core/DroneControler.h"
#include <chrono>
#include <stdio.h>

DroneControler::DroneControler(const char* device, int controller_id)
{
    m_controller_id = controller_id;
    snprintf(m_device, sizeof(m_device), "%s", device);

    m_roll = 1000;
    m_pitch = 1000;
    m_yaw = 1000;
    m_throttle = 1000;

    m_is_running = false;
}

DroneControler::~DroneControler()
{
    stop();
}

void DroneControler::start()
{
    if (m_is_running)
    {
        return;
    }

    m_is_running = true;

    // Try to open connection immediately
    fprintf(stdout, "[DroneControler] Initializing connection to %s...\n", m_device);
    if (m_hal.open_serial(m_device))
    {
        fprintf(stdout, "[DroneControler] Connection established to %s.\n", m_device);
    }
    else
    {
        fprintf(stdout, "[DroneControler] Device not ready. Reconnect thread will attempt connection.\n");
    }

    // Launch worker threads
    m_send_thread = std::thread(&DroneControler::send_loop, this);
    m_reconnect_thread = std::thread(&DroneControler::reconnect_loop, this);
}

void DroneControler::stop()
{
    m_is_running = false;

    if (m_send_thread.joinable())
    {
        m_send_thread.join();
    }

    if (m_reconnect_thread.joinable())
    {
        m_reconnect_thread.join();
    }

    m_hal.close_serial();
}

void DroneControler::update_channels(int16_t roll, int16_t pitch, int16_t yaw, int16_t throttle)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_roll = roll;
    m_pitch = pitch;
    m_yaw = yaw;
    m_throttle = throttle;
}

void DroneControler::send_command(int16_t roll, int16_t pitch, int16_t yaw, int16_t throttle)
{
    update_channels(roll, pitch, yaw, throttle);
}

bool DroneControler::is_connected()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_hal.is_connected();
}

void DroneControler::send_loop()
{
    int16_t r, p, y, t;
    bool connected;

    while (m_is_running)
    {
        connected = false;

        // Limit sending rate to 20Hz (every 50ms)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            r = m_roll;
            p = m_pitch;
            y = m_yaw;
            t = m_throttle;
            connected = m_hal.is_connected();
        }

        if (connected)
        {
            // Mode is 0, Camera is MID_VALUE (1000)
            if (!m_hal.write_packet(r, p, y, t, 1000, 0, m_controller_id))
            {
                fprintf(stdout, "[DroneControler] Write failed. Closing serial port to trigger reconnection.\n");
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_hal.close_serial();
                }
            }
        }
    }
}

void DroneControler::reconnect_loop()
{
    bool connected;

    while (m_is_running)
    {
        connected = false;

        // Attempt reconnection check every 2 seconds
        std::this_thread::sleep_for(std::chrono::seconds(2));
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            connected = m_hal.is_connected();
        }

        if (!connected && m_is_running)
        {
            fprintf(stdout, "[DroneControler] Attempting connection to %s...\n", m_device);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_hal.open_serial(m_device))
                {
                    fprintf(stdout, "[DroneControler] Connection established to %s.\n", m_device);
                }
            }
        }
    }
}
