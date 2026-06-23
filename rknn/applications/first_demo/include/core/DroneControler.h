#ifndef DRONE_CONTROLER_H
#define DRONE_CONTROLER_H

#include <thread>
#include <mutex>
#include <stdint.h>
#include "utils/DroneControlerHal.hpp"

class DroneControler : public IControlerCallback
{
private:
    DroneControlerHal m_hal;
    char m_device[256];
    int m_controller_id;

    // Current target control values
    int16_t m_roll;
    int16_t m_pitch;
    int16_t m_yaw;
    int16_t m_throttle;

    // Threads lifecycle
    bool m_is_running;
    std::thread m_send_thread;
    std::thread m_reconnect_thread;
    std::mutex m_mutex;

    // Internal thread loops
    void send_loop();
    void reconnect_loop();

public:
    DroneControler(const char* device, int controller_id);
    ~DroneControler();

    void start();
    void stop();

    // Main thread-safe entry point to update channels
    void update_channels(int16_t roll, int16_t pitch, int16_t yaw, int16_t throttle);
    
    // IControlerCallback implementation
    void send_command(int16_t roll, int16_t pitch, int16_t yaw, int16_t throttle) override;

    // Check connection status
    bool is_connected();
};

#endif
