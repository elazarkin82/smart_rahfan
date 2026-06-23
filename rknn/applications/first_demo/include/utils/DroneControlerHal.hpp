#ifndef DRONE_CONTROLER_HAL_HPP
#define DRONE_CONTROLER_HAL_HPP

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>

class IControlerCallback
{
public:
    virtual ~IControlerCallback() {}
    virtual void send_command(int16_t roll, int16_t pitch, int16_t yaw, int16_t throttle) = 0;
};

class DroneControlerHal
{
private:
    int m_fd;

public:
    DroneControlerHal() : m_fd(-1) {}

    ~DroneControlerHal()
    {
        close_serial();
    }

    bool open_serial(const char* device, int baudrate = 115200)
    {
        close_serial();

        // Open serial port in non-blocking, read/write mode
        m_fd = open(device, O_RDWR | O_NOCTTY | O_NDELAY);
        if (m_fd < 0)
        {
            return false;
        }

        // Clear non-blocking state for synchronous writes
        fcntl(m_fd, F_SETFL, 0);

        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        if (tcgetattr(m_fd, &tty) != 0)
        {
            close(m_fd);
            m_fd = -1;
            return false;
        }

        // Set speed (standard 115200 baud)
        speed_t speed = B115200;
        if (baudrate == 115200) speed = B115200;
        else if (baudrate == 9600) speed = B9600;
        else if (baudrate == 57600) speed = B57600;

        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);

        // 8N1 configuration
        tty.c_cflag &= ~PARENB;        // No parity
        tty.c_cflag &= ~CSTOPB;        // 1 stop bit
        tty.c_cflag &= ~CSIZE;
        tty.c_cflag |= CS8;            // 8 data bits
        tty.c_cflag &= ~CRTSCTS;       // No hardware flow control
        tty.c_cflag |= CREAD | CLOCAL; // Enable read, ignore carrier detect

        // Raw input mode
        tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);

        // Raw output mode (very important to prevent translating \n to \r\n)
        tty.c_oflag &= ~(OPOST | ONLCR);

        // Non-blocking read timeout (100ms)
        tty.c_cc[VMIN] = 0;
        tty.c_cc[VTIME] = 1;

        if (tcsetattr(m_fd, TCSANOW, &tty) != 0)
        {
            close(m_fd);
            m_fd = -1;
            return false;
        }

        tcflush(m_fd, TCIOFLUSH);
        return true;
    }

    void close_serial()
    {
        if (m_fd >= 0)
        {
            close(m_fd);
            m_fd = -1;
        }
    }

    bool is_connected() const
    {
        return m_fd >= 0;
    }

    bool write_packet(int16_t roll, int16_t pitch, int16_t yaw, int16_t throttle, int16_t camera, int16_t mode, int controller_id)
    {
        if (m_fd < 0)
        {
            return false;
        }

        uint8_t buffer[64];
        int offset = 0;

        // 1. Pack Header if controller_id is valid (>= 0)
        if (controller_id >= 0 && controller_id <= 255)
        {
            buffer[offset++] = 0;
            buffer[offset++] = (uint8_t)controller_id;
        }

        // Helper lambda to pack 16-bit signed int (little endian)
        auto pack_int16 = [&](int16_t value) {
            buffer[offset++] = (uint8_t)(value & 0xFF);
            buffer[offset++] = (uint8_t)((value >> 8) & 0xFF);
        };

        // 2. Pack payload (6 x int16_t)
        pack_int16(roll);
        pack_int16(pitch);
        pack_int16(yaw);
        pack_int16(throttle);
        pack_int16(camera);
        pack_int16(mode);

        // 3. Pack newline delimiter (\n = 10)
        buffer[offset++] = '\n';

        // Write to serial port
        int written = write(m_fd, buffer, offset);
        if (written != offset)
        {
            return false;
        }

        return true;
    }

    static void calculate_tracking_commands(int dx, int dy, int16_t& out_roll, int16_t& out_pitch)
    {
        const int16_t MID_VALUE = 1000;
        const float Kp = 2.5f; // Proportional feedback gain

        // Compute offsets (dx, dy are target offset from center in 256x256 space, range -128 to 128)
        float roll_offset = (float)dx * Kp;
        float pitch_offset = (float)dy * Kp;

        // Apply control commands around MID_VALUE (1000)
        int16_t r_val = MID_VALUE + (int16_t)roll_offset;
        int16_t p_val = MID_VALUE - (int16_t)pitch_offset; // Invert pitch to match drone standards

        // Constrain values to AXES_RANGE (0 to 2000)
        out_roll = std::max((int16_t)0, std::min((int16_t)2000, r_val));
        out_pitch = std::max((int16_t)0, std::min((int16_t)2000, p_val));
    }
};

#endif
