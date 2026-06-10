#ifndef STATUS_OBJECT_HPP
#define STATUS_OBJECT_HPP

#include <map>
#include <string>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>

class StatusObject
{
private:
    std::map<std::string, std::string> m_status_map;
    std::mutex m_mutex;

    // Previous CPU ticks for CPU usage calculations
    long long m_prev_user;
    long long m_prev_nice;
    long long m_prev_system;
    long long m_prev_idle;
    long long m_prev_iowait;
    long long m_prev_irq;
    long long m_prev_softirq;

    // Singleton structure
    StatusObject()
    {
        m_prev_user = 0;
        m_prev_nice = 0;
        m_prev_system = 0;
        m_prev_idle = 0;
        m_prev_iowait = 0;
        m_prev_irq = 0;
        m_prev_softirq = 0;
    }

    ~StatusObject()
    {
    }

    StatusObject(const StatusObject&) = delete;
    StatusObject& operator=(const StatusObject&) = delete;

    // Internal telemetry queries
    void query_cpu_usage(char* cpu_out, int max_len)
    {
        FILE* fp;
        long long user, nice, system, idle, iowait, irq, softirq;
        long long prev_idle_total, idle_total;
        long long prev_non_idle, non_idle;
        long long prev_total, total;
        long long total_diff, idle_diff;
        double cpu_pct;

        user = 0;
        nice = 0;
        system = 0;
        idle = 0;
        iowait = 0;
        irq = 0;
        softirq = 0;

        fp = fopen("/proc/stat", "r");
        if (fp != NULL)
        {
            if (fscanf(fp, "cpu %lld %lld %lld %lld %lld %lld %lld", &user, &nice, &system, &idle, &iowait, &irq, &softirq) == 7)
            {
                prev_idle_total = m_prev_idle + m_prev_iowait;
                idle_total = idle + iowait;

                prev_non_idle = m_prev_user + m_prev_nice + m_prev_system + m_prev_irq + m_prev_softirq;
                non_idle = user + nice + system + irq + softirq;

                prev_total = prev_idle_total + prev_non_idle;
                total = idle_total + non_idle;

                total_diff = total - prev_total;
                idle_diff = idle_total - prev_idle_total;

                if (total_diff > 0)
                {
                    cpu_pct = (double)(total_diff - idle_diff) * 100.0 / (double)total_diff;
                    snprintf(cpu_out, max_len, "%.1f%%", cpu_pct);
                }
                else
                {
                    snprintf(cpu_out, max_len, "0.0%%");
                }

                m_prev_user = user;
                m_prev_nice = nice;
                m_prev_system = system;
                m_prev_idle = idle;
                m_prev_iowait = iowait;
                m_prev_irq = irq;
                m_prev_softirq = softirq;
            }
            else
            {
                snprintf(cpu_out, max_len, "Unknown");
            }
            fclose(fp);
        }
        else
        {
            snprintf(cpu_out, max_len, "Unknown");
        }
    }

    void query_memory_usage(char* mem_out, int max_len)
    {
        FILE* fp;
        char line[128];
        long long total_mem;
        long long free_mem;
        long long used_mem;
        double mem_pct;

        total_mem = 0;
        free_mem = 0;

        fp = fopen("/proc/meminfo", "r");
        if (fp != NULL)
        {
            while (fgets(line, sizeof(line), fp) != NULL)
            {
                if (strncmp(line, "MemTotal:", 9) == 0)
                {
                    sscanf(line, "MemTotal: %lld", &total_mem);
                }
                else if (strncmp(line, "MemAvailable:", 13) == 0)
                {
                    sscanf(line, "MemAvailable: %lld", &free_mem);
                }
            }
            fclose(fp);

            if (total_mem > 0)
            {
                if (free_mem == 0)
                {
                    free_mem = total_mem / 10;
                }
                used_mem = total_mem - free_mem;
                mem_pct = (double)used_mem * 100.0 / (double)total_mem;
                snprintf(mem_out, max_len, "%.1f%% (%lld MB / %lld MB)", mem_pct, used_mem / 1024, total_mem / 1024);
            }
            else
            {
                snprintf(mem_out, max_len, "Unknown");
            }
        }
        else
        {
            snprintf(mem_out, max_len, "Unknown");
        }
    }

    void query_video_devices(char* dev_out, int max_len)
    {
        DIR* dir;
        struct dirent* entry;
        int offset;

        dir = opendir("/dev");
        offset = 0;
        dev_out[0] = '\0';

        if (dir != NULL)
        {
            while ((entry = readdir(dir)) != NULL)
            {
                if (strncmp(entry->d_name, "video", 5) == 0)
                {
                    if (offset > 0 && offset < max_len - 2)
                    {
                        offset += snprintf(dev_out + offset, max_len - offset, ", ");
                    }
                    if (offset < max_len)
                    {
                        offset += snprintf(dev_out + offset, max_len - offset, "/dev/%s", entry->d_name);
                    }
                }
            }
            closedir(dir);
        }

        if (offset == 0)
        {
            snprintf(dev_out, max_len, "None");
        }
    }

public:
    static StatusObject* instance()
    {
        static StatusObject s_instance;
        return &s_instance;
    }

    void update(const char* key, const char* value)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_status_map[key] = value;
    }

    void get_status_report(char* out, size_t max_len)
    {
        char cpu_buf[64];
        char mem_buf[64];
        char devs_buf[256];
        size_t offset;
        std::map<std::string, std::string>::const_iterator it;

        cpu_buf[0] = '\0';
        mem_buf[0] = '\0';
        devs_buf[0] = '\0';

        query_cpu_usage(cpu_buf, sizeof(cpu_buf));
        query_memory_usage(mem_buf, sizeof(mem_buf));
        query_video_devices(devs_buf, sizeof(devs_buf));

        update("cpu_usage", cpu_buf);
        update("memory_usage", mem_buf);
        update("available_devices", devs_buf);

        {
            std::lock_guard<std::mutex> lock(m_mutex);
            offset = 0;
            out[0] = '\0';

            for (it = m_status_map.begin(); it != m_status_map.end(); ++it)
            {
                if (offset < max_len)
                {
                    offset += snprintf(out + offset, max_len - offset, "%s=%s\n", it->first.c_str(), it->second.c_str());
                }
            }
        }
    }
};

#endif
