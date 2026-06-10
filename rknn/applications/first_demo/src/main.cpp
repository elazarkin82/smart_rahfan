#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include "MainService.h"

// Global pointer for signal handler access
static MainService* g_service = NULL;

void signal_handler(int sig)
{
    fprintf(stdout, "\n[main] Caught signal %d. Shutting down...\n", sig);
    if (g_service != NULL)
    {
        g_service->stop();
    }
}

int main(int argc, char* argv[])
{
    char params_path[256];

    snprintf(params_path, sizeof(params_path), "params.conf");
    if (argc > 1)
    {
        snprintf(params_path, sizeof(params_path), "%s", argv[1]);
    }

    fprintf(stdout, "[main] Starting First Demo with config: %s\n", params_path);

    // Register signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Instantiate and start MainService
    g_service = new MainService(params_path);
    g_service->start();

    // Wait/block main thread until a signal is received
    pause();

    // Clean exit
    delete g_service;
    g_service = NULL;
    
    fprintf(stdout, "[main] Application exited cleanly.\n");
    return 0;
}
