// ScopeTimer.h
#pragma once

#include <chrono>
#include <iostream>
#include <string>

#define DEBUG_SCOPE_TIMER_FN() DebugScopeTimer debugScopeTimer_##__LINE__ (__func__)
#define DEBUG_SCOPE_TIMER(label) DebugScopeTimer debugScopeTimer_##__LINE__ (label)


// DebugScopeTimer.h
#pragma once

#include <chrono>
#include <string>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

inline void debugPrint(const std::string& msg)
{
#if defined(_WIN32)
    ::OutputDebugStringA(msg.c_str());
#else
    // Fallback for non-Windows builds
    std::cerr << msg;
#endif
}

class DebugScopeTimer
{
public:
    using Clock = std::chrono::steady_clock;

    explicit DebugScopeTimer(const char* label)
        : label_(label),
        start_(Clock::now())
    {
    }

    ~DebugScopeTimer()
    {
        const auto end = Clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        const double ms = us / 1000.0;
        const double sec = ms / 1000.0;

        std::ostringstream oss;
        oss << "[TIMER] " << label_ << " took "
            << ms << " ms (" << sec << " s)\n";

        debugPrint(oss.str());
    }

private:
    const char* label_;
    Clock::time_point start_;
};
