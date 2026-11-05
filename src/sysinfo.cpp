#include "tinyinfer/sysinfo.h"

#include <fstream>
#include <thread>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#ifndef TINYINFER_BUILD_FLAGS
#define TINYINFER_BUILD_FLAGS "unknown"
#endif

namespace tinyinfer {

std::string cpu_name() {
#if defined(__APPLE__)
    char buf[256];
    size_t len = sizeof buf;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0 && len > 0) {
        return std::string(buf, len - 1);
    }
#elif defined(__linux__)
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            const size_t colon = line.find(':');
            if (colon != std::string::npos) {
                size_t start = colon + 1;
                while (start < line.size() && line[start] == ' ') ++start;
                return line.substr(start);
            }
        }
    }
#endif
    return "unknown";
}

std::string compiler_name() {
#if defined(__clang__)
    return "Clang " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__) + "." +
           std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    return "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#else
    return "unknown compiler";
#endif
}

std::string build_flags() { return TINYINFER_BUILD_FLAGS; }

int hardware_threads() {
    const unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : static_cast<int>(n);
}

}  // namespace tinyinfer
