#pragma once
// Shared pieces for the benchmark programs: repeated timing with a median,
// CSV output, and a small argument parser. Every number this project reports
// is a median over repeated runs after warm-up; nothing comes from one run.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

struct Timing {
    double median_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
    int runs = 0;
};

inline double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Run fn `warmup` times untimed, then `runs` times timed. Returns median/min/max in ms.
inline Timing time_repeated(const std::function<void()>& fn, int warmup, int runs) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    Timing t;
    t.median_ms = median(samples);
    t.min_ms = *std::min_element(samples.begin(), samples.end());
    t.max_ms = *std::max_element(samples.begin(), samples.end());
    t.runs = runs;
    return t;
}

// Keep the loop that produces `value` alive without a store the optimiser can hoist.
template <typename T>
inline void keep(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

struct Args {
    std::map<std::string, std::string> opts;
    bool flag(const std::string& k) const { return opts.count(k) != 0; }
    std::string str(const std::string& k, const std::string& def) const {
        auto it = opts.find(k);
        return it == opts.end() ? def : it->second;
    }
    long num(const std::string& k, long def) const {
        auto it = opts.find(k);
        if (it == opts.end()) return def;
        const long v = std::stol(it->second);
        if (v < 0) throw std::runtime_error("--" + k + " must be non-negative");
        return v;
    }
    static Args parse(int argc, char** argv) {
        Args a;
        for (int i = 1; i < argc; ++i) {
            std::string s = argv[i];
            if (s.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument " + s);
            s = s.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                a.opts[s] = argv[++i];
            } else {
                a.opts[s] = "1";
            }
        }
        return a;
    }
};

// Append rows to a CSV file, writing the header only when the file is new or empty.
class Csv {
public:
    Csv(const std::string& path, const std::string& header) {
        if (path.empty()) return;
        bool need_header = true;
        {
            std::ifstream in(path);
            need_header = !in || in.peek() == std::ifstream::traits_type::eof();
        }
        out_.open(path, std::ios::app);
        if (!out_) throw std::runtime_error("cannot open " + path);
        if (need_header) out_ << header << "\n";
    }
    void row(const std::string& line) {
        if (out_) out_ << line << "\n";
    }

private:
    std::ofstream out_;
};

}  // namespace bench
