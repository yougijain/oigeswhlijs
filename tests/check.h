#pragma once
// A tiny test harness: no dependencies, one executable per test file.
//
//   TEST(name) { CHECK(x == 1); CHECK_NEAR(a, b, 1e-5); CHECK_THROWS(expr); }
//
// main() runs every TEST in definition order and exits non-zero if any check
// failed. Command-line arguments are available through test_args().

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace tinytest {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& cases() {
    static std::vector<Case> v;
    return v;
}
inline int& failures() {
    static int n = 0;
    return n;
}
inline std::vector<std::string>& args() {
    static std::vector<std::string> v;
    return v;
}

struct Register {
    Register(const char* name, std::function<void()> fn) { cases().push_back({name, std::move(fn)}); }
};

inline void fail(const char* file, int line, const std::string& msg) {
    std::fprintf(stderr, "  FAIL %s:%d: %s\n", file, line, msg.c_str());
    ++failures();
}

}  // namespace tinytest

#define TEST(name)                                            \
    static void name();                                       \
    static ::tinytest::Register tinytest_reg_##name(#name, name); \
    static void name()

#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) ::tinytest::fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_EQ(a, b)                                                                       \
    do {                                                                                     \
        if (!((a) == (b))) ::tinytest::fail(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ")"); \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                              \
    do {                                                                                                   \
        const double tt_a = static_cast<double>(a), tt_b = static_cast<double>(b);                         \
        if (!(std::fabs(tt_a - tt_b) <= (tol))) {                                                          \
            ::tinytest::fail(__FILE__, __LINE__,                                                           \
                             "CHECK_NEAR(" #a ", " #b "): " + std::to_string(tt_a) + " vs " +              \
                                 std::to_string(tt_b) + ", diff " + std::to_string(std::fabs(tt_a - tt_b))); \
        }                                                                                                  \
    } while (0)

#define CHECK_THROWS(expr)                                                                   \
    do {                                                                                     \
        bool tt_threw = false;                                                               \
        try {                                                                                \
            (void)(expr);                                                                    \
        } catch (const std::exception&) {                                                    \
            tt_threw = true;                                                                 \
        }                                                                                    \
        if (!tt_threw) ::tinytest::fail(__FILE__, __LINE__, "CHECK_THROWS(" #expr ") did not throw"); \
    } while (0)

inline const std::vector<std::string>& test_args() { return ::tinytest::args(); }

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) ::tinytest::args().push_back(argv[i]);
    int failed_cases = 0;
    for (const auto& c : ::tinytest::cases()) {
        const int before = ::tinytest::failures();
        std::printf("[ RUN  ] %s\n", c.name);
        try {
            c.fn();
        } catch (const std::exception& e) {
            ::tinytest::fail("<exception>", 0, std::string("uncaught: ") + e.what());
        }
        const bool ok = ::tinytest::failures() == before;
        std::printf("[ %s ] %s\n", ok ? " OK " : "FAIL", c.name);
        if (!ok) ++failed_cases;
    }
    std::printf("%zu tests, %d failed\n", ::tinytest::cases().size(), failed_cases);
    return failed_cases == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
