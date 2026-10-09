#pragma once
// Where did this number come from? Every results file records these.

#include <string>

namespace tinyinfer {

std::string cpu_name();       // "Intel(R) Xeon(R) ..." / "Apple M2" / "unknown"
std::string compiler_name();  // "GCC 13.3.0" / "Clang 18.1.3"
std::string build_flags();    // what CMake compiled the library with
int hardware_threads();

}  // namespace tinyinfer
