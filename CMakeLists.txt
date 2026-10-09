cmake_minimum_required(VERSION 3.16)
project(tinyinfer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
# The project plan benchmarks at -O2; keep every build comparable.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

option(TINYINFER_NATIVE "Compile for the host CPU (-march=native) so SIMD kernels are enabled" ON)
option(TINYINFER_WERROR "Treat warnings as errors" ON)
option(TINYINFER_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

include(CheckCXXCompilerFlag)

set(TINYINFER_CXX_FLAGS -Wall -Wextra -Wpedantic -Wshadow)
if(TINYINFER_WERROR)
  list(APPEND TINYINFER_CXX_FLAGS -Werror)
endif()
if(TINYINFER_NATIVE)
  check_cxx_compiler_flag(-march=native TINYINFER_HAS_MARCH_NATIVE)
  if(TINYINFER_HAS_MARCH_NATIVE)
    list(APPEND TINYINFER_CXX_FLAGS -march=native)
  else()
    check_cxx_compiler_flag(-mcpu=native TINYINFER_HAS_MCPU_NATIVE)
    if(TINYINFER_HAS_MCPU_NATIVE)
      list(APPEND TINYINFER_CXX_FLAGS -mcpu=native)
    endif()
  endif()
endif()
if(TINYINFER_SANITIZE)
  list(APPEND TINYINFER_CXX_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

string(JOIN " " TINYINFER_FLAGS_STRING ${TINYINFER_CXX_FLAGS})
set(TINYINFER_FLAGS_STRING "${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}} ${TINYINFER_FLAGS_STRING}")
string(STRIP "${TINYINFER_FLAGS_STRING}" TINYINFER_FLAGS_STRING)
message(STATUS "tinyinfer flags: ${TINYINFER_FLAGS_STRING}")

add_library(tinyinfer STATIC
  src/tensor.cpp
  src/loader.cpp
  src/layers.cpp
  src/model.cpp
  src/sysinfo.cpp
  src/gemm/gemm.cpp
  src/gemm/gemm_naive.cpp
)
target_include_directories(tinyinfer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(tinyinfer PUBLIC ${TINYINFER_CXX_FLAGS})
target_compile_definitions(tinyinfer PRIVATE TINYINFER_BUILD_FLAGS="${TINYINFER_FLAGS_STRING}")

add_executable(tinyinfer_cli tools/tinyinfer_cli.cpp)
target_link_libraries(tinyinfer_cli PRIVATE tinyinfer)
set_target_properties(tinyinfer_cli PROPERTIES OUTPUT_NAME tinyinfer)

# Tests: one executable each, registered with CTest. The parity test gets the export dir.
enable_testing()
set(TINYINFER_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data/export)
function(tinyinfer_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE tinyinfer)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
  add_test(NAME ${name} COMMAND ${name} ${ARGN})
endfunction()
tinyinfer_test(test_tensor)
tinyinfer_test(test_loader)
tinyinfer_test(test_gemm)
tinyinfer_test(test_layers)
tinyinfer_test(test_parity ${TINYINFER_DATA_DIR})
