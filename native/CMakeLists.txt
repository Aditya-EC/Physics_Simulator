cmake_minimum_required(VERSION 3.18)
project(psim_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(PSIM_BUILD_TESTS "Build the native C++ tests" ON)

# Core numerical library (no Python dependency)
add_library(psim_core STATIC src/mesh_ops.cpp src/integrators.cpp src/constraints.cpp src/collision.cpp src/joints.cpp src/decompose.cpp)
target_include_directories(psim_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
    target_compile_options(psim_core PRIVATE /W4 /O2)
else()
    target_compile_options(psim_core PRIVATE -Wall -Wextra -O2)
endif()

# Python extension module
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_psim_native src/bindings.cpp)
target_link_libraries(_psim_native PRIVATE psim_core)

if(PSIM_BUILD_TESTS)
    enable_testing()
    add_executable(psim_native_tests tests/test_native.cpp)
    target_link_libraries(psim_native_tests PRIVATE psim_core)
    add_test(NAME psim_native_tests COMMAND psim_native_tests)
endif()
