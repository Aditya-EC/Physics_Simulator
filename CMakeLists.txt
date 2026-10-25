# Top-level build entry for the PSim native numerical core.
# Prefer `python build_native.py`, which configures, builds, tests and installs the module.
cmake_minimum_required(VERSION 3.18)
project(psim_studio LANGUAGES CXX)
add_subdirectory(native)
