cmake_minimum_required(VERSION 3.20)

# Build definition for the Orbit Wars native pybind11 extension. The target is
# optimized for short-lived Kaggle action calls: C++20 for modern fixed-buffer
# code, pybind11 for the Python boundary, and pthreads for bounded root search.
project(orbit_wars_beam_search LANGUAGES CXX)

# Keep the language mode explicit and portable. Extensions are disabled so the
# same source bundle can compile under Kaggle, CI, and local toolchains.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for local static analysis and editor tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The extension is loaded by Python, so CMake must discover both the interpreter
# and the development module headers/libs that match it.
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Native search evaluates root macro-actions on bounded worker threads.
find_package(Threads REQUIRED)

# pybind11 may be installed in the active Python environment rather than a
# system CMake prefix. Asking Python for --cmakedir makes the build reproducible
# across virtualenvs and CI runners.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
)

if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 is not available. Install requirements-dev.txt first.")
endif()

# Teach find_package(pybind11) about the Python-resolved package location.
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

# Build a CPython extension module named orbit_engine from all native engine
# modules. The Python entrypoint can import this module directly via PYTHONPATH.
pybind11_add_module(orbit_engine
  src/orbit_engine_bindings.cpp
  src/orbit_engine_candidate.cpp
  src/orbit_engine_engine.cpp
  src/orbit_engine_eval.cpp
  src/orbit_engine_geometry.cpp
  src/orbit_engine_internal.cpp
  src/orbit_engine_search.cpp
  src/orbit_engine_sim.cpp
  src/orbit_engine_state.cpp
)

# Keep includes private: only the extension target consumes src/*.hpp.
target_include_directories(orbit_engine PRIVATE src)

# Link the platform thread library required by std::thread in search.
target_link_libraries(orbit_engine PRIVATE Threads::Threads)

# Warnings are enabled for development builds. Release builds add the same
# high-performance flags used by main.py's JIT path: O3 for optimizer depth,
# march=native for local CPU instructions, and ffast-math for hot geometry.
target_compile_options(orbit_engine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native -ffast-math>
)

# Define NDEBUG in Release so assertions and debug-only checks do not leak into
# Kaggle's 900 ms action budget.
target_compile_definitions(orbit_engine PRIVATE
  $<$<CONFIG:Release>:NDEBUG>
)
