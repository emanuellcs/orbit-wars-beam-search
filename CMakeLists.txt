cmake_minimum_required(VERSION 3.20)

project(orbit_wars_beam_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
)

if(NOT PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "pybind11 is not available. Install requirements-dev.txt first.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

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

target_include_directories(orbit_engine PRIVATE src)
target_link_libraries(orbit_engine PRIVATE Threads::Threads)

target_compile_options(orbit_engine PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3 -march=native -ffast-math>
)

target_compile_definitions(orbit_engine PRIVATE
  $<$<CONFIG:Release>:NDEBUG>
)
