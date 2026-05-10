/**
 * @file orbit_engine_internal.hpp
 * @brief Internal math and indexing helpers shared by native engine modules.
 *
 * These functions are intentionally tiny and allocation-free. They centralize
 * angle normalization, comet array flattening, and deterministic seed mixing so
 * simulation, geometry, and search agree on low-level terminology and indexing.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit::detail {

/// @brief Circle constant used by angle helpers.
constexpr double PI = 3.141592653589793238462643383279502884;
/// @brief Full turn in radians.
constexpr double TWO_PI = 2.0 * PI;
/// @brief Shared epsilon for geometric degeneracy checks.
constexpr double EPS = 1.0e-9;

/// @brief Clamp a scalar into an inclusive interval.
/// @param value Candidate value.
/// @param lo Inclusive lower bound.
/// @param hi Inclusive upper bound.
/// @return value constrained to [lo, hi].
double clamp(double value, double lo, double hi);
/// @brief Square a scalar without invoking generic power functions.
/// @param value Scalar to square.
/// @return value * value.
double sqr(double value);
/// @brief Compute Euclidean distance between two board points.
/// @param a First point.
/// @param b Second point.
/// @return Straight-line distance in board units.
double distance(Vec2 a, Vec2 b);
/// @brief Normalize an angle into [0, 2*pi).
/// @param angle Angle in radians.
/// @return Equivalent normalized angle.
double normalize_angle(double angle);
/// @brief Compute the signed shortest angular delta from one heading to another.
/// @param from Source angle in radians.
/// @param to Target angle in radians.
/// @return Signed delta in [-pi, pi].
double angle_delta(double from, double to);
/// @brief Check whether a point remains inside the square board.
/// @param p Point to test.
/// @return true when both coordinates are within [0, BOARD_SIZE].
bool inside_board(Vec2 p);
/// @brief Test whether an observation marks a planet id as a comet.
/// @param obs Converted observation.
/// @param planet_id Planet id to search for.
/// @return true when planet_id appears in comet_planet_ids.
/// @note O(obs.comet_planet_id_count), bounded by MAX_PLANETS.
bool comet_id_in_observation(const ObservationInput& obs, int planet_id);
/// @brief Flatten a comet group and slot into a per-comet index.
/// @param group Comet group index.
/// @param slot Slot within the mirrored group.
/// @return Flat index into group/slot arrays.
int comet_slot_flat(int group, int slot);
/// @brief Flatten comet group, slot, and path point into a path sample index.
/// @param group Comet group index.
/// @param slot Slot within the mirrored group.
/// @param point Path sample index.
/// @return Flat index into path_x/path_y arrays.
int comet_path_flat(int group, int slot, int point);
/// @brief Deterministically mix a 64-bit seed for independent worker branches.
/// @param value Input seed.
/// @return Scrambled seed with high bit diffusion.
uint64_t mix64(uint64_t value);

}  // namespace orbit::detail
