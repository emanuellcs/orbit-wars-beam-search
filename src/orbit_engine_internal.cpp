/**
 * @file orbit_engine_internal.cpp
 * @brief Shared scalar math, angle normalization, comet indexing, and seed mixing.
 *
 * These helpers are intentionally small, deterministic, and allocation-free.
 * Centralizing them keeps SoA indexing and angular wrap logic consistent across
 * simulator, geometry, candidate generation, and search.
 */
#include "orbit_engine_internal.hpp"

#include <cmath>

namespace orbit::detail {

/**
 * @brief Clamp a scalar into an inclusive interval.
 * @param value Candidate value.
 * @param lo Lower inclusive bound.
 * @param hi Upper inclusive bound.
 * @return value constrained to [lo, hi].
 */
double clamp(double value, double lo, double hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

/**
 * @brief Square a scalar.
 * @param value Scalar input.
 * @return value multiplied by itself.
 */
double sqr(double value) {
    return value * value;
}

/**
 * @brief Compute Euclidean distance between two points.
 * @param a First point.
 * @param b Second point.
 * @return Distance in board units.
 */
double distance(Vec2 a, Vec2 b) {
    return std::hypot(a.x - b.x, a.y - b.y);
}

/**
 * @brief Normalize an angle into [0, 2*pi).
 * @param angle Angle in radians.
 * @return Equivalent positive-turn angle.
 */
double normalize_angle(double angle) {
    double out = std::fmod(angle, TWO_PI);
    if (out < 0.0) {
        out += TWO_PI;
    }
    return out;
}

/**
 * @brief Compute the signed shortest angular difference.
 * @param from Source angle in radians.
 * @param to Target angle in radians.
 * @return Signed delta in [-pi, pi].
 */
double angle_delta(double from, double to) {
    double delta = normalize_angle(to) - normalize_angle(from);
    if (delta > PI) {
        delta -= TWO_PI;
    } else if (delta < -PI) {
        delta += TWO_PI;
    }
    return delta;
}

/**
 * @brief Check if a point is within the board bounds.
 * @param p Point to test.
 * @return true when both coordinates are inside the closed square.
 */
bool inside_board(Vec2 p) {
    return p.x >= 0.0 && p.x <= BOARD_SIZE && p.y >= 0.0 && p.y <= BOARD_SIZE;
}

/**
 * @brief Search the observation's comet id list.
 * @param obs Fixed-buffer observation.
 * @param planet_id Planet id to test.
 * @return true when the planet id is marked as a comet.
 */
bool comet_id_in_observation(const ObservationInput& obs, int planet_id) {
    for (int i = 0; i < obs.comet_planet_id_count; ++i) {
        if (obs.comet_planet_ids[static_cast<size_t>(i)] == planet_id) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Flatten a comet group and slot into a dense index.
 * @param group Comet group index.
 * @param slot Comet slot index within the group.
 * @return Flat group/slot index.
 */
int comet_slot_flat(int group, int slot) {
    return group * MAX_COMETS_PER_GROUP + slot;
}

/**
 * @brief Flatten a comet path sample into a dense index.
 * @param group Comet group index.
 * @param slot Comet slot index within the group.
 * @param point Path sample index.
 * @return Flat group/slot/point index.
 */
int comet_path_flat(int group, int slot, int point) {
    return (group * MAX_COMETS_PER_GROUP + slot) * MAX_COMET_PATH_POINTS + point;
}

/**
 * @brief Mix a 64-bit seed for deterministic branch variation.
 * @param value Input seed.
 * @return Scrambled 64-bit value.
 * @note Constants are from SplitMix64-style avalanche mixing, which is cheap
 *       and sufficient for decorrelating worker/candidate ids.
 */
uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace orbit::detail
