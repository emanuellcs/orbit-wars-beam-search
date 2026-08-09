#pragma once

#include <cstdint>

namespace orbit {

/// @brief Maximum number of owners supported by the Kaggle rule set.
constexpr int MAX_PLAYERS = 4;
/// @brief Fixed planet capacity, including transient comet planets.
constexpr int MAX_PLANETS = 96;
/// @brief Fixed active fleet capacity used by simulation and rollouts.
constexpr int MAX_FLEETS = 4096;
/// @brief Maximum number of comet groups retained from observations.
constexpr int MAX_COMET_GROUPS = 8;
/// @brief Mirrored comet slots per group, one per quadrant.
constexpr int MAX_COMETS_PER_GROUP = 4;
/// @brief Maximum sampled path points stored for each comet.
constexpr int MAX_COMET_PATH_POINTS = 512;
/// @brief Capacity for ranked one-source tactical launch candidates.
constexpr int MAX_ATOMIC_LAUNCHES = 1536;
/// @brief Capacity for packed multi-launch macro-actions.
constexpr int MAX_MACRO_ACTIONS = 512;
/// @brief Upper bound on root candidates evaluated by search workers.
constexpr int MAX_BEAM_WIDTH = 512;
/// @brief Maximum launches emitted in a single action list.
constexpr int MAX_LAUNCHES = 128;
/// @brief Hard cap on parallel root-evaluation workers.
constexpr int MAX_SEARCH_THREADS = 20;

/// @brief Width and height of the continuous square board.
constexpr double BOARD_SIZE = 100.0;
/// @brief X coordinate of the sun and orbital center.
constexpr double CENTER_X = 50.0;
/// @brief Y coordinate of the sun and orbital center.
constexpr double CENTER_Y = 50.0;
/// @brief Radius of the central collision hazard.
constexpr double SUN_RADIUS = 10.0;
/// @brief Inner planets rotate only when orbit radius plus body radius fits.
constexpr double ROTATION_RADIUS_LIMIT = 50.0;
/// @brief Asymptotic fleet speed reached by large launches.
constexpr double DEFAULT_SHIP_SPEED = 6.0;
/// @brief Rule-defined radius for comet planets.
constexpr double COMET_RADIUS = 1.0;
/// @brief Maximum simulated turns in one Orbit Wars episode.
constexpr int EPISODE_STEPS = 500;

/// @brief Lightweight 2D point/vector used by geometry hot paths.
struct Vec2 {
    ///< X coordinate in board units.
    double x = 0.0;
    ///< Y coordinate in board units.
    double y = 0.0;
};

}  // namespace orbit
