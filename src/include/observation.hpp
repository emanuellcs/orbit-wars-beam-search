#pragma once

#include <array>
#include <cstdint>

#include "constants.hpp"

namespace orbit {

/// @brief Raw planet row decoded from a Python observation.
struct PlanetObservation {
    ///< Stable planet identifier from the Kaggle environment.
    int id = -1;
    ///< Owner id in [0, MAX_PLAYERS) or -1 for neutral.
    int owner = -1;
    ///< Current X coordinate.
    double x = 0.0;
    ///< Current Y coordinate.
    double y = 0.0;
    ///< Collision radius.
    double radius = 0.0;
    ///< Ships currently stationed on the planet.
    int ships = 0;
    ///< Ships produced per turn while owned.
    int production = 0;
};

/// @brief Raw fleet row decoded from a Python observation.
struct FleetObservation {
    ///< Stable fleet identifier from the Kaggle environment.
    int id = -1;
    ///< Owner id in [0, MAX_PLAYERS).
    int owner = -1;
    ///< Current X coordinate.
    double x = 0.0;
    ///< Current Y coordinate.
    double y = 0.0;
    ///< Heading in radians, where 0 points right and pi/2 points down.
    double angle = 0.0;
    ///< Planet id that launched the fleet, or -1 when unavailable.
    int from_planet_id = -1;
    ///< Ships carried by the fleet.
    int ships = 0;
};

/// @brief Fixed-buffer observation payload for one mirrored comet group.
struct CometGroupObservation {
    ///< Number of valid comet slots in this group.
    int planet_count = 0;
    ///< Current path sample index supplied by the environment.
    int path_index = 0;
    ///< Planet ids belonging to each comet slot.
    std::array<int, MAX_COMETS_PER_GROUP> planet_ids{};
    ///< Valid path sample count per comet slot.
    std::array<int, MAX_COMETS_PER_GROUP> path_len{};
    ///< Flattened path X coordinates, indexed by slot then point.
    std::array<double, MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_x{};
    ///< Flattened path Y coordinates, indexed by slot then point.
    std::array<double, MAX_COMETS_PER_GROUP * MAX_COMET_PATH_POINTS> path_y{};
};

/// @brief Full observation snapshot after Python-to-C++ conversion.
struct ObservationInput {
    ///< Controlled player id for this engine instance.
    int player = 0;
    ///< Current environment step.
    int step = 0;
    ///< Shared angular velocity used by orbiting planets.
    double angular_velocity = 0.0;
    ///< Kaggle overage time, preserved for diagnostics and future tuning.
    double remaining_overage_time = 0.0;
    ///< Number of valid entries in planets.
    int planet_count = 0;
    ///< Number of valid entries in fleets.
    int fleet_count = 0;
    ///< Number of valid entries in initial_planets.
    int initial_planet_count = 0;
    ///< Number of valid entries in comet_planet_ids.
    int comet_planet_id_count = 0;
    ///< Number of valid entries in comet_groups.
    int comet_group_count = 0;
    ///< Current observed planets, including active comets.
    std::array<PlanetObservation, MAX_PLANETS> planets{};
    ///< Current observed fleets.
    std::array<FleetObservation, MAX_FLEETS> fleets{};
    ///< Initial planet positions used to reconstruct orbital metadata.
    std::array<PlanetObservation, MAX_PLANETS> initial_planets{};
    ///< Planet ids that should be treated as comets.
    std::array<int, MAX_PLANETS> comet_planet_ids{};
    ///< Active comet path observations.
    std::array<CometGroupObservation, MAX_COMET_GROUPS> comet_groups{};
};

}  // namespace orbit
