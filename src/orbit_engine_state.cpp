/**
 * @file orbit_engine_state.cpp
 * @brief Observation ingestion and fixed-buffer state reconstruction.
 *
 * This file converts a Python/Kaggle observation snapshot into the engine's SoA
 * representation. It also reconstructs orbit metadata and attaches comet path
 * indices so simulation and search can predict future positions without further
 * Python interaction.
 */
#include "orbit_engine.hpp"

#include "geometry.hpp"
#include "orbit_engine_internal.hpp"

#include <algorithm>
#include <cmath>

namespace orbit {
namespace {

/**
 * @brief Find the initial observation row for a planet id.
 * @param obs Fixed-buffer observation.
 * @param planet_id Planet id to locate.
 * @return Index in obs.initial_planets, or -1 when absent.
 * @note O(initial_planet_count), bounded by MAX_PLANETS.
 */
int find_initial_planet(const ObservationInput& obs, int planet_id) {
    for (int i = 0; i < obs.initial_planet_count; ++i) {
        if (obs.initial_planets[static_cast<size_t>(i)].id == planet_id) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Copy observed comet paths into flattened native storage.
 * @param state Mutable game state receiving comet path data.
 * @param obs Fixed-buffer observation containing comet groups.
 * @note Counts are clamped to fixed capacities to protect rollout buffers.
 */
void copy_comet_paths(GameState& state, const ObservationInput& obs) {
    state.comets.group_count = std::min(obs.comet_group_count, MAX_COMET_GROUPS);
    for (int g = 0; g < state.comets.group_count; ++g) {
        const CometGroupObservation& src = obs.comet_groups[static_cast<size_t>(g)];
        state.comets.planet_count[static_cast<size_t>(g)] =
            std::min(src.planet_count, MAX_COMETS_PER_GROUP);
        state.comets.path_index[static_cast<size_t>(g)] = std::max(0, src.path_index);
        for (int s = 0; s < state.comets.planet_count[static_cast<size_t>(g)]; ++s) {
            const int flat = state.comets.slot_index(g, s);
            state.comets.planet_id[static_cast<size_t>(flat)] = src.planet_ids[static_cast<size_t>(s)];
            state.comets.path_len[static_cast<size_t>(flat)] =
                std::min(std::max(0, src.path_len[static_cast<size_t>(s)]), MAX_COMET_PATH_POINTS);
            for (int p = 0; p < state.comets.path_len[static_cast<size_t>(flat)]; ++p) {
                const int dst = state.comets.path_index_flat(g, s, p);
                const int src_index = s * MAX_COMET_PATH_POINTS + p;
                state.comets.path_x[static_cast<size_t>(dst)] = src.path_x[static_cast<size_t>(src_index)];
                state.comets.path_y[static_cast<size_t>(dst)] = src.path_y[static_cast<size_t>(src_index)];
            }
        }
    }
}

/**
 * @brief Mark planet SoA slots that correspond to active comet observations.
 * @param state Mutable state with planets and comet path ids loaded.
 * @note Comets override orbiting metadata because their path comes from samples.
 */
void attach_comet_metadata(GameState& state) {
    for (int g = 0; g < state.comets.group_count; ++g) {
        const int slots = state.comets.planet_count[static_cast<size_t>(g)];
        for (int s = 0; s < slots; ++s) {
            const int flat = state.comets.slot_index(g, s);
            const int planet_id = state.comets.planet_id[static_cast<size_t>(flat)];
            const int planet_index = state.planet_index_by_id(planet_id);
            if (planet_index < 0) {
                continue;
            }
            state.planets.is_comet[static_cast<size_t>(planet_index)] = 1;
            state.planets.comet_group[static_cast<size_t>(planet_index)] = g;
            state.planets.comet_slot[static_cast<size_t>(planet_index)] = s;
            state.planets.is_orbiting[static_cast<size_t>(planet_index)] = 0;
            state.planets.radius[static_cast<size_t>(planet_index)] = COMET_RADIUS;
        }
    }
}

}  // namespace

/**
 * @brief Reset the LaunchList to an empty logical buffer.
 * @note Clearing rows prevents stale debug output while preserving fixed storage.
 */
void LaunchList::clear() {
    count = 0;
    for (Launch& launch : launches) {
        launch = Launch{};
    }
}

/**
 * @brief Append one launch order when capacity and ship count are valid.
 * @param from_planet_id Source planet id.
 * @param angle Launch heading in radians.
 * @param ships Positive ship count.
 * @return true when appended, false when full or invalid.
 */
bool LaunchList::add(int from_planet_id, double angle, int ships) {
    if (count >= MAX_LAUNCHES || ships <= 0) {
        return false;
    }
    launches[static_cast<size_t>(count)] = Launch{from_planet_id, angle, ships};
    ++count;
    return true;
}

/**
 * @brief Clear all planet SoA fields to neutral defaults.
 * @note Uses fill on fixed arrays so state reloads are deterministic.
 */
void PlanetSoA::clear() {
    count = 0;
    alive.fill(0);
    id.fill(-1);
    ships.fill(0);
    owner.fill(-1);
    production.fill(0);
    x.fill(0.0);
    y.fill(0.0);
    radius.fill(0.0);
    is_orbiting.fill(0);
    angular_velocity.fill(0.0);
    initial_angle.fill(0.0);
    orbit_radius.fill(0.0);
    is_comet.fill(0);
    comet_group.fill(-1);
    comet_slot.fill(-1);
}

/**
 * @brief Clear all fleet SoA fields and reset generated ids.
 * @note Tombstoned and unused slots become indistinguishable after reload.
 */
void FleetSoA::clear() {
    count = 0;
    next_id = 1;
    alive.fill(0);
    id.fill(-1);
    owner.fill(-1);
    ships.fill(0);
    from_planet_id.fill(-1);
    x.fill(0.0);
    y.fill(0.0);
    angle.fill(0.0);
    speed.fill(0.0);
}

/**
 * @brief Add a fleet into a tombstoned or fresh fixed-buffer slot.
 * @param fleet_id Observation id, or negative to allocate a simulated id.
 * @param fleet_owner Owner id.
 * @param fleet_x X coordinate.
 * @param fleet_y Y coordinate.
 * @param fleet_angle Heading in radians.
 * @param from_id Source planet id.
 * @param fleet_ships Ships carried by the fleet.
 * @return SoA slot index, or -1 if MAX_FLEETS is exhausted.
 * @note The cached speed avoids recomputing the logarithmic curve every tick.
 */
int FleetSoA::add(int fleet_id, int fleet_owner, double fleet_x, double fleet_y,
                  double fleet_angle, int from_id, int fleet_ships) {
    int slot = -1;
    for (int i = 0; i < count; ++i) {
        if (alive[static_cast<size_t>(i)] == 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        if (count >= MAX_FLEETS) {
            return -1;
        }
        slot = count++;
    }
    alive[static_cast<size_t>(slot)] = 1;
    id[static_cast<size_t>(slot)] = fleet_id >= 0 ? fleet_id : next_id++;
    next_id = std::max(next_id, id[static_cast<size_t>(slot)] + 1);
    owner[static_cast<size_t>(slot)] = fleet_owner;
    ships[static_cast<size_t>(slot)] = fleet_ships;
    from_planet_id[static_cast<size_t>(slot)] = from_id;
    x[static_cast<size_t>(slot)] = fleet_x;
    y[static_cast<size_t>(slot)] = fleet_y;
    angle[static_cast<size_t>(slot)] = fleet_angle;
    speed[static_cast<size_t>(slot)] = speed_for_ships(fleet_ships);
    return slot;
}

/**
 * @brief Mark a fleet slot dead without compacting the SoA arrays.
 * @param index Fleet SoA slot to tombstone.
 * @note O(1) removal preserves fixed indices during in-progress loops.
 */
void FleetSoA::remove(int index) {
    if (index >= 0 && index < count) {
        alive[static_cast<size_t>(index)] = 0;
    }
}

/**
 * @brief Reset comet path metadata and sample storage.
 * @note Path arrays are fixed-size because comet interpolation is used in search.
 */
void CometPathStore::clear() {
    group_count = 0;
    planet_count.fill(0);
    path_index.fill(0);
    planet_id.fill(-1);
    path_len.fill(0);
    path_x.fill(0.0);
    path_y.fill(0.0);
}

/**
 * @brief Flatten a comet group/slot pair.
 * @param group Comet group index.
 * @param slot Slot within the group.
 * @return Flat per-comet index.
 */
int CometPathStore::slot_index(int group, int slot) const {
    return detail::comet_slot_flat(group, slot);
}

/**
 * @brief Flatten a comet path sample coordinate.
 * @param group Comet group index.
 * @param slot Slot within the group.
 * @param point Path point index.
 * @return Flat path coordinate index.
 */
int CometPathStore::path_index_flat(int group, int slot, int point) const {
    return detail::comet_path_flat(group, slot, point);
}

/**
 * @brief Clear queued combat arrivals.
 * @note The queue is a dense planet-owner matrix to avoid associative storage.
 */
void CombatQueue::clear() {
    ships.fill(0);
}

/**
 * @brief Accumulate arriving ships for one planet and owner.
 * @param planet_index SoA planet index.
 * @param owner Arriving owner id.
 * @param ship_count Positive arriving ship count.
 * @note Invalid coordinates are ignored so callers can stay branch-light.
 */
void CombatQueue::add(int planet_index, int owner, int ship_count) {
    if (planet_index < 0 || planet_index >= MAX_PLANETS || owner < 0 || owner >= MAX_PLAYERS || ship_count <= 0) {
        return;
    }
    ships[static_cast<size_t>(planet_index * MAX_PLAYERS + owner)] += ship_count;
}

/**
 * @brief Read accumulated arrivals for one planet and owner.
 * @param planet_index SoA planet index.
 * @param owner Owner id.
 * @return Queued ship count, or zero for invalid coordinates.
 */
int CombatQueue::at(int planet_index, int owner) const {
    if (planet_index < 0 || planet_index >= MAX_PLANETS || owner < 0 || owner >= MAX_PLAYERS) {
        return 0;
    }
    return ships[static_cast<size_t>(planet_index * MAX_PLAYERS + owner)];
}

/**
 * @brief Reset the entire game state to an empty non-terminal snapshot.
 * @note Called before every observation load and by the engine constructor.
 */
void GameState::reset() {
    player = 0;
    step = 0;
    angular_velocity = 0.0;
    remaining_overage_time = 0.0;
    done = false;
    winner = -1;
    planets.clear();
    fleets.clear();
    comets.clear();
}

/**
 * @brief Load a Kaggle observation into native SoA buffers.
 * @param obs Fixed-buffer observation created by the pybind11 bridge.
 * @note Initial positions reconstruct orbit radius/phase; comet ids attach
 *       sampled path metadata and disable circular orbit prediction.
 */
void GameState::load_from_observation(const ObservationInput& obs) {
    reset();
    player = obs.player;
    step = obs.step;
    angular_velocity = obs.angular_velocity;
    remaining_overage_time = obs.remaining_overage_time;
    planets.count = std::min(obs.planet_count, MAX_PLANETS);

    for (int i = 0; i < planets.count; ++i) {
        const PlanetObservation& p = obs.planets[static_cast<size_t>(i)];
        planets.alive[static_cast<size_t>(i)] = 1;
        planets.id[static_cast<size_t>(i)] = p.id;
        planets.owner[static_cast<size_t>(i)] = p.owner;
        planets.x[static_cast<size_t>(i)] = p.x;
        planets.y[static_cast<size_t>(i)] = p.y;
        planets.radius[static_cast<size_t>(i)] = p.radius;
        planets.ships[static_cast<size_t>(i)] = p.ships;
        planets.production[static_cast<size_t>(i)] = p.production;
        planets.is_comet[static_cast<size_t>(i)] =
            detail::comet_id_in_observation(obs, p.id) ? 1U : 0U;

        const int init_index = find_initial_planet(obs, p.id);
        const double init_x = init_index >= 0 ? obs.initial_planets[static_cast<size_t>(init_index)].x : p.x;
        const double init_y = init_index >= 0 ? obs.initial_planets[static_cast<size_t>(init_index)].y : p.y;
        const double dx = init_x - CENTER_X;
        const double dy = init_y - CENTER_Y;
        const double orbit_r = std::hypot(dx, dy);
        planets.orbit_radius[static_cast<size_t>(i)] = orbit_r;
        planets.initial_angle[static_cast<size_t>(i)] = std::atan2(dy, dx);
        const bool orbiting = planets.is_comet[static_cast<size_t>(i)] == 0 &&
                              orbit_r + p.radius < ROTATION_RADIUS_LIMIT;
        planets.is_orbiting[static_cast<size_t>(i)] = orbiting ? 1U : 0U;
        planets.angular_velocity[static_cast<size_t>(i)] = orbiting ? obs.angular_velocity : 0.0;
    }

    copy_comet_paths(*this, obs);
    attach_comet_metadata(*this);

    fleets.count = 0;
    fleets.next_id = 1;
    for (int i = 0; i < obs.fleet_count; ++i) {
        const FleetObservation& f = obs.fleets[static_cast<size_t>(i)];
        fleets.add(f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships);
    }
}

/**
 * @brief Find a live planet slot by environment id.
 * @param planet_id Planet id to find.
 * @return SoA slot index, or -1 when not present.
 */
int GameState::planet_index_by_id(int planet_id) const {
    for (int i = 0; i < planets.count; ++i) {
        if (planets.alive[static_cast<size_t>(i)] != 0 && planets.id[static_cast<size_t>(i)] == planet_id) {
            return i;
        }
    }
    return -1;
}

/**
 * @brief Predict a planet center after a fractional number of ticks.
 * @param planet_index Planet SoA slot.
 * @param ticks Future offset in ticks.
 * @return Predicted center or {0,0} for invalid/dead slots.
 * @note Comets use linear interpolation between path samples; orbiting planets
 *       use the current observed phase plus angular_velocity * ticks.
 */
Vec2 GameState::planet_position_after(int planet_index, double ticks) const {
    if (planet_index < 0 || planet_index >= planets.count ||
        planets.alive[static_cast<size_t>(planet_index)] == 0) {
        return Vec2{};
    }
    if (planets.is_comet[static_cast<size_t>(planet_index)] != 0) {
        const int group = planets.comet_group[static_cast<size_t>(planet_index)];
        const int slot = planets.comet_slot[static_cast<size_t>(planet_index)];
        if (group >= 0 && slot >= 0) {
            const int flat = comets.slot_index(group, slot);
            const int len = comets.path_len[static_cast<size_t>(flat)];
            if (len > 0) {
                const double raw = static_cast<double>(comets.path_index[static_cast<size_t>(group)]) + ticks;
                const int lo = std::min(std::max(0, static_cast<int>(std::floor(raw))), len - 1);
                const int hi = std::min(lo + 1, len - 1);
                const double frac = detail::clamp(raw - static_cast<double>(lo), 0.0, 1.0);
                const int li = comets.path_index_flat(group, slot, lo);
                const int hi_i = comets.path_index_flat(group, slot, hi);
                const double x = comets.path_x[static_cast<size_t>(li)] * (1.0 - frac) +
                                 comets.path_x[static_cast<size_t>(hi_i)] * frac;
                const double y = comets.path_y[static_cast<size_t>(li)] * (1.0 - frac) +
                                 comets.path_y[static_cast<size_t>(hi_i)] * frac;
                return Vec2{x, y};
            }
        }
    }
    if (planets.is_orbiting[static_cast<size_t>(planet_index)] != 0) {
        // Use the observed current phase rather than only initial_angle so
        // mid-game observation reloads remain aligned with Kaggle state.
        const double phase = std::atan2(planets.y[static_cast<size_t>(planet_index)] - CENTER_Y,
                                        planets.x[static_cast<size_t>(planet_index)] - CENTER_X) +
                             planets.angular_velocity[static_cast<size_t>(planet_index)] * ticks;
        const double r = planets.orbit_radius[static_cast<size_t>(planet_index)];
        return Vec2{CENTER_X + std::cos(phase) * r, CENTER_Y + std::sin(phase) * r};
    }
    return Vec2{planets.x[static_cast<size_t>(planet_index)],
                planets.y[static_cast<size_t>(planet_index)]};
}

/**
 * @brief Determine whether an owner still has any live planets or fleets.
 * @param owner_value Owner id.
 * @return true when the owner remains active in terminal checks.
 */
bool GameState::is_active_player(int owner_value) const {
    if (owner_value < 0 || owner_value >= MAX_PLAYERS) {
        return false;
    }
    for (int i = 0; i < planets.count; ++i) {
        if (planets.alive[static_cast<size_t>(i)] != 0 &&
            planets.owner[static_cast<size_t>(i)] == owner_value) {
            return true;
        }
    }
    for (int i = 0; i < fleets.count; ++i) {
        if (fleets.alive[static_cast<size_t>(i)] != 0 &&
            fleets.owner[static_cast<size_t>(i)] == owner_value) {
            return true;
        }
    }
    return false;
}

}  // namespace orbit
