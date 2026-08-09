#include "orbit_engine.hpp"

#include "geometry.hpp"
#include "orbit_engine_internal.hpp"

#include <algorithm>
#include <cmath>

namespace orbit {
namespace {

/// @brief Per-source launch spending accumulator for one turn.
struct SourceSpend {
    ///< Ships already spent from each planet SoA slot.
    std::array<int, MAX_PLANETS> spent{};

    /// @brief Reset all spend counters.
    /// @note O(MAX_PLANETS), fixed-bounded.
    void clear() {
        spent.fill(0);
    }
};

/**
 * @brief Remove comet planets that have left the valid observed path or board.
 * @param state Mutable game state.
 * @note Comets expire before launches, matching the rule that departing comets
 *       cannot be used as sources on the current turn.
 */
void expire_comets(GameState& state) {
    for (int i = 0; i < state.planets.count; ++i) {
        if (state.planets.alive[static_cast<size_t>(i)] == 0 ||
            state.planets.is_comet[static_cast<size_t>(i)] == 0) {
            continue;
        }
        const Vec2 p{state.planets.x[static_cast<size_t>(i)], state.planets.y[static_cast<size_t>(i)]};
        const int group = state.planets.comet_group[static_cast<size_t>(i)];
        const int slot = state.planets.comet_slot[static_cast<size_t>(i)];
        bool past_path = false;
        if (group >= 0 && slot >= 0) {
            const int flat = state.comets.slot_index(group, slot);
            past_path = state.comets.path_len[static_cast<size_t>(flat)] > 0 &&
                        state.comets.path_index[static_cast<size_t>(group)] >=
                            state.comets.path_len[static_cast<size_t>(flat)];
        }
        if (past_path || !detail::inside_board(p)) {
            state.planets.alive[static_cast<size_t>(i)] = 0;
            state.planets.owner[static_cast<size_t>(i)] = -1;
            state.planets.ships[static_cast<size_t>(i)] = 0;
        }
    }
}

/**
 * @brief Validate and instantiate launch orders as fleets.
 * @param state Mutable game state.
 * @param launches Joint launch list for this turn.
 * @note SourceSpend prevents multiple orders from overspending one garrison
 *       while keeping validation allocation-free.
 */
void process_launches(GameState& state, const LaunchList& launches) {
    SourceSpend spend{};
    spend.clear();
    for (int i = 0; i < launches.count; ++i) {
        const Launch& launch = launches.launches[static_cast<size_t>(i)];
        const int source = state.planet_index_by_id(launch.from_planet_id);
        if (source < 0 || launch.ships <= 0 ||
            state.planets.alive[static_cast<size_t>(source)] == 0 ||
            state.planets.owner[static_cast<size_t>(source)] < 0) {
            continue;
        }
        const int available = state.planets.ships[static_cast<size_t>(source)] -
                              spend.spent[static_cast<size_t>(source)];
        if (launch.ships > available) {
            continue;
        }
        spend.spent[static_cast<size_t>(source)] += launch.ships;
        state.planets.ships[static_cast<size_t>(source)] -= launch.ships;
        const Vec2 source_pos{state.planets.x[static_cast<size_t>(source)],
                              state.planets.y[static_cast<size_t>(source)]};
        const Vec2 spawn = point_on_heading(
            source_pos, launch.angle,
            state.planets.radius[static_cast<size_t>(source)] + 0.1);
        state.fleets.add(-1, state.planets.owner[static_cast<size_t>(source)],
                         spawn.x, spawn.y, launch.angle, launch.from_planet_id, launch.ships);
    }
}

/**
 * @brief Apply per-turn production to every live owned planet.
 * @param state Mutable game state.
 * @note Production happens after launches so freshly spent ships are not reused.
 */
void produce(GameState& state) {
    for (int i = 0; i < state.planets.count; ++i) {
        if (state.planets.alive[static_cast<size_t>(i)] != 0 &&
            state.planets.owner[static_cast<size_t>(i)] >= 0) {
            state.planets.ships[static_cast<size_t>(i)] += state.planets.production[static_cast<size_t>(i)];
        }
    }
}

/**
 * @brief Resolve all fleet arrivals queued against planets this tick.
 * @param state Mutable game state.
 * @param queue Dense planet-owner arrival matrix.
 * @note Arrivals fight each other first; only a unique survivor contests or
 *       reinforces the planet garrison.
 */
void resolve_planet_combats(GameState& state, const CombatQueue& queue) {
    for (int p = 0; p < state.planets.count; ++p) {
        if (state.planets.alive[static_cast<size_t>(p)] == 0) {
            continue;
        }
        int top_owner = -1;
        int top_ships = 0;
        int second_ships = 0;
        int top_count = 0;
        for (int owner = 0; owner < MAX_PLAYERS; ++owner) {
            const int ships = queue.at(p, owner);
            if (ships <= 0) {
                continue;
            }
            if (ships > top_ships) {
                second_ships = top_ships;
                top_ships = ships;
                top_owner = owner;
                top_count = 1;
            } else if (ships == top_ships) {
                ++top_count;
            } else if (ships > second_ships) {
                second_ships = ships;
            }
        }
        if (top_owner < 0 || top_count > 1) {
            continue;
        }
        const int survivor = top_ships - second_ships;
        if (survivor <= 0) {
            continue;
        }
        const int planet_owner = state.planets.owner[static_cast<size_t>(p)];
        if (planet_owner == top_owner) {
            state.planets.ships[static_cast<size_t>(p)] += survivor;
            continue;
        }
        const int garrison = state.planets.ships[static_cast<size_t>(p)];
        if (survivor > garrison) {
            state.planets.owner[static_cast<size_t>(p)] = top_owner;
            state.planets.ships[static_cast<size_t>(p)] = survivor - garrison;
        } else {
            state.planets.ships[static_cast<size_t>(p)] = garrison - survivor;
        }
    }
}

/**
 * @brief Move all fleets one tick using the environment's collision model.
 * @param state Mutable game state.
 * @param queue Combat queue receiving planet arrivals.
 * @param old_pos Planet centers at tick start.
 * @param next_pos Planet centers at tick end (linearised chord / comet sample).
 * @param check_collision Per-planet flag; false only for a comet's first placement.
 * @note Precedence mirrors the Kaggle engine exactly: planets are tested first
 *       with a relative swept-pair (so planet capture beats bounds/sun), then
 *       board bounds, then the sun. A fleet is removed at the first hit.
 */
void move_fleets(GameState& state, CombatQueue& queue,
                 const std::array<Vec2, MAX_PLANETS>& old_pos,
                 const std::array<Vec2, MAX_PLANETS>& next_pos,
                 const std::array<uint8_t, MAX_PLANETS>& check_collision) {
    for (int f = 0; f < state.fleets.count; ++f) {
        if (state.fleets.alive[static_cast<size_t>(f)] == 0) {
            continue;
        }
        const Vec2 start{state.fleets.x[static_cast<size_t>(f)],
                         state.fleets.y[static_cast<size_t>(f)]};
        const double angle = state.fleets.angle[static_cast<size_t>(f)];
        const double speed = state.fleets.speed[static_cast<size_t>(f)];
        const Vec2 end{start.x + std::cos(angle) * speed, start.y + std::sin(angle) * speed};

        bool removed = false;
        for (int p = 0; p < state.planets.count; ++p) {
            if (state.planets.alive[static_cast<size_t>(p)] == 0 ||
                check_collision[static_cast<size_t>(p)] == 0) {
                continue;
            }
            if (swept_pair_hit(start, end, old_pos[static_cast<size_t>(p)],
                               next_pos[static_cast<size_t>(p)],
                               state.planets.radius[static_cast<size_t>(p)])) {
                queue.add(p, state.fleets.owner[static_cast<size_t>(f)],
                          state.fleets.ships[static_cast<size_t>(f)]);
                state.fleets.remove(f);
                removed = true;
                break;
            }
        }
        if (removed) {
            continue;
        }
        if (!detail::inside_board(end)) {
            state.fleets.remove(f);
            continue;
        }
        if (point_to_segment_distance(Vec2{CENTER_X, CENTER_Y}, start, end) < SUN_RADIUS) {
            state.fleets.remove(f);
            continue;
        }
        state.fleets.x[static_cast<size_t>(f)] = end.x;
        state.fleets.y[static_cast<size_t>(f)] = end.y;
    }
}

/**
 * @brief Compute each planet's tick-end position up front (environment model).
 * @param state Mutable game state.
 * @param old_pos Output planet centers at tick start.
 * @param next_pos Output planet centers at tick end.
 * @param check_collision Output per-planet collision flag (false on first comet placement).
 * @param expired Output per-planet flag for comets whose path ended this tick.
 * @note Orbiting planets move along their orbit chord, comets advance one path
 *       sample, and a comet that finishes its path stays put for the tick so
 *       fleets can still hit it, then is removed after fleet movement.
 */
void compute_planet_motion(GameState& state, std::array<Vec2, MAX_PLANETS>& old_pos,
                           std::array<Vec2, MAX_PLANETS>& next_pos,
                           std::array<uint8_t, MAX_PLANETS>& check_collision,
                           std::array<uint8_t, MAX_PLANETS>& expired) {
    for (int g = 0; g < state.comets.group_count; ++g) {
        ++state.comets.path_index[static_cast<size_t>(g)];
    }
    for (int p = 0; p < state.planets.count; ++p) {
        if (state.planets.alive[static_cast<size_t>(p)] == 0) {
            continue;
        }
        old_pos[static_cast<size_t>(p)] =
            Vec2{state.planets.x[static_cast<size_t>(p)], state.planets.y[static_cast<size_t>(p)]};
        next_pos[static_cast<size_t>(p)] = old_pos[static_cast<size_t>(p)];
        check_collision[static_cast<size_t>(p)] = 1;
        if (state.planets.is_comet[static_cast<size_t>(p)] != 0) {
            const int group = state.planets.comet_group[static_cast<size_t>(p)];
            const int slot = state.planets.comet_slot[static_cast<size_t>(p)];
            if (group >= 0 && slot >= 0) {
                const int flat = state.comets.slot_index(group, slot);
                const int len = state.comets.path_len[static_cast<size_t>(flat)];
                const int idx = state.comets.path_index[static_cast<size_t>(group)];
                if (idx >= 0 && idx < len) {
                    const int path = state.comets.path_index_flat(group, slot, idx);
                    next_pos[static_cast<size_t>(p)] =
                        Vec2{state.comets.path_x[static_cast<size_t>(path)],
                             state.comets.path_y[static_cast<size_t>(path)]};
                    /* First placement starts from an off-board placeholder, so
                     * fleets must not be able to collide with it this tick. */
                    check_collision[static_cast<size_t>(p)] =
                        old_pos[static_cast<size_t>(p)].x >= 0.0 ? 1U : 0U;
                } else {
                    expired[static_cast<size_t>(p)] = 1;
                }
            }
        } else if (state.planets.is_orbiting[static_cast<size_t>(p)] != 0) {
            next_pos[static_cast<size_t>(p)] = state.planet_position_after(p, 1.0);
        }
    }
}

/**
 * @brief Update done/winner flags after a completed turn.
 * @param state Mutable game state.
 * @note Active-player checks include both planets and fleets, matching
 *       elimination semantics.
 */
void update_terminal(GameState& state) {
    int active_count = 0;
    int last_owner = -1;
    for (int owner = 0; owner < MAX_PLAYERS; ++owner) {
        if (state.is_active_player(owner)) {
            ++active_count;
            last_owner = owner;
        }
    }
    if (state.step >= EPISODE_STEPS - 1 || active_count <= 1) {
        state.done = true;
        state.winner = active_count == 1 ? last_owner : -1;
    }
}

}  // namespace

/**
 * @brief Reset the simulator state.
 */
void OrbitSim::reset() {
    state.reset();
}

/**
 * @brief Load a new observation into the simulator.
 * @param obs Fixed-buffer observation input.
 */
void OrbitSim::load_from_observation(const ObservationInput& obs) {
    state.load_from_observation(obs);
}

/**
 * @brief Advance the game by one rule-ordered tick.
 * @param launches Joint launch list to apply before production.
 * @note The sequence is intentionally explicit so search rollouts remain
 *       faithful to the competition rule ordering.
 */
void OrbitSim::step(const LaunchList& launches) {
    if (state.done) {
        return;
    }

    expire_comets(state);

    process_launches(state, launches);
    produce(state);

    std::array<Vec2, MAX_PLANETS> old_pos{};
    std::array<Vec2, MAX_PLANETS> next_pos{};
    std::array<uint8_t, MAX_PLANETS> check_collision{};
    std::array<uint8_t, MAX_PLANETS> expired{};
    compute_planet_motion(state, old_pos, next_pos, check_collision, expired);

    CombatQueue queue{};
    queue.clear();
    move_fleets(state, queue, old_pos, next_pos, check_collision);

    for (int p = 0; p < state.planets.count; ++p) {
        if (state.planets.alive[static_cast<size_t>(p)] == 0) {
            continue;
        }
        state.planets.x[static_cast<size_t>(p)] = next_pos[static_cast<size_t>(p)].x;
        state.planets.y[static_cast<size_t>(p)] = next_pos[static_cast<size_t>(p)].y;
    }

    for (int p = 0; p < state.planets.count; ++p) {
        if (expired[static_cast<size_t>(p)] != 0 && state.planets.alive[static_cast<size_t>(p)] != 0) {
            state.planets.alive[static_cast<size_t>(p)] = 0;
            state.planets.owner[static_cast<size_t>(p)] = -1;
            state.planets.ships[static_cast<size_t>(p)] = 0;
        }
    }

    resolve_planet_combats(state, queue);

    ++state.step;
    update_terminal(state);
}

}  // namespace orbit

