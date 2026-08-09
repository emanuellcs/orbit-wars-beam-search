#include "candidate.hpp"

#include "eval.hpp"
#include "geometry.hpp"
#include "orbit_engine_internal.hpp"

#include <algorithm>
#include <cmath>

namespace orbit {
namespace {

/**
 * @brief Build and insert one atomic launch packet when it is legal and reachable.
 * @param state Current game state.
 * @param player Controlled player id.
 * @param source Source planet SoA index.
 * @param target Target planet SoA index.
 * @param ships Ships assigned to the packet.
 * @param kind Tactical packet category.
 * @param weights Tunable scoring weights.
 * @param out Score-sorted atomic launch output.
 * @note solve_intercept supplies tau and heading so search evaluates meaningful
 *       geometry rather than wasting macro slots on sampled angles.
 */
void add_packet(const GameState& state, int player, int source, int target, int ships_hint,
                PacketKind kind, const CandidateWeights& weights, AtomicLaunchList& out) {
    if (source == target ||
        state.planets.alive[static_cast<size_t>(source)] == 0 ||
        state.planets.alive[static_cast<size_t>(target)] == 0) {
        return;
    }
    const int available = state.planets.ships[static_cast<size_t>(source)];
    if (available <= 0) {
        return;
    }
    const int garrison = state.planets.ships[static_cast<size_t>(target)];
    const int production = state.planets.production[static_cast<size_t>(target)];

    /* Solve the arrival time with a representative size first, then size the
     * capture against the garrison that will exist on arrival (production is
     * added every tick the planet is owned).  Sizing matters: a garrison+1
     * launch usually arrives one production tick too small. */
    const int guess = std::max(1, ships_hint > 0 ? ships_hint : garrison + 1);
    double eta = 0.0;
    double angle = 0.0;
    if (!solve_intercept(state, source, target, guess, eta, angle)) {
        return;
    }
    int ships = 0;
    if (kind == PacketKind::CaptureExact) {
        ships = garrison + production * static_cast<int>(std::ceil(eta)) + 1;
    } else if (kind == PacketKind::CaptureOver) {
        ships = garrison + production * static_cast<int>(std::ceil(eta)) + 4 * production + 2;
    } else if (kind == PacketKind::Harass) {
        ships = 1;
    } else {  // Reinforce or AllSafe: caller supplies the exact size.
        ships = ships_hint;
    }
    if (ships <= 0 || ships > available) {
        return;
    }
    if (ships != guess && !solve_intercept(state, source, target, ships, eta, angle)) {
        return;
    }

    const int target_owner = state.planets.owner[static_cast<size_t>(target)];
    const double prod = static_cast<double>(state.planets.production[static_cast<size_t>(target)]);
    const double owner_bonus = kind == PacketKind::Reinforce
                                   ? 0.0
                                   : (target_owner < 0
                                          ? weights.owner_neutral
                                          : (target_owner == player ? weights.owner_self : weights.owner_enemy));
    const double comet_bonus = state.planets.is_comet[static_cast<size_t>(target)] != 0
                                   ? weights.comet_bonus
                                   : 0.0;
    const double kind_bonus =
        kind == PacketKind::CaptureExact ? weights.kind_exact :
        kind == PacketKind::CaptureOver ? weights.kind_over :
        kind == PacketKind::AllSafe ? weights.kind_all_safe :
        kind == PacketKind::Reinforce ? weights.kind_reinforce : weights.kind_harass;
    AtomicLaunch launch{};
    launch.from_planet_id = state.planets.id[static_cast<size_t>(source)];
    launch.source_index = source;
    launch.target_index = target;
    launch.ships = ships;
    launch.angle = angle;
    launch.eta = eta;
    launch.kind = kind;
    // Reinforcement urgency scales with the deficit (more incoming ships means a
    // bigger send and a higher prior); regular packets keep the production/
    // ownership/ETA trade-off. Rollout evaluation remains the final arbiter.
    if (kind == PacketKind::Reinforce) {
        launch.score = weights.kind_reinforce * (1.0 + static_cast<double>(ships) / 20.0) -
                       eta * weights.eta_discount;
    } else {
        launch.score = owner_bonus + comet_bonus + prod * weights.prod_per_unit + kind_bonus -
                       eta * weights.eta_discount - static_cast<double>(ships) * weights.ship_cost;
    }
    out.insert_sorted(launch);
}

/**
 * @brief Try to add an atomic launch to a macro-action spend ledger.
 * @param state Current game state.
 * @param atom Atomic launch candidate.
 * @param spend Per-source ships already committed by the macro-action.
 * @return true when adding atom would not overspend its source.
 * @note The spend array is indexed by SoA source slot, so no map lookup is needed.
 */
bool macro_legal_add(const GameState& state, const AtomicLaunch& atom,
                     std::array<int, MAX_PLANETS>& spend) {
    const int source = atom.source_index;
    if (source < 0 || source >= state.planets.count) {
        return false;
    }
    const int available = state.planets.ships[static_cast<size_t>(source)];
    if (spend[static_cast<size_t>(source)] + atom.ships > available) {
        return false;
    }
    spend[static_cast<size_t>(source)] += atom.ships;
    return true;
}

}  // namespace

/**
 * @brief Clear the logical atomic launch list.
 */
void AtomicLaunchList::clear() {
    count = 0;
}

/**
 * @brief Insert an atomic launch into descending score order.
 * @param launch Candidate packet to insert.
 * @note Fixed-capacity insertion keeps the best MAX_ATOMIC_LAUNCHES packets and
 *       silently drops lower-priority overflow.
 */
void AtomicLaunchList::insert_sorted(const AtomicLaunch& launch) {
    if (count <= 0) {
        items[0] = launch;
        count = 1;
        return;
    }
    const int limit = std::min(count, MAX_ATOMIC_LAUNCHES - 1);
    int pos = limit;
    while (pos > 0 && items[static_cast<size_t>(pos - 1)].score < launch.score) {
        if (pos < MAX_ATOMIC_LAUNCHES) {
            items[static_cast<size_t>(pos)] = items[static_cast<size_t>(pos - 1)];
        }
        --pos;
    }
    if (pos < MAX_ATOMIC_LAUNCHES) {
        items[static_cast<size_t>(pos)] = launch;
        if (count < MAX_ATOMIC_LAUNCHES) {
            ++count;
        }
    }
}

/**
 * @brief Clear the logical macro-action list.
 */
void MacroActionList::clear() {
    count = 0;
}

/**
 * @brief Insert a macro-action into descending score order.
 * @param action Macro-action to insert.
 * @note O(MAX_MACRO_ACTIONS) worst case but fixed and small relative to rollouts.
 */
void MacroActionList::insert_sorted(const MacroAction& action) {
    const int limit = std::min(count, MAX_MACRO_ACTIONS - 1);
    int pos = limit;
    while (pos > 0 && items[static_cast<size_t>(pos - 1)].score < action.score) {
        if (pos < MAX_MACRO_ACTIONS) {
            items[static_cast<size_t>(pos)] = items[static_cast<size_t>(pos - 1)];
        }
        --pos;
    }
    if (pos < MAX_MACRO_ACTIONS) {
        items[static_cast<size_t>(pos)] = action;
        if (count < MAX_MACRO_ACTIONS) {
            ++count;
        }
    }
}

/**
 * @brief Compute a conservative reserve for one owned source planet.
 * @param state Current game state.
 * @param source_index Source planet SoA index.
 * @param player Controlled player id.
 * @return Ships to leave behind before all-safe launches.
 * @note Enemy fleets are projected over a near-term segment to avoid stripping
 *       a source that is already under direct threat.
 */
int defensive_reserve(const GameState& state, int source_index, int player) {
    int incoming = 0;
    const Vec2 source{state.planets.x[static_cast<size_t>(source_index)],
                      state.planets.y[static_cast<size_t>(source_index)]};
    for (int f = 0; f < state.fleets.count; ++f) {
        if (state.fleets.alive[static_cast<size_t>(f)] == 0 ||
            state.fleets.owner[static_cast<size_t>(f)] == player) {
            continue;
        }
        const Vec2 start{state.fleets.x[static_cast<size_t>(f)], state.fleets.y[static_cast<size_t>(f)]};
        const Vec2 end = point_on_heading(start, state.fleets.angle[static_cast<size_t>(f)],
                                          state.fleets.speed[static_cast<size_t>(f)] * 24.0);
        double t = 0.0;
        if (segment_circle_hit(start, end, source, state.planets.radius[static_cast<size_t>(source_index)], t)) {
            incoming += state.fleets.ships[static_cast<size_t>(f)];
        }
    }
    return std::max(5, std::max(state.planets.production[static_cast<size_t>(source_index)] * 4, incoming + 1));
}

/**
 * @brief Sum enemy ships projected to reach an owned planet within a window.
 * @param state Current game state.
 * @param planet_index Owned planet SoA index.
 * @param player Controlled player id.
 * @param window Forward projection in ticks.
 * @return Incoming enemy ship mass.
 * @note Uses the same segment-circle projection as the evaluator threat term.
 */
int incoming_mass(const GameState& state, int planet_index, int player, double window) {
    int mass = 0;
    const Vec2 center{state.planets.x[static_cast<size_t>(planet_index)],
                      state.planets.y[static_cast<size_t>(planet_index)]};
    for (int f = 0; f < state.fleets.count; ++f) {
        if (state.fleets.alive[static_cast<size_t>(f)] == 0 ||
            state.fleets.owner[static_cast<size_t>(f)] == player) {
            continue;
        }
        const Vec2 start{state.fleets.x[static_cast<size_t>(f)],
                         state.fleets.y[static_cast<size_t>(f)]};
        const Vec2 end = point_on_heading(start, state.fleets.angle[static_cast<size_t>(f)],
                                          state.fleets.speed[static_cast<size_t>(f)] * window);
        double t = 0.0;
        if (segment_circle_hit(start, end, center, state.planets.radius[static_cast<size_t>(planet_index)], t)) {
            mass += state.fleets.ships[static_cast<size_t>(f)];
        }
    }
    return mass;
}

/**
 * @brief Generate all ranked atomic launch candidates for one player.
 * @param state Current game state.
 * @param player Controlled player id.
 * @param weights Tunable scoring weights.
 * @param out Output atomic launch list.
 * @note Defensive reinforcements are generated first so they compete fairly on
 *       the frontier, then exact/over/harass/all-safe packets to non-owned
 *       targets. Capture sizes are time-aware (garrison + production * tau).
 */
void generate_atomic_launches(const GameState& state, int player,
                              const CandidateWeights& weights, AtomicLaunchList& out) {
    out.clear();

    for (int target = 0; target < state.planets.count; ++target) {
        if (state.planets.alive[static_cast<size_t>(target)] == 0 ||
            state.planets.owner[static_cast<size_t>(target)] != player) {
            continue;
        }
        const int incoming = incoming_mass(state, target, player, 24.0);
        const int garrison = state.planets.ships[static_cast<size_t>(target)];
        const int deficit = incoming - garrison + 1;
        if (deficit <= 0) {
            continue;
        }
        for (int source = 0; source < state.planets.count; ++source) {
            if (source == target ||
                state.planets.alive[static_cast<size_t>(source)] == 0 ||
                state.planets.owner[static_cast<size_t>(source)] != player) {
                continue;
            }
            const int reserve = defensive_reserve(state, source, player);
            if (state.planets.ships[static_cast<size_t>(source)] - reserve >= deficit) {
                add_packet(state, player, source, target, deficit, PacketKind::Reinforce, weights, out);
            }
        }
    }

    for (int source = 0; source < state.planets.count; ++source) {
        if (state.planets.alive[static_cast<size_t>(source)] == 0 ||
            state.planets.owner[static_cast<size_t>(source)] != player) {
            continue;
        }
        const int available = state.planets.ships[static_cast<size_t>(source)];
        const int reserve = defensive_reserve(state, source, player);
        const int all_safe = std::max(0, available - reserve);
        for (int target = 0; target < state.planets.count; ++target) {
            if (state.planets.alive[static_cast<size_t>(target)] == 0 ||
                state.planets.owner[static_cast<size_t>(target)] == player) {
                continue;
            }
            add_packet(state, player, source, target, 0, PacketKind::CaptureExact, weights, out);
            add_packet(state, player, source, target, 0, PacketKind::CaptureOver, weights, out);
            add_packet(state, player, source, target, 1, PacketKind::Harass, weights, out);
            add_packet(state, player, source, target, all_safe, PacketKind::AllSafe, weights, out);
        }
    }
}

/**
 * @brief Pack atomic packets into bounded, legal macro-actions.
 * @param state Current game state.
 * @param player Controlled player id.
 * @param atoms Score-sorted atomic packet list.
 * @param weights Tunable scoring weights.
 * @param eval_weights Tunable evaluator weights used for the idle prior.
 * @param out Output macro-action list.
 * @note Includes idle, single-packet actions, and greedy bundles while enforcing
 *       source spend constraints for every bundle.
 */
void pack_macro_actions(const GameState& state, int player, const AtomicLaunchList& atoms,
                        const CandidateWeights& weights, const EvalWeights& eval_weights,
                        MacroActionList& out) {
    (void)weights;  ///< per-launch scores are already baked into atoms.items[].score
    out.clear();
    MacroAction idle{};
    idle.launches.clear();
    idle.score = evaluate_state(state, player, eval_weights) * 0.001;
    out.insert_sorted(idle);

    const int single_limit = std::min(atoms.count, 128);
    for (int i = 0; i < single_limit; ++i) {
        const AtomicLaunch& atom = atoms.items[static_cast<size_t>(i)];
        MacroAction action{};
        action.launches.clear();
        action.launches.add(atom.from_planet_id, atom.angle, atom.ships);
        action.score = atom.score;
        out.insert_sorted(action);
    }

    const int start_limit = std::min(atoms.count, 64);
    for (int start = 0; start < start_limit; ++start) {
        MacroAction action{};
        action.launches.clear();
        action.score = 0.0;
        std::array<int, MAX_PLANETS> spend{};
        spend.fill(0);
        for (int i = start; i < atoms.count && action.launches.count < MAX_LAUNCHES; ++i) {
            const AtomicLaunch& atom = atoms.items[static_cast<size_t>(i)];
            if (!macro_legal_add(state, atom, spend)) {
                continue;
            }
            action.launches.add(atom.from_planet_id, atom.angle, atom.ships);
            action.score += atom.score;
            if (action.launches.count >= 4) {
                break;
            }
        }
        if (action.launches.count > 1) {
            out.insert_sorted(action);
        }
    }
}

/**
 * @brief Append deterministic launches for one owner.
 * @param state Current game state.
 * @param owner Owner id to act for.
 * @param weights Tunable scoring weights.
 * @param out Launch list to append into.
 * @note Search uses this policy for opponents during rollouts so branch scores
 *       are deterministic and reproducible across worker threads.
 */
void deterministic_launches_for_owner(const GameState& state, int owner,
                                      const CandidateWeights& weights, LaunchList& out) {
    AtomicLaunchList atoms{};
    MacroActionList macros{};
    generate_atomic_launches(state, owner, weights, atoms);
    pack_macro_actions(state, owner, atoms, weights, EvalWeights{}, macros);
    if (macros.count <= 0) {
        return;
    }
    const MacroAction& best = macros.items[0];
    for (int i = 0; i < best.launches.count; ++i) {
        const Launch& launch = best.launches.launches[static_cast<size_t>(i)];
        out.add(launch.from_planet_id, launch.angle, launch.ships);
    }
}

}  // namespace orbit

