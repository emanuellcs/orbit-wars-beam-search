/**
 * @file orbit_engine_candidate.cpp
 * @brief Tactical launch generation and legal macro-action packing.
 *
 * The search does not sample arbitrary headings. It first creates analytically
 * solved launch packets, ranks them by tactical value, then packs legal
 * multi-launch macro-actions with per-source spend accounting.
 */
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
 * @param out Score-sorted atomic launch output.
 * @note solve_intercept supplies tau and heading so search evaluates meaningful
 *       geometry rather than wasting macro slots on sampled angles.
 */
void add_packet(const GameState& state, int player, int source, int target, int ships,
                PacketKind kind, AtomicLaunchList& out) {
    if (ships <= 0 || source == target ||
        state.planets.alive[static_cast<size_t>(source)] == 0 ||
        state.planets.alive[static_cast<size_t>(target)] == 0) {
        return;
    }
    const int available = state.planets.ships[static_cast<size_t>(source)];
    if (ships > available) {
        return;
    }
    double eta = 0.0;
    double angle = 0.0;
    if (!solve_intercept(state, source, target, ships, eta, angle)) {
        return;
    }
    const int target_owner = state.planets.owner[static_cast<size_t>(target)];
    const double prod = static_cast<double>(state.planets.production[static_cast<size_t>(target)]);
    const double owner_bonus = target_owner < 0 ? 18.0 : (target_owner == player ? -40.0 : 42.0);
    const double comet_bonus = state.planets.is_comet[static_cast<size_t>(target)] != 0 ? 14.0 : 0.0;
    const double kind_bonus =
        kind == PacketKind::CaptureExact ? 10.0 :
        kind == PacketKind::CaptureOver ? 16.0 :
        kind == PacketKind::AllSafe ? 8.0 : 2.0;
    AtomicLaunch launch{};
    launch.from_planet_id = state.planets.id[static_cast<size_t>(source)];
    launch.source_index = source;
    launch.target_index = target;
    launch.ships = ships;
    launch.angle = angle;
    launch.eta = eta;
    launch.kind = kind;
    // This prior prefers production and enemy/comet targets, then discounts
    // slow arrivals and expensive launches. Rollout evaluation remains the
    // final arbiter; the prior only controls the fixed candidate frontier.
    launch.score = owner_bonus + comet_bonus + prod * 24.0 + kind_bonus -
                   eta * 0.8 - static_cast<double>(ships) * 0.08;
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
 * @brief Generate all ranked atomic launch candidates for one player.
 * @param state Current game state.
 * @param player Controlled player id.
 * @param out Output atomic launch list.
 * @note The packet basis is deliberately small: exact capture, over-capture,
 *       harassment probes, and all-safe pressure from each owned source.
 */
void generate_atomic_launches(const GameState& state, int player, AtomicLaunchList& out) {
    out.clear();
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
            const int garrison = state.planets.ships[static_cast<size_t>(target)];
            const int exact = garrison + 1;
            add_packet(state, player, source, target, exact, PacketKind::CaptureExact, out);

            const int slack = state.planets.production[static_cast<size_t>(target)] * 4 + 2;
            add_packet(state, player, source, target, exact + slack, PacketKind::CaptureOver, out);

            add_packet(state, player, source, target, 1, PacketKind::Harass, out);
            add_packet(state, player, source, target, 3, PacketKind::Harass, out);
            add_packet(state, player, source, target, 5, PacketKind::Harass, out);
            add_packet(state, player, source, target, std::min(10, available / 4), PacketKind::Harass, out);
            add_packet(state, player, source, target, all_safe, PacketKind::AllSafe, out);
        }
    }
}

/**
 * @brief Pack atomic packets into bounded, legal macro-actions.
 * @param state Current game state.
 * @param player Controlled player id.
 * @param atoms Score-sorted atomic packet list.
 * @param out Output macro-action list.
 * @note Includes idle, single-packet actions, and greedy bundles while enforcing
 *       source spend constraints for every bundle.
 */
void pack_macro_actions(const GameState& state, int player, const AtomicLaunchList& atoms,
                        MacroActionList& out) {
    out.clear();
    MacroAction idle{};
    idle.launches.clear();
    idle.score = evaluate_state(state, player) * 0.001;
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
 * @param out Launch list to append into.
 * @note Search uses this policy for opponents during rollouts so branch scores
 *       are deterministic and reproducible across worker threads.
 */
void deterministic_launches_for_owner(const GameState& state, int owner, LaunchList& out) {
    AtomicLaunchList atoms{};
    MacroActionList macros{};
    generate_atomic_launches(state, owner, atoms);
    pack_macro_actions(state, owner, atoms, macros);
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
