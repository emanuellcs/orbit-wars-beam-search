/**
 * @file orbit_engine_search.cpp
 * @brief Root-parallel fixed-buffer macro-action search.
 *
 * The search evaluates a bounded ranked frontier of macro-actions, simulates a
 * deterministic tactical prefix plus rollout for each, and returns the best
 * legal launch list before the hard deadline.
 */
#include "search.hpp"

#include "candidate.hpp"
#include "eval.hpp"
#include "orbit_engine_internal.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <thread>

namespace orbit {
namespace {

/**
 * @brief Append one launch list into another.
 * @param dst Destination launch list.
 * @param src Source launch list.
 * @note LaunchList::add enforces the fixed output capacity.
 */
void append_launches(LaunchList& dst, const LaunchList& src) {
    for (int i = 0; i < src.count; ++i) {
        const Launch& launch = src.launches[static_cast<size_t>(i)];
        dst.add(launch.from_planet_id, launch.angle, launch.ships);
    }
}

/**
 * @brief Add deterministic actions for all active owners except one.
 * @param state Current game state.
 * @param joint Joint launch list to append into.
 * @param skip_owner Owner id whose actions are already supplied.
 * @note Used to make the root candidate face plausible opponent pressure.
 */
void fill_deterministic_joint(const GameState& state, LaunchList& joint, int skip_owner) {
    for (int owner = 0; owner < MAX_PLAYERS; ++owner) {
        if (owner == skip_owner || !state.is_active_player(owner)) {
            continue;
        }
        deterministic_launches_for_owner(state, owner, joint);
    }
}

/**
 * @brief Roll out and score one macro-action from the root state.
 * @param root Root game state.
 * @param player Controlled player id.
 * @param macro Candidate macro-action to evaluate.
 * @param config Clamped search configuration.
 * @param seed Deterministic branch seed.
 * @return Heuristic score after deterministic simulation.
 * @note GameState copies are value copies of fixed arrays, avoiding runtime
 *       allocation while allowing each worker to mutate an independent state.
 */
double evaluate_macro(const GameState& root, int player, const MacroAction& macro,
                      const SearchConfig& config, uint64_t seed) {
    (void)seed;
    OrbitSim sim{};
    sim.state = root;

    LaunchList joint{};
    joint.clear();
    append_launches(joint, macro.launches);
    fill_deterministic_joint(sim.state, joint, player);
    sim.step(joint);

    for (int depth = 1; depth < config.search_depth && !sim.state.done; ++depth) {
        joint.clear();
        for (int owner = 0; owner < MAX_PLAYERS; ++owner) {
            if (sim.state.is_active_player(owner)) {
                deterministic_launches_for_owner(sim.state, owner, joint);
            }
        }
        sim.step(joint);
    }

    for (int tick = 0; tick < config.rollout_horizon && !sim.state.done; ++tick) {
        joint.clear();
        for (int owner = 0; owner < MAX_PLAYERS; ++owner) {
            if (sim.state.is_active_player(owner)) {
                deterministic_launches_for_owner(sim.state, owner, joint);
            }
        }
        sim.step(joint);
    }

    return evaluate_state(sim.state, player) + macro.score * 0.05;
}

/**
 * @brief Produce a legal fallback action from the highest-priority macro.
 * @param macros Packed macro-action list.
 * @return Launch list from the best macro, or empty if no macro exists.
 * @note This guards timeout cases so the agent still returns validated actions.
 */
LaunchList validated_fallback(const MacroActionList& macros) {
    LaunchList out{};
    out.clear();
    if (macros.count > 0) {
        append_launches(out, macros.items[0].launches);
    }
    return out;
}

}  // namespace

/**
 * @brief Choose a launch list by evaluating bounded root macro-actions.
 * @param state Current game state.
 * @param requested_config Search configuration requested by the caller.
 * @param time_budget_ms External budget in milliseconds.
 * @param seed Deterministic seed used to mix worker/candidate ids.
 * @return Best-scoring launch list or a legal fallback.
 * @note Worker count, beam width, depth, and horizon are clamped to preserve the
 *       fixed-buffer and deadline contracts.
 */
LaunchList beam_search_action(const GameState& state, const SearchConfig& requested_config,
                              int time_budget_ms, uint64_t seed) {
    SearchConfig config = requested_config;
    config.beam_width = std::min(std::max(1, config.beam_width), MAX_BEAM_WIDTH);
    config.search_depth = std::min(std::max(1, config.search_depth), 10);
    config.rollout_horizon = std::min(std::max(1, config.rollout_horizon), 96);
    const int budget = std::min(time_budget_ms > 0 ? time_budget_ms : config.hard_stop_ms,
                                config.hard_stop_ms);

    AtomicLaunchList atoms{};
    MacroActionList macros{};
    generate_atomic_launches(state, state.player, atoms);
    pack_macro_actions(state, state.player, atoms, macros);
    LaunchList fallback = validated_fallback(macros);
    if (macros.count <= 1 || budget <= 2) {
        return fallback;
    }

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::milliseconds(std::max(1, budget - 5));
    const int candidate_count = std::min(macros.count, std::min(config.beam_width, MAX_MACRO_ACTIONS));
    std::array<double, MAX_MACRO_ACTIONS> scores{};
    scores.fill(-1.0e100);

    auto worker = [&](int worker_id, std::atomic<int>& cursor) {
        while (true) {
            const int index = cursor.fetch_add(1);
            if (index >= candidate_count) {
                break;
            }
            // Check every eight candidates to keep the hot loop cheap while
            // still respecting the 900 ms action deadline with a small margin.
            if ((index & 7) == 0 && std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            scores[static_cast<size_t>(index)] =
                evaluate_macro(state, state.player, macros.items[static_cast<size_t>(index)],
                               config, detail::mix64(seed + static_cast<uint64_t>(worker_id * 4099 + index)));
        }
    };

    std::atomic<int> cursor{0};
    int thread_count = std::min(MAX_SEARCH_THREADS, candidate_count);
    if (thread_count <= 1) {
        worker(0, cursor);
    } else {
        std::array<std::thread, MAX_SEARCH_THREADS> threads{};
        for (int t = 0; t < thread_count; ++t) {
            threads[static_cast<size_t>(t)] = std::thread(worker, t, std::ref(cursor));
        }
        for (int t = 0; t < thread_count; ++t) {
            if (threads[static_cast<size_t>(t)].joinable()) {
                threads[static_cast<size_t>(t)].join();
            }
        }
    }

    int best = 0;
    double best_score = scores[0];
    for (int i = 1; i < candidate_count; ++i) {
        if (scores[static_cast<size_t>(i)] > best_score) {
            best_score = scores[static_cast<size_t>(i)];
            best = i;
        }
    }
    if (best_score <= -1.0e90) {
        return fallback;
    }
    LaunchList out{};
    out.clear();
    append_launches(out, macros.items[static_cast<size_t>(best)].launches);
    return out;
}

}  // namespace orbit
