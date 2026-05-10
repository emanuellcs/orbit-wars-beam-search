/**
 * @file search.hpp
 * @brief Root-level fixed-buffer beam-search configuration and entrypoint.
 *
 * Search evaluates a bounded set of packed macro-actions with deterministic
 * rollouts. The parameters are clamped at runtime to keep CPU and memory usage
 * predictable inside Kaggle's per-turn deadline.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit {

/// @brief Tunable native search limits.
struct SearchConfig {
    ///< Maximum root macro-actions to evaluate.
    int beam_width = 384;
    ///< Deterministic tactical prefix depth after the root action.
    int search_depth = 8;
    ///< Additional rollout ticks after the tactical prefix.
    int rollout_horizon = 64;
    ///< Native hard deadline in milliseconds.
    int hard_stop_ms = 900;
};

/// @brief Select a launch list for the current state by evaluating macro-actions.
/// @param state Current game state.
/// @param config Requested search configuration; values are clamped internally.
/// @param time_budget_ms External time budget in milliseconds.
/// @param seed Deterministic seed used for worker branch mixing.
/// @return Chosen LaunchList, or a legal fallback if no rollout finishes.
/// @note Uses fixed arrays plus bounded worker threads rather than dynamic task queues.
LaunchList beam_search_action(const GameState& state, const SearchConfig& config,
                              int time_budget_ms, uint64_t seed);

}  // namespace orbit
