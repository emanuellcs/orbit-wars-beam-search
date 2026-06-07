/**
 * @file search.hpp
 * @brief Root-level fixed-buffer beam-search configuration and entrypoint.
 *
 * Search evaluates a bounded set of packed macro-actions with deterministic
 * rollouts. The parameters are clamped at runtime to keep CPU and memory usage
 * predictable inside Kaggle's per-turn deadline.
 *
 * @note ``SearchConfig`` itself is declared in ``orbit_engine.hpp`` so that
 *       ``Engine`` can hold a value-typed member without a circular include.
 *       This header owns the search entrypoint and the runtime thread cap.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit {

/// @brief Set the maximum number of search worker threads per process.
/// @param n Requested thread count; clamped to [1, MAX_SEARCH_THREADS].
/// @note Defaults to MAX_SEARCH_THREADS; tune.py forces this to 1 to avoid
///       oversubscribing the host when n_jobs parallel trials are launched.
void set_search_thread_limit(int n);
/// @brief Return the currently configured search worker thread cap.
/// @return Effective thread limit applied to beam_search_action.
int search_thread_limit();

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
