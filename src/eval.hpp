/**
 * @file eval.hpp
 * @brief Heuristic state evaluator used by rollout search and debugging.
 *
 * Evaluation condenses material, production, territory, incoming threat, and
 * comet opportunity into one deterministic score. It is deliberately cheap so
 * many fixed-buffer rollouts can finish before the action deadline.
 *
 * @note The tunable ``EvalWeights`` bundle is declared in ``orbit_engine.hpp``
 *       so that ``Engine`` can hold it by value without a circular include.
 */
#pragma once

#include "orbit_engine.hpp"

namespace orbit {

/// @brief Score a game state from a player's perspective with default weights.
/// @param state Current or rolled-out game state.
/// @param player Player id to evaluate, or negative to use state.player.
/// @return Higher-is-better heuristic score.
/// @note Convenience overload; equivalent to evaluate_state(state, player, EvalWeights{}).
double evaluate_state(const GameState& state, int player);

/// @brief Score a game state from a player's perspective with custom weights.
/// @param state Current or rolled-out game state.
/// @param player Player id to evaluate, or negative to use state.player.
/// @param weights Tunable coefficient bundle; defaults reproduce the prior code.
/// @return Higher-is-better heuristic score.
/// @note The formula blends ships, production, territory, incoming threat, and
///       comet value to guide root macro-action selection.
double evaluate_state(const GameState& state, int player, const EvalWeights& weights);

}  // namespace orbit
